from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag_runes.index.embeddings import TextEmbedder
from rag_runes.schema import DocNode
from rag_runes.text_utils import contains_rune_query, tokenize


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def _index_text(node: DocNode) -> str:
    return "\n".join(part for part in [node.title, node.text] if part).strip()


TECHNICAL_CATALOG_QUERY_RE = re.compile(
    r"\b(code|codepoint|unicode|ucs|smp|letter|character|chart|table|"
    r"symbol|block|encoding|properties|код|юникод|символ|букв|таблиц)\b",
    flags=re.IGNORECASE,
)

CODE_CHART_RE = re.compile(r"\b1xx[0-9a-f]{1,2}\b|KHAZARIAN ROVAS LETTER", re.IGNORECASE)
ADMIN_FORM_RE = re.compile(
    r"\b(appendix|proposal summary form|administrative|requester|reference|"
    r"choose one of the following|submitted before|iso/iec 10646|"
    r"приложение|административ)\b",
    flags=re.IGNORECASE,
)
LOW_VALUE_SECTION_RE = re.compile(
    r"\b(contents|bibliography|acknowledgement|references|оглавление|"
    r"библиограф|благодарност)\b",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class SearchHit:
    node_id: str
    score: float
    dense: float
    sparse: float
    rune_boost: float


class HybridTreeIndex:
    def __init__(
        self,
        nodes: list[DocNode],
        node_ids: list[str],
        vectors: np.ndarray,
        bm25_tokens: list[list[str]],
        embedder: TextEmbedder,
        embedder_name: str,
    ) -> None:
        self.nodes = nodes
        self.node_ids = node_ids
        self.vectors = vectors
        self.bm25_tokens = bm25_tokens
        self.embedder = embedder
        self.embedder_name = embedder_name
        self.bm25 = BM25Okapi(bm25_tokens) if bm25_tokens else None

        self.node_by_id: dict[str, DocNode] = {node.node_id: node for node in nodes}
        self.id_to_pos = {node_id: idx for idx, node_id in enumerate(node_ids)}

        self.children: dict[str, list[str]] = {}
        for node in nodes:
            if node.parent_id:
                self.children.setdefault(node.parent_id, []).append(node.node_id)

    def _node_context_text(self, node: DocNode) -> str:
        parts = [node.title, node.text]
        cursor = node
        while cursor.parent_id is not None:
            parent = self.node_by_id.get(cursor.parent_id)
            if parent is None:
                break
            parts.append(parent.title)
            cursor = parent
        return "\n".join(part for part in parts if part)

    def _retrieval_quality_multiplier(self, query: str, node: DocNode) -> float:
        context = self._node_context_text(node)
        technical_query = bool(TECHNICAL_CATALOG_QUERY_RE.search(query or ""))

        if ADMIN_FORM_RE.search(context):
            return 0.25
        if LOW_VALUE_SECTION_RE.search(context):
            return 0.45
        if CODE_CHART_RE.search(context) and not technical_query:
            return 0.25

        text = node.text or ""
        if node.level == "chunk" and text:
            semicolon_count = text.count(";")
            sentence_marks = sum(text.count(mark) for mark in ".!?")
            if semicolon_count >= 5 and sentence_marks <= 1 and not technical_query:
                return 0.60

        return 1.0

    @classmethod
    def build(
        cls,
        nodes: list[DocNode],
        embedder: TextEmbedder,
        embedder_name: str,
    ) -> "HybridTreeIndex":
        indexable_levels = {"chapter", "section", "chunk", "image"}
        indexable = [node for node in nodes if node.level in indexable_levels]
        node_ids = [node.node_id for node in indexable]
        texts = [_index_text(node) for node in indexable]
        vectors = embedder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        bm25_tokens = [tokenize(text) for text in texts]
        return cls(
            nodes=nodes,
            node_ids=node_ids,
            vectors=vectors,
            bm25_tokens=bm25_tokens,
            embedder=embedder,
            embedder_name=embedder_name,
        )

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "nodes.jsonl").open("w", encoding="utf-8") as stream:
            for node in self.nodes:
                stream.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

        with (out_dir / "node_ids.json").open("w", encoding="utf-8") as stream:
            json.dump(self.node_ids, stream, ensure_ascii=False, indent=2)

        with (out_dir / "bm25_tokens.json").open("w", encoding="utf-8") as stream:
            json.dump(self.bm25_tokens, stream, ensure_ascii=False)

        with (out_dir / "meta.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "embedder_name": self.embedder_name,
                    "num_nodes": len(self.nodes),
                    "num_indexable_nodes": len(self.node_ids),
                    "vector_dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )

        np.save(out_dir / "vectors.npy", self.vectors)

    @classmethod
    def load(
        cls,
        index_dir: Path,
        embedder: TextEmbedder,
    ) -> "HybridTreeIndex":
        nodes: list[DocNode] = []
        with (index_dir / "nodes.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                payload = json.loads(line)
                nodes.append(DocNode.from_dict(payload))

        with (index_dir / "node_ids.json").open("r", encoding="utf-8") as stream:
            node_ids = json.load(stream)
        with (index_dir / "bm25_tokens.json").open("r", encoding="utf-8") as stream:
            bm25_tokens = json.load(stream)
        with (index_dir / "meta.json").open("r", encoding="utf-8") as stream:
            meta = json.load(stream)
        vectors = np.load(index_dir / "vectors.npy")

        return cls(
            nodes=nodes,
            node_ids=node_ids,
            vectors=np.asarray(vectors, dtype=np.float32),
            bm25_tokens=bm25_tokens,
            embedder=embedder,
            embedder_name=meta.get("embedder_name", "unknown"),
        )

    def descendants(
        self,
        node_id: str,
        target_levels: set[str] | None = None,
    ) -> list[str]:
        output: list[str] = []
        stack = [node_id]
        while stack:
            current = stack.pop()
            for child_id in self.children.get(current, []):
                child = self.node_by_id[child_id]
                if target_levels is None or child.level in target_levels:
                    output.append(child_id)
                stack.append(child_id)
        return output

    def hybrid_rank(
        self,
        query: str,
        candidate_ids: list[str] | None = None,
        top_k: int = 10,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        rune_weight: float = 0.12,
        quality_weight: float = 0.65,
    ) -> list[SearchHit]:
        if not self.node_ids:
            return []
        if candidate_ids is None:
            positions = np.arange(len(self.node_ids), dtype=np.int32)
        else:
            positions = np.array(
                [self.id_to_pos[node_id] for node_id in candidate_ids if node_id in self.id_to_pos],
                dtype=np.int32,
            )
            if positions.size == 0:
                return []

        query_vector = self.embedder.encode([query])[0]
        dense_raw = np.dot(self.vectors[positions], query_vector)

        if self.bm25 is not None:
            sparse_all = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
            sparse_raw = sparse_all[positions]
        else:
            sparse_raw = np.zeros_like(dense_raw, dtype=np.float32)

        dense = _normalize(dense_raw)
        sparse = _normalize(sparse_raw)
        rune_boost = np.zeros_like(dense, dtype=np.float32)
        quality = np.ones_like(dense, dtype=np.float32)

        for idx, pos in enumerate(positions):
            node = self.node_by_id[self.node_ids[pos]]
            if node.level == "chunk":
                quality[idx] = float(node.metadata.get("chunk_quality", 1.0))
            elif node.level == "image":
                quality[idx] = float(node.metadata.get("image_quality", 0.35))
            else:
                quality[idx] = 1.0
            quality[idx] *= self._retrieval_quality_multiplier(query=query, node=node)

        if contains_rune_query(query):
            for idx, pos in enumerate(positions):
                node = self.node_by_id[self.node_ids[pos]]
                rune_density = float(node.metadata.get("rune_density", 0.0))
                has_images = 1.0 if node.image_ids else 0.0
                rune_boost[idx] = rune_weight * min(1.0, rune_density + 0.1 * has_images)

        base_scores = dense_weight * dense + sparse_weight * sparse + rune_boost
        quality_multiplier = (1.0 - quality_weight) + quality_weight * quality
        quality_multiplier = np.where(
            quality < 0.5,
            quality_multiplier * quality,
            quality_multiplier,
        )
        scores = base_scores * quality_multiplier
        order = np.argsort(-scores)[:top_k]
        output: list[SearchHit] = []
        for rank_idx in order:
            pos = positions[rank_idx]
            output.append(
                SearchHit(
                    node_id=self.node_ids[pos],
                    score=float(scores[rank_idx]),
                    dense=float(dense[rank_idx]),
                    sparse=float(sparse[rank_idx]),
                    rune_boost=float(rune_boost[rank_idx]),
                )
            )
        return output
