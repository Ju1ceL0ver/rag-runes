from __future__ import annotations

import json
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

        if contains_rune_query(query):
            for idx, pos in enumerate(positions):
                node = self.node_by_id[self.node_ids[pos]]
                rune_density = float(node.metadata.get("rune_density", 0.0))
                has_images = 1.0 if node.image_ids else 0.0
                rune_boost[idx] = rune_weight * min(1.0, rune_density + 0.1 * has_images)

        scores = dense_weight * dense + sparse_weight * sparse + rune_boost
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
