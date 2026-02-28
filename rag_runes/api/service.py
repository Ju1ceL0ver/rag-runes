from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Any

from rag_runes.api.config import AppConfig
from rag_runes.generation.answer_builder import AnswerBuilder
from rag_runes.index.embeddings import create_embedder
from rag_runes.index.hybrid_index import HybridTreeIndex
from rag_runes.ingest.pdf_ingestor import PdfBookIngestor
from rag_runes.ingest.semantic_segmenter import build_segmenter
from rag_runes.ingest.tree_builder import TreeBuilder
from rag_runes.ocr.glm_ocr import build_ocr_client
from rag_runes.retrieval.reranker import create_reranker
from rag_runes.retrieval.tree_search import TreeSearcher

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    basename = Path(filename).name
    candidate = SAFE_NAME_RE.sub("_", basename).strip("._")
    if not candidate:
        token = hashlib.blake2b(basename.encode("utf-8"), digest_size=4).hexdigest()
        candidate = f"book_{token}.pdf"
    if not candidate.lower().endswith(".pdf"):
        candidate = f"{candidate}.pdf"
    return candidate


class RagPipelineService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.books_dir.mkdir(parents=True, exist_ok=True)
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        self.config.images_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._index: HybridTreeIndex | None = None
        self._errors: list[str] = []

        self.embedder = self._safe_init(
            "embedder",
            lambda: create_embedder(self.config.embedder_name),
        )
        self.ocr_client = self._safe_init(
            "ocr",
            lambda: build_ocr_client(
                endpoint=self.config.ocr_endpoint,
                model=self.config.ocr_model,
                api_key=self.config.ocr_api_key,
            ),
        )
        self.segmenter = self._safe_init(
            "segmenter",
            lambda: build_segmenter(
                mode=self.config.segmenter_mode,
                endpoint=self.config.segmenter_endpoint,
                model=self.config.segmenter_model,
                api_key=self.config.segmenter_api_key,
            ),
        )
        self.default_reranker = self._safe_init(
            "reranker",
            lambda: create_reranker(self.config.reranker_name),
        )
        self.answer_builder = self._safe_init(
            "answer",
            lambda: AnswerBuilder(
                endpoint=self.config.answer_endpoint,
                model=self.config.answer_model,
                api_key=self.config.answer_api_key,
                per_hit_chars=self.config.context_chars,
            ),
        )

        self.ingestor = (
            PdfBookIngestor(ocr_client=self.ocr_client)
            if self.ocr_client is not None
            else None
        )
        self.tree_builder = (
            TreeBuilder(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                segmenter=self.segmenter,
            )
            if self.segmenter is not None
            else None
        )

    def _safe_init(self, component: str, factory):
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"{component}: {exc}")
            return None

    def _require_ingest_ready(self) -> None:
        if self.ingestor is None or self.tree_builder is None or self.embedder is None:
            if self._errors:
                raise RuntimeError("Models are not ready: " + "; ".join(self._errors))
            raise RuntimeError("Ingest pipeline is not ready.")

    def _require_query_ready(self) -> None:
        if (
            self.embedder is None
            or self.default_reranker is None
            or self.answer_builder is None
        ):
            if self._errors:
                raise RuntimeError("Models are not ready: " + "; ".join(self._errors))
            raise RuntimeError("Query pipeline is not ready.")

    def _index_files_exist(self) -> bool:
        required = [
            "nodes.jsonl",
            "node_ids.json",
            "bm25_tokens.json",
            "meta.json",
            "vectors.npy",
        ]
        return all((self.config.index_dir / name).exists() for name in required)

    def _load_index(self) -> None:
        if not self._index_files_exist():
            self._index = None
            return
        self._index = HybridTreeIndex.load(
            index_dir=self.config.index_dir,
            embedder=self.embedder,
        )

    def initialize(self) -> None:
        with self._lock:
            if self.embedder is None:
                return
            self._load_index()
            if self._errors:
                return
            if self._index is None and self.config.bootstrap_on_startup:
                pdf_files = sorted(self.config.books_dir.glob("*.pdf"))
                if pdf_files:
                    try:
                        self._rebuild_from_pdfs(pdf_files)
                    except Exception as exc:  # noqa: BLE001
                        self._errors.append(f"bootstrap_ingest: {exc}")
                        self._index = None

    def _rebuild_from_pdfs(self, pdf_files: list[Path]) -> HybridTreeIndex:
        self._require_ingest_ready()
        all_nodes = []
        for pdf_path in pdf_files:
            artifact = self.ingestor.ingest(
                pdf_path=pdf_path,
                image_out_dir=self.config.images_dir,
            )
            all_nodes.extend(self.tree_builder.build(artifact))
        index = HybridTreeIndex.build(
            nodes=all_nodes,
            embedder=self.embedder,
            embedder_name=self.config.embedder_name,
        )
        index.save(self.config.index_dir)
        self._index = index
        return index

    def ingest_upload(self, filename: str, content: bytes) -> dict[str, Any]:
        if not content:
            raise ValueError("Uploaded file is empty")
        self._require_ingest_ready()

        safe_filename = sanitize_filename(filename)
        destination = self.config.books_dir / safe_filename
        destination.write_bytes(content)

        with self._lock:
            artifact = self.ingestor.ingest(
                pdf_path=destination,
                image_out_dir=self.config.images_dir,
            )
            new_nodes = self.tree_builder.build(artifact)

            old_nodes = self._index.nodes if self._index is not None else []
            merged_nodes = [node for node in old_nodes if node.book_id != artifact.book_id]
            merged_nodes.extend(new_nodes)

            index = HybridTreeIndex.build(
                nodes=merged_nodes,
                embedder=self.embedder,
                embedder_name=self.config.embedder_name,
            )
            index.save(self.config.index_dir)
            self._index = index

        return {
            "status": "ok",
            "book_id": artifact.book_id,
            "filename": safe_filename,
            "pages": len(artifact.pages),
            "images": len(artifact.images),
            "book_nodes": len(new_nodes),
            "total_nodes": len(self._index.nodes) if self._index else 0,
            "indexable_nodes": len(self._index.node_ids) if self._index else 0,
        }

    def query(
        self,
        question: str,
        top_k: int,
        branch_factor: int,
        reranker_name: str | None = None,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question is empty")
        self._require_query_ready()

        with self._lock:
            index = self._index
        if index is None:
            raise RuntimeError("Index is not ready. Upload at least one PDF first.")

        searcher = TreeSearcher(index=index)
        hits = searcher.retrieve(
            query=question,
            top_k=top_k,
            branch_factor=branch_factor,
        )

        reranker = (
            create_reranker(reranker_name)
            if reranker_name and reranker_name.strip()
            else self.default_reranker
        )
        reranked = reranker.rerank(question, hits)

        bundle = self.answer_builder.build(question, reranked)
        payload = bundle.to_dict()
        payload["params"] = {
            "top_k": top_k,
            "branch_factor": branch_factor,
            "reranker": reranker_name or self.config.reranker_name,
        }
        return payload

    def health(self) -> dict[str, Any]:
        with self._lock:
            index = self._index
            total_nodes = len(index.nodes) if index else 0
            indexable_nodes = len(index.node_ids) if index else 0

        return {
            "status": "ok" if not self._errors else "error",
            "index_ready": index is not None,
            "books_on_disk": len(list(self.config.books_dir.glob("*.pdf"))),
            "total_nodes": total_nodes,
            "indexable_nodes": indexable_nodes,
            "errors": list(self._errors),
            "config": {
                "books_dir": str(self.config.books_dir),
                "index_dir": str(self.config.index_dir),
                "embedder_name": self.config.embedder_name,
                "reranker_name": self.config.reranker_name,
                "segmenter_mode": self.config.segmenter_mode,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "bootstrap_on_startup": self.config.bootstrap_on_startup,
            },
        }
