from __future__ import annotations

import numpy as np

from rag_runes.retrieval.tree_search import RetrievalHit


class Reranker:
    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, alpha: float = 0.62) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        self.alpha = alpha
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, f"{hit.path}\n{hit.text}") for hit in hits]
        raw_scores = np.asarray(self.model.predict(pairs), dtype=np.float32)
        if raw_scores.size == 0:
            return hits
        min_score = float(np.min(raw_scores))
        max_score = float(np.max(raw_scores))
        if max_score - min_score < 1e-8:
            norm_scores = np.zeros_like(raw_scores, dtype=np.float32)
        else:
            norm_scores = (raw_scores - min_score) / (max_score - min_score)

        rescored: list[tuple[float, RetrievalHit]] = []
        for idx, hit in enumerate(hits):
            blended = self.alpha * hit.score + (1.0 - self.alpha) * float(norm_scores[idx])
            rescored.append((blended, hit))
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in rescored]


def create_reranker(reranker_name: str) -> Reranker:
    selected = (reranker_name or "").strip()
    if not selected:
        raise ValueError("Reranker model is not configured.")
    if selected.lower() in {"none", "off", "lexical", "token"}:
        raise ValueError(
            f"Reranker '{selected}' is disabled in strict mode. "
            "Use a real cross-encoder model."
        )
    try:
        return CrossEncoderReranker(model_name=selected)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load reranker model '{selected}': {exc}") from exc
