from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from rag_runes.text_utils import tokenize


class TextEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class HashEmbedder(TextEmbedder):
    dim: int = 512

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)

        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], byteorder="little") % self.dim
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                matrix[row, idx] += sign

            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm

        return matrix


class SentenceTransformerEmbedder(TextEmbedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=np.float32)


def create_embedder(embedder_name: str) -> TextEmbedder:
    candidate = (embedder_name or "").strip()
    if not candidate:
        raise ValueError(
            "Embedder model is not configured. Set a SentenceTransformer model name."
        )
    if candidate.lower() == "hash":
        raise ValueError(
            "Embedder 'hash' is disabled in strict mode. "
            "Use a real model, e.g. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2."
        )

    try:
        return SentenceTransformerEmbedder(candidate)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load embedder model '{candidate}': {exc}") from exc
