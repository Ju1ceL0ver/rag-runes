from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import requests
from tqdm import tqdm

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


@dataclass(slots=True)
class OllamaEmbedder(TextEmbedder):
    model_name: str
    endpoint: str = "http://127.0.0.1:11434"
    batch_size: int = 64

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        response = requests.post(
            f"{self.endpoint.rstrip('/')}/api/embed",
            json={"model": self.model_name, "input": texts},
            timeout=240,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(
                f"Ollama embedder '{self.model_name}' returned no embeddings."
            )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise RuntimeError(
                f"Ollama embedder '{self.model_name}' returned invalid shape."
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        batches: list[np.ndarray] = []
        ranges = range(0, len(texts), self.batch_size)
        batch_iter = tqdm(
            ranges,
            total=(len(texts) + self.batch_size - 1) // self.batch_size,
            desc=f"Embedding {self.model_name}",
            unit="batch",
            leave=False,
        )
        for start in batch_iter:
            batches.append(self._embed_batch(texts[start : start + self.batch_size]))
        return np.vstack(batches).astype(np.float32)


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
    if candidate.startswith("ollama:") or candidate.startswith("ollama://"):
        model_name = (
            candidate.split(":", 1)[1]
            if candidate.startswith("ollama:")
            else candidate.removeprefix("ollama://")
        )
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("Ollama embedder model is empty.")
        endpoint = os.environ.get("RAG_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
        return OllamaEmbedder(model_name=model_name, endpoint=endpoint)

    try:
        return SentenceTransformerEmbedder(candidate)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load embedder model '{candidate}': {exc}") from exc
