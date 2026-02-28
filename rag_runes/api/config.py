from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _to_bool(value: str, default: bool) -> bool:
    normalized = (value or "").strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _coerce_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    return str(value)


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return _to_bool(str(value), default=default)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def load_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _section_value(
    settings: dict[str, Any],
    section: str,
    key: str,
    default: Any = None,
) -> Any:
    section_value = settings.get(section)
    if not isinstance(section_value, dict):
        return default
    return section_value.get(key, default)


@dataclass(slots=True)
class AppConfig:
    books_dir: Path = Path("books")
    index_dir: Path = Path("data/index")

    embedder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 1200
    chunk_overlap: int = 180
    segmenter_mode: str = "llm"
    segmenter_endpoint: str | None = None
    segmenter_model: str = "minimax-m2.5:cloud"
    segmenter_api_key: str | None = None

    ocr_endpoint: str | None = None
    ocr_model: str = "glm-ocr:q8_0"
    ocr_api_key: str | None = None

    answer_endpoint: str | None = None
    answer_model: str = "minimax-m2.5:cloud"
    answer_api_key: str | None = None
    context_chars: int = 750

    bootstrap_on_startup: bool = False

    @property
    def images_dir(self) -> Path:
        return self.index_dir / "images"

    @classmethod
    def from_env(cls) -> "AppConfig":
        env_file = Path(os.environ.get("RAG_ENV_FILE", ".env"))
        load_dotenv(env_file)
        settings_path = Path(os.environ.get("RAG_SETTINGS_FILE", "config/settings.toml"))
        settings = load_settings(settings_path)

        def env_or(env_key: str, section: str, key: str, default: Any = None) -> Any:
            if env_key in os.environ:
                env_value = os.environ.get(env_key)
                if env_value is not None and str(env_value).strip() != "":
                    return env_value
            return _section_value(settings, section, key, default)

        books_dir = _coerce_str(env_or("RAG_BOOKS_DIR", "paths", "books_dir", "books")) or "books"
        index_dir = _coerce_str(
            env_or("RAG_INDEX_DIR", "paths", "index_dir", "data/index"),
            "data/index",
        ) or "data/index"

        return cls(
            books_dir=Path(books_dir),
            index_dir=Path(index_dir),
            embedder_name=_coerce_str(
                env_or(
                    "RAG_EMBEDDER",
                    "models",
                    "embedder_name",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ),
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ) or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            reranker_name=_coerce_str(
                env_or(
                    "RAG_RERANKER",
                    "models",
                    "reranker_name",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ),
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ) or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            chunk_size=_coerce_int(env_or("RAG_CHUNK_SIZE", "params", "chunk_size", 1200), 1200),
            chunk_overlap=_coerce_int(
                env_or("RAG_CHUNK_OVERLAP", "params", "chunk_overlap", 180),
                180,
            ),
            segmenter_mode=_coerce_str(env_or("RAG_SEGMENTER_MODE", "models", "segmenter_mode", "llm"), "llm")
            or "llm",
            segmenter_endpoint=_coerce_str(
                env_or("RAG_SEGMENTER_ENDPOINT", "endpoints", "segmenter_endpoint"),
            )
            or None,
            segmenter_model=_coerce_str(
                env_or("RAG_SEGMENTER_MODEL", "models", "segmenter_model", "minimax-m2.5:cloud"),
                "minimax-m2.5:cloud",
            )
            or "minimax-m2.5:cloud",
            segmenter_api_key=_coerce_str(
                env_or("RAG_SEGMENTER_API_KEY", "auth", "segmenter_api_key"),
            )
            or None,
            ocr_endpoint=_coerce_str(
                env_or("GLM_OCR_ENDPOINT", "endpoints", "ocr_endpoint"),
            )
            or None,
            ocr_model=_coerce_str(
                env_or("GLM_OCR_MODEL", "models", "ocr_model", "glm-ocr:q8_0"),
                "glm-ocr:q8_0",
            )
            or "glm-ocr:q8_0",
            ocr_api_key=_coerce_str(
                env_or("GLM_OCR_API_KEY", "auth", "ocr_api_key"),
            )
            or None,
            answer_endpoint=_coerce_str(
                env_or("RAG_ANSWER_ENDPOINT", "endpoints", "answer_endpoint"),
            )
            or None,
            answer_model=_coerce_str(
                env_or("RAG_ANSWER_MODEL", "models", "answer_model", "minimax-m2.5:cloud"),
                "minimax-m2.5:cloud",
            )
            or "minimax-m2.5:cloud",
            answer_api_key=_coerce_str(
                env_or("RAG_ANSWER_API_KEY", "auth", "answer_api_key"),
            )
            or None,
            context_chars=_coerce_int(
                env_or("RAG_CONTEXT_CHARS", "params", "context_chars", 750),
                750,
            ),
            bootstrap_on_startup=_coerce_bool(
                env_or("RAG_BOOTSTRAP_ON_STARTUP", "params", "bootstrap_on_startup", False),
                default=False,
            ),
        )
