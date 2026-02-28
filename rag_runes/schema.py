from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PageContent:
    page_num: int
    text: str
    image_ids: list[str] = field(default_factory=list)
    ocr_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImageAsset:
    image_id: str
    book_id: str
    page_num: int
    path: str
    ocr_text: str = ""
    rune_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BookArtifact:
    book_id: str
    title: str
    source_path: str
    pages: list[PageContent]
    images: list[ImageAsset]


@dataclass(slots=True)
class DocNode:
    node_id: str
    book_id: str
    parent_id: str | None
    level: str
    title: str
    text: str
    page_from: int
    page_to: int
    image_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocNode":
        return cls(
            node_id=payload["node_id"],
            book_id=payload["book_id"],
            parent_id=payload.get("parent_id"),
            level=payload["level"],
            title=payload.get("title", ""),
            text=payload.get("text", ""),
            page_from=int(payload.get("page_from", 0)),
            page_to=int(payload.get("page_to", 0)),
            image_ids=list(payload.get("image_ids", [])),
            metadata=dict(payload.get("metadata", {})),
        )
