from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_runes.ingest.semantic_segmenter import SemanticSegmenter
from rag_runes.schema import BookArtifact, DocNode
from rag_runes.text_utils import normalize_whitespace, rune_density

NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[\.\)]?\s+(.+)$")
ROMAN_RE = re.compile(r"^(?:[IVXLCM]{1,8})[\.\)]?\s+(.+)$")


@dataclass(slots=True)
class _SectionBuffer:
    title: str
    parent_id: str
    page_from: int
    page_to: int
    lines: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)


def detect_heading(line: str) -> tuple[int, str] | None:
    stripped = normalize_whitespace(line)
    if len(stripped) < 3:
        return None

    numbered = NUMBERED_RE.match(stripped)
    if numbered:
        prefix, title = numbered.groups()
        level = min(prefix.count(".") + 1, 3)
        return level, title.strip()

    roman = ROMAN_RE.match(stripped)
    if roman and len(stripped.split()) <= 10:
        return 1, stripped

    words = stripped.split()
    if len(words) > 14 or len(stripped) > 100:
        return None
    if stripped.endswith((".", "!", "?")):
        return None

    alphabetic = [ch for ch in stripped if ch.isalpha()]
    if not alphabetic:
        return None
    upper_ratio = sum(1 for ch in alphabetic if ch.isupper()) / len(alphabetic)
    title_case_ratio = sum(1 for word in words if word[:1].isupper()) / len(words)

    if upper_ratio > 0.72:
        return 1, stripped
    if title_case_ratio > 0.8:
        return 2, stripped
    return None


class TreeBuilder:
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
        segmenter: SemanticSegmenter | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if segmenter is None:
            raise ValueError("Segmenter is required in strict mode.")
        self.segmenter = segmenter

    def build(self, artifact: BookArtifact) -> list[DocNode]:
        nodes: list[DocNode] = []
        id_counters = {"chapter": 0, "section": 0, "chunk": 0, "image": 0}
        image_by_id = {img.image_id: img for img in artifact.images}
        chapter_lookup: dict[str, DocNode] = {}

        def make_id(level: str) -> str:
            id_counters[level] += 1
            return f"{artifact.book_id}:{level}:{id_counters[level]}"

        book_id = f"{artifact.book_id}:book"
        nodes.append(
            DocNode(
                node_id=book_id,
                book_id=artifact.book_id,
                parent_id=None,
                level="book",
                title=artifact.title,
                text="",
                page_from=1,
                page_to=max((page.page_num for page in artifact.pages), default=1),
            )
        )

        chapter_id = make_id("chapter")
        chapter_node = DocNode(
            node_id=chapter_id,
            book_id=artifact.book_id,
            parent_id=book_id,
            level="chapter",
            title="Введение",
            text="",
            page_from=1,
            page_to=1,
        )
        chapter_lookup[chapter_id] = chapter_node
        nodes.append(chapter_node)

        section_buffer: _SectionBuffer | None = None

        def ensure_section(page_num: int) -> _SectionBuffer:
            nonlocal section_buffer
            if section_buffer is None:
                section_buffer = _SectionBuffer(
                    title=f"Секция стр. {page_num}",
                    parent_id=chapter_id,
                    page_from=page_num,
                    page_to=page_num,
                )
            return section_buffer

        def flush_section() -> None:
            nonlocal section_buffer
            if section_buffer is None:
                return

            unique_images = sorted(set(section_buffer.image_ids))
            base_lines = [normalize_whitespace(line) for line in section_buffer.lines]
            base_lines = [line for line in base_lines if line]

            image_lines: list[str] = []
            for image_id in unique_images:
                image = image_by_id.get(image_id)
                if image is None:
                    continue
                ocr_line = normalize_whitespace(image.ocr_text)
                image_lines.append(f"[IMAGE {image_id}] {ocr_line}".strip())

            merged_lines = base_lines + image_lines
            text = "\n".join(line for line in merged_lines if line).strip()
            image_signal = max(
                (
                    image_by_id[image_id].rune_density
                    for image_id in unique_images
                    if image_id in image_by_id
                ),
                default=0.0,
            )
            merged_rune_density = max(image_signal, rune_density(text))

            if not text and not unique_images:
                section_buffer = None
                return

            section_id = make_id("section")
            section_node = DocNode(
                node_id=section_id,
                book_id=artifact.book_id,
                parent_id=section_buffer.parent_id,
                level="section",
                title=section_buffer.title,
                text=text,
                page_from=section_buffer.page_from,
                page_to=section_buffer.page_to,
                image_ids=unique_images,
                metadata={
                    "rune_density": merged_rune_density,
                    "image_count": len(unique_images),
                },
            )
            nodes.append(section_node)

            chunks = self.segmenter.split(
                text=text,
                section_title=section_buffer.title,
                max_chars=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                chunk_id = make_id("chunk")
                nodes.append(
                    DocNode(
                        node_id=chunk_id,
                        book_id=artifact.book_id,
                        parent_id=section_id,
                        level="chunk",
                        title=f"{section_buffer.title} / chunk {chunk_idx}",
                        text=chunk_text,
                        page_from=section_buffer.page_from,
                        page_to=section_buffer.page_to,
                        image_ids=unique_images,
                        metadata={
                            "rune_density": max(
                                merged_rune_density, rune_density(chunk_text)
                            ),
                            "chunk_index": chunk_idx,
                        },
                    )
                )

            for image_idx, image_id in enumerate(unique_images, start=1):
                image = image_by_id.get(image_id)
                if image is None:
                    continue
                ocr_text = normalize_whitespace(image.ocr_text)
                image_text = (
                    f"[IMAGE {image_id}] {ocr_text}"
                    if ocr_text
                    else f"[IMAGE {image_id}] Изображение без OCR текста"
                )
                image_node_id = make_id("image")
                nodes.append(
                    DocNode(
                        node_id=image_node_id,
                        book_id=artifact.book_id,
                        parent_id=section_id,
                        level="image",
                        title=f"{section_buffer.title} / image {image_idx}",
                        text=image_text,
                        page_from=image.page_num,
                        page_to=image.page_num,
                        image_ids=[image_id],
                        metadata={
                            "rune_density": float(image.rune_density),
                            "image_path": image.path,
                            "image_id": image.image_id,
                            "modality": "image",
                        },
                    )
                )

            section_buffer = None

        for page in artifact.pages:
            chapter_lookup[chapter_id].page_to = max(
                chapter_lookup[chapter_id].page_to, page.page_num
            )

            page_images = list(page.image_ids)
            attached_images = False
            page_lines = [line for line in page.text.splitlines() if line.strip()]

            for line in page_lines:
                heading = detect_heading(line)
                if heading:
                    level, title = heading
                    if level == 1:
                        flush_section()
                        chapter_id = make_id("chapter")
                        chapter_node = DocNode(
                            node_id=chapter_id,
                            book_id=artifact.book_id,
                            parent_id=book_id,
                            level="chapter",
                            title=title,
                            text="",
                            page_from=page.page_num,
                            page_to=page.page_num,
                        )
                        chapter_lookup[chapter_id] = chapter_node
                        nodes.append(chapter_node)
                        continue

                    flush_section()
                    section_buffer = _SectionBuffer(
                        title=title,
                        parent_id=chapter_id,
                        page_from=page.page_num,
                        page_to=page.page_num,
                    )
                    if page_images:
                        section_buffer.image_ids.extend(page_images)
                        attached_images = True
                    continue

                target = ensure_section(page.page_num)
                target.page_to = page.page_num
                if page_images and not attached_images:
                    target.image_ids.extend(page_images)
                    attached_images = True
                target.lines.append(normalize_whitespace(line))

            for ocr_text in page.ocr_texts:
                cleaned = normalize_whitespace(ocr_text)
                if not cleaned:
                    continue
                target = ensure_section(page.page_num)
                target.page_to = page.page_num
                if page_images and not attached_images:
                    target.image_ids.extend(page_images)
                    attached_images = True
                target.lines.append(f"[OCR] {cleaned}")

            if page_images and not attached_images:
                target = ensure_section(page.page_num)
                target.page_to = page.page_num
                target.image_ids.extend(page_images)

        flush_section()
        return nodes
