from __future__ import annotations

import logging
from pathlib import Path

import fitz
from tqdm import tqdm

from rag_runes.ocr.base import OCRClient
from rag_runes.schema import BookArtifact, ImageAsset, PageContent
from rag_runes.text_utils import normalize_whitespace, rune_density, slugify

LOGGER = logging.getLogger(__name__)


class PdfBookIngestor:
    def __init__(
        self,
        ocr_client: OCRClient,
        fail_on_ocr_error: bool = False,
        ocr_if_page_text_lt: int | None = None,
    ) -> None:
        self.ocr_client = ocr_client
        self.fail_on_ocr_error = fail_on_ocr_error
        self.ocr_if_page_text_lt = ocr_if_page_text_lt

    def ingest(
        self,
        pdf_path: Path,
        image_out_dir: Path,
        progress_desc: str | None = None,
        max_pages: int | None = None,
    ) -> BookArtifact:
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        book_id = slugify(pdf_path.stem)
        image_book_dir = image_out_dir / book_id
        image_book_dir.mkdir(parents=True, exist_ok=True)

        pages: list[PageContent] = []
        images: list[ImageAsset] = []

        with fitz.open(pdf_path) as doc:
            title = (doc.metadata.get("title") or pdf_path.stem).strip()
            total_pages = len(doc)
            if max_pages is not None:
                total_pages = min(total_pages, max_pages)
            page_iter = (doc[index] for index in range(total_pages))
            if progress_desc:
                page_iter = tqdm(
                    page_iter,
                    total=total_pages,
                    desc=progress_desc,
                    unit="page",
                    leave=False,
                )
            for page_idx, page in enumerate(page_iter, start=1):
                raw_text = page.get_text("text")
                page_lines = [
                    normalize_whitespace(line)
                    for line in raw_text.splitlines()
                    if normalize_whitespace(line)
                ]
                page_text = "\n".join(page_lines)
                page_payload = PageContent(page_num=page_idx, text=page_text)
                should_ocr_page = (
                    self.ocr_if_page_text_lt is None
                    or len(normalize_whitespace(page_text)) < self.ocr_if_page_text_lt
                )

                page_images = page.get_images(full=True)
                for img_idx, image_info in enumerate(page_images, start=1):
                    xref = image_info[0]
                    image_data = doc.extract_image(xref)
                    image_bytes = image_data.get("image")
                    if not image_bytes:
                        continue

                    extension = image_data.get("ext", "png")
                    image_id = f"{book_id}-p{page_idx}-i{img_idx}"
                    image_path = image_book_dir / f"{image_id}.{extension}"
                    image_path.write_bytes(image_bytes)

                    ocr_text = ""
                    if should_ocr_page:
                        try:
                            ocr_text = self.ocr_client.extract_text(str(image_path)).strip()
                        except Exception as exc:  # noqa: BLE001
                            if self.fail_on_ocr_error:
                                raise
                            LOGGER.warning(
                                "OCR failed for image '%s': %s",
                                image_path,
                                exc,
                            )
                            ocr_text = ""
                    page_payload.image_ids.append(image_id)
                    if ocr_text:
                        page_payload.ocr_texts.append(ocr_text)

                    images.append(
                        ImageAsset(
                            image_id=image_id,
                            book_id=book_id,
                            page_num=page_idx,
                            path=str(image_path),
                            ocr_text=ocr_text,
                            rune_density=rune_density(ocr_text),
                        )
                    )

                pages.append(page_payload)

        return BookArtifact(
            book_id=book_id,
            title=title,
            source_path=str(pdf_path),
            pages=pages,
            images=images,
        )
