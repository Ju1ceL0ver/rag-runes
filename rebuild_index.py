from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from rag_runes.api.config import AppConfig
from rag_runes.index.embeddings import create_embedder
from rag_runes.index.hybrid_index import HybridTreeIndex
from rag_runes.ingest.pdf_ingestor import PdfBookIngestor
from rag_runes.ingest.semantic_segmenter import RuleBasedSemanticSegmenter, build_segmenter
from rag_runes.ingest.tree_builder import TreeBuilder
from rag_runes.ocr.glm_ocr import build_ocr_client

LOGGER = logging.getLogger("rebuild_index")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the RAG index from all books/*.pdf")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR requests and index images without OCR text.",
    )
    parser.add_argument(
        "--ocr-timeout",
        type=int,
        default=45,
        help="Per-image OCR timeout in seconds when OCR is enabled.",
    )
    parser.add_argument(
        "--ocr-model",
        default=None,
        help="Override OCR model for this run.",
    )
    parser.add_argument(
        "--ocr-retries",
        type=int,
        default=10,
        help="OCR retry count after rate limits or timeouts.",
    )
    parser.add_argument(
        "--ocr-retry-delay",
        type=float,
        default=60.0,
        help="Base OCR retry delay in seconds. 429 Retry-After header wins when present.",
    )
    parser.add_argument(
        "--ocr-min-interval",
        type=float,
        default=20.0,
        help="Minimum delay in seconds between OCR requests.",
    )
    parser.add_argument(
        "--strict-ocr",
        action="store_true",
        help="Stop rebuild if OCR still fails after retries.",
    )
    parser.add_argument(
        "--ocr-if-page-text-lt",
        type=int,
        default=None,
        help=(
            "Run OCR only for pages whose embedded PDF text is shorter than this "
            "many characters. Omit to OCR every extracted image."
        ),
    )
    parser.add_argument(
        "--segmenter-timeout",
        type=int,
        default=45,
        help="Per-request segmenter timeout in seconds.",
    )
    parser.add_argument(
        "--segmenter-retries",
        type=int,
        default=1,
        help="Segmenter request retries before falling back to local splitting.",
    )
    parser.add_argument(
        "--local-segmenter",
        action="store_true",
        help="Use the deterministic local semantic splitter instead of the LLM segmenter.",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=None,
        help="Only ingest the first N sorted PDFs. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-pages-per-book",
        type=int,
        default=None,
        help="Only ingest the first N pages of each PDF. Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="Build into a temporary index directory and do not replace data/index.",
    )
    return parser.parse_args()


def swap_index(temp_index_dir: Path, active_index_dir: Path) -> Path | None:
    backup_dir: Path | None = None
    if active_index_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = active_index_dir.with_name(f"{active_index_dir.name}.backup_{timestamp}")
        LOGGER.info("Backing up current index to %s", backup_dir)
        shutil.move(str(active_index_dir), str(backup_dir))

    LOGGER.info("Activating rebuilt index at %s", active_index_dir)
    shutil.move(str(temp_index_dir), str(active_index_dir))
    return backup_dir


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg = AppConfig.from_env()

    pdf_files = sorted(cfg.books_dir.glob("*.pdf"))
    if args.max_books is not None:
        pdf_files = pdf_files[: max(0, args.max_books)]
    if not pdf_files:
        raise SystemExit(f"No PDF files found in {cfg.books_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_index_dir = cfg.index_dir.with_name(f"{cfg.index_dir.name}.rebuild_tmp_{timestamp}")
    if temp_index_dir.exists():
        shutil.rmtree(temp_index_dir)
    temp_index_dir.mkdir(parents=True, exist_ok=True)
    temp_images_dir = temp_index_dir / "images"
    temp_images_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading embedder: %s", cfg.embedder_name)
    embedder = create_embedder(cfg.embedder_name)
    if args.skip_ocr:
        LOGGER.info("OCR disabled for this rebuild (--skip-ocr)")
        ocr_client = build_ocr_client(endpoint=None, model="none")
    else:
        ocr_model = args.ocr_model or cfg.ocr_model
        LOGGER.info("Loading OCR client: %s", ocr_model)
        ocr_client = build_ocr_client(
            endpoint=cfg.ocr_endpoint or "",
            model=ocr_model,
            api_key=cfg.ocr_api_key,
            timeout_s=args.ocr_timeout,
            max_retries=args.ocr_retries,
            retry_delay_s=args.ocr_retry_delay,
            min_interval_s=args.ocr_min_interval,
        )
    if args.local_segmenter:
        LOGGER.info("Loading segmenter: rule-based local")
        segmenter = RuleBasedSemanticSegmenter()
    else:
        LOGGER.info("Loading segmenter: %s", cfg.segmenter_model)
        segmenter = build_segmenter(
            mode=cfg.segmenter_mode,
            endpoint=cfg.segmenter_endpoint,
            model=cfg.segmenter_model,
            api_key=cfg.segmenter_api_key,
        )
    if hasattr(segmenter, "timeout_s"):
        segmenter.timeout_s = args.segmenter_timeout
    if hasattr(segmenter, "request_retries"):
        segmenter.request_retries = args.segmenter_retries
    if hasattr(segmenter, "hard_timeout_s"):
        segmenter.hard_timeout_s = args.segmenter_timeout + 10

    ingestor = PdfBookIngestor(
        ocr_client=ocr_client,
        fail_on_ocr_error=args.strict_ocr,
        ocr_if_page_text_lt=args.ocr_if_page_text_lt,
    )
    tree_builder = TreeBuilder(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        segmenter=segmenter,
    )

    all_nodes = []
    ingest_stats: list[tuple[str, int, int, int]] = []

    for book_idx, pdf_path in enumerate(
        tqdm(pdf_files, desc="Rebuilding index", unit="book"),
        start=1,
    ):
        LOGGER.info("Ingesting %s", pdf_path.name)
        artifact = ingestor.ingest(
            pdf_path=pdf_path,
            image_out_dir=temp_images_dir,
            progress_desc=f"Book {book_idx}/{len(pdf_files)} pages",
            max_pages=args.max_pages_per_book,
        )
        LOGGER.info(
            "Building tree/chunks for %s: pages=%s images=%s",
            pdf_path.name,
            len(artifact.pages),
            len(artifact.images),
        )
        nodes = tree_builder.build(artifact)
        all_nodes.extend(nodes)
        ingest_stats.append(
            (
                artifact.book_id,
                len(artifact.pages),
                len(artifact.images),
                len(nodes),
            )
        )
        LOGGER.info(
            "Finished %s: book_id=%s pages=%s images=%s nodes=%s",
            pdf_path.name,
            artifact.book_id,
            len(artifact.pages),
            len(artifact.images),
            len(nodes),
        )

    LOGGER.info("Building hybrid index for %s total nodes", len(all_nodes))
    index = HybridTreeIndex.build(
        nodes=all_nodes,
        embedder=embedder,
        embedder_name=cfg.embedder_name,
    )
    index.save(temp_index_dir)

    if args.no_swap:
        backup_dir = None
        LOGGER.info("Leaving rebuilt index at %s (--no-swap)", temp_index_dir)
    else:
        backup_dir = swap_index(temp_index_dir=temp_index_dir, active_index_dir=cfg.index_dir)

    LOGGER.info("Rebuild complete")
    LOGGER.info("Books indexed: %s", len(ingest_stats))
    for book_id, pages, images, nodes in ingest_stats:
        LOGGER.info(
            "book_id=%s pages=%s images=%s nodes=%s",
            book_id,
            pages,
            images,
            nodes,
        )
    LOGGER.info("Total nodes: %s", len(index.nodes))
    LOGGER.info("Indexable nodes: %s", len(index.node_ids))
    LOGGER.info("Vector dimension: %s", index.vectors.shape[1] if index.vectors.size else 0)
    if backup_dir is not None:
        LOGGER.info("Previous index backup: %s", backup_dir)


if __name__ == "__main__":
    main()
