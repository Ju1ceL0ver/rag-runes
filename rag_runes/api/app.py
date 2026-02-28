from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_runes.api.config import AppConfig
from rag_runes.api.service import RagPipelineService


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=40)
    branch_factor: int = Field(default=4, ge=1, le=20)
    reranker: str | None = None


class QueryResponse(BaseModel):
    answer: str
    reading_pages: list[dict[str, Any]]
    context_hits: list[dict[str, Any]]
    params: dict[str, Any]


class IngestResponse(BaseModel):
    status: str
    book_id: str
    filename: str
    pages: int
    images: int
    book_nodes: int
    total_nodes: int
    indexable_nodes: int


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    books_on_disk: int
    total_nodes: int
    indexable_nodes: int
    errors: list[str]
    config: dict[str, Any]


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or AppConfig.from_env()
    service = RagPipelineService(cfg)

    app = FastAPI(
        title="rag-runes",
        version="0.2.0",
        description="Multimodal tree-RAG API for rune-heavy book corpora",
    )
    app.state.pipeline = service

    @app.on_event("startup")
    def _startup() -> None:
        app.state.pipeline.initialize()

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        return app.state.pipeline.health()

    @app.post("/books/upload-ingest", response_model=IngestResponse)
    async def upload_and_ingest(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = file.filename or "book.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        content = await file.read()
        try:
            return app.state.pipeline.ingest_upload(filename=filename, content=content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> dict[str, Any]:
        try:
            return app.state.pipeline.query(
                question=payload.question,
                top_k=payload.top_k,
                branch_factor=payload.branch_factor,
                reranker_name=payload.reranker,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return app


app = create_app()
