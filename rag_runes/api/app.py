from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_runes.api.config import AppConfig
from rag_runes.api.service import RagPipelineService
from rag_runes.synthetic.service import RuneSynthesisService


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
    synthetic: dict[str, Any]


class SyntheticRenderRequest(BaseModel):
    class_id: int | None = Field(default=None, ge=0)
    class_name: str | None = None
    seed: int = 42
    canvas_size: int = Field(default=256, ge=128, le=1024)
    palette_name: str = "auto"
    style: str = "engraved"
    background_mode: str = "transparent"
    defect_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    thickness: float = Field(default=1.0, ge=0.5, le=4.0)
    export_name: str | None = None


class SyntheticDatasetRequest(BaseModel):
    variants_per_class: int = Field(default=10, ge=1, le=500)
    class_ids: list[int] | None = None
    canvas_size: int = Field(default=256, ge=128, le=1024)
    base_seed: int = 42
    palette_name: str = "auto"
    style: str = "engraved"
    background_mode: str = "transparent"
    defect_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    thickness: float = Field(default=1.0, ge=0.5, le=4.0)
    output_name: str | None = None


class SyntheticRenderResponse(BaseModel):
    status: str
    class_id: int
    class_name: str
    seed: int
    score: float
    occupancy: float
    image_path: str
    mask_path: str
    cutout_path: str
    overlay_path: str
    raw_path: str
    bbox: dict[str, int]
    renderer: dict[str, Any]


class SyntheticDatasetResponse(BaseModel):
    status: str
    output_dir: str
    manifest_path: str
    classes: int
    variants_per_class: int
    samples: int
    canvas_size: int
    palette_name: str
    style: str
    background_mode: str
    defect_strength: float
    thickness: float
    checkpoint_path: str


class SyntheticSceneRequest(BaseModel):
    background_path: str = Field(min_length=1)
    instance_count: int | None = Field(default=None, ge=1, le=120)
    instance_count_min: int = Field(default=25, ge=1, le=120)
    instance_count_max: int = Field(default=45, ge=1, le=120)
    center_coverage: float = Field(default=0.70, ge=0.30, le=1.0)
    class_ids: list[int] | None = None
    seed: int = 42
    style: str = "scratch"
    defect_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    thickness: float = Field(default=1.8, ge=0.5, le=4.0)
    scale_min: float = Field(default=0.08, ge=0.02, le=0.9)
    scale_max: float = Field(default=0.16, ge=0.02, le=0.95)
    max_iou: float = Field(default=0.14, ge=0.0, le=0.95)
    export_name: str | None = None


class SyntheticSceneResponse(BaseModel):
    status: str
    background_path: str
    image_path: str
    label_path: str
    mask_path: str
    width: int
    height: int
    split: str
    instances: list[dict[str, Any]]
    stone_score_mean: float
    stone_score_max: float
    thickness: float
    instance_count: int
    scene_name: str
    scene_relpath: str
    checkpoint_path: str


class SyntheticDetectionDatasetRequest(BaseModel):
    backgrounds_dir: str | None = None
    background_path: str | None = None
    images_per_background: int = Field(default=4, ge=1, le=200)
    instances_min: int = Field(default=25, ge=1, le=120)
    instances_max: int = Field(default=45, ge=1, le=120)
    center_coverage: float = Field(default=0.70, ge=0.30, le=1.0)
    class_ids: list[int] | None = None
    base_seed: int = 42
    style: str = "scratch"
    defect_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    thickness: float = Field(default=1.8, ge=0.5, le=4.0)
    scale_min: float = Field(default=0.08, ge=0.02, le=0.9)
    scale_max: float = Field(default=0.16, ge=0.02, le=0.95)
    max_iou: float = Field(default=0.14, ge=0.0, le=0.95)
    output_name: str | None = None


class SyntheticDetectionDatasetResponse(BaseModel):
    status: str
    output_dir: str
    manifest_path: str
    dataset_yaml_path: str
    backgrounds: int
    images_per_background: int
    scenes: int
    train_scenes: int
    val_scenes: int
    instances_min: int
    instances_max: int
    style: str
    defect_strength: float
    thickness: float
    scale_min: float
    scale_max: float
    max_iou: float
    checkpoint_path: str


class SyntheticClassesResponse(BaseModel):
    ready: bool
    checkpoint_path: str | None = None
    num_classes: int | None = None
    class_names: list[str] = []
    output_dir: str | None = None
    errors: list[str] = []


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or AppConfig.from_env()
    service = RagPipelineService(cfg)
    synthetic_service: RuneSynthesisService | None = None
    synthetic_errors: list[str] = []
    try:
        synthetic_service = RuneSynthesisService(
            checkpoint_path=cfg.synthetic_checkpoint_path,
            output_dir=cfg.synthetic_output_dir,
            device=cfg.synthetic_device,
            stone_texture_dir=cfg.stone_texture_dir,
        )
    except Exception as exc:  # noqa: BLE001
        synthetic_errors.append(str(exc))

    app = FastAPI(
        title="rag-runes",
        version="0.2.0",
        description="Multimodal tree-RAG API for rune-heavy book corpora",
    )
    app.state.pipeline = service
    app.state.synthetic = synthetic_service
    app.state.synthetic_errors = synthetic_errors

    @app.on_event("startup")
    def _startup() -> None:
        app.state.pipeline.initialize()

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        rag_health = app.state.pipeline.health()
        if app.state.synthetic is None:
            synthetic_health = {"ready": False, "errors": list(app.state.synthetic_errors)}
        else:
            synthetic_health = app.state.synthetic.health()
            synthetic_health["errors"] = list(app.state.synthetic_errors)
        rag_health["synthetic"] = synthetic_health
        return rag_health

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

    @app.get("/synthetic-runes/classes", response_model=SyntheticClassesResponse)
    def synthetic_classes() -> dict[str, Any]:
        if app.state.synthetic is None:
            return {
                "ready": False,
                "errors": list(app.state.synthetic_errors),
            }
        health_payload = app.state.synthetic.health()
        health_payload["errors"] = list(app.state.synthetic_errors)
        return health_payload

    @app.post("/synthetic-runes/render", response_model=SyntheticRenderResponse)
    def synthetic_render(payload: SyntheticRenderRequest) -> dict[str, Any]:
        if app.state.synthetic is None:
            raise HTTPException(
                status_code=503,
                detail="Synthetic generator is not ready: "
                + "; ".join(app.state.synthetic_errors),
            )
        try:
            return app.state.synthetic.render_single(
                class_id=payload.class_id,
                class_name=payload.class_name,
                seed=payload.seed,
                canvas_size=payload.canvas_size,
                palette_name=payload.palette_name,
                style=payload.style,
                background_mode=payload.background_mode,
                defect_strength=payload.defect_strength,
                thickness=payload.thickness,
                export_name=payload.export_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Synthetic render failed: {exc}") from exc

    @app.post("/synthetic-runes/dataset", response_model=SyntheticDatasetResponse)
    def synthetic_dataset(payload: SyntheticDatasetRequest) -> dict[str, Any]:
        if app.state.synthetic is None:
            raise HTTPException(
                status_code=503,
                detail="Synthetic generator is not ready: "
                + "; ".join(app.state.synthetic_errors),
            )
        try:
            return app.state.synthetic.build_dataset(
                variants_per_class=payload.variants_per_class,
                class_ids=payload.class_ids,
                canvas_size=payload.canvas_size,
                base_seed=payload.base_seed,
                palette_name=payload.palette_name,
                style=payload.style,
                background_mode=payload.background_mode,
                defect_strength=payload.defect_strength,
                thickness=payload.thickness,
                output_name=payload.output_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Synthetic dataset build failed: {exc}") from exc

    @app.post("/synthetic-runes/scene", response_model=SyntheticSceneResponse)
    def synthetic_scene(payload: SyntheticSceneRequest) -> dict[str, Any]:
        if app.state.synthetic is None:
            raise HTTPException(
                status_code=503,
                detail="Synthetic generator is not ready: "
                + "; ".join(app.state.synthetic_errors),
            )
        try:
            return app.state.synthetic.render_scene(
                background_path=Path(payload.background_path),
                instance_count=payload.instance_count,
                instance_count_min=payload.instance_count_min,
                instance_count_max=payload.instance_count_max,
                center_coverage=payload.center_coverage,
                seed=payload.seed,
                class_ids=payload.class_ids,
                style=payload.style,
                defect_strength=payload.defect_strength,
                thickness=payload.thickness,
                scale_min=payload.scale_min,
                scale_max=payload.scale_max,
                max_iou=payload.max_iou,
                export_name=payload.export_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Synthetic scene render failed: {exc}") from exc

    @app.post("/synthetic-runes/detection-dataset", response_model=SyntheticDetectionDatasetResponse)
    def synthetic_detection_dataset(payload: SyntheticDetectionDatasetRequest) -> dict[str, Any]:
        if app.state.synthetic is None:
            raise HTTPException(
                status_code=503,
                detail="Synthetic generator is not ready: "
                + "; ".join(app.state.synthetic_errors),
            )
        try:
            return app.state.synthetic.build_detection_dataset(
                backgrounds_dir=Path(payload.backgrounds_dir) if payload.backgrounds_dir else None,
                background_path=Path(payload.background_path) if payload.background_path else None,
                images_per_background=payload.images_per_background,
                instances_min=payload.instances_min,
                instances_max=payload.instances_max,
                center_coverage=payload.center_coverage,
                class_ids=payload.class_ids,
                base_seed=payload.base_seed,
                style=payload.style,
                defect_strength=payload.defect_strength,
                thickness=payload.thickness,
                scale_min=payload.scale_min,
                scale_max=payload.scale_max,
                max_iou=payload.max_iou,
                output_name=payload.output_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Synthetic detection dataset build failed: {exc}") from exc

    return app


app = create_app()
