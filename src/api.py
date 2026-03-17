import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .deployment import (
        DatasetNotLoadedError,
        DeploymentError,
        DuplicateDetectionService,
        ModelUnavailableError,
        build_default_service,
        canonicalize_columns,
    )
except ImportError:
    from deployment import (
        DatasetNotLoadedError,
        DeploymentError,
        DuplicateDetectionService,
        ModelUnavailableError,
        build_default_service,
        canonicalize_columns,
    )


class DatasetPathRequest(BaseModel):
    path: str = Field(..., description="Path to a CSV or TSV file already available on the server.")


class DuplicateSearchRequest(BaseModel):
    model_name: str = Field(default="siamese_bilstm")
    blocking_keys: Optional[List[str]] = Field(default=None)
    blocking_mode: Optional[str] = Field(default=None, pattern="^(all|any)$")
    threshold: Optional[float] = None
    max_candidate_pairs: int = 250000
    top_k: int = 50


class EntryCheckRequest(BaseModel):
    model_name: str = Field(default="siamese_bilstm")
    entry: Dict[str, Any]
    blocking_keys: Optional[List[str]] = Field(default=None)
    blocking_mode: Optional[str] = Field(default=None, pattern="^(all|any)$")
    threshold: Optional[float] = None
    top_k: int = 10
    max_candidates: int = 50000


def _normalize_blocking_keys(blocking_keys: Optional[List[str]]) -> Optional[List[str]]:
    if blocking_keys is None:
        return None
    return [column for column in canonicalize_columns(blocking_keys) if column]


def create_app(service: Optional[DuplicateDetectionService] = None) -> FastAPI:
    runtime_dir = Path(os.environ.get("FRAGMENTED_ID_RUNTIME_DIR", "data/runtime"))
    default_dataset_path = os.environ.get("FRAGMENTED_ID_DEFAULT_DATASET")
    resolved_service = service or build_default_service(runtime_dir=runtime_dir)
    ui_dir = Path(__file__).resolve().parent / "ui"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        app.state.service = resolved_service
        if default_dataset_path:
            try:
                app.state.service.load_dataset_from_path(default_dataset_path)
            except Exception as exc:  # pragma: no cover
                app.state.startup_warning = f"Failed to load default dataset {default_dataset_path}: {exc}"
        yield

    app = FastAPI(
        title="NCVoters Duplicate Detection API",
        version="1.0.0",
        description=(
            "Upload a dataset, find likely duplicates inside it, or check a new record against it "
            "using the trained Siamese models or TF-IDF baseline."
        ),
        lifespan=lifespan,
    )
    app.state.service = resolved_service
    if ui_dir.exists():
        app.mount("/assets", StaticFiles(directory=ui_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def root():
        if not ui_dir.exists():
            raise HTTPException(status_code=404, detail="UI assets were not found.")
        return FileResponse(ui_dir / "index.html")

    @app.get("/api/info")
    def api_info():
        return {
            "service": "NCVoters Duplicate Detection API",
            "available_models": app.state.service.list_models(),
            "loaded_dataset": None
            if app.state.service.current_dataset is None
            else app.state.service.current_dataset.summary(),
            "startup_warning": getattr(app.state, "startup_warning", None),
            "endpoints": {
                "upload_dataset": "POST /dataset/upload",
                "load_dataset_from_path": "POST /dataset/load",
                "dataset_summary": "GET /dataset/summary",
                "clear_dataset": "DELETE /dataset",
                "find_duplicates": "POST /duplicates/find",
                "check_entry": "POST /duplicates/check-entry",
            },
        }

    @app.get("/app/state")
    def app_state():
        models = app.state.service.list_models()
        return {
            "models": models,
            "dataset": None
            if app.state.service.current_dataset is None
            else app.state.service.current_dataset.summary(),
            "startup_warning": getattr(app.state, "startup_warning", None),
            "defaults": {
                "preferred_model": next(
                    (
                        model["name"]
                        for model in models
                        if model["name"] == "siamese_bilstm"
                    ),
                    models[0]["name"] if models else None,
                ),
                "duplicate_search": {
                    "blocking_keys": ["first_name", "last_name", "zip_code"],
                    "blocking_mode": "all",
                    "top_k": 25,
                    "max_candidate_pairs": 250000,
                },
                "entry_check": {
                    "blocking_keys": ["first_name", "last_name", "zip_code"],
                    "blocking_mode": "any",
                    "top_k": 10,
                    "max_candidates": 50000,
                },
            },
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "loaded_dataset": app.state.service.current_dataset is not None,
            "available_model_count": len(app.state.service.list_models()),
        }

    @app.get("/models")
    def list_models():
        return {"models": app.state.service.list_models()}

    @app.post("/dataset/upload")
    async def upload_dataset(file: UploadFile = File(...)):
        try:
            payload = await file.read()
            summary = app.state.service.load_dataset_from_file(payload, file.filename or "uploaded.tsv")
            return {"message": "Dataset loaded.", "dataset": summary}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/dataset/load")
    def load_dataset_from_path(request: DatasetPathRequest):
        try:
            summary = app.state.service.load_dataset_from_path(request.path)
            return {"message": "Dataset loaded.", "dataset": summary}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/dataset/summary")
    def dataset_summary():
        try:
            return {"dataset": app.state.service.dataset_summary()}
        except DatasetNotLoadedError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/dataset")
    def clear_dataset():
        app.state.service.clear_dataset()
        return {"message": "Dataset cleared."}

    @app.post("/duplicates/find")
    def find_duplicates(request: DuplicateSearchRequest):
        try:
            return app.state.service.find_duplicates(
                model_name=request.model_name,
                blocking_keys=_normalize_blocking_keys(request.blocking_keys),
                blocking_mode=request.blocking_mode,
                threshold=request.threshold,
                max_candidate_pairs=request.max_candidate_pairs,
                top_k=request.top_k,
            )
        except (DatasetNotLoadedError, ModelUnavailableError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DeploymentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/duplicates/check-entry")
    def check_entry(request: EntryCheckRequest):
        try:
            return app.state.service.check_entry(
                entry=request.entry,
                model_name=request.model_name,
                blocking_keys=_normalize_blocking_keys(request.blocking_keys),
                blocking_mode=request.blocking_mode,
                threshold=request.threshold,
                top_k=request.top_k,
                max_candidates=request.max_candidates,
            )
        except (DatasetNotLoadedError, ModelUnavailableError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DeploymentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
