"""Health, system status, model registry, public configuration and bundled samples."""

from __future__ import annotations

import platform
import shutil
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from backend.app import __version__
from backend.app.schemas import HealthResponse
from backend.app.services.imaging import ALLOWED_EXTENSIONS
from backend.app.state import AppState, get_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    db_ok = state.db.ping()
    available = sum(1 for m in state.models.list_models() if m["available"])
    return HealthResponse(status="ok" if db_ok and available else "degraded", version=__version__,
                          time=datetime.now(UTC), database=db_ok, models_available=available)


@router.get("/api/system", tags=["system"])
def system(state: AppState = Depends(get_state)) -> dict:
    usage = shutil.disk_usage(state.settings.storage_path)
    stored = sum(f.stat().st_size for f in state.settings.analyses_path.rglob("*") if f.is_file())
    return {
        "version": __version__,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "compute": state.models.device_info(),
        "platform": {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version()},
        "database": {"ok": state.db.ping(), "engine": state.db.engine.dialect.name},
        "storage": {"analyses_bytes": stored, "disk_free_bytes": usage.free},
        "models": [{"key": m["key"], "available": m["available"], "loaded": m["loaded"]}
                   for m in state.models.list_models()],
        "samples": len(state.samples.list()),
    }


@router.get("/api/models", tags=["models"])
def models(state: AppState = Depends(get_state)) -> dict:
    return {"default": state.settings.default_model, "models": state.models.list_models()}


@router.get("/api/config", tags=["system"])
def config(state: AppState = Depends(get_state)) -> dict:
    s = state.settings
    return {
        "upload": {"max_file_mb": s.max_upload_mb, "min_image_side": s.min_image_side,
                   "max_image_side": s.max_image_side, "allowed_extensions": sorted(ALLOWED_EXTENSIONS)},
        "inference": {"default_model": s.default_model, "default_threshold": s.default_threshold,
                      "tile_overlap": s.tile_overlap},
        "satellite": {"min_aoi_km": s.satellite_min_aoi_km, "max_aoi_km": s.satellite_max_aoi_km},
        "training_domain": "LEVIR-CD: 0.5 m/pixel aerial/satellite RGB, binary building change",
    }


@router.get("/api/samples", tags=["samples"])
def list_samples(state: AppState = Depends(get_state)) -> dict:
    return {"samples": state.samples.list()}


@router.get("/api/samples/{sample_id}/{role}", tags=["samples"])
def sample_file(sample_id: str, role: str, state: AppState = Depends(get_state)) -> FileResponse:
    path = state.samples.path(sample_id, role)
    if path.suffix.lower() in {".tif", ".tiff"}:
        preview = path.with_suffix(".preview.png")
        if preview.is_file():
            path = preview
    return FileResponse(path)
