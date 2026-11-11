"""Satellite acquisition endpoints: providers, catalogue search, retrieval into an analysis, geocoding."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from backend.app.errors import AppError
from backend.app.satellite.base import SearchQuery, validate_bbox
from backend.app.schemas import AnalysisDetail, SatelliteRetrieveRequest, SatelliteSearchRequest
from backend.app.services.imaging import align_pair
from backend.app.services.model_manager import domain_warning, preferred_model
from backend.app.state import AppState, get_state
from ml.models.registry import MODEL_SPECS

router = APIRouter(prefix="/api/satellite", tags=["satellite"])


@router.get("/providers")
def providers(state: AppState = Depends(get_state)) -> dict:
    return {"providers": state.providers.list()}


@router.get("/geocode")
def geocode(q: Annotated[str, Query(min_length=2, max_length=120)], state: AppState = Depends(get_state)) -> dict:
    return {"query": q, "results": state.providers.geocode(q),
            "attribution": "Search results (c) OpenStreetMap contributors (Nominatim)"}


@router.post("/search")
def search(body: SatelliteSearchRequest, state: AppState = Depends(get_state)) -> dict:
    s = state.settings
    bbox = body.bbox.as_tuple()
    validate_bbox(bbox, s.satellite_min_aoi_km, s.satellite_max_aoi_km)
    provider = state.providers.get(body.provider)
    scenes = provider.search(SearchQuery(bbox, body.start, body.end, body.max_cloud_cover, body.limit))
    return {"provider": provider.describe(), "bbox": list(bbox), "count": len(scenes),
            "scenes": [scene.to_dict() for scene in scenes]}


@router.post("/retrieve", response_model=AnalysisDetail, status_code=202)
def retrieve(body: SatelliteRetrieveRequest, state: AppState = Depends(get_state)) -> AnalysisDetail:
    """Fetch both scenes clipped to the AOI, align them and start a change-detection analysis."""
    from backend.app.routers.analyses import _detail

    s = state.settings
    bbox = body.bbox.as_tuple()
    validate_bbox(bbox, s.satellite_min_aoi_km, s.satellite_max_aoi_km)
    if body.t1_scene_id == body.t2_scene_id:
        raise AppError("SAME_SCENE", "T1 and T2 must be two different acquisitions", status=422)
    model_key = body.model or preferred_model(state.models, "sentinel2", s.default_model)
    if body.threshold is not None:
        threshold = body.threshold
    elif model_key in MODEL_SPECS:
        threshold = state.models.default_threshold(model_key, s.default_threshold)
    else:
        threshold = s.default_threshold
    state.analyses.validate_request(model_key, threshold)

    provider = state.providers.get(body.provider)
    first = provider.retrieve(body.t1_scene_id, bbox)
    second = provider.retrieve(body.t2_scene_id, bbox)
    if (first.metadata.get("acquired") or "") > (second.metadata.get("acquired") or ""):
        first, second = second, first  # keep T1 = earlier acquisition
    img1, img2, warnings = align_pair(first.image, second.image)
    if note := domain_warning(model_key, "sentinel2"):
        warnings.append(note)
    row = state.analyses.create(
        t1=img1, t2=img2, truth=None, model_key=model_key, threshold=threshold, source="satellite",
        name=body.name or f"{provider.key}: {str(first.metadata.get('acquired'))[:10]} -> "
                          f"{str(second.metadata.get('acquired'))[:10]}",
        inputs={"provider": provider.key, "aoi_bbox": list(bbox), "t1_scene": first.metadata,
                "t2_scene": second.metadata, "t1_date": str(first.metadata.get("acquired"))[:10],
                "t2_date": str(second.metadata.get("acquired"))[:10],
                "attribution": first.metadata.get("attribution")},
        warnings=warnings,
    )
    return _detail(state, row)
