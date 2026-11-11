"""Analysis endpoints: create (upload / sample / re-run), inspect, re-threshold, download, delete."""

from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from backend.app.db import Analysis
from backend.app.errors import AppError
from backend.app.schemas import (
    AnalysisCreate,
    AnalysisDetail,
    AnalysisList,
    AnalysisResults,
    AnalysisSummary,
    StatsResponse,
    ThresholdUpdate,
)
from backend.app.services.imaging import align_pair, load_image, load_mask
from backend.app.services.model_manager import domain_from_resolution, domain_warning, preferred_model
from backend.app.state import AppState, get_state
from ml.models.registry import MODEL_SPECS

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


def _degrees_to_m(image) -> float | None:
    """Pixel size in metres for a geographic (degree-based) raster, else None."""
    if image.geo is None:
        return None
    lat = abs(image.geo.bounds_wgs84()[1] + image.geo.bounds_wgs84()[3]) / 2
    return max(image.geo.resolution) * 111_320 * max(0.2, math.cos(math.radians(lat)))
_CHUNK = 1024 * 1024


async def _read_limited(upload: UploadFile, label: str, limit: int) -> bytes:
    """Read an upload fully, aborting as soon as it exceeds ``limit`` bytes."""
    chunks, size = [], 0
    while chunk := await upload.read(_CHUNK):
        size += len(chunk)
        if size > limit:
            raise AppError("FILE_TOO_LARGE", f"{label} exceeds the {limit // (1024 * 1024)} MB upload limit",
                           status=413, details={"limit_bytes": limit})
        chunks.append(chunk)
    return b"".join(chunks)


def _detail(state: AppState, row: Analysis) -> AnalysisDetail:
    summary = AnalysisSummary.model_validate(row).model_dump()
    return AnalysisDetail(**summary, inputs=row.inputs or {}, artifacts=state.analyses.available_artifacts(row))


@router.post("/upload", response_model=AnalysisDetail, status_code=202)
async def upload_analysis(
    request: Request,
    t1: Annotated[UploadFile | None, File(description="Earlier image (T1)")] = None,
    t2: Annotated[UploadFile | None, File(description="Later image (T2)")] = None,
    ground_truth: Annotated[UploadFile | None, File(description="Optional binary change mask")] = None,
    model: Annotated[str | None, Form(max_length=64)] = None,
    threshold: Annotated[float | None, Form(gt=0, lt=1)] = None,
    name: Annotated[str | None, Form(max_length=120)] = None,
    state: AppState = Depends(get_state),
) -> AnalysisDetail:
    s = state.settings
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > s.max_upload_bytes * 3 + _CHUNK:
        raise AppError("FILE_TOO_LARGE", "The request body exceeds the upload limit", status=413)
    missing = [label for label, f in (("t1", t1), ("t2", t2)) if f is None or not f.filename]
    if missing:
        raise AppError("MISSING_IMAGE", "Both a T1 (earlier) and a T2 (later) image are required", status=422,
                       details={"missing": missing})

    t1_bytes = await _read_limited(t1, "T1 image", s.max_upload_bytes)
    t2_bytes = await _read_limited(t2, "T2 image", s.max_upload_bytes)
    limits = {"min_side": s.min_image_side, "max_side": s.max_image_side}
    img1 = load_image(t1.filename, t1_bytes, **limits)
    img2 = load_image(t2.filename, t2_bytes, **limits)
    img1, img2, warnings = align_pair(img1, img2)
    image_domain = domain_from_resolution(max(img1.geo.resolution) if img1.geo and img1.geo.crs.is_projected
                                          else _degrees_to_m(img1))
    model_key = model or preferred_model(state.models, image_domain, s.default_model)
    if model_key not in MODEL_SPECS:
        raise AppError("UNKNOWN_MODEL", f"Model '{model_key}' does not exist", status=404,
                       details={"available": sorted(MODEL_SPECS)})
    thr = threshold if threshold is not None else state.models.default_threshold(model_key, s.default_threshold)
    state.analyses.validate_request(model_key, thr)
    if note := domain_warning(model_key, image_domain):
        warnings.append(note)
    truth = None
    if ground_truth is not None and ground_truth.filename:
        gt_bytes = await _read_limited(ground_truth, "Ground-truth mask", s.max_upload_bytes)
        truth = load_mask(ground_truth.filename, gt_bytes, expected_shape=(img1.height, img1.width),
                          max_side=s.max_image_side)
    row = state.analyses.create(
        t1=img1, t2=img2, truth=truth, model_key=model_key, threshold=thr, source="upload",
        name=name or f"{t1.filename} vs {t2.filename}",
        inputs={"t1_filename": t1.filename, "t2_filename": t2.filename,
                "ground_truth_filename": ground_truth.filename if truth is not None else None,
                "t1_format": img1.format, "t2_format": img2.format},
        warnings=warnings,
    )
    return _detail(state, row)


@router.post("", response_model=AnalysisDetail, status_code=202)
def create_analysis(body: AnalysisCreate, state: AppState = Depends(get_state)) -> AnalysisDetail:
    s = state.settings
    model_key = body.model or s.default_model
    if body.rerun_of is not None and body.model is None:
        model_key = state.analyses.get(body.rerun_of).model_key

    def thr_for(key: str) -> float:
        if body.threshold is not None:
            return body.threshold
        return state.models.default_threshold(key, s.default_threshold) if key in MODEL_SPECS else s.default_threshold

    if body.rerun_of is not None:
        return _detail(state, state.analyses.rerun(body.rerun_of, model_key, thr_for(model_key)))

    sample = state.samples.get(body.sample_id)
    if body.model is None:
        model_key = preferred_model(state.models, sample.get("domain"), s.default_model)
    threshold = thr_for(model_key)
    state.analyses.validate_request(model_key, threshold)
    limits = {"min_side": s.min_image_side, "max_side": s.max_image_side}
    t1_path, t2_path = state.samples.path(sample["id"], "t1"), state.samples.path(sample["id"], "t2")
    img1 = load_image(t1_path.name, t1_path.read_bytes(), **limits)
    img2 = load_image(t2_path.name, t2_path.read_bytes(), **limits)
    img1, img2, warnings = align_pair(img1, img2)
    truth = None
    if sample.get("ground_truth"):
        gt_path = state.samples.path(sample["id"], "ground_truth")
        truth = load_mask(gt_path.name, gt_path.read_bytes(), expected_shape=(img1.height, img1.width),
                          max_side=s.max_image_side)
    if note := domain_warning(model_key, sample.get("domain")):
        warnings.append(note)
    row = state.analyses.create(
        t1=img1, t2=img2, truth=truth, model_key=model_key, threshold=threshold, source="sample",
        name=sample["name"],
        inputs={"sample_id": sample["id"], "dataset": sample.get("dataset"), "license": sample.get("license"),
                "attribution": sample.get("attribution"), "t1_date": sample.get("t1_date"),
                "t2_date": sample.get("t2_date")},
        warnings=warnings,
    )
    return _detail(state, row)


@router.get("", response_model=AnalysisList)
def list_analyses(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[str | None, Query(pattern="^(queued|running|completed|failed)$")] = None,
    state: AppState = Depends(get_state),
) -> AnalysisList:
    with state.db.session() as session:
        query = select(Analysis)
        count = select(func.count()).select_from(Analysis)
        if status:
            query, count = query.where(Analysis.status == status), count.where(Analysis.status == status)
        rows = session.scalars(query.order_by(Analysis.created_at.desc()).limit(limit).offset(offset)).all()
        total = session.scalar(count) or 0
    return AnalysisList(items=[AnalysisSummary.model_validate(r) for r in rows], total=total, limit=limit,
                        offset=offset)


@router.get("/stats", response_model=StatsResponse)
def stats(state: AppState = Depends(get_state)) -> StatsResponse:
    with state.db.session() as session:
        by_status = dict(session.execute(select(Analysis.status, func.count()).group_by(Analysis.status)).all())
        by_model = dict(session.execute(select(Analysis.model_key, func.count()).group_by(Analysis.model_key)).all())
        mean_frac, mean_secs = session.execute(
            select(func.avg(Analysis.changed_fraction), func.avg(Analysis.inference_seconds))
            .where(Analysis.status == "completed")
        ).one()
        last = session.scalar(select(Analysis.changed_fraction).where(Analysis.status == "completed")
                              .order_by(Analysis.finished_at.desc()).limit(1))
    return StatsResponse(
        total=sum(by_status.values()), completed=by_status.get("completed", 0), failed=by_status.get("failed", 0),
        in_progress=by_status.get("queued", 0) + by_status.get("running", 0),
        mean_changed_percent=mean_frac * 100 if mean_frac is not None else None,
        last_changed_percent=last * 100 if last is not None else None,
        mean_inference_seconds=mean_secs, by_model=by_model,
    )


@router.get("/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: str, state: AppState = Depends(get_state)) -> AnalysisDetail:
    return _detail(state, state.analyses.get(analysis_id))


@router.get("/{analysis_id}/results", response_model=AnalysisResults)
def get_results(analysis_id: str, state: AppState = Depends(get_state)) -> AnalysisResults:
    row = state.analyses.get(analysis_id)
    if row.status == "failed":
        raise AppError("ANALYSIS_FAILED", row.error_message or "The analysis failed", status=409,
                       details={"error_code": row.error_code})
    if row.status != "completed" or row.results is None:
        raise AppError("ANALYSIS_NOT_READY", f"The analysis is still {row.status}", status=409,
                       details={"status": row.status})
    detail = _detail(state, row)
    downloads = {name: f"/api/analyses/{row.id}/downloads/{name}" for name in detail.artifacts}
    return AnalysisResults(analysis=detail, results=row.results, downloads=downloads)


@router.post("/{analysis_id}/threshold", response_model=AnalysisResults)
def update_threshold(analysis_id: str, body: ThresholdUpdate, state: AppState = Depends(get_state)) -> AnalysisResults:
    state.analyses.rethreshold(analysis_id, body.threshold)
    return get_results(analysis_id, state)


@router.get("/{analysis_id}/downloads/{artifact}")
def download(analysis_id: str, artifact: str,
             attachment: Annotated[bool, Query()] = False,
             state: AppState = Depends(get_state)) -> FileResponse:
    path, media_type, filename = state.analyses.artifact_path(analysis_id, artifact)
    headers = {"Cache-Control": "no-store"}
    if attachment:
        return FileResponse(path, media_type=media_type, filename=filename, headers=headers)
    return FileResponse(path, media_type=media_type, headers=headers)


@router.delete("/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, state: AppState = Depends(get_state)) -> Response:
    state.analyses.delete(analysis_id)
    return Response(status_code=204)
