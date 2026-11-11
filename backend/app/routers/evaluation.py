"""Evaluation endpoints: offline benchmark summary and quick in-app evaluation on labelled samples."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select

from backend.app.db import EvaluationRun
from backend.app.errors import AppError
from backend.app.schemas import EvaluationRequest
from backend.app.services import artifacts
from backend.app.services.analysis import new_id
from backend.app.services.imaging import align_pair, load_image, load_mask
from backend.app.state import AppState, get_state
from ml.metrics import ConfusionCounts, compute_metrics
from ml.models.registry import MODEL_SPECS

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
_REPORT_FILES = {
    "model_comparison": "model_comparison.png",
    "model_comparison_oscd": "model_comparison_oscd.png",
    "metrics_csv": "metrics.csv",
    **{f"{key}_{kind}": f"{key}_{kind}.png" for key in MODEL_SPECS for kind in ("confusion_matrix", "examples")},
}


@router.post("")
def run_evaluation(body: EvaluationRequest, state: AppState = Depends(get_state)) -> dict:
    """Run every requested model over the labelled bundled samples and pool the pixel counts."""
    s = state.settings
    keys = body.models or [m["key"] for m in state.models.list_models() if m["available"]]
    for key in keys:
        if key not in MODEL_SPECS:
            raise AppError("UNKNOWN_MODEL", f"Model '{key}' does not exist", status=404)
        state.analyses.validate_request(key, body.threshold or state.models.default_threshold(key, s.default_threshold))
    labelled = [x for x in state.samples.list() if x["has_ground_truth"]]
    if body.sample_ids is not None:
        wanted = set(body.sample_ids)
        unknown = wanted - {x["id"] for x in state.samples.list()}
        if unknown:
            raise AppError("NOT_FOUND", f"Unknown sample ids: {sorted(unknown)}", status=404)
        labelled = [x for x in labelled if x["id"] in wanted]
    if not keys:
        raise AppError("MODEL_UNAVAILABLE", "No model has a trained checkpoint to evaluate", status=503)
    if not labelled:
        raise AppError("NO_GROUND_TRUTH", "None of the selected samples has a ground-truth mask, so "
                       "accuracy metrics cannot be computed", status=422)

    limits = {"min_side": s.min_image_side, "max_side": s.max_image_side}
    pairs = []
    for sample in labelled:
        p1, p2 = state.samples.path(sample["id"], "t1"), state.samples.path(sample["id"], "t2")
        gt = state.samples.path(sample["id"], "ground_truth")
        img1, img2, _ = align_pair(load_image(p1.name, p1.read_bytes(), **limits),
                                   load_image(p2.name, p2.read_bytes(), **limits))
        truth = load_mask(gt.name, gt.read_bytes(), expected_shape=(img1.height, img1.width),
                          max_side=s.max_image_side)
        pairs.append((sample, img1.array, img2.array, truth))

    models_out = []
    for key in keys:
        total = ConfusionCounts()
        per_sample, seconds = [], 0.0
        started = time.perf_counter()
        own = [x for x in pairs if x[0].get("domain", "vhr") == MODEL_SPECS[key].domain]
        if not own:
            continue
        thr = body.threshold or state.models.default_threshold(key, s.default_threshold)
        for sample, a, b, truth in own:
            result, _loaded = state.models.predict(key, a, b)
            mask = artifacts.threshold_mask(artifacts.probability_to_uint8(result.probability), thr)
            counts = ConfusionCounts()
            counts.update(mask, truth)
            total = total + counts
            seconds += result.seconds
            m = compute_metrics(counts)
            per_sample.append({"sample_id": sample["id"], "name": sample["name"], "f1": m.f1, "iou": m.iou,
                               "precision": m.precision, "recall": m.recall,
                               "inference_seconds": result.seconds})
        models_out.append({
            "model": key, "display_name": MODEL_SPECS[key].display_name,
            "checkpoint": state.models.describe(key)["checkpoint"],
            **compute_metrics(total).to_dict(),
            "threshold": thr, "samples": [x[0]["id"] for x in own], "dataset": own[0][0].get("dataset"),
            "mean_inference_seconds": seconds / len(own), "wall_seconds": time.perf_counter() - started,
            "device": result.device, "per_sample": per_sample,
        })
    payload = {
        "scope": "bundled labelled samples (in-app check; NOT the full test-split benchmark)",
        "samples": [x[0]["id"] for x in pairs], "threshold": body.threshold, "models": models_out,
    }
    run = EvaluationRun(id=new_id(), dataset="bundled-samples", num_pairs=len(pairs), threshold=body.threshold or 0.0,
                        results=payload)
    with state.db.session() as session:
        session.add(run)
        session.commit()
    return {"id": run.id, "created_at": run.created_at.isoformat(), **payload}


@router.get("/summary")
def summary(state: AppState = Depends(get_state)) -> dict:
    """Offline test-split benchmark (written by ``python -m ml.evaluate``) plus recent in-app runs."""
    benchmark = None
    path = state.settings.reports_path / "summary.json"
    if path.is_file():
        try:
            benchmark = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            benchmark = None
    with state.db.session() as session:
        runs = session.scalars(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(5)).all()
    figures = {name: f"/api/evaluation/reports/{name}" for name, filename in _REPORT_FILES.items()
               if (state.settings.reports_path / filename).is_file()}
    return {
        "benchmark": benchmark,
        "benchmark_available": benchmark is not None and bool(benchmark.get("models")),
        "figures": figures,
        "sample_runs": [{"id": r.id, "created_at": r.created_at.isoformat(), **r.results} for r in runs],
    }


@router.get("/reports/{name}")
def report_file(name: str, state: AppState = Depends(get_state)) -> FileResponse:
    if name not in _REPORT_FILES:
        raise AppError("NOT_FOUND", f"Report '{name}' does not exist", status=404)
    path = state.settings.reports_path / _REPORT_FILES[name]
    if not path.is_file():
        raise AppError("NOT_FOUND", f"Report '{name}' has not been generated yet", status=404)
    return FileResponse(path)
