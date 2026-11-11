"""Analysis lifecycle: persist inputs, run inference in a background worker, store results."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import Affine
from sqlalchemy import select

from backend.app.config import Settings
from backend.app.db import Analysis, Database, utcnow
from backend.app.errors import AppError, NotFoundError
from backend.app.services import artifacts
from backend.app.services import geo as geo_service
from backend.app.services.imaging import GeoInfo, LoadedImage, pair_sanity_warnings
from backend.app.services.model_manager import ModelManager
from ml.models.registry import MODEL_SPECS

log = logging.getLogger("terravision.analysis")


def new_id() -> str:
    return uuid.uuid4().hex


def _save_geo(folder: Path, geo: GeoInfo) -> None:
    (folder / "geo.json").write_text(json.dumps({
        "crs_wkt": geo.crs.to_wkt(), "transform": list(geo.transform)[:6],
        "width": geo.width, "height": geo.height,
    }))


def _load_geo(folder: Path) -> GeoInfo | None:
    path = folder / "geo.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return GeoInfo(CRS.from_wkt(data["crs_wkt"]), Affine(*data["transform"]), data["width"], data["height"])


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


class AnalysisService:
    def __init__(self, settings: Settings, db: Database, models: ModelManager) -> None:
        self.settings = settings
        self.db = db
        self.models = models
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        settings.analyses_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- helpers
    def folder(self, analysis_id: str) -> Path:
        # ids are generated server-side as 32 hex chars; reject anything else before touching disk
        if len(analysis_id) != 32 or any(c not in "0123456789abcdef" for c in analysis_id):
            raise NotFoundError("Analysis", analysis_id)
        return self.settings.analyses_path / analysis_id

    def validate_request(self, model_key: str, threshold: float) -> None:
        if model_key not in MODEL_SPECS:
            raise AppError("UNKNOWN_MODEL", f"Model '{model_key}' does not exist", status=404,
                           details={"available": sorted(MODEL_SPECS)})
        if not 0.0 < threshold < 1.0:
            raise AppError("INVALID_THRESHOLD", "Threshold must be between 0 and 1 (exclusive)", status=422)
        status = self.models.describe(model_key)
        if not status["available"]:
            raise AppError("MODEL_UNAVAILABLE",
                           f"{status['display_name']} cannot run: {status['unavailable_reason']}",
                           status=503, details={"model": model_key})

    # ----------------------------------------------------------------- create
    def create(self, *, t1: LoadedImage, t2: LoadedImage, truth: np.ndarray | None, model_key: str,
               threshold: float, source: str, name: str, inputs: dict, warnings: list[str]) -> Analysis:
        self.validate_request(model_key, threshold)
        sanity, stats = pair_sanity_warnings(t1.array, t2.array)
        analysis_id = new_id()
        folder = self.folder(analysis_id)
        folder.mkdir(parents=True)
        try:
            artifacts.save_png(folder / "t1.png", t1.array)
            artifacts.save_png(folder / "t2.png", t2.array)
            if truth is not None:
                artifacts.save_png(folder / "ground_truth.png", truth.astype(np.uint8) * 255)
            geo = t1.geo or t2.geo
            if geo is not None:
                _save_geo(folder, geo)
            row = Analysis(
                id=analysis_id, name=name[:200], status="queued", source=source, model_key=model_key,
                threshold=threshold, width=t1.width, height=t1.height, georeferenced=geo is not None,
                has_ground_truth=truth is not None,
                inputs={**inputs, "warnings": [*warnings, *t1.notes, *t2.notes, *sanity], "pair_checks": stats},
            )
            with self.db.session() as session:
                session.add(row)
                session.commit()
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        self.submit(analysis_id)
        return row

    def rerun(self, source_id: str, model_key: str, threshold: float) -> Analysis:
        """Create a new analysis that reuses the stored inputs of an existing one."""
        self.validate_request(model_key, threshold)
        with self.db.session() as session:
            original = session.get(Analysis, source_id)
        src_folder = self.folder(source_id)
        if original is None or not (src_folder / "t1.png").is_file():
            raise NotFoundError("Analysis", source_id)
        analysis_id = new_id()
        folder = self.folder(analysis_id)
        folder.mkdir(parents=True)
        for name in ("t1.png", "t2.png", "ground_truth.png", "geo.json"):
            if (src_folder / name).is_file():
                shutil.copy2(src_folder / name, folder / name)
        row = Analysis(
            id=analysis_id, name=original.name, status="queued", source="rerun", model_key=model_key,
            threshold=threshold, width=original.width, height=original.height,
            georeferenced=original.georeferenced, has_ground_truth=original.has_ground_truth,
            inputs={**(original.inputs or {}), "rerun_of": source_id},
        )
        with self.db.session() as session:
            session.add(row)
            session.commit()
        self.submit(analysis_id)
        return row

    # -------------------------------------------------------------------- run
    def submit(self, analysis_id: str) -> None:
        self._executor.submit(self._run_safely, analysis_id)

    def _run_safely(self, analysis_id: str) -> None:
        try:
            self.run(analysis_id)
        except AppError as exc:
            self._fail(analysis_id, exc.code, exc.message)
        except Exception as exc:
            log.exception("analysis failed", extra={"analysis_id": analysis_id})
            self._fail(analysis_id, "INFERENCE_FAILED", f"{type(exc).__name__}: {exc}")

    def _fail(self, analysis_id: str, code: str, message: str) -> None:
        with self.db.session() as session:
            row = session.get(Analysis, analysis_id)
            if row is not None:
                row.status, row.error_code, row.error_message = "failed", code, message
                row.finished_at = utcnow()
                session.commit()
        log.warning("analysis marked failed", extra={"analysis_id": analysis_id, "code": code})

    def run(self, analysis_id: str) -> None:
        folder = self.folder(analysis_id)
        with self.db.session() as session:
            row = session.get(Analysis, analysis_id)
            if row is None:
                return
            row.status, row.started_at = "running", utcnow()
            session.commit()
            model_key, threshold = row.model_key, row.threshold

        t1, t2 = _read_rgb(folder / "t1.png"), _read_rgb(folder / "t2.png")
        truth = None
        if (folder / "ground_truth.png").is_file():
            with Image.open(folder / "ground_truth.png") as img:
                truth = np.asarray(img.convert("L")) > 127
        geo = _load_geo(folder)

        result, loaded = self.models.predict(model_key, t1, t2)
        prob_u8 = artifacts.probability_to_uint8(result.probability)
        artifacts.save_png(folder / "probability.png", prob_u8)
        artifacts.save_png(folder / "confidence.png", artifacts.render_confidence(prob_u8))
        ckpt = self.models.describe(model_key)["checkpoint"]
        tags = {"TERRAVISION_MODEL": model_key, "TERRAVISION_CHECKPOINT": ckpt["version"]}
        outputs = artifacts.build_outputs(folder, t2=t2, prob_u8=prob_u8, threshold=threshold, truth=truth,
                                          geo=geo, tags=tags)
        if geo is not None:
            rgba = np.dstack([t2, np.full(t2.shape[:2], 255, dtype=np.uint8)])
            geo_service.web_mercator_overlay(folder / "map_t2.png", rgba, geo)

        results = {
            "analysis_id": analysis_id,
            "model": {"key": model_key, "display_name": loaded.spec.display_name, "family": loaded.spec.family,
                      "parameters": loaded.parameters, "checkpoint": ckpt},
            "inference": {"seconds": result.seconds, "tiles": result.tiles, "tile_size": loaded.spec.tile_size,
                          "tile_overlap": self.settings.tile_overlap, "device": result.device,
                          "amp": result.amp},
            "image": {"width": int(t1.shape[1]), "height": int(t1.shape[0])},
            "georeferenced": geo is not None,
            "has_ground_truth": truth is not None,
            **outputs,
        }
        self._write_results(folder, results)
        with self.db.session() as session:
            row = session.get(Analysis, analysis_id)
            if row is None:  # deleted while running
                shutil.rmtree(folder, ignore_errors=True)
                return
            row.status, row.finished_at = "completed", utcnow()
            row.results = results
            row.changed_fraction = outputs["statistics"]["changed_fraction"]
            row.inference_seconds = result.seconds
            session.commit()
        log.info("analysis completed", extra={"analysis_id": analysis_id, "model": model_key,
                                              "seconds": round(result.seconds, 3), "device": result.device,
                                              "changed_percent": outputs["statistics"]["changed_percent"]})

    @staticmethod
    def _write_results(folder: Path, results: dict) -> None:
        (folder / "results.json").write_text(json.dumps(results, indent=2))

    # ------------------------------------------------------------ re-threshold
    def rethreshold(self, analysis_id: str, threshold: float) -> Analysis:
        if not 0.0 < threshold < 1.0:
            raise AppError("INVALID_THRESHOLD", "Threshold must be between 0 and 1 (exclusive)", status=422)
        folder = self.folder(analysis_id)
        with self.db.session() as session:
            row = session.get(Analysis, analysis_id)
            if row is None:
                raise NotFoundError("Analysis", analysis_id)
            if row.status != "completed" or not (folder / "probability.png").is_file():
                raise AppError("ANALYSIS_NOT_READY", "Only completed analyses can be re-thresholded", status=409)
            with Image.open(folder / "probability.png") as img:
                prob_u8 = np.asarray(img.convert("L"))
            truth = None
            if (folder / "ground_truth.png").is_file():
                with Image.open(folder / "ground_truth.png") as img:
                    truth = np.asarray(img.convert("L")) > 127
            results = dict(row.results or {})
            tags = {"TERRAVISION_MODEL": row.model_key,
                    "TERRAVISION_CHECKPOINT": results.get("model", {}).get("checkpoint", {}).get("version", "")}
            outputs = artifacts.build_outputs(folder, t2=_read_rgb(folder / "t2.png"), prob_u8=prob_u8,
                                              threshold=threshold, truth=truth, geo=_load_geo(folder), tags=tags)
            results.update(outputs)
            self._write_results(folder, results)
            row.results, row.threshold = results, threshold
            row.changed_fraction = outputs["statistics"]["changed_fraction"]
            session.commit()
            return row

    # ------------------------------------------------------------------ query
    def get(self, analysis_id: str) -> Analysis:
        self.folder(analysis_id)  # id format check
        with self.db.session() as session:
            row = session.get(Analysis, analysis_id)
        if row is None:
            raise NotFoundError("Analysis", analysis_id)
        return row

    def available_artifacts(self, row: Analysis) -> list[str]:
        folder = self.folder(row.id)
        return [name for name, (filename, *_rest) in artifacts.ARTIFACTS.items() if (folder / filename).is_file()]

    def artifact_path(self, analysis_id: str, artifact: str) -> tuple[Path, str, str]:
        row = self.get(analysis_id)
        if artifact not in artifacts.ARTIFACTS:
            raise AppError("UNKNOWN_ARTIFACT", f"'{artifact}' is not a downloadable artefact", status=404,
                           details={"available": self.available_artifacts(row)})
        filename, media_type, needs_geo, needs_truth = artifacts.ARTIFACTS[artifact]
        path = self.folder(analysis_id) / filename
        if not path.is_file():
            if needs_geo and not row.georeferenced:
                raise AppError("NOT_GEOREFERENCED", "This artefact needs georeferenced input imagery; the "
                               "uploaded images carry no spatial reference", status=409)
            if needs_truth and not row.has_ground_truth:
                raise AppError("NO_GROUND_TRUTH", "This artefact needs a ground-truth mask, which was not "
                               "provided for this analysis", status=409)
            raise AppError("ARTIFACT_NOT_READY", f"'{artifact}' is not available yet (status: {row.status})",
                           status=409)
        return path, media_type, f"terravision_{analysis_id[:8]}_{filename}"

    def delete(self, analysis_id: str) -> None:
        folder = self.folder(analysis_id)
        with self.db.session() as session:
            row = session.get(Analysis, analysis_id)
            if row is None:
                raise NotFoundError("Analysis", analysis_id)
            running = row.status == "running"
            session.delete(row)
            session.commit()
        if not running:  # a running job removes its own folder when it notices the row is gone
            shutil.rmtree(folder, ignore_errors=True)

    # ------------------------------------------------------------ maintenance
    def recover_interrupted(self) -> int:
        """Mark jobs left queued/running by a previous process as failed (they cannot resume)."""
        with self.db.session() as session:
            rows = session.scalars(select(Analysis).where(Analysis.status.in_(("queued", "running")))).all()
            for row in rows:
                row.status, row.error_code = "failed", "INTERRUPTED"
                row.error_message = "The server stopped before this analysis finished."
                row.finished_at = utcnow()
            session.commit()
            return len(rows)

    def cleanup_orphans(self) -> int:
        """Remove analysis folders that have no database row."""
        with self.db.session() as session:
            known = set(session.scalars(select(Analysis.id)).all())
        removed = 0
        for child in self.settings.analyses_path.iterdir():
            if child.is_dir() and child.name not in known:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        return removed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
