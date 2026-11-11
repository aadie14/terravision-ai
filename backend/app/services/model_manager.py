"""Model lifecycle: load each checkpoint once, keep it in memory, serialise GPU access."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from backend.app.config import Settings
from backend.app.errors import AppError
from ml.checkpoint import CheckpointError, load_model, read_checkpoint
from ml.inference import InferenceResult, predict_pair, resolve_device
from ml.models.registry import MODEL_SPECS, ModelSpec, count_parameters

log = logging.getLogger("terravision.models")


@dataclass
class LoadedModel:
    spec: ModelSpec
    module: torch.nn.Module
    info: dict
    parameters: int
    mtime: float = 0.0


DOMAIN_LABELS = {
    "vhr": "very-high-resolution imagery (LEVIR-CD, 0.5 m/pixel)",
    "sentinel2": "Sentinel-2 imagery (OSCD, 10 m/pixel)",
}


def domain_from_resolution(resolution_m: float | None) -> str | None:
    """Best guess of the imagery domain from a ground pixel size in metres (None when unknown)."""
    if resolution_m is None:
        return None
    return "sentinel2" if resolution_m >= 5.0 else "vhr" if resolution_m <= 2.0 else None


def domain_warning(model_key: str, image_domain: str | None) -> str | None:
    spec = MODEL_SPECS[model_key]
    if image_domain is None or image_domain == spec.domain:
        return None
    return (f"This imagery looks like {DOMAIN_LABELS[image_domain]}, but {spec.display_name} was trained on "
            f"{DOMAIN_LABELS[spec.domain]}. Predictions are out-of-distribution and may be unreliable; choose a "
            f"{'Sentinel-2' if image_domain == 'sentinel2' else 'LEVIR-CD'} model for this input.")


def preferred_model(models: ModelManager, domain: str | None, fallback: str) -> str:
    """The default-family model for ``domain`` when it has a usable checkpoint, else ``fallback``."""
    if domain is None:
        return fallback
    family = MODEL_SPECS[fallback].family
    candidates = sorted((k for k, s in MODEL_SPECS.items() if s.domain == domain),
                        key=lambda k: MODEL_SPECS[k].family != family)
    for key in candidates:
        if models.describe(key)["available"]:
            return key
    return fallback


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded: dict[str, LoadedModel] = {}
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # one inference at a time on the single GPU
        self._device: torch.device | None = None
        self._status_cache: dict[str, tuple[float, dict]] = {}

    # ------------------------------------------------------------------ device
    @property
    def device(self) -> torch.device:
        if self._device is None:
            try:
                self._device = resolve_device(self.settings.device)
            except RuntimeError as exc:
                log.warning("Requested device unavailable, falling back to CPU: %s", exc)
                self._device = torch.device("cpu")
        return self._device

    def device_info(self) -> dict:
        info: dict = {
            "device": self.device.type,
            "requested": self.settings.device,
            "cuda_available": torch.cuda.is_available(),
            "amp": bool(self.settings.amp and self.device.type == "cuda"),
            "torch": torch.__version__,
        }
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_mb": total // 2**20,
                "gpu_memory_free_mb": free // 2**20,
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() // 2**20,
                "cuda_version": torch.version.cuda,
            })
        return info

    # --------------------------------------------------------------- registry
    def checkpoint_file(self, key: str) -> Path:
        return self.settings.checkpoint_path / MODEL_SPECS[key].default_checkpoint

    def _benchmark(self, key: str) -> dict | None:
        path = self.settings.reports_path / f"{key}_metrics.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        keep = ("dataset", "evaluation", "split", "num_patches", "num_images", "threshold", "precision", "recall",
                "f1", "iou", "accuracy", "tp", "fp", "tn", "fn", "latency_ms_median", "latency_ms_p95",
                "throughput_patches_per_s", "peak_gpu_mb", "device", "gpu_name", "evaluated_utc", "checkpoint_sha256",
                "confusion_matrix")
        return {k: data.get(k) for k in keep}

    def describe(self, key: str) -> dict:
        spec = MODEL_SPECS[key]
        path = self.checkpoint_file(key)
        entry: dict = {
            "key": key, "display_name": spec.display_name, "family": spec.family,
            "architecture": spec.architecture, "reference": spec.reference, "tile_size": spec.tile_size,
            "domain": spec.domain, "domain_label": DOMAIN_LABELS[spec.domain],
            "checkpoint_file": spec.default_checkpoint, "loaded": key in self._loaded,
            "available": False, "unavailable_reason": None, "parameters": None, "checkpoint": None,
            "benchmark": None,
        }
        loaded = self._loaded.get(key)
        if loaded is not None and (not path.is_file() or path.stat().st_mtime != loaded.mtime):
            loaded = None
        if loaded is not None:
            entry.update(available=True, parameters=loaded.parameters, checkpoint=self._public_ckpt(loaded.info))
        else:
            mtime = path.stat().st_mtime if path.is_file() else -1.0
            cached = self._status_cache.get(key)
            if cached is None or cached[0] != mtime:
                status: dict = {"available": False, "unavailable_reason": None, "checkpoint": None}
                try:
                    payload = read_checkpoint(path)
                    status["available"] = True
                    status["checkpoint"] = self._public_ckpt({"meta": payload["meta"], "file": path.name,
                                                              "sha256": None})
                    status["parameters"] = payload["meta"].get("parameters")
                    del payload
                except CheckpointError as exc:
                    status["unavailable_reason"] = str(exc)
                self._status_cache[key] = (mtime, status)
                cached = self._status_cache[key]
            entry.update(cached[1])
        entry["benchmark"] = self._benchmark(key)
        if entry["benchmark"] and entry["checkpoint"] and entry["checkpoint"].get("sha256"):
            entry["benchmark"]["matches_loaded_checkpoint"] = (
                entry["benchmark"].get("checkpoint_sha256") == entry["checkpoint"]["sha256"]
            )
        return entry

    @staticmethod
    def _public_ckpt(info: dict) -> dict:
        meta = info["meta"]
        sha = info.get("sha256")
        return {
            "file": info["file"],
            "sha256": sha,
            "version": f"{meta.get('run_name', 'unknown')}@epoch{meta.get('epoch', '?')}"
                       + (f"#{sha[:8]}" if sha else ""),
            "run_name": meta.get("run_name"),
            "epoch": meta.get("epoch"),
            "val_f1": meta.get("val_f1"),
            "val_iou": meta.get("val_iou"),
            "trained_on": meta.get("dataset"),
            "saved_utc": meta.get("saved_utc"),
            "seed": meta.get("seed"),
            "threshold": meta.get("threshold"),
            "threshold_selection": meta.get("threshold_selection"),
        }

    def default_threshold(self, key: str, fallback: float) -> float:
        """The checkpoint's validation-selected threshold, else the configured default."""
        ckpt = self.describe(key).get("checkpoint") or {}
        value = ckpt.get("threshold")
        return float(value) if isinstance(value, int | float) and 0 < value < 1 else fallback

    def list_models(self) -> list[dict]:
        return [self.describe(key) for key in MODEL_SPECS]

    # ----------------------------------------------------------------- loading
    def get(self, key: str) -> LoadedModel:
        if key not in MODEL_SPECS:
            raise AppError("UNKNOWN_MODEL", f"Model '{key}' does not exist", status=404,
                           details={"available": sorted(MODEL_SPECS)})
        with self._load_lock:
            path = self.checkpoint_file(key)
            mtime = path.stat().st_mtime if path.is_file() else -1.0
            cached = self._loaded.get(key)
            if cached is not None and cached.mtime == mtime:
                return cached
            if cached is not None:
                # The checkpoint file was replaced (e.g. training saved a better epoch): never keep
                # serving the stale weights under the new file's name (BUG-004).
                log.info("checkpoint changed on disk, reloading", extra={"model": key})
                del self._loaded[key]
            try:
                module, info = load_model(path, expected_key=key)
            except CheckpointError as exc:
                raise AppError("MODEL_UNAVAILABLE",
                               f"{MODEL_SPECS[key].display_name} has no usable trained checkpoint: {exc}",
                               status=503, details={"model": key}) from exc
            try:
                module.to(self.device)
            except RuntimeError as exc:  # e.g. CUDA out of memory -> degrade to CPU
                log.warning("Could not move %s to %s (%s); using CPU", key, self.device, exc)
                self._device = torch.device("cpu")
                module.to(self._device)
            loaded = LoadedModel(MODEL_SPECS[key], module, info, count_parameters(module), mtime)
            self._loaded[key] = loaded
            log.info("model loaded", extra={"model": key, "device": self.device.type,
                                            "checkpoint": info["file"], "sha256": info["sha256"][:12]})
            return loaded

    def predict(self, key: str, t1: np.ndarray, t2: np.ndarray) -> tuple[InferenceResult, LoadedModel]:
        loaded = self.get(key)
        kwargs = {"tile": loaded.spec.tile_size, "overlap": self.settings.tile_overlap,
                  "batch_size": self.settings.inference_batch_size, "amp": self.settings.amp}
        with self._infer_lock:
            try:
                result = predict_pair(loaded.module, t1, t2, device=self.device, **kwargs)
            except torch.OutOfMemoryError:
                log.warning("CUDA out of memory; retrying with batch size 1")
                torch.cuda.empty_cache()
                kwargs["batch_size"] = 1
                result = predict_pair(loaded.module, t1, t2, device=self.device, **kwargs)
        return result, loaded

    def unload_all(self) -> None:
        with self._load_lock:
            self._loaded.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
