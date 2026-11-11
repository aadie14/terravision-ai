"""Rendering and post-processing of analysis artefacts from a stored probability map."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.app.services import geo as geo_service
from backend.app.services.imaging import GeoInfo
from ml.metrics import metrics_from_masks

CHANGE_COLOR = (255, 59, 92)  # overlay colour for predicted change
ERROR_COLORS = {"tp": (46, 204, 113), "fp": (255, 59, 92), "fn": (52, 152, 219)}

# Artefact name -> (file name, media type, requires geo, requires ground truth)
ARTIFACTS: dict[str, tuple[str, str, bool, bool]] = {
    "t1": ("t1.png", "image/png", False, False),
    "t2": ("t2.png", "image/png", False, False),
    "probability": ("probability.png", "image/png", False, False),
    "confidence": ("confidence.png", "image/png", False, False),
    "mask": ("mask.png", "image/png", False, False),
    "overlay": ("overlay.png", "image/png", False, False),
    "ground_truth": ("ground_truth.png", "image/png", False, True),
    "error_map": ("error_map.png", "image/png", False, True),
    "results": ("results.json", "application/json", False, False),
    "mask_geotiff": ("mask.tif", "image/tiff", True, False),
    "regions_geojson": ("regions.geojson", "application/geo+json", True, False),
    "map_overlay": ("map_overlay.png", "image/png", True, False),
    "map_t2": ("map_t2.png", "image/png", True, False),
}


def save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path, optimize=False)


def probability_to_uint8(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)


def threshold_mask(prob_u8: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask from the stored 8-bit probability map (same rule used everywhere)."""
    return prob_u8 > round(threshold * 255)


def render_overlay(base: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = base.astype(np.float32)
    color = np.array(CHANGE_COLOR, dtype=np.float32)
    out[mask] = out[mask] * (1 - alpha) + color * alpha
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, CHANGE_COLOR, 1)
    return out


def render_confidence(prob_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.applyColorMap(prob_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def render_error_map(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    out = np.full((*pred.shape, 3), 18, dtype=np.uint8)
    out[pred & truth] = ERROR_COLORS["tp"]
    out[pred & ~truth] = ERROR_COLORS["fp"]
    out[~pred & truth] = ERROR_COLORS["fn"]
    return out


def prediction_statistics(prob_u8: np.ndarray, mask: np.ndarray) -> dict:
    """Descriptive statistics that need no ground truth."""
    prob = prob_u8.astype(np.float32) / 255.0
    changed = int(np.count_nonzero(mask))
    total = int(mask.size)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([], dtype=np.int64)
    hist, _ = np.histogram(prob, bins=10, range=(0.0, 1.0))
    return {
        "total_pixels": total,
        "changed_pixels": changed,
        "changed_fraction": changed / total if total else 0.0,
        "changed_percent": 100.0 * changed / total if total else 0.0,
        "region_count": len(areas),
        "largest_region_pixels": int(areas.max()) if len(areas) else 0,
        "mean_probability": float(prob.mean()),
        "max_probability": float(prob.max()),
        "mean_probability_in_change": float(prob[mask].mean()) if changed else None,
        "uncertain_fraction": float(np.count_nonzero((prob > 0.3) & (prob < 0.7)) / total) if total else 0.0,
        "probability_histogram": {"bin_edges": [round(i / 10, 1) for i in range(11)],
                                  "counts": [int(c) for c in hist]},
    }


def build_outputs(folder: Path, *, t2: np.ndarray, prob_u8: np.ndarray, threshold: float,
                  truth: np.ndarray | None, geo: GeoInfo | None, tags: dict | None = None) -> dict:
    """(Re)generate every threshold-dependent artefact and return the threshold-dependent results."""
    mask = threshold_mask(prob_u8, threshold)
    save_png(folder / "mask.png", mask.astype(np.uint8) * 255)
    save_png(folder / "overlay.png", render_overlay(t2, mask))
    out: dict = {"threshold": threshold, "statistics": prediction_statistics(prob_u8, mask), "metrics": None,
                 "geospatial": None}
    if truth is not None:
        out["metrics"] = metrics_from_masks(mask, truth).to_dict()
        out["metrics"]["ground_truth_changed_fraction"] = float(truth.mean())
        save_png(folder / "error_map.png", render_error_map(mask, truth))
    if geo is not None:
        geo_service.write_geotiff(folder / "mask.tif", mask.astype(np.uint8), geo, nodata=None,
                                  tags={"TERRAVISION_THRESHOLD": threshold, **(tags or {})})
        collection, summary = geo_service.change_regions(mask, geo)
        (folder / "regions.geojson").write_text(json.dumps(collection))
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask] = (*CHANGE_COLOR, 255)
        bounds = geo_service.web_mercator_overlay(folder / "map_overlay.png", rgba, geo)
        total_area = geo_service.footprint_area_m2(geo)
        out["geospatial"] = {**geo.describe(), **summary, "footprint_area_m2": total_area,
                             "footprint_area_km2": total_area / 1e6, "map_overlay_bounds": bounds}
    return out
