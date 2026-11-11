"""Shared pytest fixtures.

The API tests run against a real FastAPI app wired to a temporary storage folder, a temporary
SQLite database and a *tiny* FC-Siam-diff checkpoint written into a temporary checkpoint
directory. That checkpoint has random weights and exists only to exercise the plumbing
(loading, inference, artefacts, persistence); no accuracy claim is ever derived from it.
Everything runs on CPU so the suite does not depend on, or disturb, the GPU.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
import torch
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from backend.app.config import Settings
from backend.app.main import create_app
from ml.checkpoint import save_checkpoint
from ml.models.registry import build_model

TINY_KWARGS = {"base": 4, "dropout": 0.0}


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def scene(height: int = 96, width: int = 96, seed: int = 0) -> np.ndarray:
    """Deterministic textured RGB image (smooth structure + noise) as uint8 [H, W, 3]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = 110 + 50 * np.sin(xx / 9.0) + 40 * np.cos(yy / 7.0)
    img = np.stack([base, base * 0.9 + 10, base * 0.8 + 20], axis=2) + rng.normal(0, 6, (height, width, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def with_building(img: np.ndarray, box: tuple[int, int, int, int] = (20, 20, 50, 60)) -> tuple[np.ndarray, np.ndarray]:
    """Paint a bright rectangle ("new building"); return (changed image, boolean mask)."""
    out = img.copy()
    y0, x0, y1, x1 = box
    out[y0:y1, x0:x1] = (235, 235, 225)
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = True
    return out, mask


def geotiff_bytes(array: np.ndarray, *, crs: str | None = "EPSG:32643", origin: tuple[float, float] = (780000.0, 1450000.0),
                  pixel: float = 10.0, dtype: str = "uint8", with_transform: bool = True) -> bytes:
    """Encode [H, W, 3] (or [H, W]) as an in-memory GeoTIFF."""
    bands = array[None] if array.ndim == 2 else np.transpose(array, (2, 0, 1))
    profile = {"driver": "GTiff", "height": bands.shape[1], "width": bands.shape[2], "count": bands.shape[0],
               "dtype": dtype}
    if crs:
        profile["crs"] = crs
    if with_transform:
        profile["transform"] = from_origin(origin[0], origin[1], pixel, pixel)
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(bands.astype(dtype))
        return mem.read()


def write_tiny_checkpoint(path: Path, key: str = "fc_siam_diff", kwargs: dict | None = None) -> None:
    torch.manual_seed(0)
    kwargs = TINY_KWARGS if kwargs is None else kwargs
    model = build_model(key, **kwargs)
    save_checkpoint(path, key, model, dict(kwargs),
                    meta={"run_name": "pytest-tiny", "epoch": 1, "val_f1": None, "val_iou": None, "threshold": 0.5,
                          "dataset": "synthetic (test fixture)", "seed": 0, "tile_size": 256,
                          "parameters": sum(p.numel() for p in model.parameters())})


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    write_tiny_checkpoint(folder / "fc_siam_diff_levir.pt")
    return folder


@pytest.fixture
def samples_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "samples"
    (folder / "synthetic").mkdir(parents=True)
    t1 = scene(128, 128, seed=1)
    t2, mask = with_building(t1, (30, 30, 80, 90))
    Image.fromarray(t1).save(folder / "synthetic/a_t1.png")
    Image.fromarray(t2).save(folder / "synthetic/a_t2.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "synthetic/a_gt.png")
    Image.fromarray(t1).save(folder / "synthetic/b_t1.png")
    Image.fromarray(scene(128, 128, seed=2)).save(folder / "synthetic/b_t2.png")
    manifest = {"samples": [
        {"id": "syn-a", "name": "Synthetic A", "description": "labelled", "dataset": "synthetic", "split": "test",
         "domain": "vhr", "georeferenced": False, "t1": "synthetic/a_t1.png", "t2": "synthetic/a_t2.png",
         "ground_truth": "synthetic/a_gt.png", "license": "test", "attribution": "test"},
        {"id": "syn-b", "name": "Synthetic B", "description": "unlabelled", "dataset": "synthetic", "split": None,
         "domain": "sentinel2", "georeferenced": False, "t1": "synthetic/b_t1.png", "t2": "synthetic/b_t2.png",
         "ground_truth": None, "license": "test", "attribution": "test"},
    ]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


@pytest.fixture
def settings(tmp_path: Path, checkpoint_dir: Path, samples_dir: Path) -> Settings:
    return Settings(
        storage_dir=tmp_path / "storage", checkpoint_dir=checkpoint_dir, reports_dir=tmp_path / "reports",
        samples_dir=samples_dir, device="cpu", default_model="fc_siam_diff", max_upload_mb=2,
        min_image_side=32, max_image_side=1024, tile_overlap=32, inference_batch_size=2,
        cdse_client_id="", cdse_client_secret="", log_level="WARNING",
    )


class MockSatellite:
    """Programmable httpx transport standing in for the remote STAC / geocoding services."""

    def __init__(self) -> None:
        self.handler = lambda request: httpx.Response(200, json={"features": []})
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


@pytest.fixture
def satellite() -> MockSatellite:
    return MockSatellite()


@pytest.fixture
def client(settings: Settings, satellite: MockSatellite):
    app = create_app(settings, provider_transport=httpx.MockTransport(satellite), log_to_file=False)
    with TestClient(app) as test_client:
        yield test_client


def wait_for(client: TestClient, analysis_id: str, timeout: float = 60.0) -> dict:
    """Poll an analysis until it leaves the queued/running states."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/analyses/{analysis_id}").json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"analysis {analysis_id} did not finish within {timeout}s")


def upload(client: TestClient, t1: bytes, t2: bytes, *, gt: bytes | None = None, names=("t1.png", "t2.png", "gt.png"),
           data: dict | None = None):
    files = {"t1": (names[0], t1), "t2": (names[1], t2)}
    if gt is not None:
        files["ground_truth"] = (names[2], gt)
    return client.post("/api/analyses/upload", files=files, data=data or {})
