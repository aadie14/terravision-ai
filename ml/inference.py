"""Sliding-window bitemporal inference for rasters of arbitrary size."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ml.data.dataset import normalize


@dataclass
class InferenceResult:
    probability: np.ndarray  # float32 [H, W] in 0..1
    seconds: float
    tiles: int
    device: str
    amp: bool


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile, stride))
    pos.append(length - tile)
    return pos


def _blend_window(tile: int, device: torch.device) -> torch.Tensor:
    """Separable Hann-like window (never zero) so overlapping tiles blend smoothly."""
    ramp = torch.hann_window(tile, periodic=False, device=device).clamp_min(0.05)
    return ramp[:, None] * ramp[None, :]


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cpu")


@torch.inference_mode()
def predict_pair(
    model: nn.Module,
    t1: np.ndarray,
    t2: np.ndarray,
    *,
    device: torch.device,
    tile: int = 256,
    overlap: int = 64,
    batch_size: int = 8,
    amp: bool = True,
) -> InferenceResult:
    """Return the per-pixel change probability for two aligned uint8 RGB images [H, W, 3]."""
    if t1.shape != t2.shape:
        raise ValueError(f"Image shapes differ: {t1.shape} vs {t2.shape}")
    if t1.ndim != 3 or t1.shape[2] != 3 or t1.dtype != np.uint8:
        raise ValueError("Images must be uint8 arrays shaped [H, W, 3]")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")

    h, w = t1.shape[:2]
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        mode = "reflect" if (h > pad_h and w > pad_w) else "edge"
        t1 = np.pad(t1, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        t2 = np.pad(t2, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    ph, pw = t1.shape[:2]

    use_amp = amp and device.type == "cuda"
    stride = tile - overlap
    coords = [(y, x) for y in _positions(ph, tile, stride) for x in _positions(pw, tile, stride)]
    window = _blend_window(tile, device)
    prob_sum = torch.zeros((ph, pw), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((ph, pw), dtype=torch.float32, device=device)

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        a = torch.from_numpy(np.stack([t1[y : y + tile, x : x + tile] for y, x in chunk])).to(device)
        b = torch.from_numpy(np.stack([t2[y : y + tile, x : x + tile] for y, x in chunk])).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(normalize(a), normalize(b))
        probs = torch.sigmoid(logits.float())[:, 0]
        for (y, x), p in zip(chunk, probs, strict=True):
            prob_sum[y : y + tile, x : x + tile] += p * window
            weight_sum[y : y + tile, x : x + tile] += window
    probability = (prob_sum / weight_sum)[:h, :w].clamp_(0, 1).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return InferenceResult(probability.astype(np.float32), seconds, len(coords), device.type, use_amp)
