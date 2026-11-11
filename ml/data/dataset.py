"""PyTorch dataset over the memmapped 256x256 LEVIR-CD patches produced by ``prepare.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet statistics: both backbones consume inputs normalised this way.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def normalize(batch_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B, H, W, 3] (or [B, 3, H, W] float in 0..1) -> normalised float [B, 3, H, W]."""
    x = batch_uint8
    if x.dtype == torch.uint8:
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class PatchPairDataset(Dataset):
    """Yields ``(A uint8 HWC, B uint8 HWC, mask uint8 HW)`` tensors for one split."""

    def __init__(self, root: str | Path, split: str, limit: int | None = None) -> None:
        root = Path(root)
        for suffix in ("A", "B", "M"):
            if not (root / f"{split}_{suffix}.npy").exists():
                raise FileNotFoundError(
                    f"{root / f'{split}_{suffix}.npy'} not found - run `python -m ml.data.prepare` first"
                )
        self.a = np.load(root / f"{split}_A.npy", mmap_mode="r")
        self.b = np.load(root / f"{split}_B.npy", mmap_mode="r")
        self.m = np.load(root / f"{split}_M.npy", mmap_mode="r")
        self.n = len(self.a) if limit is None else min(limit, len(self.a))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.a[idx])),
            torch.from_numpy(np.array(self.b[idx])),
            torch.from_numpy(np.array(self.m[idx])),
        )


def augment(
    a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-level geometric + photometric augmentation on normalised tensors.

    a, b: float [B, 3, H, W]; m: float [B, 1, H, W]. Geometric transforms are applied
    identically to both dates and the mask. Photometric jitter is applied per date so the
    model learns to ignore illumination / seasonal colour shifts. With probability 0.5 the
    temporal order is swapped (change is symmetric for binary change detection).
    """

    def rand() -> float:
        return float(torch.rand((), generator=generator))

    if rand() < 0.5:
        a, b, m = a.flip(-1), b.flip(-1), m.flip(-1)
    if rand() < 0.5:
        a, b, m = a.flip(-2), b.flip(-2), m.flip(-2)
    k = int(torch.randint(0, 4, (), generator=generator))
    if k:
        a, b, m = (torch.rot90(t, k, (-2, -1)) for t in (a, b, m))
    if rand() < 0.5:
        a, b = b, a
    for t in (a, b):
        if rand() < 0.8:
            gain = 1.0 + (torch.rand(t.shape[0], 1, 1, 1, generator=generator) - 0.5) * 0.4
            bias = (torch.rand(t.shape[0], 3, 1, 1, generator=generator) - 0.5) * 0.4
            t.mul_(gain.to(t.device)).add_(bias.to(t.device))
    return a, b, m
