"""Validate LEVIR-CD and tile it into 256x256 patches stored as NumPy memmaps.

The official 1024x1024 image pairs are cut into a non-overlapping 4x4 grid (16 patches per
pair). Tiling happens *inside* each official split, so no source image ever contributes
patches to more than one split (no train/test leakage).

Outputs, per split, under ``--out``:
    {split}_A.npy, {split}_B.npy   uint8  [N, 256, 256, 3]
    {split}_M.npy                  uint8  [N, 256, 256]   (0 = no change, 1 = change)
    manifest.json                  reproducible split manifest (file names, counts, hashes)

Usage:  python -m ml.data.prepare --root data/raw/LEVIR-CD --out data/processed/levir256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
PATCH = 256


class DatasetError(RuntimeError):
    """Raised when the raw dataset is missing files or is structurally invalid."""


def list_split(root: Path, split: str) -> list[str]:
    """Return sorted file names present in A, B and label for ``split``; validate pairing."""
    names: dict[str, set[str]] = {}
    for sub in ("A", "B", "label"):
        folder = root / split / sub
        if not folder.is_dir():
            raise DatasetError(f"Missing folder: {folder}")
        names[sub] = {p.name for p in folder.glob("*.png")}
    if not names["A"]:
        raise DatasetError(f"No images found in {root / split / 'A'}")
    if not (names["A"] == names["B"] == names["label"]):
        missing = (names["A"] ^ names["B"]) | (names["A"] ^ names["label"])
        raise DatasetError(f"Unpaired files in split '{split}': {sorted(missing)[:5]} ...")
    return sorted(names["A"])


def check_no_leakage(files: dict[str, list[str]]) -> None:
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            overlap = set(files[a]) & set(files[b])
            if overlap:
                raise DatasetError(f"Leakage: {len(overlap)} files in both {a} and {b}")


def tile(arr: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """Cut an HxW[xC] array into non-overlapping patches (drops any ragged border)."""
    h, w = arr.shape[:2]
    rows, cols = h // patch, w // patch
    arr = arr[: rows * patch, : cols * patch]
    if arr.ndim == 3:
        out = arr.reshape(rows, patch, cols, patch, arr.shape[2]).swapaxes(1, 2)
        return out.reshape(-1, patch, patch, arr.shape[2])
    out = arr.reshape(rows, patch, cols, patch).swapaxes(1, 2)
    return out.reshape(-1, patch, patch)


def prepare(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {split: list_split(root, split) for split in SPLITS}
    check_no_leakage(files)

    manifest: dict = {
        "dataset": "LEVIR-CD",
        "source": "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "patch_size": PATCH,
        "tiling": "non-overlapping grid inside each official split",
        "splits": {},
    }
    for split in SPLITS:
        names = files[split]
        first = np.asarray(Image.open(root / split / "A" / names[0]).convert("RGB"))
        per_image = (first.shape[0] // PATCH) * (first.shape[1] // PATCH)
        n = len(names) * per_image
        a_mm = np.lib.format.open_memmap(out / f"{split}_A.npy", "w+", np.uint8, (n, PATCH, PATCH, 3))
        b_mm = np.lib.format.open_memmap(out / f"{split}_B.npy", "w+", np.uint8, (n, PATCH, PATCH, 3))
        m_mm = np.lib.format.open_memmap(out / f"{split}_M.npy", "w+", np.uint8, (n, PATCH, PATCH))
        changed = 0
        for i, name in enumerate(names):
            a = np.asarray(Image.open(root / split / "A" / name).convert("RGB"))
            b = np.asarray(Image.open(root / split / "B" / name).convert("RGB"))
            m = np.asarray(Image.open(root / split / "label" / name).convert("L"))
            if a.shape != b.shape or a.shape[:2] != m.shape or a.shape != first.shape:
                raise DatasetError(f"Shape mismatch for {split}/{name}: {a.shape} {b.shape} {m.shape}")
            sl = slice(i * per_image, (i + 1) * per_image)
            a_mm[sl] = tile(a)
            b_mm[sl] = tile(b)
            mask = (tile(m) > 127).astype(np.uint8)
            m_mm[sl] = mask
            changed += int(mask.sum())
            if (i + 1) % 50 == 0 or i + 1 == len(names):
                print(f"[{split}] {i + 1}/{len(names)} images", flush=True)
        for mm in (a_mm, b_mm, m_mm):
            mm.flush()
        manifest["splits"][split] = {
            "num_images": len(names),
            "num_patches": n,
            "changed_pixel_fraction": changed / (n * PATCH * PATCH),
            "files_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "files": names,
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/raw/LEVIR-CD"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/levir256"))
    args = parser.parse_args(argv)
    manifest = prepare(args.root, args.out)
    for split, info in manifest["splits"].items():
        print(f"{split}: {info['num_images']} images -> {info['num_patches']} patches, "
              f"{info['changed_pixel_fraction'] * 100:.2f}% changed pixels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
