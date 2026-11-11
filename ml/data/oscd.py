"""OSCD (Onera Satellite Change Detection) - Sentinel-2 urban change, 24 city pairs.

Download: Hugging Face mirror ``hkristen/oscd`` at a pinned revision, with the SHA-256 checksums
TorchGeo uses. Dataset: R. Caye Daudt, B. Le Saux, A. Boulch, Y. Gousseau, "Urban Change Detection
for Multispectral Earth Observation Using Convolutional Neural Networks", IGARSS 2018
(doi:10.1109/IGARSS.2018.8518015).

Preparation (``prepare``):
* RGB = Sentinel-2 bands B04, B03, B02 (10 m). Raw 16-bit Level-1C digital numbers are converted to
  8-bit with the *same* linear gain the application sees in Earth Search true-colour (TCI) imagery
  (``RGB_GAIN``), so a model trained here receives the same radiometry at inference time.
* Official split: 14 train cities, 10 test cities. Three train cities are held out (deterministically)
  as validation for model selection, so no test city is ever used for training or selection.
* Train / val cities are cut into 256x256 tiles with stride 128 (small images are reflect-padded,
  padded mask pixels are marked ignore=255). Test cities are stored whole and evaluated with the
  same sliding-window inference the application uses.

Usage:
    python -m ml.data.oscd download --root data/raw/OSCD
    python -m ml.data.oscd prepare  --root data/raw/OSCD --out data/processed/oscd256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ml.data.download import _safe_extract

_REV = "4958d786c1389ede1511d91a6ecf1a75c4074933"
_BASE = f"https://hf.co/datasets/hkristen/oscd/resolve/{_REV}"
ARCHIVES = {
    "Onera Satellite Change Detection dataset - Images.zip":
        "940b87887511058a933e67cd6d0e43e2eb825a55d8e79a50983dee7f23003656",
    "Onera Satellite Change Detection dataset - Train Labels.zip":
        "89fb54cd12ad0dbea6c447528139dec305b865294215434bf6dd170fb8fd3ca5",
    "Onera Satellite Change Detection dataset - Test Labels.zip":
        "2e195eaa1b788b99fa93ea8073e3780bc0b763000b0c49dbf70548acf1e5d67d",
}
IMAGES = "Onera Satellite Change Detection dataset - Images"
PATCH, STRIDE = 256, 128
IGNORE = 255
# 8-bit value = clip(DN * RGB_GAIN). Fitted against Earth Search Sentinel-2 TCI ('visual') pixels of
# two real scenes (scripts/fit_tci_gain.py, reports/tci_gain_fit.json): TCI = DN * 255 / 2500 with
# RMSE 2-5 grey levels, so OSCD training data gets the radiometry the app receives at inference time.
RGB_GAIN = 255 / 2500.0


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        folder = root / name.removesuffix(".zip")
        if folder.is_dir() and any(folder.iterdir()):
            print(f"[{name}] already extracted")
            continue
        archive = root / name
        if not archive.exists():
            url = f"{_BASE}/{urllib.request.quote(name)}"
            print(f"[{name}] downloading")
            request = urllib.request.Request(url, headers={"User-Agent": "terravision-dataset-fetch"})
            tmp = archive.with_suffix(".part")
            with urllib.request.urlopen(request) as response, tmp.open("wb") as out:  # noqa: S310
                while block := response.read(1 << 20):
                    out.write(block)
            tmp.replace(archive)
        actual = sha256sum(archive)
        if actual != expected:
            raise RuntimeError(f"{name}: sha256 mismatch (expected {expected}, got {actual})")
        print(f"[{name}] verified, extracting")
        _safe_extract(archive, root)
        archive.unlink()


def _band(folder: Path, band: str) -> np.ndarray:
    matches = sorted(folder.glob(f"*{band}.tif"))
    if not matches:
        raise FileNotFoundError(f"band {band} missing in {folder}")
    with Image.open(matches[0]) as img:
        return np.asarray(img, dtype=np.float32)


def load_rgb(city_dir: Path, index: int) -> np.ndarray:
    folder = city_dir / f"imgs_{index}_rect"
    bands = [_band(folder, b) for b in ("B04", "B03", "B02")]
    return np.clip(np.stack(bands, axis=2) * RGB_GAIN, 0, 255).round().astype(np.uint8)


def load_mask(labels_dir: Path, city: str) -> np.ndarray:
    with Image.open(labels_dir / city / "cm" / "cm.png") as img:
        arr = np.asarray(img.convert("L"))
    return (arr > (arr.max() // 2 if arr.max() > 1 else 0)).astype(np.uint8)


def city_lists(root: Path) -> dict[str, list[str]]:
    def listed(split: str) -> list[str]:
        folder = root / f"Onera Satellite Change Detection dataset - {split} Labels"
        return sorted(p.name for p in folder.iterdir() if (p / "cm" / "cm.png").is_file())

    train, test = listed("Train"), listed("Test")
    if set(train) & set(test):
        raise RuntimeError("OSCD train and test cities overlap")
    val = train[2::5][:3]  # deterministic, spread alphabetically
    return {"train": [c for c in train if c not in val], "val": val, "test": test}


def _tiles(arr: np.ndarray, fill) -> list[np.ndarray]:
    h, w = arr.shape[:2]
    ph, pw = max(0, PATCH - h), max(0, PATCH - w)
    if ph or pw:
        pad = ((0, ph), (0, pw)) + (((0, 0),) if arr.ndim == 3 else ())
        arr = np.pad(arr, pad, mode="reflect") if fill is None else np.pad(arr, pad, constant_values=fill)
        h, w = arr.shape[:2]
    ys = sorted({*range(0, h - PATCH + 1, STRIDE), h - PATCH})
    xs = sorted({*range(0, w - PATCH + 1, STRIDE), w - PATCH})
    return [arr[y : y + PATCH, x : x + PATCH] for y in ys for x in xs]


def prepare(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cities = city_lists(root)
    images = root / IMAGES
    labels = {"train": root / "Onera Satellite Change Detection dataset - Train Labels",
              "val": root / "Onera Satellite Change Detection dataset - Train Labels",
              "test": root / "Onera Satellite Change Detection dataset - Test Labels"}
    manifest = {"dataset": "OSCD", "source": f"https://huggingface.co/datasets/hkristen/oscd (rev {_REV})",
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "bands": "B04,B03,B02 (10 m)",
                "rgb_gain": RGB_GAIN, "patch_size": PATCH, "stride": STRIDE, "ignore_index": IGNORE,
                "cities": cities, "splits": {}}
    (out / "test").mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        a_tiles, b_tiles, m_tiles = [], [], []
        changed = total = 0
        for city in cities[split]:
            a, b = load_rgb(images / city, 1), load_rgb(images / city, 2)
            m = load_mask(labels[split], city)
            if not (a.shape == b.shape and a.shape[:2] == m.shape):
                raise RuntimeError(f"{city}: shape mismatch {a.shape} {b.shape} {m.shape}")
            changed += int(m.sum())
            total += m.size
            if split == "test":
                np.savez_compressed(out / "test" / f"{city}.npz", a=a, b=b, m=m)
                continue
            a_tiles += _tiles(a, None)
            b_tiles += _tiles(b, None)
            m_tiles += _tiles(m, IGNORE)
        info = {"cities": cities[split], "changed_pixel_fraction": changed / max(total, 1), "pixels": total}
        if split != "test":
            np.save(out / f"{split}_A.npy", np.stack(a_tiles))
            np.save(out / f"{split}_B.npy", np.stack(b_tiles))
            np.save(out / f"{split}_M.npy", np.stack(m_tiles))
            info["num_patches"] = len(a_tiles)
        manifest["splits"][split] = info
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_test_cities(out: Path) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    rows = []
    for path in sorted((out / "test").glob("*.npz")):
        data = np.load(path)
        rows.append((path.stem, data["a"], data["b"], data["m"]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["download", "prepare"])
    parser.add_argument("--root", type=Path, default=Path("data/raw/OSCD"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/oscd256"))
    args = parser.parse_args(argv)
    if args.action == "download":
        download(args.root)
    else:
        manifest = prepare(args.root, args.out)
        for split, info in manifest["splits"].items():
            print(f"{split}: {len(info['cities'])} cities, {info.get('num_patches', 'full images')} patches, "
                  f"{info['changed_pixel_fraction'] * 100:.2f}% changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
