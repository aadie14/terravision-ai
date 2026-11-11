"""Build the bundled demonstration samples under ``samples/`` and write ``samples/manifest.json``.

1. LEVIR-CD: four full 1024x1024 pairs (with labels) copied from the official *test* split,
   chosen deterministically by changed-pixel fraction (max, ~75th pct, median, a no-change pair).
2. Sentinel-2: one georeferenced bitemporal pair over north Bengaluru retrieved through the
   Earth Search provider (skipped with a message when offline; existing files are kept).

Usage:  python scripts/build_samples.py [--skip-satellite]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings
from backend.app.errors import AppError
from backend.app.satellite.base import SearchQuery
from backend.app.satellite.earth_search import EarthSearchProvider
from backend.app.services.geo import write_geotiff
from backend.app.services.imaging import align_pair

SAMPLES = ROOT / "samples"
LEVIR_TEST = ROOT / "data" / "raw" / "LEVIR-CD" / "test"
LEVIR_LICENSE = ("LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from "
                 "Google Earth is subject to the Google Earth terms of use.")
LEVIR_ATTRIBUTION = ("H. Chen and Z. Shi, 'A Spatial-Temporal Attention-Based Method and a New Dataset for "
                     "Remote Sensing Image Change Detection', Remote Sensing 12(10):1662, 2020.")
# North Bengaluru (Yelahanka / Kattigenahalli area): ~6.5 x 5.5 km
BENGALURU_BBOX = (77.600, 13.090, 77.660, 13.140)


def levir_samples() -> list[dict]:
    names = sorted(p.name for p in (LEVIR_TEST / "label").glob("*.png"))
    if not names:
        print("LEVIR-CD test split not found - run `python -m ml.data.download` first")
        return []
    fractions = np.array([(np.asarray(Image.open(LEVIR_TEST / "label" / n).convert("L")) > 127).mean()
                          for n in names])
    order = np.argsort(fractions)
    nonzero = order[fractions[order] > 0]
    picks = {
        "dense-change": int(nonzero[-1]),
        "moderate-change": int(nonzero[int(len(nonzero) * 0.75)]),
        "sparse-change": int(nonzero[len(nonzero) // 2]),
    }
    zero = order[fractions[order] == 0]
    if len(zero):
        picks["no-change"] = int(zero[0])
    out_dir = SAMPLES / "levir"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, idx in picks.items():
        name = names[idx]
        stem = Path(name).stem
        for sub, suffix in (("A", "t1"), ("B", "t2"), ("label", "gt")):
            shutil.copy2(LEVIR_TEST / sub / name, out_dir / f"{stem}_{suffix}.png")
        entries.append({
            "id": f"levir-{stem}", "name": f"LEVIR-CD {stem} ({label.replace('-', ' ')})",
            "description": f"Official test-split pair {name}; {fractions[idx] * 100:.2f}% of pixels are labelled "
                           "as building change. 1024x1024 px at 0.5 m/pixel.",
            "dataset": "LEVIR-CD", "split": "test", "domain": "vhr", "georeferenced": False,
            "resolution_note": "0.5 m/pixel (not georeferenced: the dataset ships without coordinates)",
            "t1": f"levir/{stem}_t1.png", "t2": f"levir/{stem}_t2.png", "ground_truth": f"levir/{stem}_gt.png",
            "license": LEVIR_LICENSE, "attribution": LEVIR_ATTRIBUTION,
            "source_url": "https://justchenhao.github.io/LEVIR/",
        })
        print(f"LEVIR sample {stem}: {label}, {fractions[idx] * 100:.2f}% change")
    return entries


def sentinel_sample() -> dict | None:
    settings = get_settings()
    out_dir = SAMPLES / "sentinel2"
    meta_path = out_dir / "bengaluru_north.json"
    if meta_path.is_file() and (out_dir / "bengaluru_north_t1.tif").is_file():
        print("Sentinel-2 sample already present, keeping it")
        entry = json.loads(meta_path.read_text())
        entry.pop("in_training_domain", None)
        entry["domain"] = "sentinel2"
        entry["description"] = entry["description"].replace(
            " Demonstrates the GeoTIFF / map workflow; the models were trained on 0.5 m imagery, so this is "
            "out-of-distribution.", " Real-world demo for the Sentinel-2 (OSCD-trained) models; no ground truth.")
        return entry
    provider = EarthSearchProvider(settings.earth_search_url, settings.satellite_timeout_s)
    try:
        scenes = {}
        for label, (start, end) in {"t1": (date(2019, 1, 15), date(2019, 3, 15)),
                                    "t2": (date(2025, 1, 15), date(2025, 3, 15))}.items():
            found = provider.search(SearchQuery(BENGALURU_BBOX, start, end, max_cloud_cover=5, limit=5))
            found = [s for s in found if s.aoi_coverage > 0.999]
            if not found:
                print(f"No cloud-free full-coverage scene for {label}")
                return None
            scenes[label] = found[0]
            print(f"{label}: {found[0].id}  {found[0].acquired}  cloud {found[0].cloud_cover}%")
        first = provider.retrieve(scenes["t1"].id, BENGALURU_BBOX)
        second = provider.retrieve(scenes["t2"].id, BENGALURU_BBOX)
    except AppError as exc:
        print(f"Sentinel-2 sample skipped: {exc.message}")
        return None
    img1, img2, notes = align_pair(first.image, second.image)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, img, meta in (("t1", img1, first.metadata), ("t2", img2, second.metadata)):
        write_geotiff(out_dir / f"bengaluru_north_{label}.tif", img.array, img.geo,
                      tags={"SCENE_ID": meta["scene_id"], "ACQUIRED": meta["acquired"],
                            "ATTRIBUTION": meta["attribution"]})
        Image.fromarray(img.array).save(out_dir / f"bengaluru_north_{label}.preview.png")
    t1_date, t2_date = first.metadata["acquired"][:10], second.metadata["acquired"][:10]
    entry = {
        "id": "sentinel2-bengaluru-north",
        "name": f"Sentinel-2 north Bengaluru {t1_date[:4]} vs {t2_date[:4]} (georeferenced)",
        "description": (f"Sentinel-2 L2A true-colour, {img1.width}x{img1.height} px at 10 m/pixel, "
                        f"{img1.geo.crs.to_string()}. Scenes {first.metadata['scene_id']} and "
                        f"{second.metadata['scene_id']}. No ground truth. Demonstrates the GeoTIFF / map "
                        "workflow; the models were trained on 0.5 m imagery, so this is out-of-distribution."),
        "dataset": "Sentinel-2 L2A (Earth Search)", "split": None, "domain": "sentinel2",
        "georeferenced": True, "resolution_note": "10 m/pixel",
        "t1": "sentinel2/bengaluru_north_t1.tif", "t2": "sentinel2/bengaluru_north_t2.tif",
        "ground_truth": None, "t1_date": t1_date, "t2_date": t2_date,
        "license": "Copernicus Sentinel data: free, full and open access; attribution required.",
        "attribution": f"Contains modified Copernicus Sentinel data {t1_date[:4]}, {t2_date[:4]}; retrieved "
                       "via Element 84 Earth Search (AWS Open Data).",
        "source_url": "https://earth-search.aws.element84.com/v1",
        "alignment_notes": notes,
    }
    meta_path.write_text(json.dumps(entry, indent=2))
    return entry


OSCD_ROOT = ROOT / "data" / "raw" / "OSCD"
OSCD_CITIES = ("lasvegas", "dubai", "chongqing")  # official TEST cities with the most labelled change


def oscd_samples() -> list[dict]:
    """Georeferenced OSCD test-city pairs with ground truth (Sentinel-2, 10 m)."""
    import rasterio
    from rasterio.crs import CRS

    from backend.app.services.imaging import GeoInfo
    from ml.data.oscd import IMAGES, load_mask, load_rgb

    images = OSCD_ROOT / IMAGES
    labels = OSCD_ROOT / "Onera Satellite Change Detection dataset - Test Labels"
    if not labels.is_dir():
        print("OSCD not found - run `python -m ml.data.oscd download` first")
        return []
    out_dir = SAMPLES / "oscd"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for city in OSCD_CITIES:
        a, b = load_rgb(images / city, 1), load_rgb(images / city, 2)
        m = load_mask(labels, city)
        # imgs_1 holds the original georeferenced 10 m bands; use them only when the grid matches the rectified image.
        with rasterio.open(next((images / city / "imgs_1").glob("*B04.tif"))) as src:
            geo = GeoInfo(CRS.from_user_input(src.crs), src.transform, src.width, src.height) \
                if (src.height, src.width) == m.shape and src.crs is not None else None
        for label, img in (("t1", a), ("t2", b)):
            if geo is not None:
                write_geotiff(out_dir / f"{city}_{label}.tif", img, geo, tags={"SOURCE": f"OSCD {city} {label}"})
            Image.fromarray(img).save(out_dir / f"{city}_{label}{'.preview' if geo else ''}.png")
        Image.fromarray(m * 255).save(out_dir / f"{city}_gt.png")
        dates = dict(line.split(": ") for line in (images / city / "dates.txt").read_text().split("\n") if ": " in line)
        d1, d2 = (f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in (dates["date_1"], dates["date_2"]))
        ext = "tif" if geo else "png"
        entries.append({
            "id": f"oscd-{city}", "name": f"OSCD {city} {d1[:4]} vs {d2[:4]} (Sentinel-2, ground truth)",
            "description": f"Official OSCD test city, never used for training or model selection. Sentinel-2 RGB "
                           f"(B04/B03/B02) at 10 m, {m.shape[1]}x{m.shape[0]} px, {m.mean() * 100:.2f}% labelled "
                           "urban change." + (" Georeferenced (EPSG:4326)." if geo else ""),
            "dataset": "OSCD", "split": "test", "domain": "sentinel2", "georeferenced": geo is not None,
            "resolution_note": "10 m/pixel",
            "t1": f"oscd/{city}_t1.{ext}", "t2": f"oscd/{city}_t2.{ext}", "ground_truth": f"oscd/{city}_gt.png",
            "t1_date": d1, "t2_date": d2,
            "license": "Imagery: Copernicus Sentinel data (free, full and open). Labels: OSCD by Caye Daudt et al.; "
                       "the dataset README requires citation and states no further licence terms.",
            "attribution": "R. Caye Daudt, B. Le Saux, A. Boulch, Y. Gousseau, 'Urban Change Detection for "
                           "Multispectral Earth Observation Using Convolutional Neural Networks', IGARSS 2018. "
                           f"Contains modified Copernicus Sentinel data {d1[:4]}, {d2[:4]}.",
            "source_url": "https://rcdaudt.github.io/oscd/",
        })
        print(f"OSCD sample {city}: {m.mean() * 100:.2f}% change, georeferenced={geo is not None}")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-satellite", action="store_true")
    args = parser.parse_args()
    SAMPLES.mkdir(exist_ok=True)
    entries = levir_samples() + oscd_samples()
    if not args.skip_satellite:
        sentinel = sentinel_sample()
        if sentinel:
            entries.append(sentinel)
    (SAMPLES / "manifest.json").write_text(json.dumps({"samples": entries}, indent=2))
    print(f"wrote {SAMPLES / 'manifest.json'} with {len(entries)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
