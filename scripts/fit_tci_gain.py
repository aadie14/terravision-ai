"""Fit the linear gain that maps Sentinel-2 16-bit reflectance DNs to Earth Search 8-bit TCI values.

OSCD ships raw band DNs, while the application's satellite workflow receives the 8-bit true-colour
('visual') asset. Training on OSCD with the same mapping keeps train / inference radiometry consistent.
For each scene the red/green/blue band windows and the visual window over the same AOI are read and
TCI = gain * DN is fitted on unsaturated pixels. Measured result: one gain (DN 2500 -> 255) fits both a
baseline-02.11 and a baseline-05.11 scene, i.e. the TCI is computed from the stored DNs.

Usage:  python scripts/fit_tci_gain.py   ->  reports/tci_gain_fit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
BBOX = (77.600, 13.090, 77.660, 13.140)
SCENES = ["S2A_43PGQ_20190315_0_L2A", "S2B_43PGQ_20250206_0_L2A"]
API = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/"
ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "AWS_NO_SIGN_REQUEST": "YES"}


def read(href: str) -> np.ndarray:
    with rasterio.Env(**ENV), rasterio.open(href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *BBOX)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        return src.read(window=window).astype(np.float64)


def main() -> int:
    results = []
    for scene in SCENES:
        item = httpx.get(API + scene, timeout=60).json()
        props = item["properties"]
        baseline = props.get("s2:processing_baseline")
        offset = 0.0  # fit on DNs exactly as stored: Earth Search's TCI does not remove the BOA offset
        visual = read(item["assets"]["visual"]["href"])
        gains = []
        for i, band in enumerate(("red", "green", "blue")):
            dn = read(item["assets"][band]["href"])[0] - offset
            tci = visual[i]
            h, w = min(dn.shape[0], tci.shape[0]), min(dn.shape[1], tci.shape[1])
            dn, tci = dn[:h, :w].ravel(), tci[:h, :w].ravel()
            keep = (tci > 5) & (tci < 250) & (dn > 0)
            gain = float((dn[keep] * tci[keep]).sum() / (dn[keep] ** 2).sum())  # least squares through 0
            resid = tci[keep] - gain * dn[keep]
            gains.append({"band": band, "gain": gain, "dn_for_255": 255 / gain, "rmse": float(np.sqrt((resid ** 2).mean())),
                          "pixels": int(keep.sum())})
        results.append({"scene": scene, "processing_baseline": baseline, "bands": gains})
        print(scene, [(g["band"], round(g["dn_for_255"]), round(g["rmse"], 2)) for g in gains])
    all_gains = [g["gain"] for r in results for g in r["bands"]]
    summary = {"scenes": results, "mean_gain": float(np.mean(all_gains)), "dn_for_255": float(255 / np.mean(all_gains)),
               "note": "OSCD images are Level-1C (top of atmosphere) while these TCIs are Level-2A (bottom of "
                       "atmosphere): the gain matches the DN scale, not the atmospheric correction."}
    (ROOT / "reports" / "tci_gain_fit.json").write_text(json.dumps(summary, indent=2))
    print("mean DN for 255:", round(summary["dn_for_255"], 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
