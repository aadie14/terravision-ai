"""Georeferenced export tests: GeoTIFF round-trip, vectorisation, geodesic area, map overlay."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin

from backend.app.services import artifacts, geo
from backend.app.services.imaging import GeoInfo


def utm_geo(width: int = 100, height: int = 80, pixel: float = 10.0) -> GeoInfo:
    return GeoInfo(CRS.from_epsg(32643), from_origin(780000.0, 1450000.0, pixel, pixel), width, height)


def test_geotiff_roundtrip_preserves_crs_transform_and_values(tmp_path):
    info = utm_geo()
    mask = np.zeros((80, 100), np.uint8)
    mask[10:30, 20:60] = 1
    path = tmp_path / "mask.tif"
    geo.write_geotiff(path, mask, info, tags={"TERRAVISION_MODEL": "x"})
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32643
        assert src.transform == info.transform
        assert (src.width, src.height, src.count) == (100, 80, 1)
        assert np.array_equal(src.read(1), mask)
        assert src.tags()["TERRAVISION_MODEL"] == "x"


def test_multiband_geotiff(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 255, (80, 100, 3), dtype=np.uint8)
    geo.write_geotiff(tmp_path / "rgb.tif", rgb, utm_geo())
    with rasterio.open(tmp_path / "rgb.tif") as src:
        assert src.count == 3 and np.array_equal(src.read(2), rgb[..., 1])


def test_change_regions_area_matches_pixel_area_in_utm():
    info = utm_geo()
    mask = np.zeros((80, 100), bool)
    mask[10:30, 20:60] = True  # 20 x 40 px at 10 m => 80,000 m2
    mask[50:60, 70:80] = True  # 10 x 10 px => 10,000 m2
    collection, summary = geo.change_regions(mask, info)
    assert summary["region_count"] == 2 and len(collection["features"]) == 2
    # UTM scale distortion is < 0.1% this close to the zone's central meridian band
    assert summary["changed_area_m2"] == pytest.approx(90_000, rel=5e-3)
    assert summary["changed_area_ha"] == pytest.approx(9.0, rel=5e-3)
    areas = sorted(f["properties"]["area_m2"] for f in collection["features"])
    assert areas[0] == pytest.approx(10_000, rel=5e-3) and areas[1] == pytest.approx(80_000, rel=5e-3)


def test_change_regions_are_wgs84_lon_lat():
    collection, _ = geo.change_regions(np.ones((8, 8), bool), utm_geo(8, 8))
    ring = collection["features"][0]["geometry"]["coordinates"][0]
    for lon, lat in ring:
        assert 77 < lon < 79 and 12 < lat < 14


def test_area_is_geodesic_not_naive_for_web_mercator():
    # 100 x 100 px of 10 "metre" Web-Mercator pixels at ~60 deg N cover about a quarter of the naive area.
    info = GeoInfo(CRS.from_epsg(3857), from_origin(1_113_194.9, 8_399_737.9, 10.0, 10.0), 100, 100)
    _, summary = geo.change_regions(np.ones((100, 100), bool), info)
    naive = 100 * 100 * 10.0 * 10.0
    assert summary["changed_area_m2"] == pytest.approx(naive * np.cos(np.radians(60)) ** 2, rel=0.02)


def test_empty_mask_yields_no_regions():
    collection, summary = geo.change_regions(np.zeros((16, 16), bool), utm_geo(16, 16))
    assert collection["features"] == [] and summary["region_count"] == 0 and summary["changed_area_m2"] == 0


def test_footprint_area():
    assert geo.footprint_area_m2(utm_geo()) == pytest.approx(100 * 80 * 100.0, rel=5e-3)


def test_web_mercator_overlay_bounds_and_image(tmp_path):
    info = utm_geo()
    rgba = np.zeros((80, 100, 4), np.uint8)
    rgba[20:40, 20:40] = (255, 59, 92, 255)
    bounds = geo.web_mercator_overlay(tmp_path / "overlay.png", rgba, info)
    west, south, east, north = info.bounds_wgs84()
    assert bounds["west"] == pytest.approx(west, abs=2e-3) and bounds["east"] == pytest.approx(east, abs=2e-3)
    assert bounds["south"] == pytest.approx(south, abs=2e-3) and bounds["north"] == pytest.approx(north, abs=2e-3)
    assert bounds["south"] < bounds["north"] and bounds["west"] < bounds["east"]
    with Image.open(tmp_path / "overlay.png") as img:
        assert img.mode == "RGBA"
        alpha = np.asarray(img)[..., 3]
        assert 0 < (alpha > 0).mean() < 0.2


# ---------------------------------------------------------- artefact rendering
def test_threshold_rule_and_statistics():
    prob = np.zeros((10, 10), np.uint8)
    prob[:5] = 200
    mask = artifacts.threshold_mask(prob, 0.5)
    assert mask.sum() == 50
    assert artifacts.threshold_mask(prob, 0.9).sum() == 0
    stats = artifacts.prediction_statistics(prob, mask)
    assert stats["changed_pixels"] == 50 and stats["changed_percent"] == 50.0
    assert stats["region_count"] == 1 and stats["largest_region_pixels"] == 50
    assert sum(stats["probability_histogram"]["counts"]) == 100
    assert stats["mean_probability_in_change"] == pytest.approx(200 / 255)


def test_statistics_for_empty_mask():
    prob = np.zeros((8, 8), np.uint8)
    stats = artifacts.prediction_statistics(prob, artifacts.threshold_mask(prob, 0.5))
    assert stats["changed_pixels"] == 0 and stats["region_count"] == 0
    assert stats["mean_probability_in_change"] is None


def test_build_outputs_without_geo_or_truth_reports_no_metrics(tmp_path):
    t2 = np.full((32, 32, 3), 100, np.uint8)
    prob = np.zeros((32, 32), np.uint8)
    prob[4:12, 4:12] = 255
    out = artifacts.build_outputs(tmp_path, t2=t2, prob_u8=prob, threshold=0.5, truth=None, geo=None)
    assert out["metrics"] is None and out["geospatial"] is None
    assert (tmp_path / "mask.png").is_file() and (tmp_path / "overlay.png").is_file()
    assert not (tmp_path / "mask.tif").exists() and not (tmp_path / "error_map.png").exists()


def test_build_outputs_with_geo_and_truth(tmp_path):
    info = utm_geo(32, 32)
    t2 = np.full((32, 32, 3), 100, np.uint8)
    prob = np.zeros((32, 32), np.uint8)
    prob[4:12, 4:12] = 255
    truth = np.zeros((32, 32), bool)
    truth[4:12, 4:16] = True
    out = artifacts.build_outputs(tmp_path, t2=t2, prob_u8=prob, threshold=0.5, truth=truth, geo=info)
    assert out["metrics"]["tp"] == 64 and out["metrics"]["fn"] == 32 and out["metrics"]["fp"] == 0
    assert out["geospatial"]["crs"] == "EPSG:32643"
    assert out["geospatial"]["changed_area_m2"] == pytest.approx(6400, rel=5e-3)
    for name in ("mask.tif", "regions.geojson", "map_overlay.png", "error_map.png"):
        assert (tmp_path / name).is_file()
