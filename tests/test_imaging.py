"""Image decoding / validation and bitemporal pair alignment tests."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.errors import AppError
from backend.app.services.imaging import (
    align_pair,
    detect_format,
    load_image,
    load_mask,
    pair_sanity_warnings,
    structural_correlation,
)
from tests.conftest import geotiff_bytes, jpeg_bytes, png_bytes, scene, with_building

LIMITS = {"min_side": 32, "max_side": 1024}


def code(excinfo) -> str:
    return excinfo.value.code


def test_png_and_jpeg_decode_to_rgb_uint8():
    img = scene(64, 80)
    for name, data in (("a.png", png_bytes(img)), ("a.jpg", jpeg_bytes(img)), ("A.JPEG", jpeg_bytes(img))):
        loaded = load_image(name, data, **LIMITS)
        assert loaded.array.shape == (64, 80, 3) and loaded.array.dtype == np.uint8
        assert loaded.geo is None
    assert np.array_equal(load_image("a.png", png_bytes(img), **LIMITS).array, img)


def test_grayscale_and_rgba_are_converted_with_a_note():
    gray = load_image("g.png", png_bytes(scene(64, 64)[..., 0]), **LIMITS)
    assert gray.array.shape == (64, 64, 3) and any("Converted" in n for n in gray.notes)
    rgba = np.dstack([scene(64, 64), np.full((64, 64), 255, np.uint8)])
    assert load_image("r.png", png_bytes(rgba), **LIMITS).array.shape == (64, 64, 3)


@pytest.mark.parametrize("name", ["report.pdf", "image.gif", "archive.zip", "noextension", "x.bmp"])
def test_unsupported_extension_rejected(name):
    with pytest.raises(AppError) as exc:
        detect_format(name, b"whatever")
    assert code(exc) == "UNSUPPORTED_FILE_TYPE" and exc.value.status == 415


def test_extension_must_match_content():
    with pytest.raises(AppError) as exc:
        load_image("fake.png", jpeg_bytes(scene(64, 64)), **LIMITS)
    assert code(exc) == "CORRUPTED_IMAGE"


def test_empty_file_rejected():
    with pytest.raises(AppError) as exc:
        load_image("a.png", b"", **LIMITS)
    assert code(exc) == "EMPTY_FILE"


def test_truncated_png_rejected():
    data = png_bytes(scene(64, 64))
    with pytest.raises(AppError) as exc:
        load_image("a.png", data[: len(data) // 2], **LIMITS)
    assert code(exc) == "CORRUPTED_IMAGE"


def test_garbage_with_valid_magic_rejected():
    with pytest.raises(AppError) as exc:
        load_image("a.tif", b"II*\x00" + b"\x00" * 200, **LIMITS)
    assert code(exc) == "CORRUPTED_IMAGE"


@pytest.mark.parametrize("shape,expected", [((31, 64), "IMAGE_TOO_SMALL"), ((32, 32), None), ((1024, 1024), None),
                                             ((1025, 64), "IMAGE_TOO_LARGE")])
def test_dimension_boundaries(shape, expected):
    data = png_bytes(np.zeros((*shape, 3), np.uint8))
    if expected is None:
        assert load_image("a.png", data, **LIMITS).array.shape[:2] == shape
    else:
        with pytest.raises(AppError) as exc:
            load_image("a.png", data, **LIMITS)
        assert code(exc) == expected


# ------------------------------------------------------------------- GeoTIFF
def test_geotiff_keeps_crs_transform_and_bounds():
    loaded = load_image("a.tif", geotiff_bytes(scene(64, 96)), **LIMITS)
    assert loaded.geo is not None
    assert loaded.geo.crs.to_epsg() == 32643
    assert loaded.geo.resolution == (10.0, 10.0)
    west, south, east, north = loaded.geo.bounds
    assert (west, north) == (780000.0, 1450000.0)
    assert east - west == pytest.approx(960.0) and north - south == pytest.approx(640.0)
    described = loaded.geo.describe()
    assert described["crs_is_projected"] is True
    assert 77 < described["bounds_wgs84"]["west"] < 79 and 12 < described["bounds_wgs84"]["south"] < 14


def test_plain_tiff_without_georeferencing_is_accepted_as_image():
    loaded = load_image("a.tif", geotiff_bytes(scene(64, 64), crs=None, with_transform=False), **LIMITS)
    assert loaded.geo is None and any("no georeferencing" in n for n in loaded.notes)


def test_transform_without_crs_is_invalid_crs():
    with pytest.raises(AppError) as exc:
        load_image("a.tif", geotiff_bytes(scene(64, 64), crs=None, with_transform=True), **LIMITS)
    assert code(exc) == "INVALID_CRS"


def test_uint16_raster_is_stretched_to_8bit():
    data = (scene(64, 64).astype(np.uint16) * 40) + 500
    loaded = load_image("a.tif", geotiff_bytes(data, dtype="uint16"), **LIMITS)
    assert loaded.array.dtype == np.uint8
    assert loaded.array.max() > 200 and loaded.array.min() < 50
    assert any("stretched" in n for n in loaded.notes)


def test_single_band_raster_replicated():
    loaded = load_image("a.tif", geotiff_bytes(scene(64, 64)[..., 0]), **LIMITS)
    assert loaded.array.shape == (64, 64, 3)
    assert np.array_equal(loaded.array[..., 0], loaded.array[..., 2])


def test_two_band_raster_rejected():
    with pytest.raises(AppError) as exc:
        load_image("a.tif", geotiff_bytes(scene(64, 64)[..., :2]), **LIMITS)
    assert code(exc) == "UNSUPPORTED_CHANNELS"


# ----------------------------------------------------------------- pair checks
def test_mismatched_dimensions_rejected_for_plain_images():
    a = load_image("a.png", png_bytes(scene(64, 64)), **LIMITS)
    b = load_image("b.png", png_bytes(scene(64, 80)), **LIMITS)
    with pytest.raises(AppError) as exc:
        align_pair(a, b)
    assert code(exc) == "IMAGE_DIMENSION_MISMATCH"
    assert exc.value.details == {"t1": [64, 64], "t2": [80, 64]}


def test_identical_grids_pass_through_untouched():
    a = load_image("a.tif", geotiff_bytes(scene(64, 64, 1)), **LIMITS)
    b = load_image("b.tif", geotiff_bytes(scene(64, 64, 2)), **LIMITS)
    out1, out2, warnings = align_pair(a, b)
    assert warnings == [] and out2 is b and out1 is a


def test_same_crs_shifted_grid_is_resampled_onto_t1():
    a = load_image("a.tif", geotiff_bytes(scene(100, 100, 1)), **LIMITS)
    # T2 covers a superset footprint at a different resolution
    b = load_image("b.tif", geotiff_bytes(scene(220, 220, 2), origin=(779900.0, 1450100.0), pixel=5.0), **LIMITS)
    _, aligned, warnings = align_pair(a, b)
    assert aligned.array.shape == a.array.shape
    assert aligned.geo.transform == a.geo.transform
    assert any("resampled" in w for w in warnings)
    assert aligned.array.std() > 1  # real content, not an empty raster


def test_different_crs_is_reprojected_onto_t1():
    a = load_image("a.tif", geotiff_bytes(scene(100, 100, 1)), **LIMITS)
    west, _south, east, north = a.geo.bounds_wgs84()
    pad = 0.002
    pixel = (east - west + 2 * pad) / 160
    b = load_image("b.tif", geotiff_bytes(scene(160, 160, 2), crs="EPSG:4326", origin=(west - pad, north + pad),
                                          pixel=pixel), **LIMITS)
    _, aligned, warnings = align_pair(a, b)
    assert aligned.geo.crs == a.geo.crs and aligned.array.shape == a.array.shape
    assert any("reprojected" in w and "EPSG:4326" in w for w in warnings)


def test_non_overlapping_rasters_rejected():
    a = load_image("a.tif", geotiff_bytes(scene(64, 64)), **LIMITS)
    b = load_image("b.tif", geotiff_bytes(scene(64, 64), origin=(500000.0, 1000000.0)), **LIMITS)
    with pytest.raises(AppError) as exc:
        align_pair(a, b)
    assert code(exc) == "GEOSPATIAL_MISALIGNMENT"
    assert exc.value.details["overlap_fraction"] == 0.0


def test_partially_overlapping_rasters_rejected():
    a = load_image("a.tif", geotiff_bytes(scene(64, 64)), **LIMITS)
    b = load_image("b.tif", geotiff_bytes(scene(64, 64), origin=(780320.0, 1450000.0)), **LIMITS)
    with pytest.raises(AppError) as exc:
        align_pair(a, b)
    assert code(exc) == "GEOSPATIAL_MISALIGNMENT"
    assert 0.4 < exc.value.details["overlap_fraction"] < 0.6


def test_one_georeferenced_image_emits_warning_and_shares_geo():
    a = load_image("a.tif", geotiff_bytes(scene(64, 64)), **LIMITS)
    b = load_image("b.png", png_bytes(scene(64, 64, 2)), **LIMITS)
    out1, out2, warnings = align_pair(a, b)
    assert out2.geo is not None and out1.geo is not None
    assert any("Only one" in w for w in warnings)


# ------------------------------------------------------------------------ mask
def test_mask_loading_binarises_0_255_and_0_1():
    truth = np.zeros((64, 64), np.uint8)
    truth[10:20, 10:20] = 255
    assert load_mask("m.png", png_bytes(truth), expected_shape=(64, 64), max_side=1024).sum() == 100
    assert load_mask("m.png", png_bytes(truth // 255), expected_shape=(64, 64), max_side=1024).sum() == 100


def test_mask_dimension_mismatch_rejected():
    with pytest.raises(AppError) as exc:
        load_mask("m.png", png_bytes(np.zeros((32, 32), np.uint8)), expected_shape=(64, 64), max_side=1024)
    assert code(exc) == "MASK_DIMENSION_MISMATCH"


# --------------------------------------------------------------------- sanity
def city(seed: int, size: int = 256) -> np.ndarray:
    """Synthetic scene with many sharp-edged blocks, like roads / roofs in real imagery."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 90, np.uint8)
    for _ in range(40):
        y, x = rng.integers(0, size - 30, 2)
        h, w = rng.integers(8, 30, 2)
        img[y : y + h, x : x + w] = rng.integers(60, 220, 3)
    return img


def test_related_pair_has_high_structural_correlation_unrelated_low():
    base = city(1)
    changed, _ = with_building(base, (100, 100, 130, 140))  # one new building in an otherwise unchanged scene
    darker = (changed * 0.8).astype(np.uint8)  # illumination change must not matter
    assert structural_correlation(base, changed) > 0.6
    assert structural_correlation(base, darker) > 0.6
    assert structural_correlation(base, city(2)) < 0.1  # a different place
    assert structural_correlation(base, np.full_like(base, 128)) == 0.0  # blank image: no structure at all


def test_sanity_warnings_flag_identical_unrelated_and_blank_images():
    base = scene(96, 96, 1)
    notes, stats = pair_sanity_warnings(base, base.copy())
    assert stats.get("identical") is True and any("pixel-identical" in n for n in notes)
    notes, _ = pair_sanity_warnings(city(1), city(2))
    assert any("very little structure" in n for n in notes)
    related, _ = with_building(city(1), (100, 100, 130, 140))
    notes, stats = pair_sanity_warnings(city(1), related)
    assert notes == [] and stats["structural_correlation"] > 0.6
    notes, _ = pair_sanity_warnings(base, np.full_like(base, 128))
    assert any("almost uniform" in n for n in notes)
