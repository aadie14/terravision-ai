"""Image decoding, validation and bitemporal pair checks (PNG / JPEG / GeoTIFF)."""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

from backend.app.errors import AppError

log = logging.getLogger("terravision.imaging")

ALLOWED_EXTENSIONS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".tif": "tiff", ".tiff": "tiff"}
_MAGIC = {
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpeg": (b"\xff\xd8\xff",),
    "tiff": (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"),
}
# Guard against decompression bombs independent of the configurable side limit.
Image.MAX_IMAGE_PIXELS = 64_000_000


@dataclass
class GeoInfo:
    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return rasterio.transform.array_bounds(self.height, self.width, self.transform)

    @property
    def resolution(self) -> tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)

    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        return transform_bounds(self.crs, "EPSG:4326", *self.bounds, densify_pts=21)

    def describe(self) -> dict:
        west, south, east, north = self.bounds_wgs84()
        return {
            "crs": self.crs.to_string(),
            "crs_is_projected": bool(self.crs.is_projected),
            "crs_units": self.crs.linear_units if self.crs.is_projected else "degree",
            "bounds": list(self.bounds),
            "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
            "resolution": list(self.resolution),
            "transform": list(self.transform)[:6],
        }


@dataclass
class LoadedImage:
    array: np.ndarray  # uint8 [H, W, 3]
    format: str
    geo: GeoInfo | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]


def detect_format(filename: str, data: bytes) -> str:
    """Validate extension *and* magic bytes; return 'png' | 'jpeg' | 'tiff'."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AppError("UNSUPPORTED_FILE_TYPE",
                       f"'{Path(filename or '').name or 'file'}' is not a supported image. "
                       "Use PNG, JPEG or GeoTIFF.", status=415, details={"extension": ext})
    fmt = ALLOWED_EXTENSIONS[ext]
    if not data:
        raise AppError("EMPTY_FILE", f"'{Path(filename).name}' is empty")
    if not any(data.startswith(sig) for sig in _MAGIC[fmt]):
        raise AppError("CORRUPTED_IMAGE",
                       f"'{Path(filename).name}' does not contain valid {fmt.upper()} data", status=422)
    return fmt


def _check_side(name: str, width: int, height: int, min_side: int, max_side: int) -> None:
    if min(width, height) < min_side:
        raise AppError("IMAGE_TOO_SMALL", f"'{name}' is {width}x{height}px; the minimum side is {min_side}px",
                       status=422, details={"width": width, "height": height})
    if max(width, height) > max_side:
        raise AppError("IMAGE_TOO_LARGE", f"'{name}' is {width}x{height}px; the maximum side is {max_side}px",
                       status=413, details={"width": width, "height": height})


def _stretch_to_uint8(bands: np.ndarray, nodata_mask: np.ndarray | None) -> np.ndarray:
    """Per-band 2-98 percentile stretch for non-8-bit rasters -> uint8 [C, H, W]."""
    out = np.zeros(bands.shape, dtype=np.uint8)
    for i, band in enumerate(bands.astype(np.float32)):
        valid = band[~nodata_mask] if nodata_mask is not None and (~nodata_mask).any() else band
        lo, hi = np.percentile(valid, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        out[i] = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def _load_tiff(name: str, data: bytes, min_side: int, max_side: int) -> LoadedImage:
    notes: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.MemoryFile(data) as mem, mem.open() as src:
                _check_side(name, src.width, src.height, min_side, max_side)
                if src.count >= 3:
                    indexes = [1, 2, 3]
                    if src.count > 3:
                        notes.append(f"Raster has {src.count} bands; bands 1-3 were used as RGB.")
                elif src.count == 1:
                    indexes = [1, 1, 1]
                    notes.append("Single-band raster replicated to 3 channels.")
                else:
                    raise AppError("UNSUPPORTED_CHANNELS",
                                   f"'{name}' has {src.count} bands; 1 or at least 3 are required", status=422)
                bands = src.read(indexes)
                nodata_mask = None
                if src.nodata is not None:
                    nodata_mask = np.all(bands == src.nodata, axis=0)
                if bands.dtype != np.uint8:
                    notes.append(f"{bands.dtype} raster stretched to 8-bit (2-98 percentile per band).")
                    bands = _stretch_to_uint8(bands, nodata_mask)
                geo = None
                has_transform = src.transform is not None and not src.transform.is_identity
                if src.crs is not None and has_transform:
                    try:
                        crs = CRS.from_user_input(src.crs)
                        transform_bounds(crs, "EPSG:4326", *src.bounds)
                    except Exception as exc:
                        raise AppError("INVALID_CRS", f"'{name}' has a coordinate reference system that "
                                       f"cannot be interpreted: {exc}", status=422) from exc
                    if abs(src.transform.b) > 1e-9 or abs(src.transform.d) > 1e-9:
                        notes.append("Raster is rotated; georeferencing was ignored.")
                    else:
                        geo = GeoInfo(crs, src.transform, src.width, src.height)
                elif has_transform:
                    raise AppError("INVALID_CRS", f"'{name}' has a geotransform but no coordinate reference "
                                   "system; cannot georeference it reliably", status=422)
                else:
                    notes.append("TIFF has no georeferencing; treated as a plain image.")
    except AppError:
        raise
    except (RasterioIOError, ValueError, OSError) as exc:
        raise AppError("CORRUPTED_IMAGE", f"'{name}' could not be decoded as a TIFF raster", status=422) from exc
    array = np.ascontiguousarray(np.transpose(bands, (1, 2, 0)))
    return LoadedImage(array, "tiff", geo, notes)


def load_image(filename: str, data: bytes, *, min_side: int, max_side: int) -> LoadedImage:
    """Decode an uploaded image into uint8 RGB, validating type, integrity, size and channels."""
    name = Path(filename).name
    fmt = detect_format(filename, data)
    if fmt == "tiff":
        return _load_tiff(name, data, min_side, max_side)
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # integrity check; the image must be reopened afterwards
        with Image.open(io.BytesIO(data)) as img:
            _check_side(name, img.width, img.height, min_side, max_side)
            notes = []
            if img.mode not in {"RGB", "RGBA", "L", "P", "LA"}:
                raise AppError("UNSUPPORTED_CHANNELS", f"'{name}' uses colour mode {img.mode}, which is not "
                               "supported", status=422)
            if img.mode != "RGB":
                notes.append(f"Converted from {img.mode} to RGB.")
            array = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except AppError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise AppError("CORRUPTED_IMAGE", f"'{name}' is corrupted or cannot be decoded", status=422) from exc
    return LoadedImage(np.ascontiguousarray(array), fmt, None, notes)


def load_mask(filename: str, data: bytes, *, expected_shape: tuple[int, int], max_side: int) -> np.ndarray:
    """Decode a ground-truth mask to a boolean [H, W] array matching ``expected_shape``."""
    image = load_image(filename, data, min_side=1, max_side=max_side)
    gray = image.array[..., 0] if (image.array[..., 0] == image.array[..., 1]).all() else image.array.max(axis=2)
    if gray.shape != expected_shape:
        raise AppError("MASK_DIMENSION_MISMATCH",
                       f"Ground-truth mask is {gray.shape[1]}x{gray.shape[0]}px but the images are "
                       f"{expected_shape[1]}x{expected_shape[0]}px", status=422)
    threshold = 0 if gray.max() <= 1 else 127
    return gray > threshold


def _same_grid(a: GeoInfo, b: GeoInfo) -> bool:
    return (a.crs == b.crs and a.width == b.width and a.height == b.height
            and a.transform.almost_equals(b.transform, precision=max(a.resolution) * 0.01))


def _overlap_fraction(a: GeoInfo, b: GeoInfo) -> float:
    """Share of raster ``a``'s footprint covered by raster ``b`` (in a's CRS)."""
    aw, as_, ae, an = a.bounds
    bw, bs, be, bn = transform_bounds(b.crs, a.crs, *b.bounds, densify_pts=21)
    iw, ih = min(ae, be) - max(aw, bw), min(an, bn) - max(as_, bs)
    if iw <= 0 or ih <= 0:
        return 0.0
    return (iw * ih) / ((ae - aw) * (an - as_))


def align_pair(t1: LoadedImage, t2: LoadedImage) -> tuple[LoadedImage, LoadedImage, list[str]]:
    """Validate that two images form a usable bitemporal pair; resample T2 onto T1 when both
    are georeferenced on different grids. Raises AppError for incompatible pairs."""
    warnings_out: list[str] = []
    if t1.geo and t2.geo:
        if _same_grid(t1.geo, t2.geo):
            return t1, t2, warnings_out
        overlap = _overlap_fraction(t1.geo, t2.geo)
        if overlap < 0.9:
            raise AppError("GEOSPATIAL_MISALIGNMENT",
                           f"The two rasters cover different areas (T2 covers {overlap * 100:.0f}% of T1's "
                           "footprint; at least 90% is required)", status=422,
                           details={"overlap_fraction": overlap, "t1_crs": t1.geo.crs.to_string(),
                                    "t2_crs": t2.geo.crs.to_string()})
        r1, r2 = max(t1.geo.resolution), max(t2.geo.resolution)
        if t1.geo.crs == t2.geo.crs and max(r1, r2) / min(r1, r2) > 4:
            raise AppError("RESOLUTION_MISMATCH", "The rasters differ in pixel resolution by more than 4x",
                           status=422, details={"t1_resolution": r1, "t2_resolution": r2})
        dest = np.zeros((3, t1.height, t1.width), dtype=np.uint8)
        reproject(
            source=np.transpose(t2.array, (2, 0, 1)), destination=dest,
            src_transform=t2.geo.transform, src_crs=t2.geo.crs,
            dst_transform=t1.geo.transform, dst_crs=t1.geo.crs, resampling=Resampling.bilinear,
        )
        what = "reprojected" if t1.geo.crs != t2.geo.crs else "resampled"
        warnings_out.append(
            f"T2 was {what} onto the T1 grid ({t2.geo.crs.to_string()} -> {t1.geo.crs.to_string()}, "
            f"bilinear)."
        )
        aligned = LoadedImage(np.ascontiguousarray(np.transpose(dest, (1, 2, 0))), t2.format,
                              GeoInfo(t1.geo.crs, t1.geo.transform, t1.width, t1.height), t2.notes)
        return t1, aligned, warnings_out

    if (t1.height, t1.width) != (t2.height, t2.width):
        raise AppError("IMAGE_DIMENSION_MISMATCH",
                       f"T1 is {t1.width}x{t1.height}px but T2 is {t2.width}x{t2.height}px. Non-georeferenced "
                       "images must be co-registered and have identical dimensions.", status=422,
                       details={"t1": [t1.width, t1.height], "t2": [t2.width, t2.height]})
    if bool(t1.geo) != bool(t2.geo):
        warnings_out.append("Only one of the two images is georeferenced; pixel alignment is assumed and "
                            "the available georeferencing is used for both.")
        geo = t1.geo or t2.geo
        t1, t2 = (LoadedImage(t1.array, t1.format, geo, t1.notes), LoadedImage(t2.array, t2.format, geo, t2.notes))
    return t1, t2, warnings_out


# Calibrated on 40 LEVIR-CD test pairs (2026-11-11): same-place pairs score median 0.47 (5th percentile
# 0.16); deliberately mismatched pairs score median -0.03 (95th percentile 0.15). 0.10 flags most
# unrelated pairs while rarely flagging genuine ones. The check only ever produces a warning.
LOW_STRUCTURE_THRESHOLD = 0.10


def structural_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of gradient magnitudes at reduced resolution (-1..1).

    Co-registered acquisitions of the same place share most edges (roads, field boundaries)
    even when colours differ; unrelated images do not. Used only to *warn* the user.
    """
    def edges(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 2.0)

    ea, eb = edges(a), edges(b)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    denom = float(np.sqrt((ea**2).sum() * (eb**2).sum()))
    return float((ea * eb).sum() / denom) if denom > 1e-6 else 0.0


def pair_sanity_warnings(t1: np.ndarray, t2: np.ndarray) -> tuple[list[str], dict]:
    """Non-blocking checks that surface suspicious pairs instead of silently accepting them."""
    notes: list[str] = []
    stats: dict = {}
    if np.array_equal(t1, t2):
        notes.append("T1 and T2 are pixel-identical; no change can be detected.")
        stats["identical"] = True
        return notes, stats
    corr = structural_correlation(t1, t2)
    stats["structural_correlation"] = corr
    if corr < LOW_STRUCTURE_THRESHOLD:
        notes.append(
            f"The two images share very little structure (edge correlation {corr:.2f}). They may show "
            "different places or be badly mis-registered; treat the result with caution. (Heuristic check: "
            "it can also trigger for a genuine pair in which most of the scene changed.)"
        )
    for label, img in (("T1", t1), ("T2", t2)):
        if float(img.std()) < 2.0:
            notes.append(f"{label} is almost uniform (no visible content).")
    return notes, stats
