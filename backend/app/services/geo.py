"""Georeferenced exports: GeoTIFF mask, GeoJSON change regions, geodesic area, web-map overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Geod
from rasterio.features import shapes
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds, transform_geom
from shapely.geometry import shape

from backend.app.services.imaging import GeoInfo

_GEOD = Geod(ellps="WGS84")
MAX_REGIONS = 5000


def write_geotiff(path: Path, array: np.ndarray, geo: GeoInfo, *, nodata: int | None = None,
                  tags: dict | None = None) -> None:
    """Write a uint8 single-band [H, W] or multi-band [H, W, C] array with ``geo`` georeferencing."""
    bands = array[None] if array.ndim == 2 else np.transpose(array, (2, 0, 1))
    profile = {
        "driver": "GTiff", "height": bands.shape[1], "width": bands.shape[2], "count": bands.shape[0],
        "dtype": "uint8", "crs": geo.crs, "transform": geo.transform, "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands.astype(np.uint8))
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def change_regions(mask: np.ndarray, geo: GeoInfo) -> tuple[dict, dict]:
    """Vectorise the change mask.

    Returns ``(geojson_feature_collection_in_WGS84, summary)``. Region areas are geodesic
    areas on the WGS84 ellipsoid, so they are valid for any source CRS (including Web
    Mercator, where pixel-count * pixel-size would be badly wrong away from the equator).
    """
    features = []
    total_area = 0.0
    truncated = False
    mask_u8 = mask.astype(np.uint8)
    for i, (geom, value) in enumerate(shapes(mask_u8, mask=mask_u8 == 1, transform=geo.transform)):
        if value != 1:
            continue
        if i >= MAX_REGIONS:
            truncated = True
            break
        geom_wgs = transform_geom(geo.crs, "EPSG:4326", geom, precision=7)
        area, _ = _GEOD.geometry_area_perimeter(shape(geom_wgs))
        area = abs(area)
        total_area += area
        features.append({"type": "Feature", "geometry": geom_wgs,
                         "properties": {"region_id": len(features) + 1, "area_m2": round(area, 2)}})
    collection = {
        "type": "FeatureCollection",
        "crs_note": "Coordinates are WGS84 longitude/latitude (EPSG:4326, RFC 7946).",
        "features": features,
    }
    summary = {
        "region_count": len(features),
        "changed_area_m2": total_area,
        "changed_area_ha": total_area / 10_000.0,
        "area_method": "geodesic polygon area on the WGS84 ellipsoid",
        "regions_truncated": truncated,
    }
    return collection, summary


def footprint_area_m2(geo: GeoInfo) -> float:
    west, south, east, north = geo.bounds
    ring = {"type": "Polygon", "coordinates": [[(west, south), (east, south), (east, north), (west, north),
                                                 (west, south)]]}
    area, _ = _GEOD.geometry_area_perimeter(shape(transform_geom(geo.crs, "EPSG:4326", ring)))
    return abs(area)


def web_mercator_overlay(path: Path, rgba: np.ndarray, geo: GeoInfo, max_side: int = 2048) -> dict:
    """Reproject an RGBA [H, W, 4] image to EPSG:3857 and save it as a PNG for Leaflet.

    Leaflet stretches an image overlay linearly between its corner coordinates in Web
    Mercator, so the raster must itself be in EPSG:3857 for the overlay to sit correctly.
    Returns the overlay bounds in WGS84 (south, west, north, east).
    """
    transform, width, height = calculate_default_transform(geo.crs, "EPSG:3857", geo.width, geo.height, *geo.bounds)
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
        transform = transform * transform.scale(1 / scale, 1 / scale)
    dest = np.zeros((4, height, width), dtype=np.uint8)
    reproject(source=np.transpose(rgba, (2, 0, 1)), destination=dest, src_transform=geo.transform,
              src_crs=geo.crs, dst_transform=transform, dst_crs="EPSG:3857", resampling=Resampling.nearest)
    Image.fromarray(np.transpose(dest, (1, 2, 0)), "RGBA").save(path)
    west, south, east, north = transform_bounds(
        "EPSG:3857", "EPSG:4326", *rasterio.transform.array_bounds(height, width, transform)
    )
    return {"south": south, "west": west, "north": north, "east": east}
