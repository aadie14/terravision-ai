"""Shared STAC API (v1) search helper for Sentinel-2 L2A collections."""

from __future__ import annotations

from backend.app.satellite.base import SceneCandidate, SearchQuery, aoi_coverage


def search_body(query: SearchQuery, collection: str) -> dict:
    return {
        "collections": [collection],
        "bbox": list(query.bbox),
        "datetime": f"{query.start.isoformat()}T00:00:00Z/{query.end.isoformat()}T23:59:59Z",
        "limit": max(1, min(query.limit * 3, 100)),  # over-fetch: partial-coverage scenes are dropped
        "query": {"eo:cloud_cover": {"lte": query.max_cloud_cover}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }


def item_crs(properties: dict) -> str | None:
    if properties.get("proj:code"):
        return str(properties["proj:code"])
    if properties.get("proj:epsg"):
        return f"EPSG:{properties['proj:epsg']}"
    return None


def to_candidate(item: dict, provider: str, collection: str, query: SearchQuery) -> SceneCandidate:
    props = item.get("properties", {})
    assets = item.get("assets", {})
    thumb = assets.get("thumbnail", {}).get("href")
    return SceneCandidate(
        id=item["id"], provider=provider, collection=collection,
        acquired=props.get("datetime") or props.get("start_datetime") or "",
        cloud_cover=props.get("eo:cloud_cover"),
        platform=props.get("platform"),
        crs=item_crs(props),
        resolution_m=10.0,
        aoi_coverage=aoi_coverage(item.get("geometry"), query.bbox),
        thumbnail_url=thumb if isinstance(thumb, str) and thumb.startswith("https://") else None,
        extra={"mgrs_tile": props.get("grid:code") or props.get("s2:mgrs_tile"),
               "processing_baseline": props.get("s2:processing_baseline") or props.get("processing:version")},
    )


def rank(candidates: list[SceneCandidate], limit: int) -> list[SceneCandidate]:
    """Full-coverage scenes first, then by cloud cover."""
    candidates.sort(key=lambda c: (c.aoi_coverage < 0.999, c.cloud_cover if c.cloud_cover is not None else 101))
    return candidates[:limit]
