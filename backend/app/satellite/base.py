"""Satellite imagery provider abstraction.

A provider can *search* a catalogue for scenes intersecting an area of interest and
*retrieve* a georeferenced, model-ready RGB raster (CRS, transform and acquisition metadata
preserved) for one scene clipped to that area. Web-map tiles are deliberately not a provider:
they are rendered pictures without reliable acquisition dates or radiometry.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date

import httpx
from shapely.geometry import box, shape

from backend.app.errors import AppError
from backend.app.services.imaging import LoadedImage

BBox = tuple[float, float, float, float]  # west, south, east, north (WGS84 degrees)


@dataclass
class SearchQuery:
    bbox: BBox
    start: date
    end: date
    max_cloud_cover: float = 30.0
    limit: int = 10


@dataclass
class SceneCandidate:
    id: str
    provider: str
    collection: str
    acquired: str  # ISO 8601 UTC
    cloud_cover: float | None
    platform: str | None
    crs: str | None
    resolution_m: float
    aoi_coverage: float  # 0..1 share of the AOI inside the scene footprint
    thumbnail_url: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedScene:
    image: LoadedImage
    metadata: dict


def bbox_size_km(bbox: BBox) -> tuple[float, float]:
    west, south, east, north = bbox
    mid_lat = math.radians((south + north) / 2)
    return (east - west) * 111.320 * math.cos(mid_lat), (north - south) * 110.574


def validate_bbox(bbox: BBox, min_km: float, max_km: float) -> None:
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
        raise AppError("INVALID_AOI", "The area of interest must be a valid west<east, south<north box in "
                       "WGS84 degrees", status=422)
    width, height = bbox_size_km(bbox)
    if min(width, height) < min_km:
        raise AppError("AOI_TOO_SMALL", f"The area of interest is {width:.2f} x {height:.2f} km; each side "
                       f"must be at least {min_km:g} km", status=422)
    if max(width, height) > max_km:
        raise AppError("AOI_TOO_LARGE", f"The area of interest is {width:.1f} x {height:.1f} km; each side "
                       f"must be at most {max_km:g} km", status=422)


def aoi_coverage(geometry: dict | None, bbox: BBox) -> float:
    if not geometry:
        return 0.0
    aoi = box(*bbox)
    try:
        footprint = shape(geometry)
        if not footprint.is_valid:
            footprint = footprint.buffer(0)
        return float(footprint.intersection(aoi).area / aoi.area)
    except Exception:
        return 0.0


class ProviderError(AppError):
    """The remote imagery service failed or returned something unusable."""

    def __init__(self, provider: str, message: str, status: int = 502, code: str = "SATELLITE_PROVIDER_ERROR"):
        super().__init__(code, f"{provider}: {message}", status=status, details={"provider": provider})


class SatelliteProvider(ABC):
    key: str
    name: str
    description: str
    data_license: str
    documentation_url: str
    requires_credentials: bool = False
    verified: bool = False  # True only when retrieval has been exercised end-to-end in this project

    def __init__(self, timeout: float, transport: httpx.BaseTransport | None = None) -> None:
        self.timeout = timeout
        self._transport = transport

    def client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self._transport,
                            headers={"User-Agent": "TerraVisionAI/0.1 (research prototype)"},
                            follow_redirects=True)

    @abstractmethod
    def configured(self) -> bool:
        """Whether the provider can currently retrieve imagery (credentials present if needed)."""

    @abstractmethod
    def search(self, query: SearchQuery) -> list[SceneCandidate]: ...

    @abstractmethod
    def retrieve(self, scene_id: str, bbox: BBox) -> RetrievedScene: ...

    def describe(self) -> dict:
        return {
            "key": self.key, "name": self.name, "description": self.description,
            "data_license": self.data_license, "documentation_url": self.documentation_url,
            "requires_credentials": self.requires_credentials, "configured": self.configured(),
            "verified": self.verified,
        }

    # ------------------------------------------------------------ shared I/O
    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            with self.client() as client:
                response = client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, "the service did not respond in time", status=504,
                                code="SATELLITE_PROVIDER_TIMEOUT") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"could not reach the service ({type(exc).__name__})") from exc
        if response.status_code in (401, 403):
            raise ProviderError(self.name, "the service rejected the credentials", status=502,
                                code="SATELLITE_PROVIDER_AUTH")
        if response.status_code >= 400:
            raise ProviderError(self.name, f"the service returned HTTP {response.status_code}")
        return response

    def _json(self, method: str, url: str, **kwargs) -> dict:
        response = self._request(method, url, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(self.name, "the service returned an unreadable response") from exc
