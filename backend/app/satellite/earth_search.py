"""Element 84 Earth Search provider: Sentinel-2 L2A cloud-optimised GeoTIFFs on AWS Open Data.

Catalogue: STAC API at https://earth-search.aws.element84.com/v1 (no authentication).
Imagery:   the ``visual`` asset of each item is the 10 m true-colour (TCI) 8-bit COG; only
           the window covering the area of interest is read over HTTP range requests.
Licence:   Copernicus Sentinel data - free, full and open access (attribution required).
"""

from __future__ import annotations

from urllib.parse import quote, urlparse

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

from backend.app.satellite import stac
from backend.app.satellite.base import (
    BBox,
    ProviderError,
    RetrievedScene,
    SatelliteProvider,
    SceneCandidate,
    SearchQuery,
    aoi_coverage,
)
from backend.app.services.imaging import GeoInfo, LoadedImage

COLLECTION = "sentinel-2-l2a"
# Asset hosts this provider is allowed to read from (guards against a tampered catalogue response).
ALLOWED_ASSET_HOSTS = (
    "sentinel-cogs.s3.us-west-2.amazonaws.com",
    "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
)
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "AWS_NO_SIGN_REQUEST": "YES",
}


class EarthSearchProvider(SatelliteProvider):
    key = "earth_search"
    name = "Earth Search (Sentinel-2 L2A on AWS)"
    description = ("Element 84 Earth Search STAC catalogue of Copernicus Sentinel-2 Level-2A scenes; 10 m "
                   "true-colour imagery read directly from public cloud-optimised GeoTIFFs. No account needed.")
    data_license = "Copernicus Sentinel data (free, full and open; attribution required)"
    documentation_url = "https://element84.com/earth-search/"
    requires_credentials = False

    def __init__(self, base_url: str, timeout: float, transport=None) -> None:
        super().__init__(timeout, transport)
        self.base_url = base_url.rstrip("/")

    def configured(self) -> bool:
        return True

    def search(self, query: SearchQuery) -> list[SceneCandidate]:
        data = self._json("POST", f"{self.base_url}/search", json=stac.search_body(query, COLLECTION))
        features = data.get("features")
        if not isinstance(features, list):
            raise ProviderError(self.name, "the catalogue response did not contain a feature list")
        candidates = [stac.to_candidate(item, self.key, COLLECTION, query) for item in features
                      if "visual" in item.get("assets", {})]
        return stac.rank(candidates, query.limit)

    def _item(self, scene_id: str) -> dict:
        return self._json("GET", f"{self.base_url}/collections/{COLLECTION}/items/{quote(scene_id, safe='')}")

    def retrieve(self, scene_id: str, bbox: BBox) -> RetrievedScene:
        item = self._item(scene_id)
        href = item.get("assets", {}).get("visual", {}).get("href", "")
        parsed = urlparse(href)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_ASSET_HOSTS:
            raise ProviderError(self.name, "the scene has no true-colour asset on an allowed host")
        coverage = aoi_coverage(item.get("geometry"), bbox)
        if coverage < 0.99:
            raise ProviderError(self.name, f"scene {scene_id} covers only {coverage * 100:.0f}% of the area of "
                                "interest; choose a scene with full coverage", status=422,
                                code="SCENE_PARTIAL_COVERAGE")
        try:
            with rasterio.Env(GDAL_HTTP_TIMEOUT=str(int(self.timeout)), **_GDAL_ENV), rasterio.open(href) as src:
                bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
                window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
                window = window.intersection(Window(0, 0, src.width, src.height))
                if window.width < 32 or window.height < 32:
                    raise ProviderError(self.name, "the area of interest is too small at 10 m resolution",
                                        status=422, code="AOI_TOO_SMALL")
                bands = src.read([1, 2, 3], window=window)
                transform = window_transform(window, src.transform)
                crs = src.crs
        except ProviderError:
            raise
        except (RasterioError, OSError, ValueError) as exc:
            raise ProviderError(self.name, f"the imagery could not be read ({type(exc).__name__})") from exc
        array = np.ascontiguousarray(np.transpose(bands, (1, 2, 0)).astype(np.uint8))
        nodata_fraction = float(np.all(array == 0, axis=2).mean())
        if nodata_fraction > 0.2:
            raise ProviderError(self.name, f"{nodata_fraction * 100:.0f}% of the area of interest has no data "
                                "in this scene", status=422, code="SCENE_PARTIAL_COVERAGE")
        geo = GeoInfo(crs, transform, array.shape[1], array.shape[0])
        props = item.get("properties", {})
        metadata = {
            "provider": self.key, "provider_name": self.name, "scene_id": scene_id, "collection": COLLECTION,
            "acquired": props.get("datetime"), "cloud_cover": props.get("eo:cloud_cover"),
            "platform": props.get("platform"), "asset": "visual (TCI, 10 m, 8-bit)", "asset_href": href,
            "crs": crs.to_string(), "resolution_m": 10.0, "nodata_fraction": nodata_fraction,
            "license": self.data_license,
            "attribution": f"Contains modified Copernicus Sentinel data {str(props.get('datetime', ''))[:4]}",
        }
        return RetrievedScene(LoadedImage(array, "tiff", geo, []), metadata)
