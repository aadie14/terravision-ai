"""Copernicus Data Space Ecosystem (CDSE) provider.

Search:    public STAC API  https://stac.dataspace.copernicus.eu/v1  (no authentication).
Retrieval: Sentinel Hub Process API  https://sh.dataspace.copernicus.eu/api/v1/process
           authenticated with an OAuth2 *client-credentials* token obtained from
           https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
           The OAuth client is created by the user in the CDSE dashboard and supplied through
           TV_CDSE_CLIENT_ID / TV_CDSE_CLIENT_SECRET. Nothing is retrieved without it.
Docs:      https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html

STATUS: retrieval is implemented against the documented API but has NOT been exercised
end-to-end in this project because no CDSE credentials were available (``verified = False``).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from backend.app.satellite import stac
from backend.app.satellite.base import (
    BBox,
    ProviderError,
    RetrievedScene,
    SatelliteProvider,
    SceneCandidate,
    SearchQuery,
    bbox_size_km,
)
from backend.app.services.imaging import _load_tiff

COLLECTION = "sentinel-2-l2a"

# True-colour evalscript: reflectance * 2.5 gain, 8-bit output (per Sentinel Hub S2 L2A examples).
EVALSCRIPT = """//VERSION=3
function setup() {
  return { input: ["B04", "B03", "B02"], output: { bands: 3, sampleType: "UINT8" } };
}
function evaluatePixel(s) {
  return [s.B04, s.B03, s.B02].map(v => Math.max(0, Math.min(255, 255 * 2.5 * v)));
}
"""


class CDSEProvider(SatelliteProvider):
    key = "cdse"
    name = "Copernicus Data Space Ecosystem"
    description = ("Official Copernicus catalogue (STAC) with Sentinel-2 L2A retrieval through the Sentinel Hub "
                   "Process API. Retrieval needs a free CDSE account and an OAuth client.")
    data_license = "Copernicus Sentinel data (free, full and open; attribution required)"
    documentation_url = "https://documentation.dataspace.copernicus.eu/"
    requires_credentials = True
    verified = False

    def __init__(self, stac_url: str, token_url: str, process_url: str, client_id: str, client_secret: str,
                 timeout: float, transport=None) -> None:
        super().__init__(timeout, transport)
        self.stac_url = stac_url.rstrip("/")
        self.token_url = token_url
        self.process_url = process_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: tuple[str, float] | None = None

    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def search(self, query: SearchQuery) -> list[SceneCandidate]:
        data = self._json("POST", f"{self.stac_url}/search", json=stac.search_body(query, COLLECTION))
        features = data.get("features")
        if not isinstance(features, list):
            raise ProviderError(self.name, "the catalogue response did not contain a feature list")
        return stac.rank([stac.to_candidate(item, self.key, COLLECTION, query) for item in features], query.limit)

    def _access_token(self) -> str:
        if self._token and self._token[1] > time.time() + 30:
            return self._token[0]
        data = self._json("POST", self.token_url, data={
            "grant_type": "client_credentials", "client_id": self._client_id,
            "client_secret": self._client_secret,
        })
        token = data.get("access_token")
        if not token:
            raise ProviderError(self.name, "no access token was issued", code="SATELLITE_PROVIDER_AUTH")
        self._token = (token, time.time() + float(data.get("expires_in", 300)))
        return token

    def retrieve(self, scene_id: str, bbox: BBox) -> RetrievedScene:
        if not self.configured():
            raise ProviderError(self.name, "retrieval needs CDSE OAuth credentials (set TV_CDSE_CLIENT_ID and "
                                "TV_CDSE_CLIENT_SECRET)", status=503, code="SATELLITE_PROVIDER_NOT_CONFIGURED")
        item = self._json("GET", f"{self.stac_url}/collections/{COLLECTION}/items/{scene_id}")
        acquired = item.get("properties", {}).get("datetime")
        if not acquired:
            raise ProviderError(self.name, f"scene {scene_id} has no acquisition time")
        moment = datetime.fromisoformat(acquired.replace("Z", "+00:00"))
        width_km, height_km = bbox_size_km(bbox)
        body = {
            "input": {
                "bounds": {"bbox": list(bbox),
                           "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
                "data": [{"type": COLLECTION, "dataFilter": {"timeRange": {
                    "from": (moment - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": (moment + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }}}],
            },
            "output": {"width": max(32, round(width_km * 100)), "height": max(32, round(height_km * 100)),
                       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": EVALSCRIPT,
        }
        response = self._request("POST", self.process_url, json=body,
                                 headers={"Authorization": f"Bearer {self._access_token()}",
                                          "Accept": "image/tiff"})
        image = _load_tiff(f"{scene_id}.tif", response.content, 32, 8192)
        if image.geo is None:
            raise ProviderError(self.name, "the returned raster carries no georeferencing")
        props = item.get("properties", {})
        metadata = {
            "provider": self.key, "provider_name": self.name, "scene_id": scene_id, "collection": COLLECTION,
            "acquired": acquired, "cloud_cover": props.get("eo:cloud_cover"), "platform": props.get("platform"),
            "asset": "B04/B03/B02 true colour via Process API (8-bit, gain 2.5)",
            "crs": image.geo.crs.to_string(), "resolution_m": 10.0, "license": self.data_license,
            "attribution": f"Contains modified Copernicus Sentinel data {acquired[:4]}",
        }
        return RetrievedScene(image, metadata)
