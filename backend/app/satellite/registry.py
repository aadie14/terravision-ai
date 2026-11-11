"""Builds the configured satellite providers and offers place-name search (geocoding)."""

from __future__ import annotations

import httpx

from backend.app.config import Settings
from backend.app.errors import AppError
from backend.app.satellite.base import ProviderError, SatelliteProvider
from backend.app.satellite.cdse import CDSEProvider
from backend.app.satellite.earth_search import EarthSearchProvider


class ProviderRegistry:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport
        earth = EarthSearchProvider(settings.earth_search_url, settings.satellite_timeout_s, transport)
        cdse = CDSEProvider(settings.cdse_stac_url, settings.cdse_token_url, settings.cdse_process_url,
                            settings.cdse_client_id, settings.cdse_client_secret,
                            settings.satellite_timeout_s, transport)
        self._providers: dict[str, SatelliteProvider] = {earth.key: earth, cdse.key: cdse}

    def list(self) -> list[dict]:
        return [p.describe() for p in self._providers.values()]

    def get(self, key: str) -> SatelliteProvider:
        if key not in self._providers:
            raise AppError("UNKNOWN_PROVIDER", f"Satellite provider '{key}' does not exist", status=404,
                           details={"available": sorted(self._providers)})
        return self._providers[key]

    def geocode(self, text: str, limit: int = 5) -> list[dict]:
        """Place-name search through OpenStreetMap Nominatim (public endpoint, light use only)."""
        try:
            with httpx.Client(timeout=15, transport=self._transport,
                              headers={"User-Agent": "TerraVisionAI/0.1 (research prototype)"}) as client:
                response = client.get(self.settings.geocoder_url,
                                      params={"q": text, "format": "jsonv2", "limit": limit})
            response.raise_for_status()
            rows = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Place search", "the geocoding service is unavailable",
                                code="GEOCODER_UNAVAILABLE") from exc
        results = []
        for row in rows if isinstance(rows, list) else []:
            try:
                south, north, west, east = (float(v) for v in row["boundingbox"])
                results.append({"name": row["display_name"], "lat": float(row["lat"]), "lon": float(row["lon"]),
                                "bbox": [west, south, east, north]})
            except (KeyError, TypeError, ValueError):
                continue
        return results
