"""Satellite provider tests with a mocked remote service (no network access needed)."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from backend.app.errors import AppError
from backend.app.satellite.base import RetrievedScene, SearchQuery, aoi_coverage, bbox_size_km, validate_bbox
from backend.app.satellite.cdse import CDSEProvider
from backend.app.satellite.earth_search import EarthSearchProvider
from backend.app.services.imaging import GeoInfo, LoadedImage
from tests.conftest import scene, wait_for, with_building

BBOX = {"west": 77.60, "south": 13.09, "east": 77.66, "north": 13.14}
BBOX_T = (77.60, 13.09, 77.66, 13.14)
SEARCH = {"provider": "earth_search", "bbox": BBOX, "start": "2019-01-01", "end": "2019-03-31"}


def stac_item(item_id: str, cloud: float, *, cover: bool = True, when: str = "2019-03-15T05:27:57Z") -> dict:
    w, s, e, n = (77.0, 12.5, 78.0, 13.5) if cover else (77.63, 12.5, 78.0, 13.5)
    return {
        "id": item_id,
        "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
        "properties": {"datetime": when, "eo:cloud_cover": cloud, "platform": "sentinel-2a", "proj:code": "EPSG:32643",
                       "grid:code": "MGRS-43PGQ"},
        "assets": {"visual": {"href": f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/{item_id}/TCI.tif"},
                   "thumbnail": {"href": f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/{item_id}/thumb.jpg"}},
    }


def error_code(response) -> str:
    return response.json()["error"]["code"]


# ------------------------------------------------------------------- helpers
def test_bbox_helpers():
    width, height = bbox_size_km(BBOX_T)
    assert width == pytest.approx(6.5, abs=0.2) and height == pytest.approx(5.5, abs=0.2)
    validate_bbox(BBOX_T, 1, 25)
    for bad, code in [((77.6, 13.09, 77.6005, 13.14), "AOI_TOO_SMALL"), ((77, 13, 78, 14), "AOI_TOO_LARGE"),
                      ((77.66, 13.09, 77.60, 13.14), "INVALID_AOI"), ((77.6, 86, 77.66, 87), "INVALID_AOI")]:
        with pytest.raises(AppError) as exc:
            validate_bbox(bad, 1, 25)
        assert exc.value.code == code


def test_aoi_coverage_full_partial_and_missing_geometry():
    assert aoi_coverage(stac_item("a", 1)["geometry"], BBOX_T) == pytest.approx(1.0)
    assert aoi_coverage(stac_item("a", 1, cover=False)["geometry"], BBOX_T) == pytest.approx(0.5, abs=0.01)
    assert aoi_coverage(None, BBOX_T) == 0.0
    assert aoi_coverage({"type": "Nonsense"}, BBOX_T) == 0.0


# -------------------------------------------------------------------- search
def test_providers_endpoint_reports_configuration(client):
    providers = {p["key"]: p for p in client.get("/api/satellite/providers").json()["providers"]}
    assert providers["earth_search"]["configured"] is True and providers["earth_search"]["requires_credentials"] is False
    assert providers["cdse"]["configured"] is False and providers["cdse"]["requires_credentials"] is True
    assert providers["cdse"]["verified"] is False


def test_search_parses_ranks_and_sends_a_correct_stac_query(client, satellite):
    satellite.handler = lambda request: httpx.Response(200, json={"features": [
        stac_item("S2A_cloudy", 18.0), stac_item("S2A_partial", 0.1, cover=False), stac_item("S2A_clear", 0.5),
        {**stac_item("S2A_no_visual", 0.0), "assets": {}},
    ]})
    response = client.post("/api/satellite/search", json={**SEARCH, "max_cloud_cover": 20, "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert [s["id"] for s in body["scenes"]] == ["S2A_clear", "S2A_cloudy", "S2A_partial"]  # full coverage first
    first = body["scenes"][0]
    assert first["cloud_cover"] == 0.5 and first["crs"] == "EPSG:32643" and first["resolution_m"] == 10.0
    assert first["aoi_coverage"] == pytest.approx(1.0) and first["acquired"].startswith("2019-03-15")
    assert body["scenes"][2]["aoi_coverage"] == pytest.approx(0.5, abs=0.01)

    request = satellite.requests[0]
    assert request.method == "POST" and str(request.url) == "https://earth-search.aws.element84.com/v1/search"
    import json
    sent = json.loads(request.content)
    assert sent["collections"] == ["sentinel-2-l2a"] and sent["bbox"] == list(BBOX_T)
    assert sent["datetime"] == "2019-01-01T00:00:00Z/2019-03-31T23:59:59Z"
    assert sent["query"] == {"eo:cloud_cover": {"lte": 20.0}}
    assert "TerraVisionAI" in request.headers["user-agent"]


def test_search_with_no_matches_returns_empty_list(client):
    body = client.post("/api/satellite/search", json=SEARCH).json()
    assert body["count"] == 0 and body["scenes"] == []


@pytest.mark.parametrize("status", [500, 502, 503, 429])
def test_provider_http_failure_maps_to_502(client, satellite, status):
    satellite.handler = lambda request: httpx.Response(status, text="upstream broke")
    response = client.post("/api/satellite/search", json=SEARCH)
    assert response.status_code == 502 and error_code(response) == "SATELLITE_PROVIDER_ERROR"
    assert response.json()["error"]["details"]["provider"].startswith("Earth Search")
    assert "upstream broke" not in response.text  # upstream bodies are never echoed


def test_provider_timeout_maps_to_504(client, satellite):
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    satellite.handler = timeout
    response = client.post("/api/satellite/search", json=SEARCH)
    assert response.status_code == 504 and error_code(response) == "SATELLITE_PROVIDER_TIMEOUT"


def test_provider_network_failure_and_garbage_response(client, satellite):
    def refuse(request):
        raise httpx.ConnectError("no route", request=request)

    satellite.handler = refuse
    assert client.post("/api/satellite/search", json=SEARCH).status_code == 502
    satellite.handler = lambda request: httpx.Response(200, text="<html>not json</html>")
    assert error_code(client.post("/api/satellite/search", json=SEARCH)) == "SATELLITE_PROVIDER_ERROR"
    satellite.handler = lambda request: httpx.Response(200, json={"unexpected": True})
    assert error_code(client.post("/api/satellite/search", json=SEARCH)) == "SATELLITE_PROVIDER_ERROR"


def test_search_request_validation(client):
    def post(**overrides):
        return client.post("/api/satellite/search", json={**SEARCH, **overrides})

    assert post(end="2018-12-31").status_code == 422  # end before start
    assert post(start="2010-01-01").status_code == 422  # before Sentinel-2 era
    assert post(max_cloud_cover=140).status_code == 422
    assert post(bbox={**BBOX, "west": 500}).status_code == 422
    assert error_code(post(bbox={"west": 77, "south": 13, "east": 78, "north": 14})) == "AOI_TOO_LARGE"
    assert error_code(post(bbox={"west": 77.6, "south": 13.09, "east": 77.6004, "north": 13.14})) == "AOI_TOO_SMALL"
    assert error_code(post(provider="landsat")) == "UNKNOWN_PROVIDER"


# ------------------------------------------------------------------ retrieve
def test_retrieve_rejects_scene_with_partial_coverage_or_foreign_asset_host():
    def handler(request: httpx.Request) -> httpx.Response:
        if "partial" in str(request.url):
            return httpx.Response(200, json=stac_item("partial", 1.0, cover=False))
        item = stac_item("evil", 1.0)
        item["assets"]["visual"]["href"] = "https://attacker.example/steal.tif"
        return httpx.Response(200, json=item)

    provider = EarthSearchProvider("https://earth-search.aws.element84.com/v1", 5, httpx.MockTransport(handler))
    with pytest.raises(AppError) as exc:
        provider.retrieve("partial", BBOX_T)
    assert exc.value.code == "SCENE_PARTIAL_COVERAGE"
    with pytest.raises(AppError) as exc:
        provider.retrieve("evil", BBOX_T)
    assert exc.value.code == "SATELLITE_PROVIDER_ERROR" and "allowed host" in exc.value.message


def test_cdse_search_works_without_credentials_but_retrieve_requires_them(client, satellite):
    satellite.handler = lambda request: httpx.Response(200, json={"features": [stac_item("S2B_x", 2.0)]})
    body = client.post("/api/satellite/search", json={**SEARCH, "provider": "cdse"}).json()
    assert body["count"] == 1 and body["provider"]["configured"] is False
    assert str(satellite.requests[-1].url) == "https://stac.dataspace.copernicus.eu/v1/search"

    response = client.post("/api/satellite/retrieve", json={"provider": "cdse", "bbox": BBOX, "t1_scene_id": "a",
                                                           "t2_scene_id": "b"})
    assert response.status_code == 503 and error_code(response) == "SATELLITE_PROVIDER_NOT_CONFIGURED"


def test_cdse_token_and_process_request_follow_documented_api():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if url.endswith("/token"):
            return httpx.Response(200, json={"access_token": "tok-123", "expires_in": 600})
        if "/items/" in url:
            return httpx.Response(200, json=stac_item("S2B_x", 1.0, when="2025-02-06T05:25:21Z"))
        return httpx.Response(401)  # the process call is rejected in this test

    provider = CDSEProvider("https://stac.example/v1", "https://id.example/token", "https://sh.example/process",
                            "client", "secret", 5, httpx.MockTransport(handler))
    assert provider.configured()
    with pytest.raises(AppError) as exc:
        provider.retrieve("S2B_x", BBOX_T)
    assert exc.value.code == "SATELLITE_PROVIDER_AUTH"
    token_request = next(r for r in seen if str(r.url).endswith("/token"))
    assert b"grant_type=client_credentials" in token_request.content
    process = seen[-1]
    assert process.headers["authorization"] == "Bearer tok-123"
    import json
    body = json.loads(process.content)
    assert body["input"]["data"][0]["type"] == "sentinel-2-l2a"
    assert body["input"]["bounds"]["bbox"] == list(BBOX_T)
    assert body["output"]["responses"][0]["format"]["type"] == "image/tiff"
    assert body["input"]["data"][0]["dataFilter"]["timeRange"]["from"].startswith("2025-02-06T04:55")


def test_retrieve_validation(client):
    def post(**overrides):
        return client.post("/api/satellite/retrieve", json={"provider": "earth_search", "bbox": BBOX,
                                                            "t1_scene_id": "a", "t2_scene_id": "b", **overrides})

    assert error_code(post(t2_scene_id="a")) == "SAME_SCENE"
    assert post(t1_scene_id="../../etc/passwd").status_code == 422
    assert error_code(post(model="swin_siamese")) == "MODEL_UNAVAILABLE"
    assert error_code(post(provider="nope")) == "UNKNOWN_PROVIDER"


def test_retrieve_runs_a_georeferenced_analysis_and_orders_scenes_by_date(client, monkeypatch):
    """Map workflow end-to-end with the remote imagery replaced by synthetic georeferenced rasters."""
    base = scene(120, 140, seed=7)
    changed, _ = with_building(base)
    geo = GeoInfo(CRS.from_epsg(32643), from_origin(780000.0, 1450000.0, 10.0, 10.0), 140, 120)

    def fake_retrieve(scene_id: str, bbox):
        later = scene_id == "S2B_2025"
        return RetrievedScene(
            LoadedImage(changed if later else base, "tiff", geo, []),
            {"scene_id": scene_id, "acquired": "2025-02-06T05:25:21Z" if later else "2019-03-15T05:27:57Z",
             "cloud_cover": 0.1, "attribution": "Contains modified Copernicus Sentinel data"},
        )

    provider = client.app.state.services.providers.get("earth_search")
    monkeypatch.setattr(provider, "retrieve", fake_retrieve)
    # scenes deliberately passed in the wrong temporal order
    response = client.post("/api/satellite/retrieve", json={"provider": "earth_search", "bbox": BBOX,
                                                           "t1_scene_id": "S2B_2025", "t2_scene_id": "S2A_2019"})
    assert response.status_code == 202
    created = response.json()
    assert created["source"] == "satellite" and created["georeferenced"] is True
    assert created["inputs"]["t1_scene"]["scene_id"] == "S2A_2019" and created["inputs"]["t2_date"] == "2025-02-06"
    # no Sentinel-2 checkpoint in the fixture -> LEVIR-CD fallback, explicitly flagged as out-of-distribution
    assert created["model_key"] == "fc_siam_diff"
    assert any("out-of-distribution" in w for w in created["inputs"]["warnings"])
    assert wait_for(client, created["id"])["status"] == "completed"
    results = client.get(f"/api/analyses/{created['id']}/results").json()
    assert results["results"]["geospatial"]["crs"] == "EPSG:32643"
    assert results["results"]["metrics"] is None
    assert "mask_geotiff" in results["downloads"] and "map_overlay" in results["downloads"]


def test_provider_failure_during_retrieve_creates_no_analysis(client, satellite):
    satellite.handler = lambda request: httpx.Response(503)
    response = client.post("/api/satellite/retrieve", json={"provider": "earth_search", "bbox": BBOX,
                                                           "t1_scene_id": "a", "t2_scene_id": "b"})
    assert response.status_code == 502
    assert client.get("/api/analyses").json()["total"] == 0


# ------------------------------------------------------------------- geocode
def test_geocode_parses_results_and_handles_failure(client, satellite):
    satellite.handler = lambda request: httpx.Response(200, json=[
        {"display_name": "Yelahanka, Bengaluru", "lat": "13.1007", "lon": "77.5963",
         "boundingbox": ["13.06", "13.14", "77.55", "77.64"]},
        {"display_name": "broken row"},
    ])
    body = client.get("/api/satellite/geocode", params={"q": "Yelahanka"}).json()
    assert body["results"] == [{"name": "Yelahanka, Bengaluru", "lat": 13.1007, "lon": 77.5963,
                                "bbox": [77.55, 13.06, 77.64, 13.14]}]
    assert "OpenStreetMap" in body["attribution"]
    satellite.handler = lambda request: httpx.Response(500)
    response = client.get("/api/satellite/geocode", params={"q": "Yelahanka"})
    assert response.status_code == 502 and error_code(response) == "GEOCODER_UNAVAILABLE"
    assert client.get("/api/satellite/geocode", params={"q": "a"}).status_code == 422


def test_search_query_dataclass_defaults():
    query = SearchQuery(BBOX_T, date(2019, 1, 1), date(2019, 2, 1))
    assert query.max_cloud_cover == 30.0 and query.limit == 10
