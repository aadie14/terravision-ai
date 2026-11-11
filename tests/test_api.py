"""API tests: endpoints, validation, the full upload -> inference -> results -> download flow,
persistence and failure handling. Uses a real app instance with temporary storage (see conftest)."""

from __future__ import annotations

import io
import json

import httpx
import numpy as np
import rasterio
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.db import Analysis
from backend.app.main import create_app
from tests.conftest import (
    geotiff_bytes,
    jpeg_bytes,
    png_bytes,
    scene,
    upload,
    wait_for,
    with_building,
    write_tiny_checkpoint,
)


def error_code(response) -> str:
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "details"}
    return body["error"]["code"]


def pair(size=(96, 96)):
    t1 = scene(*size, seed=1)
    t2, mask = with_building(t1)
    return t1, t2, mask


# ------------------------------------------------------------- read endpoints
def test_health_reports_ok_with_database_and_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] is True and body["models_available"] == 1


def test_system_reports_cpu_device_and_storage(client):
    body = client.get("/api/system").json()
    assert body["compute"]["device"] == "cpu" and body["compute"]["amp"] is False
    assert body["database"]["ok"] is True and body["samples"] == 2
    assert {m["key"] for m in body["models"]} == {"swin_siamese", "fc_siam_diff", "swin_siamese_s2", "fc_siam_diff_s2"}


def test_models_lists_availability_and_reason_for_missing_checkpoint(client):
    body = client.get("/api/models").json()
    by_key = {m["key"]: m for m in body["models"]}
    assert by_key["fc_siam_diff"]["available"] is True
    assert by_key["fc_siam_diff"]["checkpoint"]["run_name"] == "pytest-tiny"
    assert by_key["swin_siamese"]["available"] is False
    assert "not found" in by_key["swin_siamese"]["unavailable_reason"]
    assert by_key["swin_siamese"]["benchmark"] is None


def test_config_exposes_limits_but_no_paths_or_secrets(client):
    body = client.get("/api/config").json()
    assert body["upload"]["max_file_mb"] == 2
    assert ".tif" in body["upload"]["allowed_extensions"]
    text = json.dumps(body).lower()
    assert "secret" not in text and "storage" not in text and ":\\" not in text


def test_unknown_route_uses_error_envelope(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404 and error_code(response) == "NOT_FOUND"


def test_security_and_request_id_headers(client):
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert len(response.headers["x-request-id"]) == 12


def test_cors_allows_only_configured_origin(client):
    ok = client.options("/api/models", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/api/models", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers


# ----------------------------------------------------- full upload workflow
def test_upload_inference_results_download_flow(client):
    t1, t2, mask = pair()
    response = upload(client, png_bytes(t1), png_bytes(t2), gt=png_bytes(mask.astype(np.uint8) * 255),
                      data={"threshold": "0.4", "name": "flow test"})
    assert response.status_code == 202
    created = response.json()
    assert created["status"] in {"queued", "running", "completed"} and len(created["id"]) == 32
    assert created["has_ground_truth"] is True and created["georeferenced"] is False

    done = wait_for(client, created["id"])
    assert done["status"] == "completed", done
    assert done["inference_seconds"] > 0 and 0 <= done["changed_fraction"] <= 1

    results = client.get(f"/api/analyses/{created['id']}/results").json()
    r = results["results"]
    assert r["model"]["key"] == "fc_siam_diff" and r["model"]["checkpoint"]["version"].startswith("pytest-tiny@epoch1#")
    assert r["threshold"] == 0.4 and r["inference"]["device"] == "cpu"
    assert r["image"] == {"width": 96, "height": 96}
    assert r["statistics"]["total_pixels"] == 96 * 96
    assert r["metrics"] is not None
    m = r["metrics"]
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 96 * 96
    assert m["tp"] + m["fn"] == int(mask.sum())
    assert r["geospatial"] is None and r["georeferenced"] is False

    for artifact, kind in [("t1", "PNG"), ("t2", "PNG"), ("mask", "PNG"), ("overlay", "PNG"), ("probability", "PNG"),
                           ("confidence", "PNG"), ("ground_truth", "PNG"), ("error_map", "PNG")]:
        download = client.get(results["downloads"][artifact])
        assert download.status_code == 200, artifact
        img = Image.open(io.BytesIO(download.content))
        assert img.format == kind and img.size == (96, 96)
    mask_png = np.asarray(Image.open(io.BytesIO(client.get(results["downloads"]["mask"]).content)))
    assert set(np.unique(mask_png)) <= {0, 255}
    assert (mask_png > 0).sum() == r["statistics"]["changed_pixels"]

    as_json = client.get(results["downloads"]["results"]).json()
    assert as_json["analysis_id"] == created["id"]
    attachment = client.get(results["downloads"]["mask"] + "?attachment=true")
    assert "attachment" in attachment.headers["content-disposition"]
    assert f"terravision_{created['id'][:8]}_mask.png" in attachment.headers["content-disposition"]


def test_results_without_ground_truth_have_no_accuracy_metrics(client):
    t1, t2, _ = pair()
    created = upload(client, png_bytes(t1), jpeg_bytes(t2), names=("a.png", "b.jpg", "")).json()
    assert wait_for(client, created["id"])["status"] == "completed"
    results = client.get(f"/api/analyses/{created['id']}/results").json()
    assert results["results"]["metrics"] is None and results["results"]["has_ground_truth"] is False
    assert "ground_truth" not in results["downloads"] and "error_map" not in results["downloads"]
    response = client.get(f"/api/analyses/{created['id']}/downloads/error_map")
    assert response.status_code == 409 and error_code(response) == "NO_GROUND_TRUTH"


def test_non_georeferenced_analysis_refuses_geotiff_export(client):
    t1, t2, _ = pair()
    created = upload(client, png_bytes(t1), png_bytes(t2)).json()
    wait_for(client, created["id"])
    response = client.get(f"/api/analyses/{created['id']}/downloads/mask_geotiff")
    assert response.status_code == 409 and error_code(response) == "NOT_GEOREFERENCED"


def test_no_change_pair_completes_with_identical_warning_and_undefined_metrics(client):
    t1 = scene(96, 96, seed=5)
    empty = np.zeros((96, 96), np.uint8)
    created = upload(client, png_bytes(t1), png_bytes(t1), gt=png_bytes(empty)).json()
    done = wait_for(client, created["id"])
    assert done["status"] == "completed"
    assert any("pixel-identical" in w for w in done["inputs"]["warnings"])
    metrics = client.get(f"/api/analyses/{created['id']}/results").json()["results"]["metrics"]
    assert metrics["tp"] == 0 and metrics["fn"] == 0
    assert metrics["recall"] is None and "recall" in metrics["undefined"]


def test_rethreshold_updates_mask_statistics_and_metrics_without_new_inference(client):
    t1, t2, mask = pair()
    created = upload(client, png_bytes(t1), png_bytes(t2), gt=png_bytes(mask.astype(np.uint8) * 255)).json()
    wait_for(client, created["id"])
    before = client.get(f"/api/analyses/{created['id']}/results").json()["results"]
    low = client.post(f"/api/analyses/{created['id']}/threshold", json={"threshold": 0.05}).json()["results"]
    high = client.post(f"/api/analyses/{created['id']}/threshold", json={"threshold": 0.95}).json()["results"]
    assert low["threshold"] == 0.05 and high["threshold"] == 0.95
    assert low["statistics"]["changed_pixels"] >= before["statistics"]["changed_pixels"] >= high["statistics"]["changed_pixels"]
    assert low["inference"]["seconds"] == before["inference"]["seconds"]  # inference was not repeated
    assert client.get(f"/api/analyses/{created['id']}").json()["threshold"] == 0.95
    for bad in (0, 1, -0.2, 1.5):
        response = client.post(f"/api/analyses/{created['id']}/threshold", json={"threshold": bad})
        assert response.status_code == 422 and error_code(response) == "VALIDATION_ERROR"


# ---------------------------------------------------- GeoTIFF end-to-end flow
def test_geotiff_upload_prediction_and_georeferenced_exports(client):
    t1, t2, _ = pair((120, 150))
    created = upload(client, geotiff_bytes(t1), geotiff_bytes(t2), names=("t1.tif", "t2.tif", "")).json()
    assert created["georeferenced"] is True
    assert wait_for(client, created["id"])["status"] == "completed"
    results = client.get(f"/api/analyses/{created['id']}/results").json()
    g = results["results"]["geospatial"]
    assert g["crs"] == "EPSG:32643" and g["resolution"] == [10.0, 10.0]
    assert g["footprint_area_km2"] > 0 and "geodesic" in g["area_method"]
    assert g["map_overlay_bounds"]["south"] < g["map_overlay_bounds"]["north"]

    tif = client.get(results["downloads"]["mask_geotiff"])
    assert tif.status_code == 200
    with rasterio.MemoryFile(tif.content) as mem, mem.open() as src:
        assert src.crs.to_epsg() == 32643 and (src.width, src.height) == (150, 120)
        assert src.transform.a == 10.0 and src.transform.c == 780000.0 and src.transform.f == 1450000.0
        exported = src.read(1)
        assert set(np.unique(exported)) <= {0, 1}
        assert int(exported.sum()) == results["results"]["statistics"]["changed_pixels"]
        assert src.tags()["TERRAVISION_MODEL"] == "fc_siam_diff"

    geojson = client.get(results["downloads"]["regions_geojson"]).json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == g["region_count"]
    overlay = Image.open(io.BytesIO(client.get(results["downloads"]["map_overlay"]).content))
    assert overlay.mode == "RGBA"
    assert client.get(results["downloads"]["map_t2"]).status_code == 200


def test_geotiff_pair_on_different_grids_is_aligned_with_warning(client):
    t1 = scene(100, 100, seed=1)
    t2 = scene(220, 220, seed=2)
    created = upload(client, geotiff_bytes(t1), geotiff_bytes(t2, origin=(779900.0, 1450100.0), pixel=5.0),
                     names=("t1.tif", "t2.tif", "")).json()
    assert any("resampled onto the T1 grid" in w for w in created["inputs"]["warnings"])
    assert (created["width"], created["height"]) == (100, 100)
    assert wait_for(client, created["id"])["status"] == "completed"


def test_geotiff_pair_covering_different_places_is_rejected(client):
    response = upload(client, geotiff_bytes(scene(64, 64)), geotiff_bytes(scene(64, 64), origin=(500000.0, 1000000.0)),
                      names=("t1.tif", "t2.tif", ""))
    assert response.status_code == 422 and error_code(response) == "GEOSPATIAL_MISALIGNMENT"


def test_geotiff_with_transform_but_no_crs_is_rejected(client):
    bad = geotiff_bytes(scene(64, 64), crs=None, with_transform=True)
    response = upload(client, bad, bad, names=("t1.tif", "t2.tif", ""))
    assert response.status_code == 422 and error_code(response) == "INVALID_CRS"


# ----------------------------------------------- negative and boundary cases
def test_missing_image_rejected(client):
    response = client.post("/api/analyses/upload", files={"t1": ("t1.png", png_bytes(scene()))})
    assert response.status_code == 422 and error_code(response) == "MISSING_IMAGE"
    assert response.json()["error"]["details"]["missing"] == ["t2"]
    response = client.post("/api/analyses/upload", data={"model": "fc_siam_diff"})
    assert response.status_code == 422 and error_code(response) == "MISSING_IMAGE"


def test_corrupted_image_rejected(client):
    good = png_bytes(scene())
    response = upload(client, good, good[:200])
    assert response.status_code == 422 and error_code(response) == "CORRUPTED_IMAGE"


def test_unsupported_file_type_rejected(client):
    response = upload(client, png_bytes(scene()), b"%PDF-1.7 not an image", names=("t1.png", "t2.pdf", ""))
    assert response.status_code == 415 and error_code(response) == "UNSUPPORTED_FILE_TYPE"


def test_mismatched_dimensions_rejected(client):
    response = upload(client, png_bytes(scene(96, 96)), png_bytes(scene(96, 128)))
    assert response.status_code == 422 and error_code(response) == "IMAGE_DIMENSION_MISMATCH"


def test_mask_with_wrong_dimensions_rejected(client):
    t1, t2, _ = pair()
    response = upload(client, png_bytes(t1), png_bytes(t2), gt=png_bytes(np.zeros((50, 50), np.uint8)))
    assert response.status_code == 422 and error_code(response) == "MASK_DIMENSION_MISMATCH"


def test_oversized_upload_rejected(client):
    rng = np.random.default_rng(0)
    big = png_bytes(rng.integers(0, 255, (1000, 1000, 3), dtype=np.uint8))  # ~3 MB of incompressible noise
    assert len(big) > 2 * 1024 * 1024
    response = upload(client, big, big)
    assert response.status_code == 413 and error_code(response) == "FILE_TOO_LARGE"


def test_image_side_boundaries(client):
    too_small = png_bytes(np.zeros((16, 64, 3), np.uint8))
    response = upload(client, too_small, too_small)
    assert response.status_code == 422 and error_code(response) == "IMAGE_TOO_SMALL"
    too_big = png_bytes(np.zeros((1025, 40, 3), np.uint8))
    response = upload(client, too_big, too_big)
    assert response.status_code == 413 and error_code(response) == "IMAGE_TOO_LARGE"
    smallest = png_bytes(scene(32, 32))
    assert upload(client, smallest, smallest).status_code == 202


def test_unknown_model_rejected(client):
    t1, t2, _ = pair()
    response = upload(client, png_bytes(t1), png_bytes(t2), data={"model": "resnet50"})
    assert response.status_code == 404 and error_code(response) == "UNKNOWN_MODEL"


def test_missing_checkpoint_gives_model_unavailable_and_stores_nothing(client, settings):
    t1, t2, _ = pair()
    response = upload(client, png_bytes(t1), png_bytes(t2), data={"model": "swin_siamese"})
    assert response.status_code == 503 and error_code(response) == "MODEL_UNAVAILABLE"
    assert client.get("/api/analyses").json()["total"] == 0
    assert list(settings.analyses_path.iterdir()) == []


def test_invalid_threshold_rejected(client):
    t1, t2, _ = pair()
    for value in ("0", "1", "1.7", "abc"):
        response = upload(client, png_bytes(t1), png_bytes(t2), data={"threshold": value})
        assert response.status_code == 422 and error_code(response) == "VALIDATION_ERROR"


def test_unknown_and_malformed_analysis_ids(client):
    for bad in ("0" * 32, "..%2F..%2Fsecret", "not-an-id", "A" * 32):
        response = client.get(f"/api/analyses/{bad}")
        assert response.status_code == 404 and error_code(response) == "NOT_FOUND"
    response = client.get(f"/api/analyses/{'0' * 32}/downloads/mask")
    assert response.status_code == 404


def test_unknown_artifact_rejected(client):
    t1, t2, _ = pair()
    created = upload(client, png_bytes(t1), png_bytes(t2)).json()
    wait_for(client, created["id"])
    for name in ("geo.json", "..%2Fterravision.sqlite3", "passwd"):
        response = client.get(f"/api/analyses/{created['id']}/downloads/{name}")
        assert response.status_code == 404


def test_results_of_failed_analysis_return_error(client, monkeypatch):
    services = client.app.state.services

    def boom(*args, **kwargs):
        raise RuntimeError("simulated inference crash")

    monkeypatch.setattr(services.models, "predict", boom)
    t1, t2, _ = pair()
    created = upload(client, png_bytes(t1), png_bytes(t2)).json()
    done = wait_for(client, created["id"])
    assert done["status"] == "failed" and done["error_code"] == "INFERENCE_FAILED"
    assert "simulated inference crash" in done["error_message"]
    response = client.get(f"/api/analyses/{created['id']}/results")
    assert response.status_code == 409 and error_code(response) == "ANALYSIS_FAILED"
    stats = client.get("/api/analyses/stats").json()
    assert stats["failed"] == 1 and stats["completed"] == 0


def test_requested_gpu_unavailable_falls_back_to_cpu(tmp_path, checkpoint_dir, samples_dir, monkeypatch, settings):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gpu_settings = settings.model_copy(update={"device": "cuda", "storage_dir": tmp_path / "storage_gpu"})
    with TestClient(create_app(gpu_settings, log_to_file=False)) as gpu_client:
        system = gpu_client.get("/api/system").json()
        assert system["compute"]["requested"] == "cuda" and system["compute"]["device"] == "cpu"
        t1, t2, _ = pair()
        created = upload(gpu_client, png_bytes(t1), png_bytes(t2)).json()
        assert wait_for(gpu_client, created["id"])["status"] == "completed"


# ------------------------------------------------- samples, re-run, evaluation
def test_samples_are_listed_with_provenance(client):
    samples = client.get("/api/samples").json()["samples"]
    assert [s["id"] for s in samples] == ["syn-a", "syn-b"]
    assert samples[0]["has_ground_truth"] is True and samples[1]["has_ground_truth"] is False
    assert client.get("/api/samples/syn-a/t1").status_code == 200
    assert client.get("/api/samples/syn-b/ground_truth").status_code == 404
    assert client.get("/api/samples/nope/t1").status_code == 404
    assert client.get("/api/samples/syn-a/secret").status_code == 404


def test_sample_analysis_and_rerun(client):
    response = client.post("/api/analyses", json={"sample_id": "syn-a"})
    assert response.status_code == 202
    first = wait_for(client, response.json()["id"])
    assert first["status"] == "completed" and first["source"] == "sample" and first["has_ground_truth"] is True

    # syn-b is Sentinel-2 imagery; no Sentinel-2 checkpoint exists in the fixture, so the server falls back to the
    # LEVIR-CD model and must flag the domain mismatch instead of silently presenting the result.
    out_of_domain = client.post("/api/analyses", json={"sample_id": "syn-b"}).json()
    assert out_of_domain["model_key"] == "fc_siam_diff"
    assert any("out-of-distribution" in w and "Sentinel-2" in w for w in out_of_domain["inputs"]["warnings"])
    assert not any("out-of-distribution" in w for w in first["inputs"]["warnings"])  # matching domain: no warning

    rerun = client.post("/api/analyses", json={"rerun_of": first["id"], "threshold": 0.3})
    assert rerun.status_code == 202
    second = wait_for(client, rerun.json()["id"])
    assert second["status"] == "completed" and second["source"] == "rerun"
    assert second["inputs"]["rerun_of"] == first["id"] and second["id"] != first["id"]


def test_create_analysis_validation(client):
    assert client.post("/api/analyses", json={}).status_code == 422
    assert client.post("/api/analyses", json={"sample_id": "syn-a", "rerun_of": "0" * 32}).status_code == 422
    assert error_code(client.post("/api/analyses", json={"sample_id": "missing"})) == "NOT_FOUND"
    assert error_code(client.post("/api/analyses", json={"rerun_of": "0" * 32})) == "NOT_FOUND"
    response = client.post("/api/analyses", json={"sample_id": "syn-a", "model": "swin_siamese"})
    assert response.status_code == 503 and error_code(response) == "MODEL_UNAVAILABLE"


def test_evaluation_on_labelled_samples_and_summary(client):
    summary = client.get("/api/evaluation/summary").json()
    assert summary["benchmark_available"] is False and summary["benchmark"] is None and summary["sample_runs"] == []

    run = client.post("/api/evaluation", json={"threshold": 0.5})
    assert run.status_code == 200
    body = run.json()
    assert body["samples"] == ["syn-a"]  # only the labelled sample is used
    model = body["models"][0]
    assert model["model"] == "fc_siam_diff"
    assert model["tp"] + model["fp"] + model["tn"] + model["fn"] == 128 * 128
    assert "NOT the full test-split benchmark" in body["scope"]

    summary = client.get("/api/evaluation/summary").json()
    assert len(summary["sample_runs"]) == 1 and summary["sample_runs"][0]["id"] == body["id"]


def test_evaluation_rejects_unlabelled_selection_and_missing_models(client):
    response = client.post("/api/evaluation", json={"sample_ids": ["syn-b"]})
    assert response.status_code == 422 and error_code(response) == "NO_GROUND_TRUTH"
    response = client.post("/api/evaluation", json={"models": ["swin_siamese"]})
    assert response.status_code == 503 and error_code(response) == "MODEL_UNAVAILABLE"
    response = client.post("/api/evaluation", json={"models": ["nope"]})
    assert response.status_code == 404


def test_benchmark_summary_is_served_when_report_exists(client, settings):
    settings.reports_path.mkdir(parents=True)
    report = {"generated_utc": "2026-01-01T00:00:00+00:00", "dataset": "LEVIR-CD", "split": "test", "note": "x",
              "models": [{"model": "fc_siam_diff", "f1": 0.5}]}
    (settings.reports_path / "summary.json").write_text(json.dumps(report))
    summary = client.get("/api/evaluation/summary").json()
    assert summary["benchmark_available"] is True and summary["benchmark"]["models"][0]["f1"] == 0.5
    assert client.get("/api/evaluation/reports/model_comparison").status_code == 404
    assert client.get("/api/evaluation/reports/..%2Fsummary.json").status_code == 404


# ------------------------------------------------- history, stats, persistence
def test_history_pagination_filter_stats_and_delete(client, settings):
    t1, t2, _ = pair()
    ids = []
    for _ in range(3):
        ids.append(upload(client, png_bytes(t1), png_bytes(t2)).json()["id"])
    for analysis_id in ids:
        assert wait_for(client, analysis_id)["status"] == "completed"

    listing = client.get("/api/analyses").json()
    assert listing["total"] == 3 and [i["id"] for i in listing["items"]] == ids[::-1]  # newest first
    page = client.get("/api/analyses?limit=2&offset=2").json()
    assert len(page["items"]) == 1 and page["total"] == 3
    assert client.get("/api/analyses?status=failed").json()["total"] == 0
    assert client.get("/api/analyses?status=bogus").status_code == 422
    assert client.get("/api/analyses?limit=0").status_code == 422

    stats = client.get("/api/analyses/stats").json()
    assert stats["total"] == 3 and stats["completed"] == 3 and stats["failed"] == 0
    assert stats["by_model"] == {"fc_siam_diff": 3} and stats["mean_changed_percent"] is not None

    folder = settings.analyses_path / ids[0]
    assert folder.is_dir()
    assert client.delete(f"/api/analyses/{ids[0]}").status_code == 204
    assert not folder.exists()  # files are cleaned up with the record
    assert client.get(f"/api/analyses/{ids[0]}").status_code == 404
    assert client.delete(f"/api/analyses/{ids[0]}").status_code == 404
    assert client.get("/api/analyses").json()["total"] == 2


def test_analyses_persist_across_application_restarts(settings):
    t1, t2, _ = pair()
    with TestClient(create_app(settings, log_to_file=False)) as first:
        analysis_id = upload(first, png_bytes(t1), png_bytes(t2)).json()["id"]
        assert wait_for(first, analysis_id)["status"] == "completed"
    with TestClient(create_app(settings, log_to_file=False)) as second:
        assert second.get(f"/api/analyses/{analysis_id}").json()["status"] == "completed"
        assert second.get(f"/api/analyses/{analysis_id}/results").status_code == 200
        assert second.get(f"/api/analyses/{analysis_id}/downloads/mask").status_code == 200


def test_interrupted_jobs_are_failed_and_orphan_folders_removed_on_startup(settings):
    with TestClient(create_app(settings, log_to_file=False)) as first:
        services = first.app.state.services
        with services.db.session() as session:
            session.add(Analysis(id="a" * 32, name="stuck", status="running", source="upload",
                                 model_key="fc_siam_diff", threshold=0.5, inputs={}))
            session.commit()
        (settings.analyses_path / ("a" * 32)).mkdir()
        (settings.analyses_path / ("b" * 32)).mkdir()  # orphan: no database row
    with TestClient(create_app(settings, log_to_file=False)) as second:
        body = second.get(f"/api/analyses/{'a' * 32}").json()
        assert body["status"] == "failed" and body["error_code"] == "INTERRUPTED"
        assert not (settings.analyses_path / ("b" * 32)).exists()
        assert (settings.analyses_path / ("a" * 32)).exists()


def test_model_is_loaded_once_and_reused(client, monkeypatch):
    import backend.app.services.model_manager as mm

    calls = []
    original = mm.load_model

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(mm, "load_model", counting)
    t1, t2, _ = pair()
    for _ in range(3):
        wait_for(client, upload(client, png_bytes(t1), png_bytes(t2)).json()["id"])
    assert len(calls) == 1
    by_key = {m["key"]: m for m in client.get("/api/models").json()["models"]}
    assert by_key["fc_siam_diff"]["loaded"] is True and by_key["swin_siamese"]["loaded"] is False


def test_swin_checkpoint_becomes_available_when_file_appears(client, settings):
    assert client.get("/health").json()["models_available"] == 1
    write_tiny_checkpoint(settings.checkpoint_path / "swin_siamese_levir.pt", key="fc_siam_diff")
    # a checkpoint holding the wrong architecture is readable but must fail at load time, not crash
    t1, t2, _ = pair()
    created = upload(client, png_bytes(t1), png_bytes(t2), data={"model": "swin_siamese"}).json()
    done = wait_for(client, created["id"])
    assert done["status"] == "failed" and done["error_code"] == "MODEL_UNAVAILABLE"


def test_provider_transport_is_not_used_by_core_workflow(client, satellite):
    t1, t2, _ = pair()
    wait_for(client, upload(client, png_bytes(t1), png_bytes(t2)).json()["id"])
    assert satellite.requests == []  # the core demo works fully offline
    assert isinstance(httpx.MockTransport(satellite), httpx.BaseTransport)


def test_sentinel2_model_is_chosen_automatically_for_sentinel2_inputs(client, settings):
    """With a Sentinel-2 checkpoint present, 10 m GeoTIFFs and Sentinel-2 samples default to it - no warning."""
    write_tiny_checkpoint(settings.checkpoint_path / "fc_siam_diff_oscd.pt", key="fc_siam_diff_s2")
    t1, t2, _ = pair((120, 150))
    created = upload(client, geotiff_bytes(t1), geotiff_bytes(t2), names=("t1.tif", "t2.tif", "")).json()
    assert created["model_key"] == "fc_siam_diff_s2"  # 10 m UTM pixels -> Sentinel-2 domain
    assert not any("out-of-distribution" in w for w in created["inputs"]["warnings"])
    assert wait_for(client, created["id"])["status"] == "completed"

    sample = client.post("/api/analyses", json={"sample_id": "syn-b"}).json()
    assert sample["model_key"] == "fc_siam_diff_s2"
    # an explicit LEVIR-CD choice on 10 m data is honoured but flagged
    forced = upload(client, geotiff_bytes(t1), geotiff_bytes(t2), names=("t1.tif", "t2.tif", ""),
                    data={"model": "fc_siam_diff"}).json()
    assert forced["model_key"] == "fc_siam_diff"
    assert any("out-of-distribution" in w for w in forced["inputs"]["warnings"])
    models = {m["key"]: m for m in client.get("/api/models").json()["models"]}
    assert models["fc_siam_diff_s2"]["available"] is True and models["fc_siam_diff_s2"]["domain"] == "sentinel2"


def test_replaced_checkpoint_is_reloaded_instead_of_serving_stale_weights(client, settings):
    """BUG-004: a backend started during training kept serving the checkpoint it loaded first."""
    import os

    t1, t2, _ = pair()
    first = wait_for(client, upload(client, png_bytes(t1), png_bytes(t2)).json()["id"])
    v1 = client.get(f"/api/analyses/{first['id']}/results").json()["results"]["model"]["checkpoint"]

    path = settings.checkpoint_path / "fc_siam_diff_levir.pt"
    torch.manual_seed(123)
    from ml.checkpoint import save_checkpoint
    from ml.models.registry import build_model
    from tests.conftest import TINY_KWARGS

    save_checkpoint(path, "fc_siam_diff", build_model("fc_siam_diff", **TINY_KWARGS), dict(TINY_KWARGS),
                    meta={"run_name": "pytest-tiny", "epoch": 2, "dataset": "synthetic", "threshold": 0.5})
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 5))

    second = wait_for(client, upload(client, png_bytes(t1), png_bytes(t2)).json()["id"])
    v2 = client.get(f"/api/analyses/{second['id']}/results").json()["results"]["model"]["checkpoint"]
    assert v1["epoch"] == 1 and v2["epoch"] == 2
    assert v1["sha256"] != v2["sha256"]


def test_checkpoint_threshold_is_the_default_and_explicit_threshold_wins(client, settings):
    """Checkpoints carry a validation-selected threshold (ml.calibrate); it is used unless the user sets one."""
    from ml.checkpoint import read_checkpoint

    path = settings.checkpoint_path / "fc_siam_diff_levir.pt"
    payload = read_checkpoint(path)
    payload["meta"]["threshold"] = 0.3
    torch.save(payload, path)
    t1, t2, _ = pair()
    assert upload(client, png_bytes(t1), png_bytes(t2)).json()["threshold"] == 0.3
    assert upload(client, png_bytes(t1), png_bytes(t2), data={"threshold": "0.7"}).json()["threshold"] == 0.7
    assert client.get("/api/models").json()["models"][2]["checkpoint"]["threshold"] == 0.3
