"""Single source of truth for the academic documents (requirements, test cases, bugs, plan, viva).

Numbers that come from experiments or test runs are NOT written here: ``build_docs.py`` reads them
from ``reports/evaluation``, ``experiments`` and ``evidence/tests`` at build time.
"""

from __future__ import annotations

PROJECT = {
    "title": "Vision Transformers for Satellite Land-Use Monitoring",
    "app": "TerraVision AI - Satellite Change Intelligence",
    "build_date": "11 November 2026",
}


PROBLEM_STATEMENT = (
    "Urban planners, environmental agencies and disaster-response teams need to know where land use has "
    "changed between two satellite acquisitions, but comparing image pairs by eye does not scale and classical "
    "pixel differencing is swamped by illumination, seasonal and registration differences. Convolutional "
    "change-detection networks improved on this, yet their local receptive fields limit the long-range context "
    "needed to separate real structural change from such pseudo-change. This project designs, implements and "
    "evaluates a Siamese Swin Transformer that predicts pixel-level building/urban change from bitemporal "
    "imagery, benchmarks it against a CNN baseline (FC-Siam-diff) on the same held-out data, and delivers it as "
    "a working application (TerraVision AI) so an analyst can upload or fetch an image pair and obtain a "
    "verifiable change map."
)

OBJECTIVES = [
    "Implement a Siamese Swin Transformer change-detection network with a trained decoder (not just a "
    "pre-trained backbone) that fits an 8 GB consumer GPU.",
    "Implement and train a CNN baseline (FC-Siam-diff) and evaluate both models on the identical held-out "
    "LEVIR-CD test split with the same metrics and threshold.",
    "Make the models work on real Sentinel-2 imagery: fine-tune both on OSCD (10 m) and measure the domain shift "
    "between 0.5 m aerial and 10 m satellite imagery on held-out test cities.",
    "Build a reproducible data pipeline (verified download, leakage-free tiling, split manifest) and a "
    "configurable training / evaluation pipeline that logs every experiment in machine-readable form.",
    "Deliver an end-to-end application: validated upload or satellite retrieval, background inference, "
    "interactive visualisation, ground-truth metrics, georeferenced export and persistent history.",
    "Verify the system with automated tests that include boundary and negative cases, and document bugs with "
    "root causes.",
]

GAPS = [
    ("G1 - Hierarchical transformers remain heavy for commodity hardware",
     "Pure-transformer designs such as SwinSUNet (Zhang et al., 2022) and ChangeFormer (Bandara & Patel, 2022) "
     "report strong accuracy but use transformer decoders, long schedules and large-memory GPUs; BIT (Chen et "
     "al., 2022) is light but keeps a CNN backbone and a single-scale transformer. There is little evidence on "
     "how far a hierarchical Swin encoder can go when it must train and run inside 8 GB of VRAM.",
     "ADDRESSED in Phase 1: Swin-V2-Tiny encoder with a convolutional FPN decoder, mixed precision and "
     "sliding-window inference, trained and benchmarked on an RTX 5050 (8 GB)."),
    ("G2 - Benchmarks are narrow in geography, sensor and change type",
     "LEVIR-CD (Chen & Shi, 2020) covers building change in Texas at 0.5 m; OSCD (Daudt et al., 2018) covers "
     "urban change at 10-60 m with only 24 city pairs. Models are rarely evaluated across sensors, resolutions or "
     "regions, so reported scores say little about transfer to, for example, Sentinel-2 over Indian cities.",
     "PARTLY ADDRESSED in Phase 1: the domain gap is measured (LEVIR-CD-only models score F1 of about 0-1% on the "
     "OSCD Sentinel-2 test cities) and closed in part by fine-tuning on OSCD RGB with radiometry matched to the "
     "application's imagery source. Multispectral input, more regions and multi-class change remain for Phase 2."),
    ("G3 - Research models are not delivered as verifiable, geospatially correct tools",
     "The surveyed papers publish architectures and scores on pre-cut patches. They do not address what an "
     "analyst needs in practice: validation of arbitrary input pairs, CRS-aware alignment, inference on full "
     "rasters, georeferenced outputs, honest reporting when ground truth is absent, and traceability of which "
     "checkpoint produced a result.",
     "ADDRESSED in Phase 1: TerraVision AI wraps both models in a tested application with input validation, "
     "reprojection, GeoTIFF/GeoJSON export, geodesic area and checkpoint versioning."),
]

# id, requirement, type, priority, design element, code, verification
REQUIREMENTS = [
    ("R1", "Accept a bitemporal image pair (PNG, JPEG, GeoTIFF) and an optional ground-truth mask; validate type, "
           "integrity, size, channels and pair compatibility before any processing.", "Functional", "High",
     "imaging.load_image / align_pair; POST /api/analyses/upload; FileDrop component",
     "backend/app/services/imaging.py, backend/app/routers/analyses.py", "tests/test_imaging.py, tests/test_api.py"),
    ("R2", "Predict a pixel-level binary change mask with a Siamese Swin Transformer whose encoder and decoder "
           "are trained on a change-detection dataset.", "Functional", "High",
     "SiameseSwinCD (shared Swin-V2-T encoder, TemporalFusion x4, FPNDecoder); predict_pair",
     "ml/models/swin_siamese.py, ml/inference.py", "tests/test_models.py; reports/evaluation/swin_siamese_metrics.json"),
    ("R3", "Provide a CNN baseline (FC-Siam-diff) selectable at run time through a model registry.", "Functional",
     "High", "FCSiamDiff; MODEL_SPECS registry; ModelManager; ModelSelect component",
     "ml/models/fc_siam_diff.py, ml/models/registry.py, backend/app/services/model_manager.py",
     "tests/test_models.py, tests/test_api.py"),
    ("R4", "Compute TP, FP, TN, FN, precision, recall, F1, IoU, pixel accuracy and a confusion matrix, only when "
           "ground truth is available; guard zero denominators.", "Functional", "High",
     "ConfusionCounts / compute_metrics; artifacts.build_outputs; MetricsCard",
     "ml/metrics.py, backend/app/services/artifacts.py", "tests/test_metrics.py (cross-checked with scikit-learn)"),
    ("R5", "Visualise before / after images, a comparison slider, the change mask, a coloured overlay and a "
           "confidence map; allow threshold and overlay-opacity adjustment.", "Functional", "High",
     "Results page (CompareSlider, maskOverlay); POST /api/analyses/{id}/threshold (re-threshold stored "
     "probability map)", "frontend/src/pages/Results.tsx, backend/app/services/analysis.py",
     "frontend/src/test/app.test.tsx, frontend/e2e/demo.spec.ts, tests/test_api.py"),
    ("R6", "Download the mask, overlay and results (PNG / JSON), plus a GeoTIFF mask and GeoJSON regions when the "
           "input is georeferenced.", "Functional", "Medium",
     "ARTIFACTS whitelist; GET /api/analyses/{id}/downloads/{artifact}; geo.write_geotiff / change_regions",
     "backend/app/services/artifacts.py, backend/app/services/geo.py", "tests/test_api.py, tests/test_geo.py"),
    ("R7", "Persist every analysis with status, model, threshold and results; show history and live overview "
           "statistics.", "Functional", "Medium",
     "SQLite via SQLAlchemy (Analysis, EvaluationRun); /api/analyses, /api/analyses/stats; Overview and History pages",
     "backend/app/db.py, frontend/src/pages/History.tsx", "tests/test_api.py (persistence, restart recovery)"),
    ("R8", "Compare the transformer and the CNN on the same held-out split and present the measured results in a "
           "dashboard.", "Functional", "High",
     "ml.evaluate (test split, fixed threshold); GET /api/evaluation/summary; Compare page (Recharts)",
     "ml/evaluate.py, backend/app/routers/evaluation.py, frontend/src/pages/Compare.tsx",
     "tests/test_data_training.py, tests/test_api.py; reports/evaluation/"),
    ("R9", "Handle GeoTIFF input: validate CRS, extract bounds and resolution, reproject / resample T2 onto the T1 "
           "grid, export a georeferenced mask and display results on a map.", "Functional", "Medium",
     "GeoInfo; align_pair (rasterio.warp.reproject); web_mercator_overlay; GeoMap component",
     "backend/app/services/imaging.py, backend/app/services/geo.py, frontend/src/components/maps.tsx",
     "tests/test_imaging.py, tests/test_geo.py, frontend/e2e/demo.spec.ts"),
    ("R10", "Map-based acquisition: search a place, draw an area of interest, choose two date windows, list "
            "candidate scenes with cloud cover, retrieve georeferenced imagery and run detection.", "Functional",
     "Medium", "SatelliteProvider abstraction (EarthSearchProvider, CDSEProvider); /api/satellite/*; Map page",
     "backend/app/satellite/, frontend/src/pages/MapPage.tsx", "tests/test_satellite.py, frontend/e2e/demo.spec.ts"),
    ("R18", "Detect change on real Sentinel-2 (10 m) imagery: provide models trained on Sentinel-2 data, choose the "
            "model that matches the imagery automatically, and flag any model / imagery domain mismatch.",
     "Functional", "High",
     "swin_siamese_s2 / fc_siam_diff_s2 (OSCD fine-tune); preferred_model + domain_warning; ml.data.oscd; "
     "ml.calibrate",
     "ml/data/oscd.py, ml/models/registry.py, backend/app/services/model_manager.py",
     "tests/test_api.py (domain selection), tests/test_data_training.py (OSCD), e2e 08-09; "
     "reports/evaluation/*_s2_metrics.json"),
    ("R11", "Reproducible experiments: verified dataset download, leakage-free tiling with a split manifest, "
            "configurable training with seeds, machine-readable logs; the test split is never used for training "
            "or model selection.", "Functional", "High",
     "ml.data.download (MD5), ml.data.prepare (manifest, leakage check), ml.train (config.json, history.jsonl)",
     "ml/data/, ml/train.py, ml/configs/", "tests/test_data_training.py; experiments/*/"),
    ("R12", "Every failure returns a consistent, machine-readable error; background failures are recorded on the "
            "analysis; the UI shows loading, empty and error states.", "Non-functional", "High",
     "AppError + error envelope handlers; AnalysisService._run_safely; ErrorBanner / EmptyState / toasts",
     "backend/app/errors.py, frontend/src/components/ui.tsx", "tests/test_api.py, frontend/src/test/app.test.tsx"),
    ("R13", "Run on an 8 GB consumer GPU with mixed precision and fall back to CPU automatically; load each model "
            "once and serialise GPU access.", "Non-functional", "Medium",
     "ModelManager (load once, inference lock, OOM retry, CPU fallback); autocast; sliding window",
     "backend/app/services/model_manager.py, ml/inference.py", "tests/test_models.py, tests/test_api.py; benchmark latency"),
    ("R14", "Safe file handling for a local prototype: upload size limits, magic-byte checks, server-generated "
            "identifiers, whitelisted artefact names, no filesystem paths in responses, restricted CORS.",
     "Non-functional", "High", "detect_format; _read_limited; AnalysisService.folder id check; CORS middleware",
     "backend/app/services/imaging.py, backend/app/routers/analyses.py, backend/app/main.py", "tests/test_api.py"),
    ("R15", "Trustworthy output: never serve an untrained or mismatched checkpoint, show the checkpoint version "
            "with every result, never fabricate coordinates, areas or metrics.", "Non-functional", "High",
     "ml.checkpoint validation ('trained' flag, strict load); checkpoint version + SHA-256; geo only when present",
     "ml/checkpoint.py, backend/app/services/artifacts.py", "tests/test_models.py, tests/test_api.py"),
    ("R16", "Responsive, accessible dark dashboard with eight pages; every control performs a real action.",
     "Non-functional", "Medium", "Layout (responsive sidebar), Tailwind theme, ARIA roles/labels on controls",
     "frontend/src/components/Layout.tsx, frontend/src/pages/", "frontend/src/test/app.test.tsx, frontend/e2e/demo.spec.ts"),
    ("R17", "Automated test suite covering unit, API, integration, boundary and negative cases, with recorded "
            "execution logs.", "Non-functional", "High", "pytest + HTTPX TestClient, Vitest + Testing Library, Playwright",
     "tests/, frontend/src/test/, frontend/e2e/", "evidence/tests/"),
]

DESIGN_DECISIONS = [
    ("Encoder: Swin-V2-Tiny at 256x256 with window 8",
     "A hierarchical encoder gives the four-scale pyramid a dense-prediction decoder needs, and windowed "
     "attention is linear in image size. The V2 'window8_256' weights are pre-trained at exactly the 256x256 tile "
     "size used here.",
     "Plain ViT-B/16 (single scale, quadratic attention, weaker boundaries); Swin-V1-T at 224 with window 7 (its "
     "window does not divide a 256 px tile's 64x64 token grid, forcing position-bias interpolation); larger Swin "
     "variants (do not train comfortably in 8 GB)."),
    ("Decoder: per-scale [F1, F2, |F1-F2|] fusion + convolutional FPN",
     "Keeps the transformer where it matters (representation) and uses a small convolutional decoder so the "
     "whole model is about 30 M parameters and trains at under 2 GB of VRAM.",
     "A Swin transformer decoder as in SwinSUNet (heavier, slower); pure feature differencing without "
     "concatenation (discards which date a feature came from)."),
    ("Loss: BCE + batch-level soft Dice",
     "About 55% of LEVIR-CD tiles contain no change. Dice computed over the whole batch stays informative; the "
     "first implementation used per-tile Dice and the CNN collapsed to 'no change' (BUG-002).",
     "Per-tile Dice (rejected after BUG-002); BCE alone (biased towards the 95% negative class); focal loss "
     "(extra hyper-parameters, not needed once Dice was fixed)."),
    ("Data: pre-tiled uint8 NumPy memmaps",
     "Decoding the three 1024x1024 PNGs of a pair costs about 172 ms per sample against about 4 ms for a memmapped "
     "tile (measured, reports/dataloader_benchmark.json); a single-process loader then suffices, which avoids "
     "Windows multiprocessing overhead.",
     "On-the-fly PNG decoding with worker processes; random crops from full images (slower, and tiling must stay "
     "inside each official split to avoid leakage anyway)."),
    ("Persistence: SQLite + per-analysis folders",
     "A single-user local prototype needs zero-administration storage; SQLAlchemy keeps the door open for "
     "PostgreSQL. Artefacts are files because they are large binary rasters.",
     "PostgreSQL / PostGIS (operational overhead with no Phase-1 benefit); storing images as database BLOBs."),
    ("Jobs: one in-process worker thread",
     "There is one GPU, so inference must be serialised anyway; a ThreadPoolExecutor(max_workers=1) gives "
     "background execution with no extra services and works natively on Windows.",
     "Celery / Redis (extra infrastructure); running inference inside the request (blocks the API and times out "
     "on large rasters)."),
    ("Store the probability map, re-threshold on demand",
     "The 8-bit probability map is saved once; changing the threshold recomputes mask, metrics and exports in "
     "milliseconds without touching the GPU.",
     "Re-running inference per threshold; thresholding only in the browser (downloads and metrics would then "
     "disagree with what is shown)."),
    ("Sentinel-2 models: fine-tune the LEVIR-CD networks on OSCD RGB with TCI-matched radiometry",
     "OSCD has only 11 training cities, so both networks start from their LEVIR-CD weights (building-change "
     "features) instead of ImageNet. OSCD's 16-bit bands are scaled with the gain measured on real Earth Search "
     "true-colour pixels (DN 2500 -> 255; reports/tci_gain_fit.json) so training data looks like what the app "
     "receives. Padded border pixels are excluded from loss and metrics (ignore index 255).",
     "Training on all 13 bands (the app's imagery source delivers RGB only); per-image percentile stretching "
     "(would not match the fixed TCI scaling at inference); training from ImageNet only (too little data)."),
    ("Decision threshold chosen on the validation split",
     "Change is 2-3% of OSCD pixels, so 0.5 is not necessarily the best operating point. ml.calibrate sweeps "
     "0.05-0.95 on the validation cities and stores the best threshold in the checkpoint; the app uses it as the "
     "default and the user can still override it.",
     "Fixed 0.5 for every model; tuning on the test split (would leak test information)."),
    ("Imagery: Earth Search COGs as the default provider, CDSE optional",
     "Element 84 Earth Search serves Sentinel-2 L2A as public cloud-optimised GeoTIFFs: no account is needed and "
     "only the AOI window is read. CDSE is implemented behind the same interface but needs the user's own OAuth "
     "client.",
     "Web-map tiles (rendered pictures without dates or radiometry - explicitly rejected); downloading whole "
     "Sentinel-2 products (about 1 GB each)."),
]

DEVIATIONS = [
    ("CDSE is not the default imagery provider",
     "The brief preferred the Copernicus Data Space Ecosystem. Retrieval there needs OAuth credentials that were "
     "not available, so Earth Search (same Sentinel-2 L2A data, no account) is the verified default and CDSE "
     "remains implemented but unverified."),
    ("CNN baseline uses dropout 0.1 and stopped early",
     "The FC-Siam-diff paper uses dropout 0.2; 0.1 was used because the overfit diagnostic showed 0.2 slows "
     "convergence within a 40-epoch budget. Early stopping (patience 10) ended training at epoch 25, so the "
     "baseline is weaker than figures reported in the literature for longer schedules."),
    ("Swin batch size was reduced automatically",
     "The configuration requested batch 8; the conservative VRAM heuristic in train.py reduced it to 5 with "
     "gradient accumulation 2 (effective batch 10) because a second training run was sharing the GPU."),
    ("Sentinel-2 / OSCD work moved from Phase 2 into Phase 1",
     "The LEVIR-CD models produced 0.00% change on real Sentinel-2 imagery (BUG-006), so OSCD training, a "
     "cross-domain evaluation and Sentinel-2 model variants were added in Phase 1. Only the RGB bands are used."),
    ("Linting uses oxlint instead of ESLint",
     "The current Vite React-TS template ships oxlint with TypeScript 7; it was kept rather than adding an "
     "ESLint toolchain whose TypeScript-7 support was not verified."),
    ("Docker files are provided but untested",
     "Docker is not installed on the development machine; native Windows execution is the supported and "
     "verified path."),
    ("Database migrations are a schema-version table only",
     "With two tables and no released schema to migrate from, Alembic was deferred; create_all plus a "
     "schema_version row is used."),
]


# id, kind, feature/requirement, input, expected, actual (from the recorded run), automated test
TEST_CASES = [
    ("T1", "", "R1/R5 Upload to results flow", "96x96 PNG pair + mask, threshold 0.4",
     "202 with id; status reaches 'completed'; results list model, checkpoint version, statistics, metrics; "
     "all artefacts download as 96x96 PNG", "As expected", "test_api::test_upload_inference_results_download_flow"),
    ("T2", "", "R4 Metric correctness", "Random masks, 4 seeds",
     "Precision, recall, F1, IoU, accuracy and confusion matrix equal scikit-learn", "Equal to 1e-12",
     "test_metrics::test_metrics_match_sklearn"),
    ("T3", "", "R2 Swin forward pass", "2x3x256x256 tensors", "Logits 2x1x256x256, finite; 4-scale pyramid "
     "96/192/384/768 channels", "As expected", "test_models::test_swin_forward_shape_and_parameter_count"),
    ("T4", "", "R9 GeoTIFF to georeferenced export", "120x150 UTM 43N GeoTIFF pair",
     "Mask GeoTIFF keeps EPSG:32643, transform and size; changed pixels equal the PNG mask; GeoJSON region "
     "count matches", "As expected", "test_api::test_geotiff_upload_prediction_and_georeferenced_exports"),
    ("T5", "", "R9 Reprojection", "T1 in UTM, T2 in EPSG:4326 covering the same area",
     "T2 reprojected onto the T1 grid with a recorded warning", "As expected",
     "test_imaging::test_different_crs_is_reprojected_onto_t1"),
    ("T6", "", "R6 Geodesic area", "Web-Mercator raster at 60 deg N, all pixels changed",
     "Area about 25% of pixel-count x pixel-size (cos^2 60 deg), not the naive value", "Within 2%",
     "test_geo::test_area_is_geodesic_not_naive_for_web_mercator"),
    ("T7", "", "R7 Persistence", "Create analysis, restart the application",
     "Analysis, results and artefacts still served", "As expected",
     "test_api::test_analyses_persist_across_application_restarts"),
    ("T8", "", "R11 Train then evaluate", "Synthetic dataset, 2 CPU epochs",
     "Loadable trained checkpoint, history.jsonl, config.json, metrics JSON/CSV, charts", "As expected",
     "test_data_training::test_train_then_evaluate_smoke"),
    ("T9", "", "R5/R8 End-to-end in the browser", "Bundled LEVIR-CD pair, real Swin checkpoint",
     "Result page shows F1 > 50%, all views, threshold change, downloads, CNN re-run", "Pass (F1 on this pair "
     "above 90%)", "e2e/demo.spec.ts::02 sample analysis with ground truth"),
    ("T27", "", "R18 Real Sentinel-2 with ground truth", "OSCD Las Vegas test city (never trained on), automatic "
     "model choice", "Sentinel-2 Swin model selected, no domain warning, F1 > 30%, georeferenced map overlay",
     "Pass (F1 about 60% on this city)", "e2e/demo.spec.ts::08 real Sentinel-2 test city with ground truth"),
    ("T28", "", "R18 Domain-aware model choice", "10 m UTM GeoTIFF pair with and without an explicit LEVIR model",
     "Sentinel-2 model chosen automatically; explicit LEVIR choice honoured but flagged out-of-distribution",
     "As expected", "test_api::test_sentinel2_model_is_chosen_automatically_for_sentinel2_inputs"),
    ("T29", "", "R11 Ignore padded pixels", "Tile with 25% padded (ignore=255) pixels, wrong predictions there",
     "Loss identical to the same tile with correct predictions in the padding", "As expected",
     "test_data_training::test_ignore_pixels_do_not_contribute_to_loss_or_validation_counts"),
    ("T10", "boundary", "R1 Image side limits", "31 px, 32 px, 1024 px, 1025 px sides (limits 32-1024)",
     "31 rejected IMAGE_TOO_SMALL; 32 and 1024 accepted; 1025 rejected IMAGE_TOO_LARGE", "As expected",
     "test_imaging::test_dimension_boundaries"),
    ("T11", "boundary", "R13 Arbitrary raster size", "64x64, 257x100, 300x420, 512x512 inputs",
     "Probability map has exactly the input size, values in [0, 1]", "As expected",
     "test_models::test_sliding_window_output_matches_input_size"),
    ("T12", "boundary", "R4 Empty mask / no-change pair", "Identical images, all-zero ground truth",
     "Completes; warning 'pixel-identical'; recall reported undefined (not 0 or 1); accuracy defined",
     "As expected", "test_api::test_no_change_pair_completes_with_identical_warning_and_undefined_metrics"),
    ("T13", "boundary", "R5 Threshold limits", "Threshold 0, 1, -0.2, 1.5", "422 VALIDATION_ERROR each",
     "As expected", "test_api::test_rethreshold_updates_mask_statistics_and_metrics_without_new_inference"),
    ("T14", "boundary", "R1 Oversized upload", "3 MB PNG with a 2 MB limit", "413 FILE_TOO_LARGE",
     "As expected", "test_api::test_oversized_upload_rejected"),
    ("T15", "negative", "R1 Missing image", "Only T1 supplied", "422 MISSING_IMAGE, details.missing = [t2]",
     "As expected", "test_api::test_missing_image_rejected"),
    ("T16", "negative", "R1 Corrupted image", "PNG truncated to 200 bytes", "422 CORRUPTED_IMAGE",
     "As expected", "test_api::test_corrupted_image_rejected"),
    ("T17", "negative", "R1 Unsupported file", "PDF named t2.pdf", "415 UNSUPPORTED_FILE_TYPE", "As expected",
     "test_api::test_unsupported_file_type_rejected"),
    ("T18", "negative", "R1 Mismatched dimensions", "96x96 vs 128x96 PNG", "422 IMAGE_DIMENSION_MISMATCH",
     "As expected", "test_api::test_mismatched_dimensions_rejected"),
    ("T19", "negative", "R15 Missing checkpoint", "Request Swin with no checkpoint file",
     "503 MODEL_UNAVAILABLE; nothing stored", "As expected",
     "test_api::test_missing_checkpoint_gives_model_unavailable_and_stores_nothing"),
    ("T20", "negative", "R9 Invalid CRS", "GeoTIFF with a geotransform but no CRS", "422 INVALID_CRS",
     "As expected", "test_api::test_geotiff_with_transform_but_no_crs_is_rejected"),
    ("T21", "negative", "R9 Rasters of different places", "Two UTM GeoTIFFs with no overlap",
     "422 GEOSPATIAL_MISALIGNMENT", "As expected", "test_api::test_geotiff_pair_covering_different_places_is_rejected"),
    ("T22", "negative", "R13 GPU unavailable", "TV_DEVICE=cuda with CUDA reported absent",
     "Service starts, reports device=cpu, analysis completes", "As expected",
     "test_api::test_requested_gpu_unavailable_falls_back_to_cpu"),
    ("T23", "negative", "R10 Satellite provider failure", "Provider returns 500/502/503/429, times out, or "
     "returns non-JSON", "502 SATELLITE_PROVIDER_ERROR or 504 ..._TIMEOUT; upstream body never echoed; no analysis "
     "created", "As expected", "test_satellite::test_provider_http_failure_maps_to_502 (+ timeout, garbage)"),
    ("T24", "negative", "R12 Inference crash", "Model raises during prediction",
     "Analysis marked failed with INFERENCE_FAILED; results endpoint returns 409 ANALYSIS_FAILED", "As expected",
     "test_api::test_results_of_failed_analysis_return_error"),
    ("T25", "negative", "R14 Path traversal", "Analysis ids and artefact names containing ../",
     "404, no file access", "As expected", "test_api::test_unknown_and_malformed_analysis_ids"),
    ("T26", "negative", "R12 UI error state", "Server rejects the upload",
     "Error banner with message and code; no navigation", "As expected",
     "app.test.tsx::shows the server validation error when the upload is rejected"),
]

BUGS = [
    ("BUG-001", "Training crashed at start-up with JSONDecodeError on the CNN config file.",
     "The config had been rewritten by Windows PowerShell, which adds a UTF-8 byte-order mark; json.loads on "
     "text read as plain UTF-8 rejects the BOM.",
     "Config files are read with encoding='utf-8-sig' (ml/train.py; same for the sample manifest). Regression "
     "test writes a BOM-prefixed config.",
     "Before: evidence/bugs/BUG-001_before_config_bom_traceback.log. After: "
     "test_data_training::test_train_cli_rejects_unknown_config_keys passes with a BOM config."),
    ("BUG-002", "The CNN baseline predicted 'no change' for every pixel: validation F1 stayed at 0.0000 for four "
                "epochs while the loss barely moved.",
     "Dice loss was computed per tile. About 55% of tiles have no change, and for an empty tile the per-tile "
     "Dice term is about 1 for any non-zero probability mass, so those tiles dominated the gradient and pushed "
     "all outputs to zero.",
     "Dice is computed over the whole batch (ml/losses.py). Confirmed with an overfit diagnostic before "
     "retraining; both models were retrained from scratch so the comparison stays fair.",
     "Before: evidence/bugs/BUG-002_before/ (v1 history: F1 = 0 for epochs 1-4). After: "
     "evidence/bugs/BUG-002_after/ (v2 history: F1 0.39 at epoch 1, 0.69 by epoch 6)."),
    ("BUG-003", "A checkpoint written by the real training script could not be loaded: 'Weights only load "
                "failed ... Unsupported global: torch.torch_version.TorchVersion'.",
     "torch.__version__ is a TorchVersion object, not a str; it was stored in the checkpoint metadata, and the "
     "safe weights_only unpickler refuses unknown classes. Found by the train-then-evaluate test before the "
     "application ever used a real checkpoint.",
     "The version is stored as str(); the loader allow-lists TorchVersion and normalises metadata "
     "(ml/train.py, ml/checkpoint.py).",
     "Before: evidence/bugs/BUG-003_before_checkpoint_torchversion.log (1 failed, 49 passed). After: "
     "evidence/tests/backend_run_final.log (155 passed)."),
    ("BUG-004", "A backend started while training was still running kept serving the Swin weights from epoch 18 "
                "after training had saved the final epoch-29 checkpoint under the same file name.",
     "ModelManager cached a loaded model for the life of the process and never checked whether the checkpoint "
     "file had been replaced, so results silently came from stale weights that did not match the benchmark.",
     "The cache records the file's modification time and reloads when it changes "
     "(backend/app/services/model_manager.py).",
     "Before: evidence/bugs/BUG-004/before_stale_epoch18_checkpoint_served.jpg. After: "
     "evidence/bugs/BUG-004/after_final_epoch29_checkpoint_served.png and "
     "test_api::test_replaced_checkpoint_is_reloaded_instead_of_serving_stale_weights."),
    ("BUG-005", "The 'these images may show different places' warning scored a genuinely related pair below its "
                "threshold (edge correlation 0.14 with a 0.15 threshold).",
     "The threshold was a guess. Measured on 40 LEVIR-CD test pairs, genuine pairs score median 0.47 (5th "
     "percentile 0.16) and deliberately mismatched pairs median -0.03 (95th percentile 0.15), so 0.15 sat inside "
     "the genuine range.",
     "Threshold calibrated to 0.10 from those measurements, the warning text now states it is a heuristic, and "
     "the test uses realistic edge-rich scenes (backend/app/services/imaging.py).",
     "Before: first full run 153 passed / 1 failed. After: evidence/tests/backend_run_final.log."),
    ("BUG-006", "Real Sentinel-2 imagery gave no output: the north-Bengaluru map analysis reported 0.00% changed "
                "pixels, 0 regions (reported by the user).",
     "Domain shift, not a code fault: the only models were trained on 0.5 m LEVIR-CD aerial imagery, where a "
     "building spans dozens of pixels; at 10 m it spans one or two. Measured on the OSCD test cities, the "
     "LEVIR-only Swin model reaches F1 0.0%. The app ran such a model on Sentinel-2 and only printed a warning.",
     "Added Sentinel-2 models fine-tuned on OSCD with TCI-matched radiometry (ml/data/oscd.py), a validation-"
     "selected threshold (ml/calibrate.py), automatic model choice by imagery domain and a domain-mismatch "
     "warning (backend/app/services/model_manager.py), and a cross-domain benchmark.",
     "Before: evidence/bugs/BUG-006/before_levir_model_0pct_on_sentinel2.png. After: "
     "evidence/bugs/BUG-006/after_sentinel2_model_on_bengaluru.png (detections; no ground truth available) and "
     "after_oscd_lasvegas_measured.png (held-out city with ground truth, measured F1)."),
]

INTEGRATION_NOTE = (
    "Module 1 is the ML package (ml/: data, models, training, evaluation, inference) and Module 2 is the "
    "application (FastAPI backend + React frontend). They connect through exactly two contracts: the checkpoint "
    "file format validated by ml.checkpoint.load_model, and ml.inference.predict_pair, which the backend's "
    "ModelManager calls. Integration exposed three defects that unit tests of either module alone could not: the "
    "checkpoint written by training was rejected by the application's safe loader (BUG-003), a long-running "
    "backend kept serving a checkpoint that training had since replaced (BUG-004), a model was applied to imagery "
    "from a domain it was never trained on (BUG-006, now prevented by domain-aware model selection), and the "
    "benchmark report "
    "could describe a different checkpoint from the one loaded - now guarded by comparing SHA-256 hashes. The "
    "frontend and backend integrate through a Vite proxy so result images can be read into a canvas "
    "same-origin; the Playwright suite exercises the full chain frontend -> API -> GPU inference -> results."
)

PHASE2 = [
    ("Strengthen the CNN baseline: longer schedule without early stop, learning-rate search, 3 seeds; report "
     "mean and standard deviation for both models", "High",
     "Reuse ml.train with new configs; add a seed sweep and aggregate in ml.evaluate", "Phase-2 weeks 1-2"),
    ("Improve Sentinel-2 accuracy: all 13 OSCD bands, Level-1C vs Level-2A harmonisation, more training regions "
     "(e.g. Indian cities), 5-fold city cross-validation", "High",
     "Extend ml.data.oscd and the patch embedding to multispectral input; add an L2A band reader to the provider",
     "Phase-2 weeks 2-5"),
    ("Multi-class (semantic) change: land-cover transitions rather than a binary mask", "High",
     "Adopt a semantic change dataset (e.g. SECOND); add a multi-class head and per-class IoU",
     "Phase-2 weeks 5-8"),
    ("Robustness to clouds, seasons and water-level change for Sentinel-2 retrieval (the Bengaluru result "
     "flags filled lakes and farmland as change)", "High",
     "Use the L2A scene-classification layer to mask clouds; same-season scene pairing; photometric augmentation",
     "Phase-2 weeks 6-8"),
    ("Wider transformer comparison: BIT, ChangeFormer, Swin-S, plus ablations of the fusion module", "Medium",
     "Implement behind the existing ModelSpec registry; same split, same metrics", "Phase-2 weeks 8-10"),
    ("Explainability: attention / gradient-based maps and calibrated uncertainty", "Medium",
     "Grad-CAM on fused features, test-time augmentation variance, reliability diagrams", "Phase-2 weeks 9-11"),
    ("Verify the Copernicus (CDSE) provider end-to-end; scheduled monitoring of saved areas of interest",
     "Medium", "Create an OAuth client; add a scheduler and change alerts", "Phase-2 weeks 10-12"),
    ("Deployment hardening: authentication, Docker build verification, PostgreSQL option, load testing",
     "Low", "Build and test the provided Docker files; add Alembic migrations; Locust load test",
     "Phase-2 weeks 12-14"),
]

LIMITATIONS = [
    "Two imagery domains are validated: 0.5 m LEVIR-CD-like aerial RGB and 10 m Sentinel-2 RGB (OSCD). Sentinel-2 "
    "accuracy is moderate (see the OSCD benchmark): small or thin changes are missed, and on the Bengaluru pair "
    "seasonal farmland and lakes that filled with water are also flagged. Deforestation and disaster damage are "
    "not supported.",
    "OSCD is small (11 training cities) and is Level-1C top-of-atmosphere data while the application receives "
    "Level-2A true colour; only the DN scale is matched, not the atmospheric correction. The CNN's selected "
    "threshold (0.05) lies at the edge of the search grid.",
    "Single training run per model (seed 42); no confidence intervals. The CNN baseline stopped early at epoch "
    "25 and is weaker than figures reported in the literature for FC-Siam-diff trained longer.",
    "The Copernicus Data Space (CDSE) retrieval path is implemented from the documentation but has not been run "
    "against the real service (no credentials). Earth Search retrieval is verified.",
    "The pair-similarity warning is a calibrated heuristic; it can miss unrelated pairs or flag genuine pairs "
    "with very large change.",
    "Local research prototype: no authentication, single worker, SQLite. Docker files are untested. No load, "
    "security or reliability testing has been done.",
    "Frontend unit-test coverage is partial (map drawing and charts are covered by the Playwright run, not by "
    "unit tests).",
]



# Keep tables in ID order (entries are grouped by topic above).
REQUIREMENTS.sort(key=lambda r: int(r[0][1:]))
TEST_CASES.sort(key=lambda t: int(t[0][1:]))
