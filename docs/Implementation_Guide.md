# TerraVision AI - Implementation guide

Vision Transformers for Satellite Land-Use Monitoring. Research prototype, Phase 1.

## 1. What this is

A bitemporal change-detection system: a Siamese Swin Transformer (primary) and an FC-Siam-diff CNN (baseline),
trained on LEVIR-CD, served by a FastAPI backend and a React dashboard.

## 2. Repository layout

| Path | Contents |
|---|---|
| `ml/` | datasets (`data/`), models (`models/`), `train.py`, `evaluate.py`, `inference.py`, `checkpoint.py`, `metrics.py` |
| `backend/app/` | FastAPI app: `routers/`, `services/`, `satellite/`, `schemas.py`, `config.py`, `db.py` |
| `frontend/` | React + Vite + TypeScript dashboard, unit tests (`src/test`), end-to-end tests (`e2e`) |
| `tests/` | backend and ML tests (pytest) |
| `scripts/` | `start.ps1`, `build_samples.py`, `build_docs.py` |
| `samples/` | bundled demonstration pairs with `manifest.json` (provenance, licence) |
| `checkpoints/` | trained weights (git-ignored; see section 5) |
| `reports/evaluation/` | benchmark JSON / CSV / figures written by `ml.evaluate` |
| `experiments/` | per-run `config.json`, `history.jsonl`, `summary.json` |
| `docs/`, `evidence/` | documentation and recorded evidence |

## 3. Setup (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend; npm install; cd ..
```

RTX 50-series GPUs need the CUDA 12.8 (`cu128`) wheels. For CPU only use the `cpu` index. Exact versions used
are in `requirements.lock.txt`. Docker is optional and untested (`Dockerfile.backend`, `docker-compose.yml`).

## 4. Run

```powershell
.\scripts\start.ps1
```

or in two terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
cd frontend; npm run dev
```

Frontend: http://localhost:5173 - API: http://127.0.0.1:8000 (interactive docs at `/docs`).
Configuration: copy `.env.example` to `.env` (all `TV_*` variables are optional).

## 5. Reproduce the models

```powershell
.\.venv\Scripts\python.exe -m ml.data.download          # ~2.7 GB, MD5-verified, official split
.\.venv\Scripts\python.exe -m ml.data.prepare           # 256x256 tiles + manifest
.\.venv\Scripts\python.exe -m ml.train --config ml/configs/swin_siamese.json
.\.venv\Scripts\python.exe -m ml.train --config ml/configs/fc_siam_diff.json
.\.venv\Scripts\python.exe -m ml.data.oscd download      # OSCD (Sentinel-2), SHA-256 verified
.\.venv\Scripts\python.exe -m ml.data.oscd prepare       # RGB at the measured TCI gain, city split
.\.venv\Scripts\python.exe -m ml.train --config ml/configs/swin_siamese_s2.json   # fine-tune from LEVIR-CD
.\.venv\Scripts\python.exe -m ml.train --config ml/configs/fc_siam_diff_s2.json
.\.venv\Scripts\python.exe -m ml.calibrate --models swin_siamese_s2 fc_siam_diff_s2   # threshold on val cities
.\.venv\Scripts\python.exe -m ml.evaluate --cross-domain # both test sets + domain-shift check
.\.venv\Scripts\python.exe scripts\build_samples.py     # bundled samples (Sentinel-2 part needs internet)
```

Dataset: LEVIR-CD, 445 / 64 / 128 images ->
7120 / 1024 / 2048 patches
(train / val / test). Source: Hugging Face mirror `satellite-image-deep-learning/LEVIR-CD` at a pinned revision
(the URLs and checksums TorchGeo uses). Licence: academic use only.

Sentinel-2: OSCD (Caye Daudt et al., IGARSS 2018) from the Hugging Face mirror `hkristen/oscd` at a pinned revision
(TorchGeo's SHA-256 checksums). Train cities abudhabi, aguasclaras, beirut, bercy, bordeaux, cupertino, mumbai, nantes, paris, pisa, saclay_e; validation cities
beihai, hongkong, rennes; test cities brasilia, chongqing, dubai, lasvegas, milano, montpellier, norcia, rio, saclay_w, valencia.
Bands B04/B03/B02 are scaled with the gain measured on real Earth Search true-colour pixels (DN
2503 -> 255; `reports/tci_gain_fit.json`). Thresholds chosen on the validation cities:
Swin 0.2, CNN 0.05
(`reports/threshold_calibration.json`).

Checkpoints are produced locally by the commands above; no third-party change-detection checkpoint is used. The
Swin encoder starts from timm `swinv2_tiny_window8_256.ms_in1k` (ImageNet-1k) and is fine-tuned end-to-end.

## 6. Measured results (11 November 2026)

On the held-out LEVIR-CD test split (2048 patches of 256x256 px at 0.5 m, threshold 0.5), the Siamese Swin Transformer reaches F1 91.35% and IoU 84.07%; the FC-Siam-diff baseline reaches F1 77.89% and IoU 63.79%. On real Sentinel-2 imagery (the 10 OSCD test cities at 10 m, whole images, thresholds selected on validation cities) the fine-tuned Swin model reaches F1 47.09% and IoU 30.80% against F1 39.70% and IoU 24.77% for the CNN, whereas the LEVIR-CD-only Swin model scores F1 0.00% on the same cities. The transformer costs 31x the parameters and 3.8x the per-tile latency (22 ms vs 6 ms on NVIDIA GeForce RTX 5050 Laptop GPU). All results come from a single training run per model (seed 42).

| Test data | Model | Precision | Recall | F1 | IoU | Accuracy | Threshold | Params | Latency / tile |
|---|---|---|---|---|---|---|---|---|---|
| LEVIR-CD | Siamese Swin Transformer | 92.45% | 90.27% | 91.35% | 84.07% | 99.13% | 0.5 | 29.98 M | 21.9 ms |
| LEVIR-CD | FC-Siam-diff (CNN baseline) | 90.53% | 68.35% | 77.89% | 63.79% | 98.02% | 0.5 | 0.97 M | 5.7 ms |
| OSCD | Siamese Swin Transformer (Sentinel-2) | 66.85% | 36.35% | 47.09% | 30.80% | 95.78% | 0.2 | 29.98 M | 21.9 ms |
| OSCD | FC-Siam-diff CNN (Sentinel-2) | 55.89% | 30.79% | 39.70% | 24.77% | 95.17% | 0.05 | 0.97 M | 5.3 ms |

Domain shift - the OSCD test cities scored by models with and without Sentinel-2 training:

| Model (scored on OSCD test cities) | Trained on | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| Siamese Swin Transformer | LEVIR-CD only | n/a | 0.00% | 0.00% | 0.00% |
| FC-Siam-diff (CNN baseline) | LEVIR-CD only | 100.00% | 0.30% | 0.59% | 0.30% |
| Siamese Swin Transformer (Sentinel-2) | LEVIR-CD, then OSCD fine-tune | 66.85% | 36.35% | 47.09% | 30.80% |
| FC-Siam-diff CNN (Sentinel-2) | LEVIR-CD, then OSCD fine-tune | 55.89% | 30.79% | 39.70% | 24.77% |

| Model | Run | Batch | Grad. accum. | LR | Epochs | Best val F1 | Train time | Peak VRAM (train) |
|---|---|---|---|---|---|---|---|---|
| Siamese Swin Transformer | swin_siamese_levir_v2 (LEVIR-CD) | 5 | 2 | 0.0002 | 30 (best 29) | 91.82% | 127 min | 1915 MB |
| FC-Siam-diff (CNN baseline) | fc_siam_diff_levir_v2 (LEVIR-CD) | 16 | 1 | 0.001 | 25 (best 15) | 79.25% | 74 min | 1499 MB |
| Siamese Swin Transformer (Sentinel-2) | swin_siamese_oscd_v1 (OSCD) | 8 | 1 | 0.0001 | 43 (best 23) | 45.35% | 5 min | 2608 MB |
| FC-Siam-diff CNN (Sentinel-2) | fc_siam_diff_oscd_v1 (OSCD) | 16 | 1 | 0.0005 | 39 (best 14) | 32.98% | 2 min | 1507 MB |

Hardware: NVIDIA GeForce RTX 5050 Laptop GPU (8150 MB), PyTorch 2.11.0+cu128, mixed precision. The two LEVIR-CD
models were trained at the same time on the one GPU, so their training times are slower than a solo run. Application
inference on a 1024x1024 pair (25 tiles) took about 0.6-1.0 s with the Swin model once loaded, and 4.5 s on the
first request including model load (values from the analysis history of the demo runs).

## 7. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest --cov          # 161 tests, 85% coverage
cd frontend; npm test                                 # 22 tests
npx playwright test                                   # 9 end-to-end tests (servers must be running)
.\.venv\Scripts\python.exe -m ruff check . ; cd frontend; npm run typecheck; npm run lint
```

## 8. Demonstration

1. Start: `.\scripts\start.ps1`. Browser opens the Overview page.
2. **System Status** page: API online, database connected, 4 models ready (2 for LEVIR-CD, 2 for Sentinel-2), GPU.
3. **New Analysis -> Bundled samples**: select "LEVIR-CD test_45 (dense change)"; the matching *Siamese Swin Transformer* (LEVIR-CD) is selected; **Run change detection**.
4. Results page: before / after slider; tabs *Side by side*, *Overlay*, *Change mask*, *Confidence*.
5. Read *Changed pixels*, *Inference time*, model and checkpoint version.
6. Tab *Ground truth*: ground-truth mask and TP / FP / FN error map; F1, IoU and the confusion matrix on the right.
7. Move the **Threshold** slider: mask and metrics update without new inference. Adjust **Overlay opacity**.
8. **Downloads**: mask PNG, overlay PNG, results JSON.
9. **Re-run with FC-Siam-diff** (top right): the same pair with the CNN - compare F1 / IoU.
10. **Model Comparison**: LEVIR-CD and OSCD (Sentinel-2) benchmarks, the domain-shift table, and **Run now** for the live sample evaluation.
11. **Analysis History**: both runs listed; open or delete.
12. **Real Sentinel-2 with ground truth**: *Bundled samples* -> "OSCD lasvegas" (a held-out test city). The
    Sentinel-2 Swin model is selected automatically. Show F1 / IoU against ground truth, the *Ground truth* tab and
    the *Map* tab (the overlay sits on Las Vegas). Mention the model never saw this city.
13. **New Analysis -> Upload**: upload `samples/sentinel2/bengaluru_north_t1.tif` and `_t2.tif`. The 10 m GeoTIFFs
    are recognised and the Sentinel-2 model is used. The *Map* tab shows detected change over north Bengaluru;
    *Change mask (GeoTIFF)* and *GeoJSON* downloads appear. No accuracy metrics (no ground truth); point out real
    construction detected, and false positives on lakes that filled with water and on seasonal farmland.
14. **Satellite Map** (needs internet): search "Yelahanka", *Draw area*, *Find candidate scenes*, inspect dates and
    cloud cover, *Retrieve & detect change*.
15. Error handling: upload `samples/levir/test_45_t1.png` with `samples/sentinel2/bengaluru_north_t2.preview.png`
    -> "IMAGE_DIMENSION_MISMATCH"; choose a `.txt` file -> rejected in the browser; stop the backend -> "Backend offline".

Steps 1-13 and 15 work offline once dependencies, checkpoints and samples are installed (basemap tiles in the
*Map* tab need internet; the result overlay itself does not).

## 9. Bundled samples

| Sample | Dataset | Ground truth | Georeferenced | Licence |
|---|---|---|---|---|
| LEVIR-CD test_45 (dense change) | LEVIR-CD | yes | no | LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from Google Earth is subject to the Google Earth terms of use. |
| LEVIR-CD test_74 (moderate change) | LEVIR-CD | yes | no | LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from Google Earth is subject to the Google Earth terms of use. |
| LEVIR-CD test_101 (sparse change) | LEVIR-CD | yes | no | LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from Google Earth is subject to the Google Earth terms of use. |
| LEVIR-CD test_108 (no change) | LEVIR-CD | yes | no | LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from Google Earth is subject to the Google Earth terms of use. |
| OSCD lasvegas 2015 vs 2018 (Sentinel-2, ground truth) | OSCD | yes | yes | Imagery: Copernicus Sentinel data (free, full and open). Labels: OSCD by Caye Daudt et al.; the dataset README requires citation and states no further licence terms. |
| OSCD dubai 2015 vs 2018 (Sentinel-2, ground truth) | OSCD | yes | yes | Imagery: Copernicus Sentinel data (free, full and open). Labels: OSCD by Caye Daudt et al.; the dataset README requires citation and states no further licence terms. |
| OSCD chongqing 2017 vs 2018 (Sentinel-2, ground truth) | OSCD | yes | yes | Imagery: Copernicus Sentinel data (free, full and open). Labels: OSCD by Caye Daudt et al.; the dataset README requires citation and states no further licence terms. |
| Sentinel-2 north Bengaluru 2019 vs 2025 (georeferenced) | Sentinel-2 L2A (Earth Search) | no | yes | Copernicus Sentinel data: free, full and open access; attribution required. |

## 10. Known limitations

- Two imagery domains are validated: 0.5 m LEVIR-CD-like aerial RGB and 10 m Sentinel-2 RGB (OSCD). Sentinel-2 accuracy is moderate (see the OSCD benchmark): small or thin changes are missed, and on the Bengaluru pair seasonal farmland and lakes that filled with water are also flagged. Deforestation and disaster damage are not supported.
- OSCD is small (11 training cities) and is Level-1C top-of-atmosphere data while the application receives Level-2A true colour; only the DN scale is matched, not the atmospheric correction. The CNN's selected threshold (0.05) lies at the edge of the search grid.
- Single training run per model (seed 42); no confidence intervals. The CNN baseline stopped early at epoch 25 and is weaker than figures reported in the literature for FC-Siam-diff trained longer.
- The Copernicus Data Space (CDSE) retrieval path is implemented from the documentation but has not been run against the real service (no credentials). Earth Search retrieval is verified.
- The pair-similarity warning is a calibrated heuristic; it can miss unrelated pairs or flag genuine pairs with very large change.
- Local research prototype: no authentication, single worker, SQLite. Docker files are untested. No load, security or reliability testing has been done.
- Frontend unit-test coverage is partial (map drawing and charts are covered by the Playwright run, not by unit tests).
