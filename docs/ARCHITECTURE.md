# Architecture and design

TerraVision AI is a three-layer local application: a **React** single-page frontend, a **FastAPI** backend
and an **ML package** (PyTorch), with SQLite and the filesystem as data stores and optional external
imagery services. The layers meet at two narrow interfaces: the HTTP/JSON API, and the checkpoint format plus
`ml.inference.predict_pair`.

![System architecture](design/diagrams/01_system_architecture.png)

## 1. Components

| Component | Responsibility | Code |
|---|---|---|
| Frontend pages | Eight routed pages; typed API client; presentation only | `frontend/src/pages`, `frontend/src/lib/api.ts` |
| Routers | HTTP contracts, validation, response models | `backend/app/routers/` |
| imaging | Decode, validate, stretch, align / reproject image pairs; pair sanity heuristics | `backend/app/services/imaging.py` |
| AnalysisService | Persist inputs, queue and run jobs, re-threshold, artefacts, clean-up, crash recovery | `backend/app/services/analysis.py` |
| ModelManager | Checkpoint status, load-once cache with change detection, GPU lock, CPU fallback, OOM retry, domain-aware model choice | `backend/app/services/model_manager.py` |
| artifacts / geo | Masks, overlays, confidence and error maps; GeoTIFF, GeoJSON, geodesic area, Web-Mercator overlay | `backend/app/services/artifacts.py`, `geo.py` |
| Satellite providers | Provider interface, STAC search, windowed COG reads, CDSE Process API, geocoding | `backend/app/satellite/` |
| Models | SiameseSwinCD, FCSiamDiff behind a registry of four trained variants | `ml/models/` |
| Data | LEVIR-CD and OSCD download, validation, tiling, manifests | `ml/data/` |
| Training / evaluation | Config-driven training, threshold calibration, benchmark reports | `ml/train.py`, `ml/calibrate.py`, `ml/evaluate.py` |
| Inference | Sliding-window bitemporal prediction with blending and AMP | `ml/inference.py` |

## 2. Request lifecycle (upload)

1. `POST /api/analyses/upload` reads both files with a size limit, checks extension, magic bytes and decodability.
2. `imaging.align_pair` rejects incompatible pairs (different sizes, different places) or reprojects T2 onto
   T1's grid when both are georeferenced and overlap by at least 90%.
3. The model is chosen (explicit, or matching the imagery domain) with its calibrated threshold; inputs are
   saved and a row is inserted with status `queued`; the API returns `202 Accepted`.
4. A single background worker runs sliding-window inference on the GPU under a lock, stores the 8-bit
   probability map, then the mask, overlay, metrics (only with ground truth) and georeferenced exports (only
   with georeferenced input), and marks the row `completed` (or `failed` with an error code).
5. The frontend polls `GET /api/analyses/{id}` and then loads `/results` and the artefacts.
   Moving the threshold slider re-thresholds the stored probability map - no new inference.

## 3. Data model

| Store | Contents |
|---|---|
| `analyses` (SQLite) | id, name, status, source, model_key, threshold, timestamps, width, height, georeferenced, has_ground_truth, changed_fraction, inference_seconds, error_code, error_message, inputs (JSON), results (JSON) |
| `evaluation_runs` (SQLite) | id, created_at, dataset, num_pairs, threshold, results (JSON) |
| `storage/analyses/<id>/` | t1/t2 PNG, ground_truth, geo.json, probability, confidence, mask, overlay, error_map, mask.tif, regions.geojson, map overlays, results.json |
| `checkpoints/*.pt` | format_version, model_key, model_kwargs, state_dict, trained, meta |
| `reports/evaluation/` | <model>_metrics.json, metrics.csv, summary.json, cross_domain.json, figures |

## 4. HTTP API

| Method and path | Purpose |
|---|---|
| GET /health | Liveness, database and model availability |
| GET /api/system | Compute device, GPU memory, storage, versions |
| GET /api/models | Model registry with checkpoint and benchmark information |
| GET /api/config | Public limits and defaults |
| GET /api/samples, /api/samples/{id}/{role} | Bundled samples and their images |
| POST /api/analyses/upload | Create an analysis from uploaded T1, T2 (+ mask) - multipart |
| POST /api/analyses | Create an analysis from a bundled sample, or re-run stored inputs with another model |
| GET /api/analyses, /api/analyses/stats | History (paginated, filterable) and overview statistics |
| GET /api/analyses/{id} | Status and details of one analysis |
| GET /api/analyses/{id}/results | Full results and download links (409 until completed) |
| POST /api/analyses/{id}/threshold | Re-threshold the stored probability map |
| GET /api/analyses/{id}/downloads/{artifact} | Whitelisted artefacts: images, JSON, GeoTIFF, GeoJSON |
| DELETE /api/analyses/{id} | Delete the record and its files |
| POST /api/evaluation | Evaluate available models on labelled bundled samples |
| GET /api/evaluation/summary, /reports/{name} | Offline test-split benchmark and figures |
| GET /api/satellite/providers, /geocode | Imagery providers; place search |
| POST /api/satellite/search | Candidate Sentinel-2 scenes for an AOI and date window |
| POST /api/satellite/retrieve | Fetch two scenes for the AOI and start an analysis |

All errors use one envelope: `{"error": {"code", "message", "details"}}`. Interactive schema at
`http://127.0.0.1:8000/docs` when the backend is running.

## 5. Diagrams

All diagrams are Mermaid sources in [`docs/design/diagrams`](design/diagrams) (GitHub renders them below)
with PNG and SVG renders alongside. Each element corresponds to real code (see the component table).

### 5.1 System architecture

Rendered: [PNG](design/diagrams/01_system_architecture.png) - [SVG](design/diagrams/01_system_architecture.svg) - source [`01_system_architecture.mmd`](design/diagrams/01_system_architecture.mmd)

```mermaid
flowchart LR
  user([Analyst / evaluator])
  subgraph FE["Frontend - React + Vite + TypeScript (port 5173)"]
    pages["Pages: Overview, New Analysis, Satellite Map,<br/>Results, Model Comparison, History, Research, System"]
    apiclient["lib/api.ts<br/>typed API client"]
    pages --> apiclient
  end
  subgraph BE["Backend - FastAPI (port 8000)"]
    routers["Routers: system, analyses,<br/>evaluation, satellite"]
    analysis["AnalysisService<br/>(background worker, 1 thread)"]
    imaging["imaging.py<br/>decode, validate, align"]
    manager["ModelManager<br/>(load once, GPU lock)"]
    artifacts["artifacts.py / geo.py<br/>mask, overlay, GeoTIFF, GeoJSON"]
    providers["ProviderRegistry<br/>EarthSearch, CDSE"]
    routers --> analysis
    routers --> imaging
    routers --> providers
    analysis --> manager
    analysis --> artifacts
  end
  subgraph ML["ML package (PyTorch)"]
    swin["SiameseSwinCD<br/>Swin-V2-T encoder + FPN decoder"]
    cnn["FCSiamDiff<br/>CNN baseline"]
    infer["inference.predict_pair<br/>sliding window + AMP"]
    train["train.py / evaluate.py<br/>(offline)"]
  end
  subgraph ST["Local storage"]
    db[("SQLite<br/>analyses, evaluation_runs")]
    files[("storage/analyses/ID/<br/>inputs + artefacts")]
    ckpt[("checkpoints/*.pt")]
    reports[("reports/evaluation/<br/>benchmark JSON, CSV, PNG")]
    samples[("samples/<br/>bundled pairs")]
  end
  subgraph EXT["External services (optional, online)"]
    es["Element 84 Earth Search<br/>STAC + Sentinel-2 COGs"]
    cdse["Copernicus Data Space<br/>STAC + Process API"]
    osm["OpenStreetMap<br/>tiles + Nominatim"]
  end
  user --> pages
  apiclient -- "HTTP/JSON, multipart (Vite proxy)" --> routers
  pages -. "basemap tiles" .-> osm
  manager --> infer
  infer --> swin
  infer --> cnn
  manager --> ckpt
  train --> ckpt
  train --> reports
  analysis --> db
  analysis --> files
  routers --> reports
  routers --> samples
  providers --> es
  providers --> cdse
  providers --> osm
```
### 5.2 Application flowchart (analysis request)

Rendered: [PNG](design/diagrams/02_flowchart.png) - [SVG](design/diagrams/02_flowchart.svg) - source [`02_flowchart.mmd`](design/diagrams/02_flowchart.mmd)

```mermaid
flowchart TD
  start([Start analysis request]) --> mode{Input mode?}
  mode -- Upload --> read["Read files with size limit"]
  mode -- Bundled sample --> sample["Load sample files from manifest"]
  mode -- Satellite map --> aoi{"AOI size and dates valid?"}
  aoi -- No --> e1[/"422 AOI / validation error"/]
  aoi -- Yes --> fetch["Provider.retrieve T1 and T2<br/>clipped to AOI"]
  fetch --> pok{"Provider OK and<br/>scene covers AOI?"}
  pok -- No --> e2[/"502 / 504 / 422 provider error"/]
  pok -- Yes --> order["Order scenes by acquisition date"]
  read --> fmt{"Extension, magic bytes,<br/>integrity, size OK?"}
  fmt -- No --> e3[/"413 / 415 / 422 error envelope"/]
  fmt -- Yes --> geo
  sample --> geo
  order --> geo
  geo{"Both images georeferenced?"}
  geo -- Yes --> grid{"Same CRS and grid?"}
  grid -- Yes --> model
  grid -- No --> ov{"Footprint overlap >= 90%?"}
  ov -- No --> e4[/"422 GEOSPATIAL_MISALIGNMENT"/]
  ov -- Yes --> warp["Reproject / resample T2 onto T1 grid<br/>(warning recorded)"] --> model
  geo -- No --> dims{"Identical dimensions?"}
  dims -- No --> e5[/"422 IMAGE_DIMENSION_MISMATCH"/]
  dims -- Yes --> model
  model{"Model has a trained<br/>checkpoint?"}
  model -- No --> e6[/"503 MODEL_UNAVAILABLE"/]
  model -- Yes --> store["Save T1, T2, mask, geo.json<br/>insert row: status = queued"]
  store --> resp[/"202 Accepted + analysis id"/]
  store --> worker["Background worker: status = running"]
  worker --> tiles["Cut into 256x256 tiles with 64 px overlap<br/>(pad if smaller than a tile)"]
  tiles --> fwd["Siamese forward pass per batch<br/>(AMP on CUDA, OOM: retry with batch 1)"]
  fwd --> blend["Sigmoid, Hann-weighted blend<br/>gives probability map"]
  blend --> thr["Threshold gives binary change mask"]
  thr --> gt{"Ground truth provided?"}
  gt -- Yes --> met["Precision, recall, F1, IoU, accuracy,<br/>confusion matrix, error map"]
  gt -- No --> desc["Descriptive statistics only"]
  met --> isgeo
  desc --> isgeo
  isgeo{"Georeferenced?"}
  isgeo -- Yes --> gexp["GeoTIFF mask, GeoJSON regions,<br/>geodesic area, Web-Mercator overlay"]
  isgeo -- No --> png["PNG artefacts only"]
  gexp --> done
  png --> done
  done["Write results.json, status = completed"] --> finish([Results page polls and renders])
  worker -. exception .-> failed["status = failed + error code"]
```
### 5.3 UML use-case diagram

Rendered: [PNG](design/diagrams/03_use_case.png) - [SVG](design/diagrams/03_use_case.svg) - source [`03_use_case.mmd`](design/diagrams/03_use_case.mmd)

```mermaid
flowchart LR
  analyst(["Analyst"])
  evaluator(["Evaluator"])
  researcher(["Researcher"])
  provider(["Satellite imagery provider"])
  subgraph system["TerraVision AI"]
    uc1(["Upload image pair (PNG / JPEG / GeoTIFF)"])
    uc2(["Run analysis on a bundled sample"])
    uc3(["Select AOI and dates on the map"])
    uc4(["Search and retrieve Sentinel-2 scenes"])
    uc5(["Run change detection"])
    uc6(["Validate and align image pair"])
    uc7(["View mask, overlay, slider, confidence"])
    uc8(["Adjust threshold and overlay opacity"])
    uc9(["Compare against ground truth (F1, IoU)"])
    uc10(["Download PNG / JSON / GeoTIFF / GeoJSON"])
    uc11(["Re-run with the other model"])
    uc12(["Compare CNN vs transformer benchmark"])
    uc13(["Browse and delete analysis history"])
    uc14(["Check system and model status"])
    uc15(["Train and evaluate models (CLI)"])
  end
  analyst --> uc1
  analyst --> uc2
  analyst --> uc3
  analyst --> uc7
  analyst --> uc8
  analyst --> uc10
  analyst --> uc11
  analyst --> uc13
  evaluator --> uc2
  evaluator --> uc9
  evaluator --> uc12
  evaluator --> uc14
  researcher --> uc15
  researcher --> uc12
  uc3 -. include .-> uc4
  uc1 -. include .-> uc6
  uc4 -. include .-> uc6
  uc1 -. include .-> uc5
  uc2 -. include .-> uc5
  uc4 -. include .-> uc5
  uc9 -. extend .-> uc7
  uc4 --> provider
```
### 5.4 UML class diagram

Rendered: [PNG](design/diagrams/04_class_diagram.png) - [SVG](design/diagrams/04_class_diagram.svg) - source [`04_class_diagram.mmd`](design/diagrams/04_class_diagram.mmd)

```mermaid
classDiagram
  direction LR
  class Settings {
    +device: str
    +max_upload_mb: int
    +tile_overlap: int
    +default_model: str
    +db_url()
    +resolve(path)
  }
  class AppState {
    +settings
    +db
    +models
    +analyses
    +samples
    +providers
  }
  class Database {
    +engine
    +create_all()
    +session()
    +ping()
  }
  class Analysis {
    +id: str
    +status: str
    +source: str
    +model_key: str
    +threshold: float
    +georeferenced: bool
    +has_ground_truth: bool
    +changed_fraction: float
    +inputs: JSON
    +results: JSON
  }
  class EvaluationRun {
    +id: str
    +dataset: str
    +num_pairs: int
    +threshold: float
    +results: JSON
  }
  class AnalysisService {
    -executor: ThreadPoolExecutor
    +create(t1, t2, truth, model_key, threshold)
    +rerun(source_id, model_key, threshold)
    +run(id)
    +rethreshold(id, threshold)
    +artifact_path(id, name)
    +delete(id)
    +recover_interrupted()
    +cleanup_orphans()
  }
  class ModelManager {
    -loaded: dict
    -infer_lock: Lock
    +device()
    +describe(key)
    +list_models()
    +get(key)
    +predict(key, t1, t2)
    +unload_all()
  }
  class LoadedModel {
    +spec: ModelSpec
    +module: Module
    +info: dict
    +parameters: int
    +mtime: float
  }
  class ModelSpec {
    +key: str
    +display_name: str
    +family: str
    +tile_size: int
    +builder
    +default_checkpoint: str
  }
  class SiameseSwinCD {
    +encoder: SwinV2Tiny
    +fusions: TemporalFusion x4
    +decoder: FPNDecoder
    +head
    +encode(x)
    +forward(t1, t2)
    +param_groups(lr)
  }
  class TemporalFusion {
    +project: Conv1x1
    +refine: Conv3x3
    +forward(f1, f2)
  }
  class FPNDecoder {
    +smooth: Conv x4
    +aggregate
    +forward(feats)
  }
  class FCSiamDiff {
    +enc1..enc4
    +up4..up1
    +head
    +encode(x)
    +forward(t1, t2)
  }
  class LoadedImage {
    +array: uint8 HxWx3
    +format: str
    +geo: GeoInfo
    +notes: list
  }
  class GeoInfo {
    +crs: CRS
    +transform: Affine
    +width: int
    +height: int
    +bounds()
    +resolution()
    +bounds_wgs84()
    +describe()
  }
  class InferenceResult {
    +probability: float32 HxW
    +seconds: float
    +tiles: int
    +device: str
    +amp: bool
  }
  class ConfusionCounts {
    +tp: int
    +fp: int
    +tn: int
    +fn: int
    +update(pred, target)
  }
  class BinaryMetrics {
    +precision
    +recall
    +f1
    +iou
    +accuracy
    +undefined: list
    +to_dict()
  }
  class SatelliteProvider {
    <<abstract>>
    +key: str
    +name: str
    +requires_credentials: bool
    +configured()
    +search(query)
    +retrieve(scene_id, bbox)
    +describe()
  }
  class EarthSearchProvider {
    +base_url: str
    +search(query)
    +retrieve(scene_id, bbox)
  }
  class CDSEProvider {
    +stac_url: str
    +token_url: str
    +process_url: str
    +search(query)
    +retrieve(scene_id, bbox)
  }
  class ProviderRegistry {
    +list()
    +get(key)
    +geocode(text)
  }
  class SceneCandidate {
    +id: str
    +acquired: str
    +cloud_cover: float
    +crs: str
    +aoi_coverage: float
  }
  class SampleService {
    +list()
    +get(id)
    +path(id, role)
  }
  class AppError {
    +code: str
    +message: str
    +status: int
    +details: dict
  }
  AppState o-- Settings
  AppState o-- Database
  AppState o-- ModelManager
  AppState o-- AnalysisService
  AppState o-- SampleService
  AppState o-- ProviderRegistry
  Database "1" --> "*" Analysis
  Database "1" --> "*" EvaluationRun
  AnalysisService --> Database
  AnalysisService --> ModelManager
  AnalysisService ..> LoadedImage
  AnalysisService ..> BinaryMetrics
  ModelManager "1" o-- "*" LoadedModel
  LoadedModel --> ModelSpec
  ModelSpec ..> SiameseSwinCD : builds
  ModelSpec ..> FCSiamDiff : builds
  ModelManager ..> InferenceResult
  SiameseSwinCD *-- TemporalFusion
  SiameseSwinCD *-- FPNDecoder
  LoadedImage o-- GeoInfo
  BinaryMetrics ..> ConfusionCounts
  SatelliteProvider <|-- EarthSearchProvider
  SatelliteProvider <|-- CDSEProvider
  ProviderRegistry o-- SatelliteProvider
  SatelliteProvider ..> SceneCandidate
  SatelliteProvider ..> LoadedImage
  SatelliteProvider ..> AppError
```
### 5.5 UML sequence diagram (upload analysis)

Rendered: [PNG](design/diagrams/05_sequence.png) - [SVG](design/diagrams/05_sequence.svg) - source [`05_sequence.mmd`](design/diagrams/05_sequence.mmd)

```mermaid
sequenceDiagram
  autonumber
  actor U as User
  participant FE as React frontend
  participant API as FastAPI router (analyses)
  participant IMG as imaging.py
  participant AS as AnalysisService
  participant DB as SQLite
  participant W as Worker thread
  participant MM as ModelManager
  participant NN as SiameseSwinCD (GPU)
  participant ART as artifacts.py / geo.py
  U->>FE: choose T1, T2 (+ mask), model, threshold
  FE->>API: POST /api/analyses/upload (multipart)
  API->>IMG: load_image(T1), load_image(T2), load_mask()
  IMG-->>API: LoadedImage x2 (uint8 RGB, optional GeoInfo)
  API->>IMG: align_pair(T1, T2)
  alt incompatible pair
    IMG-->>API: AppError (e.g. IMAGE_DIMENSION_MISMATCH)
    API-->>FE: 422 error envelope
    FE-->>U: error banner + toast
  else valid pair
    API->>AS: create(...)
    AS->>DB: INSERT analysis (status = queued)
    AS->>W: submit(analysis_id)
    API-->>FE: 202 Accepted (id, status)
    FE-->>U: navigate to /results/:id (progress bar)
    W->>DB: UPDATE status = running
    W->>MM: predict(model_key, T1, T2)
    MM->>MM: get() loads checkpoint once (reloads if file changed)
    loop each batch of 256x256 tiles
      MM->>NN: forward(T1 tiles, T2 tiles) under autocast
      NN-->>MM: change logits
    end
    MM-->>W: InferenceResult (probability map, seconds)
    W->>ART: build_outputs(prob, threshold, truth, geo)
    ART-->>W: mask.png, overlay.png, metrics, GeoTIFF / GeoJSON
    W->>DB: UPDATE status = completed, results JSON
    loop every 1 s until terminal state
      FE->>API: GET /api/analyses/ID
      API->>DB: SELECT
      API-->>FE: status
    end
    FE->>API: GET /api/analyses/ID/results
    API-->>FE: results + download links
    FE->>API: GET downloads t1, t2, mask
    FE-->>U: slider, overlay, metrics, downloads
    opt threshold slider moved
      FE->>API: POST /api/analyses/ID/threshold
      API->>ART: build_outputs(stored probability, new threshold)
      API-->>FE: updated results (no new inference)
    end
  end
```
### 5.6 DFD level 0 (context diagram)

Rendered: [PNG](design/diagrams/06_dfd_level0.png) - [SVG](design/diagrams/06_dfd_level0.svg) - source [`06_dfd_level0.mmd`](design/diagrams/06_dfd_level0.mmd)

```mermaid
flowchart LR
  user[/"User (analyst / evaluator)"/]
  provider[/"Satellite imagery provider<br/>(Earth Search / CDSE)"/]
  geocoder[/"Geocoding service<br/>(Nominatim)"/]
  researcher[/"Researcher (training CLI)"/]
  system(("0<br/>TerraVision AI<br/>change-detection system"))
  user -- "image pair, ground-truth mask,<br/>AOI, dates, model, threshold" --> system
  system -- "change mask, overlay, statistics,<br/>metrics, downloads, history, errors" --> user
  system -- "STAC search query, scene request" --> provider
  provider -- "scene metadata, georeferenced imagery" --> system
  system -- "place name" --> geocoder
  geocoder -- "coordinates, bounding box" --> system
  researcher -- "LEVIR-CD dataset, training config" --> system
  system -- "checkpoints, training logs,<br/>benchmark reports" --> researcher
```
### 5.7 DFD level 1

Rendered: [PNG](design/diagrams/07_dfd_level1.png) - [SVG](design/diagrams/07_dfd_level1.svg) - source [`07_dfd_level1.mmd`](design/diagrams/07_dfd_level1.mmd)

```mermaid
flowchart TB
  user[/"User"/]
  provider[/"Satellite provider"/]
  researcher[/"Researcher"/]
  p1(("1.0<br/>Validate and<br/>ingest inputs"))
  p2(("2.0<br/>Acquire satellite<br/>imagery"))
  p3(("3.0<br/>Run change-detection<br/>inference"))
  p4(("4.0<br/>Post-process and<br/>evaluate"))
  p5(("5.0<br/>Serve results<br/>and history"))
  p6(("6.0<br/>Train and benchmark<br/>models"))
  d1[("D1 Analysis database<br/>(SQLite)")]
  d2[("D2 Analysis file store<br/>(storage/analyses)")]
  d3[("D3 Model checkpoints")]
  d4[("D4 Benchmark reports")]
  d5[("D5 Bundled samples")]
  d6[("D6 LEVIR-CD patches")]
  user -- "T1, T2, mask, model, threshold" --> p1
  user -- "AOI, date windows, scene choice" --> p2
  p2 -- "search query / scene id" --> provider
  provider -- "metadata, clipped rasters" --> p2
  p2 -- "georeferenced T1, T2" --> p1
  d5 -- "sample pair" --> p1
  p1 -- "validation errors" --> user
  p1 -- "analysis record (queued)" --> d1
  p1 -- "normalised T1, T2, mask, geo.json" --> d2
  d2 -- "T1, T2" --> p3
  d3 -- "trained weights" --> p3
  p3 -- "probability map" --> p4
  d2 -- "ground truth, georeference" --> p4
  p4 -- "mask, overlay, GeoTIFF, GeoJSON, results.json" --> d2
  p4 -- "status, statistics, metrics" --> d1
  d1 -- "analysis records" --> p5
  d2 -- "artefact files" --> p5
  d4 -- "benchmark summary" --> p5
  p5 -- "results, history, comparison, downloads" --> user
  user -- "new threshold" --> p4
  researcher -- "training configuration" --> p6
  d6 -- "train / val / test patches" --> p6
  p6 -- "best-epoch weights" --> d3
  p6 -- "metrics, confusion matrices, charts" --> d4
```

## 6. Technology choices

| Technology | Why |
|---|---|
| PyTorch + timm | Reference Swin-V2 with ImageNet weights; mixed precision; CUDA 12.8 wheels for the RTX 50 series |
| FastAPI + Pydantic | Typed request / response contracts, automatic OpenAPI docs at /docs |
| SQLAlchemy + SQLite | Zero-administration persistence with a path to PostgreSQL |
| Rasterio / pyproj / shapely | CRS-aware raster I/O, reprojection, vectorisation, geodesic area |
| React 19 + Vite + TypeScript (strict) | Component UI with compile-time checking of API types |
| Tailwind, Recharts, Leaflet | Dark theme, charts from live data, interactive map |
| pytest, Vitest, Playwright | Unit / API, component and real-browser end-to-end tests |

## 7. Design decisions and rejected alternatives

| Decision | Rationale | Alternatives rejected |
|---|---|---|
| Encoder: Swin-V2-Tiny at 256x256 with window 8 | A hierarchical encoder gives the four-scale pyramid a dense-prediction decoder needs, and windowed attention is linear in image size. The V2 'window8_256' weights are pre-trained at exactly the 256x256 tile size used here. | Plain ViT-B/16 (single scale, quadratic attention, weaker boundaries); Swin-V1-T at 224 with window 7 (its window does not divide a 256 px tile's 64x64 token grid, forcing position-bias interpolation); larger Swin variants (do not train comfortably in 8 GB). |
| Decoder: per-scale [F1, F2, \|F1-F2\|] fusion + convolutional FPN | Keeps the transformer where it matters (representation) and uses a small convolutional decoder so the whole model is about 30 M parameters and trains at under 2 GB of VRAM. | A Swin transformer decoder as in SwinSUNet (heavier, slower); pure feature differencing without concatenation (discards which date a feature came from). |
| Loss: BCE + batch-level soft Dice | About 55% of LEVIR-CD tiles contain no change. Dice computed over the whole batch stays informative; the first implementation used per-tile Dice and the CNN collapsed to 'no change' (BUG-002). | Per-tile Dice (rejected after BUG-002); BCE alone (biased towards the 95% negative class); focal loss (extra hyper-parameters, not needed once Dice was fixed). |
| Data: pre-tiled uint8 NumPy memmaps | Decoding the three 1024x1024 PNGs of a pair costs about 172 ms per sample against about 4 ms for a memmapped tile (measured, reports/dataloader_benchmark.json); a single-process loader then suffices, which avoids Windows multiprocessing overhead. | On-the-fly PNG decoding with worker processes; random crops from full images (slower, and tiling must stay inside each official split to avoid leakage anyway). |
| Persistence: SQLite + per-analysis folders | A single-user local prototype needs zero-administration storage; SQLAlchemy keeps the door open for PostgreSQL. Artefacts are files because they are large binary rasters. | PostgreSQL / PostGIS (operational overhead with no Phase-1 benefit); storing images as database BLOBs. |
| Jobs: one in-process worker thread | There is one GPU, so inference must be serialised anyway; a ThreadPoolExecutor(max_workers=1) gives background execution with no extra services and works natively on Windows. | Celery / Redis (extra infrastructure); running inference inside the request (blocks the API and times out on large rasters). |
| Store the probability map, re-threshold on demand | The 8-bit probability map is saved once; changing the threshold recomputes mask, metrics and exports in milliseconds without touching the GPU. | Re-running inference per threshold; thresholding only in the browser (downloads and metrics would then disagree with what is shown). |
| Sentinel-2 models: fine-tune the LEVIR-CD networks on OSCD RGB with TCI-matched radiometry | OSCD has only 11 training cities, so both networks start from their LEVIR-CD weights (building-change features) instead of ImageNet. OSCD's 16-bit bands are scaled with the gain measured on real Earth Search true-colour pixels (DN 2500 -> 255; reports/tci_gain_fit.json) so training data looks like what the app receives. Padded border pixels are excluded from loss and metrics (ignore index 255). | Training on all 13 bands (the app's imagery source delivers RGB only); per-image percentile stretching (would not match the fixed TCI scaling at inference); training from ImageNet only (too little data). |
| Decision threshold chosen on the validation split | Change is 2-3% of OSCD pixels, so 0.5 is not necessarily the best operating point. ml.calibrate sweeps 0.05-0.95 on the validation cities and stores the best threshold in the checkpoint; the app uses it as the default and the user can still override it. | Fixed 0.5 for every model; tuning on the test split (would leak test information). |
| Imagery: Earth Search COGs as the default provider, CDSE optional | Element 84 Earth Search serves Sentinel-2 L2A as public cloud-optimised GeoTIFFs: no account is needed and only the AOI window is read. CDSE is implemented behind the same interface but needs the user's own OAuth client. | Web-map tiles (rendered pictures without dates or radiometry - explicitly rejected); downloading whole Sentinel-2 products (about 1 GB each). |

## 8. Deviations from the original brief

| Deviation | Reason |
|---|---|
| CDSE is not the default imagery provider | The brief preferred the Copernicus Data Space Ecosystem. Retrieval there needs OAuth credentials that were not available, so Earth Search (same Sentinel-2 L2A data, no account) is the verified default and CDSE remains implemented but unverified. |
| CNN baseline uses dropout 0.1 and stopped early | The FC-Siam-diff paper uses dropout 0.2; 0.1 was used because the overfit diagnostic showed 0.2 slows convergence within a 40-epoch budget. Early stopping (patience 10) ended training at epoch 25, so the baseline is weaker than figures reported in the literature for longer schedules. |
| Swin batch size was reduced automatically | The configuration requested batch 8; the conservative VRAM heuristic in train.py reduced it to 5 with gradient accumulation 2 (effective batch 10) because a second training run was sharing the GPU. |
| Sentinel-2 / OSCD work moved from Phase 2 into Phase 1 | The LEVIR-CD models produced 0.00% change on real Sentinel-2 imagery (BUG-006), so OSCD training, a cross-domain evaluation and Sentinel-2 model variants were added in Phase 1. Only the RGB bands are used. |
| Linting uses oxlint instead of ESLint | The current Vite React-TS template ships oxlint with TypeScript 7; it was kept rather than adding an ESLint toolchain whose TypeScript-7 support was not verified. |
| Docker files are provided but untested | Docker is not installed on the development machine; native Windows execution is the supported and verified path. |
| Database migrations are a schema-version table only | With two tables and no released schema to migrate from, Alembic was deferred; create_all plus a schema_version row is used. |

## 9. Requirement traceability

| ID | Requirement | Type | Priority | Design element | Code | Verification |
|---|---|---|---|---|---|---|
| R1 | Accept a bitemporal image pair (PNG, JPEG, GeoTIFF) and an optional ground-truth mask; validate type, integrity, size, channels and pair compatibility before any processing. | Functional | High | imaging.load_image / align_pair; POST /api/analyses/upload; FileDrop component | backend/app/services/imaging.py, backend/app/routers/analyses.py | tests/test_imaging.py, tests/test_api.py |
| R2 | Predict a pixel-level binary change mask with a Siamese Swin Transformer whose encoder and decoder are trained on a change-detection dataset. | Functional | High | SiameseSwinCD (shared Swin-V2-T encoder, TemporalFusion x4, FPNDecoder); predict_pair | ml/models/swin_siamese.py, ml/inference.py | tests/test_models.py; reports/evaluation/swin_siamese_metrics.json |
| R3 | Provide a CNN baseline (FC-Siam-diff) selectable at run time through a model registry. | Functional | High | FCSiamDiff; MODEL_SPECS registry; ModelManager; ModelSelect component | ml/models/fc_siam_diff.py, ml/models/registry.py, backend/app/services/model_manager.py | tests/test_models.py, tests/test_api.py |
| R4 | Compute TP, FP, TN, FN, precision, recall, F1, IoU, pixel accuracy and a confusion matrix, only when ground truth is available; guard zero denominators. | Functional | High | ConfusionCounts / compute_metrics; artifacts.build_outputs; MetricsCard | ml/metrics.py, backend/app/services/artifacts.py | tests/test_metrics.py (cross-checked with scikit-learn) |
| R5 | Visualise before / after images, a comparison slider, the change mask, a coloured overlay and a confidence map; allow threshold and overlay-opacity adjustment. | Functional | High | Results page (CompareSlider, maskOverlay); POST /api/analyses/{id}/threshold (re-threshold stored probability map) | frontend/src/pages/Results.tsx, backend/app/services/analysis.py | frontend/src/test/app.test.tsx, frontend/e2e/demo.spec.ts, tests/test_api.py |
| R6 | Download the mask, overlay and results (PNG / JSON), plus a GeoTIFF mask and GeoJSON regions when the input is georeferenced. | Functional | Medium | ARTIFACTS whitelist; GET /api/analyses/{id}/downloads/{artifact}; geo.write_geotiff / change_regions | backend/app/services/artifacts.py, backend/app/services/geo.py | tests/test_api.py, tests/test_geo.py |
| R7 | Persist every analysis with status, model, threshold and results; show history and live overview statistics. | Functional | Medium | SQLite via SQLAlchemy (Analysis, EvaluationRun); /api/analyses, /api/analyses/stats; Overview and History pages | backend/app/db.py, frontend/src/pages/History.tsx | tests/test_api.py (persistence, restart recovery) |
| R8 | Compare the transformer and the CNN on the same held-out split and present the measured results in a dashboard. | Functional | High | ml.evaluate (test split, fixed threshold); GET /api/evaluation/summary; Compare page (Recharts) | ml/evaluate.py, backend/app/routers/evaluation.py, frontend/src/pages/Compare.tsx | tests/test_data_training.py, tests/test_api.py; reports/evaluation/ |
| R9 | Handle GeoTIFF input: validate CRS, extract bounds and resolution, reproject / resample T2 onto the T1 grid, export a georeferenced mask and display results on a map. | Functional | Medium | GeoInfo; align_pair (rasterio.warp.reproject); web_mercator_overlay; GeoMap component | backend/app/services/imaging.py, backend/app/services/geo.py, frontend/src/components/maps.tsx | tests/test_imaging.py, tests/test_geo.py, frontend/e2e/demo.spec.ts |
| R10 | Map-based acquisition: search a place, draw an area of interest, choose two date windows, list candidate scenes with cloud cover, retrieve georeferenced imagery and run detection. | Functional | Medium | SatelliteProvider abstraction (EarthSearchProvider, CDSEProvider); /api/satellite/*; Map page | backend/app/satellite/, frontend/src/pages/MapPage.tsx | tests/test_satellite.py, frontend/e2e/demo.spec.ts |
| R11 | Reproducible experiments: verified dataset download, leakage-free tiling with a split manifest, configurable training with seeds, machine-readable logs; the test split is never used for training or model selection. | Functional | High | ml.data.download (MD5), ml.data.prepare (manifest, leakage check), ml.train (config.json, history.jsonl) | ml/data/, ml/train.py, ml/configs/ | tests/test_data_training.py; experiments/*/ |
| R12 | Every failure returns a consistent, machine-readable error; background failures are recorded on the analysis; the UI shows loading, empty and error states. | Non-functional | High | AppError + error envelope handlers; AnalysisService._run_safely; ErrorBanner / EmptyState / toasts | backend/app/errors.py, frontend/src/components/ui.tsx | tests/test_api.py, frontend/src/test/app.test.tsx |
| R13 | Run on an 8 GB consumer GPU with mixed precision and fall back to CPU automatically; load each model once and serialise GPU access. | Non-functional | Medium | ModelManager (load once, inference lock, OOM retry, CPU fallback); autocast; sliding window | backend/app/services/model_manager.py, ml/inference.py | tests/test_models.py, tests/test_api.py; benchmark latency |
| R14 | Safe file handling for a local prototype: upload size limits, magic-byte checks, server-generated identifiers, whitelisted artefact names, no filesystem paths in responses, restricted CORS. | Non-functional | High | detect_format; _read_limited; AnalysisService.folder id check; CORS middleware | backend/app/services/imaging.py, backend/app/routers/analyses.py, backend/app/main.py | tests/test_api.py |
| R15 | Trustworthy output: never serve an untrained or mismatched checkpoint, show the checkpoint version with every result, never fabricate coordinates, areas or metrics. | Non-functional | High | ml.checkpoint validation ('trained' flag, strict load); checkpoint version + SHA-256; geo only when present | ml/checkpoint.py, backend/app/services/artifacts.py | tests/test_models.py, tests/test_api.py |
| R16 | Responsive, accessible dark dashboard with eight pages; every control performs a real action. | Non-functional | Medium | Layout (responsive sidebar), Tailwind theme, ARIA roles/labels on controls | frontend/src/components/Layout.tsx, frontend/src/pages/ | frontend/src/test/app.test.tsx, frontend/e2e/demo.spec.ts |
| R17 | Automated test suite covering unit, API, integration, boundary and negative cases, with recorded execution logs. | Non-functional | High | pytest + HTTPX TestClient, Vitest + Testing Library, Playwright | tests/, frontend/src/test/, frontend/e2e/ | evidence/tests/ |
| R18 | Detect change on real Sentinel-2 (10 m) imagery: provide models trained on Sentinel-2 data, choose the model that matches the imagery automatically, and flag any model / imagery domain mismatch. | Functional | High | swin_siamese_s2 / fc_siam_diff_s2 (OSCD fine-tune); preferred_model + domain_warning; ml.data.oscd; ml.calibrate | ml/data/oscd.py, ml/models/registry.py, backend/app/services/model_manager.py | tests/test_api.py (domain selection), tests/test_data_training.py (OSCD), e2e 08-09; reports/evaluation/*_s2_metrics.json |

## 10. Security measures (local research prototype)

Present: extension + magic-byte + decode validation, size limits, server-generated ids, whitelisted artefact
names, no filesystem paths in responses, CORS restricted to the frontend origin, asset-host allow-list for
satellite reads, safe weights-only checkpoint loading. Not present: authentication, rate limiting, HTTPS -
this is not a production deployment.
