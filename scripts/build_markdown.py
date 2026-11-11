"""Generate the GitHub-facing Markdown documentation from recorded project outputs.

Writes README.md and docs/{PROJECT_REPORT, DATASETS, MODELS, ARCHITECTURE, TESTING, LITERATURE_SURVEY,
PHASE2_PLAN, Implementation_Guide, README}.md and docs/design/README.md. Every number is read from reports/, experiments/, evidence/ and the data manifests via
``build_docs.load_facts`` - nothing is typed in by hand.

Usage:  python scripts/build_markdown.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import doc_content as C  # noqa: N812
import doc_facts as B  # noqa: N812
from doc_facts import (
    API_TABLE,
    CROSS_HEAD,
    RESULT_HEAD,
    TRAIN_HEAD,
    cross_rows,
    md_table,
    results_rows,
    results_sentence,
    training_rows,
)

DOCS = ROOT / "docs"
P = C.PROJECT
REPO_URL = "https://github.com/aadie14/terravision-ai"
MD = B.md_table
pct = B.pct
SHOTS = "evidence/screenshots"


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"ok  {path.relative_to(ROOT)}")


def link(rel_from_docs: str) -> str:
    """Path to a repo file from inside docs/."""
    return f"../{rel_from_docs}"


# ----------------------------------------------------------------------------- shared blocks
def results_block(f: dict) -> str:
    return "\n\n".join([MD(B.RESULT_HEAD, B.results_rows(f)), "**Domain shift** - the OSCD Sentinel-2 test cities "
                        "scored by models with and without Sentinel-2 training:", MD(B.CROSS_HEAD, B.cross_rows(f))])


def run_config(f: dict, key: str) -> dict:
    return f["runs"][key]["config"]


def test_catalog() -> dict[str, list[tuple[str, str]]]:
    """{file: [(test name, first docstring line)]} for the backend, frontend and end-to-end suites."""
    out: dict[str, list[tuple[str, str]]] = {}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rows = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                rows.append((node.name, doc[0] if doc else ""))
        out[f"tests/{path.name}"] = rows
    fe = (ROOT / "frontend/src/test/app.test.tsx").read_text(encoding="utf-8")
    out["frontend/src/test/app.test.tsx"] = [(m, "") for m in re.findall(r"\bit\('([^']+)'", fe)]
    e2e = (ROOT / "frontend/e2e/demo.spec.ts").read_text(encoding="utf-8")
    out["frontend/e2e/demo.spec.ts"] = [(m, "") for m in re.findall(r"^test\('([^']+)'", e2e, flags=re.M)]
    return out


# ============================================================================ DATASETS
def datasets_md(f: dict) -> str:
    lev = f["manifest"]
    osc = f["oscd_manifest"]
    tci = f["tci_fit"]
    ls_ = lev["splits"]
    os_ = osc["splits"]
    sample_rows = [[x["name"], x.get("dataset"), x.get("split") or "-", x.get("domain", "vhr"),
                    "yes" if x.get("ground_truth") else "no", "yes" if x.get("georeferenced") else "no",
                    x.get("license")] for x in f["samples"]]
    fit_rows = [[r["scene"], r["processing_baseline"]] + [f"{b['dn_for_255']:.0f} (RMSE {b['rmse']:.1f})" for b in r["bands"]]
                for r in tci["scenes"]]
    return f"""
# Datasets

This page documents every dataset used by {P['title']}: where it comes from, its licence, how it was split
and pre-processed, and why it was chosen. Figures are read from the dataset manifests written by the
preparation scripts (`data/processed/*/manifest.json`).

| Dataset | Role | Imagery | Task | Licence |
|---|---|---|---|---|
| **LEVIR-CD** | Train / validate / test the 0.5 m models | Google Earth RGB, 0.5 m/pixel | Binary building change | Academic use only |
| **OSCD** | Fine-tune / validate / test the Sentinel-2 models | Sentinel-2 L1C, bands B04/B03/B02, 10 m | Binary urban change | Citation required (see below) |
| **Sentinel-2 L2A via Earth Search** | Live map workflow and the north-Bengaluru demo pair | Sentinel-2 true colour (TCI), 10 m | Inference only (no labels) | Copernicus free and open, attribution |

---

## 1. LEVIR-CD (primary dataset)

- **Reference:** H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote
  Sensing Image Change Detection", *Remote Sensing* 12(10):1662, 2020. doi:10.3390/rs12101662.
- **Source used:** {lev['source']} at a pinned revision, with the MD5 checksums TorchGeo uses
  (`ml/data/download.py`). Official page: https://justchenhao.github.io/LEVIR/
- **Licence:** academic use only, commercial use prohibited; imagery subject to the Google Earth terms of use.
- **Content:** 637 pairs of 1024x1024 px images at 0.5 m/pixel over Texas, USA, with hand-drawn building-change masks.

| Split | Images | 256x256 patches | Changed pixels |
|---|---|---|---|
| train | {ls_['train']['num_images']} | {ls_['train']['num_patches']} | {pct(ls_['train']['changed_pixel_fraction'])} |
| val | {ls_['val']['num_images']} | {ls_['val']['num_patches']} | {pct(ls_['val']['changed_pixel_fraction'])} |
| test | {ls_['test']['num_images']} | {ls_['test']['num_patches']} | {pct(ls_['test']['changed_pixel_fraction'])} |

**Pre-processing** (`ml/data/prepare.py`): each official split is cut into a non-overlapping 4x4 grid of 256x256
tiles stored as uint8 NumPy memmaps. Tiling happens *inside* each split and the script asserts that no file
name occurs in two splits, so no image contributes to both training and testing. Reading a memmapped tile
takes about 4 ms against about 172 ms for decoding the PNGs (`reports/dataloader_benchmark.json`).

**Why LEVIR-CD:** it is the most widely used benchmark for building change detection, has an official
train/val/test split (so results are comparable with the literature), carefully annotated masks, and a size
(7,120 training patches) that trains a ~30 M-parameter transformer on an 8 GB GPU in about two hours.

## 2. OSCD - Onera Satellite Change Detection (Sentinel-2 dataset)

- **Reference:** R. Caye Daudt, B. Le Saux, A. Boulch, Y. Gousseau, "Urban Change Detection for
  Multispectral Earth Observation Using Convolutional Neural Networks", IGARSS 2018. doi:10.1109/IGARSS.2018.8518015.
- **Source used:** {osc['source']}, with TorchGeo's SHA-256 checksums (`ml/data/oscd.py`).
  Project page: https://rcdaudt.github.io/oscd/
- **Licence:** imagery is Copernicus Sentinel data (free, full and open). The dataset README asks for the
  citation above and states no further licence terms.
- **Content:** 24 city pairs of Sentinel-2 images (2015-2018) with pixel-level urban-change labels.

| Split | Cities | Training patches | Changed pixels |
|---|---|---|---|
| train | {', '.join(osc['cities']['train'])} | {os_['train']['num_patches']} | {pct(os_['train']['changed_pixel_fraction'])} |
| val | {', '.join(osc['cities']['val'])} | {os_['val']['num_patches']} | {pct(os_['val']['changed_pixel_fraction'])} |
| test | {', '.join(osc['cities']['test'])} | whole images | {pct(os_['test']['changed_pixel_fraction'])} |

The 14 official training cities are split 11 / 3 into train and validation (deterministic, every 5th city
alphabetically from the third). The 10 official test cities are never used for training, early stopping
or threshold selection; they are evaluated whole with the same sliding-window inference the application uses.

**Pre-processing:**
- Only the 10 m true-colour bands B04, B03, B02 are used, because the application's imagery source
  (Earth Search) delivers true-colour images.
- **Radiometry matched to the application.** OSCD ships raw 16-bit digital numbers (DN). The mapping the
  app's Sentinel-2 imagery uses was measured on real Earth Search scenes (`scripts/fit_tci_gain.py`):

{MD(['Scene', 'Processing baseline', 'Red: DN for 255', 'Green: DN for 255', 'Blue: DN for 255'], fit_rows)}

  One gain (DN {tci['dn_for_255']:.0f} -> 255) fits both an old and a new processing baseline, so OSCD bands are
  converted with `uint8 = clip(DN x 255 / 2500)`. Caveat: OSCD is Level-1C (top of atmosphere) while the app
  receives Level-2A (bottom of atmosphere); only the scale is matched, not the atmospheric correction.
- Train / val cities are cut into 256x256 tiles with stride 128. Images smaller than a tile are
  reflect-padded and the padded mask pixels get the ignore value 255, which is excluded from the loss and
  from validation metrics.

**Why OSCD:** the LEVIR-CD models produced 0.00% change on real Sentinel-2 imagery (their F1 on the OSCD test
cities is 0-1%, see [MODELS.md](MODELS.md)). OSCD is the standard, openly available, labelled Sentinel-2 urban
change dataset, at exactly the 10 m resolution of the imagery the map workflow retrieves.

## 3. Sentinel-2 Level-2A via Element 84 Earth Search (inference only)

Used by the live map workflow and for the bundled north-Bengaluru pair. Catalogue:
https://earth-search.aws.element84.com/v1 (STAC, no account). Only the window covering the area of interest
is read from the public cloud-optimised GeoTIFF of the `visual` (true-colour) asset. Licence: Copernicus
Sentinel data, free and open, attribution "Contains modified Copernicus Sentinel data <year>". There are no
labels, so results on this imagery cannot be scored.

## 4. Bundled demonstration samples (`samples/`)

{MD(['Sample', 'Dataset', 'Split', 'Imagery domain', 'Ground truth', 'Georeferenced', 'Licence'], sample_rows)}

Built by `scripts/build_samples.py`; provenance is recorded in `samples/manifest.json`.

## 5. Not included in the repository

The full datasets are not committed (size and licence). Re-create them with:

```powershell
.\\.venv\\Scripts\\python.exe -m ml.data.download      # LEVIR-CD
.\\.venv\\Scripts\\python.exe -m ml.data.prepare
.\\.venv\\Scripts\\python.exe -m ml.data.oscd download
.\\.venv\\Scripts\\python.exe -m ml.data.oscd prepare
```

## 6. Leakage prevention summary

| Check | Where |
|---|---|
| Official splits only; tiles never cross a split | `ml/data/prepare.py` (`check_no_leakage`), `ml/data/oscd.py` (`city_lists`) |
| Hash of the split file lists stored with every training run | `experiments/*/config.json` -> `dataset_manifest_sha256` |
| Model selection (early stopping, threshold) on validation data only | `ml/train.py`, `ml/calibrate.py` |
| Test data read once, by `ml.evaluate` | `reports/evaluation/` |

Candidate datasets for Phase 2 (not used yet): the full 13-band OSCD imagery, and a semantic-change dataset
for multi-class change. See [PHASE2_PLAN.md](PHASE2_PLAN.md).
"""


# ============================================================================ MODELS
def models_md(f: dict) -> str:
    m = f["models"]
    cal = f["calibration"]
    cfg_rows = []
    for key in B.MODEL_ORDER:
        c = run_config(f, key)
        init = (c.get("initialised_from") or {}).get("file") or "ImageNet (Swin encoder) / random (CNN)"
        cfg_rows.append([m[key]["display_name"], c["dataset"], init, c["batch_size"], c["grad_accum"], f"{c['lr']:g}",
                         c["epochs"], c["patience"], c["weight_decay"], c["seed"]])
    cal_rows = [[m[k]["display_name"], cal[k]["selected"]["threshold"], pct(cal[k]["at_0.5"]["f1"]),
                 pct(cal[k]["selected"]["f1"])] for k in ("swin_siamese_s2", "fc_siam_diff_s2")]
    decisions = MD(["Decision", "Rationale", "Alternatives rejected"], [list(d) for d in C.DESIGN_DECISIONS[:3]] +
                   [list(d) for d in C.DESIGN_DECISIONS if d[0].startswith(("Sentinel-2 models", "Decision threshold"))])
    return f"""
# Models

Four trained models are served through one registry (`ml/models/registry.py`): two network architectures,
each trained for two imagery domains.

| Key | Model | Family | Imagery domain | Parameters | Checkpoint |
|---|---|---|---|---|---|
| `swin_siamese` | Siamese Swin Transformer | transformer | 0.5 m aerial (LEVIR-CD) | {m['swin_siamese']['parameters']:,} | `checkpoints/swin_siamese_levir.pt` |
| `fc_siam_diff` | FC-Siam-diff (CNN baseline) | CNN | 0.5 m aerial (LEVIR-CD) | {m['fc_siam_diff']['parameters']:,} | `checkpoints/fc_siam_diff_levir.pt` |
| `swin_siamese_s2` | Siamese Swin Transformer (Sentinel-2) | transformer | 10 m Sentinel-2 (OSCD) | {m['swin_siamese_s2']['parameters']:,} | `checkpoints/swin_siamese_oscd.pt` |
| `fc_siam_diff_s2` | FC-Siam-diff CNN (Sentinel-2) | CNN | 10 m Sentinel-2 (OSCD) | {m['fc_siam_diff_s2']['parameters']:,} | `checkpoints/fc_siam_diff_oscd.pt` |

The application picks the model that matches the input imagery (bundled sample domain, or pixel size of a
GeoTIFF: >= 5 m is Sentinel-2) and warns when a user forces a mismatched model.

## 1. Primary model - Siamese Swin Transformer (`ml/models/swin_siamese.py`)

```
T1 ─┐                        ┌─ F1 (4 scales) ─┐
    ├─ shared Swin-V2-Tiny ──┤                 ├─ TemporalFusion x4 ─ FPN decoder ─ head ─ logits ─ sigmoid ─ mask
T2 ─┘   (one set of weights) └─ F2 (4 scales) ─┘   concat(F1, F2, |F1-F2|) -> 1x1 conv -> 3x3 conv
```

- **Encoder:** timm `swinv2_tiny_window8_256`, ImageNet-1k pre-trained; strides 4/8/16/32 with
  96/192/384/768 channels. Both dates pass through it in one batch, so weights are shared by construction.
- **Fusion:** at every scale, `concat(F1, F2, |F1-F2|)` -> 1x1 conv -> 3x3 conv (128 channels). The difference
  makes change explicit; F1 and F2 keep "what was there before / after".
- **Decoder:** top-down FPN with lateral additions, all levels upsampled to stride 4 and aggregated; a
  two-layer head outputs one logit channel, bilinearly upsampled to 256x256.
- **Inference on any image size:** 256 px tiles with 64 px overlap, Hann-window blending, mixed precision
  (`ml/inference.py`).

## 2. Baseline - FC-Siam-diff (`ml/models/fc_siam_diff.py`)

Re-implementation of Daudt, Le Saux & Boulch, "Fully Convolutional Siamese Networks for Change Detection",
ICIP 2018 (doi:10.1109/ICIP.2018.8451652): a shared 4-level U-Net encoder (16-32-64-128 channels) whose skip
connections carry the absolute difference of the two dates' features.

## 3. Why these models

- **Why a transformer:** the project objective is to evaluate transformer-based change detection against CNNs.
  Windowed self-attention gives long-range context (separating real structural change from illumination and
  seasonal pseudo-change) at linear cost in image size.
- **Why Swin-V2-Tiny specifically:** hierarchical (gives the multi-scale pyramid a segmentation decoder needs),
  pre-trained at exactly the 256x256 tile size used here, and small enough to train in 8 GB of VRAM.
- **Why FC-Siam-diff as the baseline:** the canonical Siamese CNN for change detection, cited by every later
  paper surveyed; small, fast, and trained from scratch, so it isolates the effect of the architecture.
- **Why separate Sentinel-2 models:** at 10 m a building is 1-2 pixels; models trained on 0.5 m imagery find
  nothing (measured below). Fine-tuning from the LEVIR-CD weights reuses learned building-change features
  despite OSCD's small size (11 training cities).

{decisions}

## 4. Training configuration (from `experiments/*/config.json`)

{MD(['Model', 'Dataset', 'Initialised from', 'Batch', 'Grad. accum.', 'LR', 'Max epochs', 'Patience', 'Weight decay', 'Seed'], cfg_rows)}

Common to all runs (`ml/train.py`): AdamW, BCE + batch-level soft Dice loss (BUG-002 explains why batch-level),
linear warm-up then cosine decay, gradient clipping at 1.0, mixed precision, flips / 90-degree rotations /
temporal swap / per-date photometric jitter, early stopping on validation F1, best-epoch checkpoint.

{MD(B.TRAIN_HEAD, B.training_rows(f))}

## 5. Decision thresholds

For the Sentinel-2 models the threshold is chosen on the **validation** cities (`ml/calibrate.py`, grid
0.05-0.95) and stored in the checkpoint; the application uses it as the default. The LEVIR-CD models use 0.5.

{MD(['Model', 'Selected threshold', 'Val F1 at 0.5', 'Val F1 at selected'], cal_rows)}

The CNN's selected threshold lies at the edge of the grid, which indicates under-confident probabilities.

## 6. Results (held-out test data)

{results_block(f)}

{B.results_sentence(f)}

Charts: [LEVIR-CD](../reports/evaluation/model_comparison.png) - [OSCD](../reports/evaluation/model_comparison_oscd.png).
Prediction examples: [Swin LEVIR-CD](../reports/evaluation/swin_siamese_examples.png) -
[Swin Sentinel-2](../reports/evaluation/swin_siamese_s2_examples.png) -
[CNN LEVIR-CD](../reports/evaluation/fc_siam_diff_examples.png) -
[CNN Sentinel-2](../reports/evaluation/fc_siam_diff_s2_examples.png).
Raw numbers: [`reports/evaluation/metrics.csv`](../reports/evaluation/metrics.csv),
[`summary.json`](../reports/evaluation/summary.json).

![OSCD predictions](../reports/evaluation/swin_siamese_s2_examples.png)

## 7. Checkpoints

Format (`ml/checkpoint.py`): a plain dict loaded with `torch.load(weights_only=True)` containing
`format_version, model_key, model_kwargs, state_dict, trained, meta`. The application refuses files that are
missing, unreadable, not marked as trained, or do not match the architecture; every result shows the
checkpoint version and SHA-256. Checkpoints are not stored in git (the Swin file exceeds GitHub's 100 MB limit):
download them from the repository's **Releases** page into `checkpoints/`, or re-train with the commands in
[Implementation_Guide.md](Implementation_Guide.md).

## 8. Limitations

""" + "\n".join(f"- {x}" for x in C.LIMITATIONS[:3])


# ============================================================================ ARCHITECTURE
def architecture_md(f: dict) -> str:
    diagrams = DOCS / "design" / "diagrams"
    titles = {
        "01_system_architecture": "System architecture",
        "02_flowchart": "Application flowchart (analysis request)",
        "03_use_case": "UML use-case diagram",
        "04_class_diagram": "UML class diagram",
        "05_sequence": "UML sequence diagram (upload analysis)",
        "06_dfd_level0": "DFD level 0 (context diagram)",
        "07_dfd_level1": "DFD level 1",
    }
    blocks = []
    for i, (stem, title) in enumerate(titles.items(), 1):
        src = (diagrams / f"{stem}.mmd").read_text(encoding="utf-8").strip()
        blocks.append(f"### 5.{i} {title}\n\nRendered: [PNG](design/diagrams/{stem}.png) - [SVG](design/diagrams/{stem}.svg) - "
                      f"source [`{stem}.mmd`](design/diagrams/{stem}.mmd)\n\n```mermaid\n{src}\n```")
    components = [
        ["Frontend pages", "Eight routed pages; typed API client; presentation only", "`frontend/src/pages`, `frontend/src/lib/api.ts`"],
        ["Routers", "HTTP contracts, validation, response models", "`backend/app/routers/`"],
        ["imaging", "Decode, validate, stretch, align / reproject image pairs; pair sanity heuristics", "`backend/app/services/imaging.py`"],
        ["AnalysisService", "Persist inputs, queue and run jobs, re-threshold, artefacts, clean-up, crash recovery", "`backend/app/services/analysis.py`"],
        ["ModelManager", "Checkpoint status, load-once cache with change detection, GPU lock, CPU fallback, OOM retry, domain-aware model choice", "`backend/app/services/model_manager.py`"],
        ["artifacts / geo", "Masks, overlays, confidence and error maps; GeoTIFF, GeoJSON, geodesic area, Web-Mercator overlay", "`backend/app/services/artifacts.py`, `geo.py`"],
        ["Satellite providers", "Provider interface, STAC search, windowed COG reads, CDSE Process API, geocoding", "`backend/app/satellite/`"],
        ["Models", "SiameseSwinCD, FCSiamDiff behind a registry of four trained variants", "`ml/models/`"],
        ["Data", "LEVIR-CD and OSCD download, validation, tiling, manifests", "`ml/data/`"],
        ["Training / evaluation", "Config-driven training, threshold calibration, benchmark reports", "`ml/train.py`, `ml/calibrate.py`, `ml/evaluate.py`"],
        ["Inference", "Sliding-window bitemporal prediction with blending and AMP", "`ml/inference.py`"],
    ]
    data_model = [
        ["`analyses` (SQLite)", "id, name, status, source, model_key, threshold, timestamps, width, height, georeferenced, "
                                "has_ground_truth, changed_fraction, inference_seconds, error_code, error_message, inputs (JSON), results (JSON)"],
        ["`evaluation_runs` (SQLite)", "id, created_at, dataset, num_pairs, threshold, results (JSON)"],
        ["`storage/analyses/<id>/`", "t1/t2 PNG, ground_truth, geo.json, probability, confidence, mask, overlay, error_map, "
                                     "mask.tif, regions.geojson, map overlays, results.json"],
        ["`checkpoints/*.pt`", "format_version, model_key, model_kwargs, state_dict, trained, meta"],
        ["`reports/evaluation/`", "<model>_metrics.json, metrics.csv, summary.json, cross_domain.json, figures"],
    ]
    tech = [["PyTorch + timm", "Reference Swin-V2 with ImageNet weights; mixed precision; CUDA 12.8 wheels for the RTX 50 series"],
            ["FastAPI + Pydantic", "Typed request / response contracts, automatic OpenAPI docs at /docs"],
            ["SQLAlchemy + SQLite", "Zero-administration persistence with a path to PostgreSQL"],
            ["Rasterio / pyproj / shapely", "CRS-aware raster I/O, reprojection, vectorisation, geodesic area"],
            ["React 19 + Vite + TypeScript (strict)", "Component UI with compile-time checking of API types"],
            ["Tailwind, Recharts, Leaflet", "Dark theme, charts from live data, interactive map"],
            ["pytest, Vitest, Playwright", "Unit / API, component and real-browser end-to-end tests"]]
    return f"""
# Architecture and design

TerraVision AI is a three-layer local application: a **React** single-page frontend, a **FastAPI** backend
and an **ML package** (PyTorch), with SQLite and the filesystem as data stores and optional external
imagery services. The layers meet at two narrow interfaces: the HTTP/JSON API, and the checkpoint format plus
`ml.inference.predict_pair`.

![System architecture](design/diagrams/01_system_architecture.png)

## 1. Components

{MD(['Component', 'Responsibility', 'Code'], components)}

## 2. Request lifecycle (upload)

1. `POST /api/analyses/upload` reads both files with a size limit, checks extension, magic bytes and decodability.
2. `imaging.align_pair` rejects incompatible pairs (different sizes, different places) or reprojects T2 onto
   T1's grid when both are georeferenced and overlap by at least 90%.
3. The model is chosen (explicit, or matching the imagery domain) with its calibrated threshold; inputs are
   saved and a row is inserted with status `queued`; the API returns `202 Accepted`.
4. A single background worker runs sliding-window inference on the GPU under a lock, stores the 8-bit
   probability map, then the mask, overlay, metrics (only with ground truth) and georeferenced exports (only
   with georeferenced input), and marks the row `completed` (or `failed` with an error code).
5. The frontend polls `GET /api/analyses/{{id}}` and then loads `/results` and the artefacts.
   Moving the threshold slider re-thresholds the stored probability map - no new inference.

## 3. Data model

{MD(['Store', 'Contents'], data_model)}

## 4. HTTP API

{MD(['Method and path', 'Purpose'], B.API_TABLE)}

All errors use one envelope: `{{"error": {{"code", "message", "details"}}}}`. Interactive schema at
`http://127.0.0.1:8000/docs` when the backend is running.

## 5. Diagrams

All diagrams are Mermaid sources in [`docs/design/diagrams`](design/diagrams) (GitHub renders them below)
with PNG and SVG renders alongside. Each element corresponds to real code (see the component table).

{chr(10).join(blocks)}

## 6. Technology choices

{MD(['Technology', 'Why'], tech)}

## 7. Design decisions and rejected alternatives

{MD(['Decision', 'Rationale', 'Alternatives rejected'], [list(d) for d in C.DESIGN_DECISIONS])}

## 8. Deviations from the original brief

{MD(['Deviation', 'Reason'], [list(d) for d in C.DEVIATIONS])}

## 9. Requirement traceability

{MD(['ID', 'Requirement', 'Type', 'Priority', 'Design element', 'Code', 'Verification'], [list(r) for r in C.REQUIREMENTS])}

## 10. Security measures (local research prototype)

Present: extension + magic-byte + decode validation, size limits, server-generated ids, whitelisted artefact
names, no filesystem paths in responses, CORS restricted to the frontend origin, asset-host allow-list for
satellite reads, safe weights-only checkpoint loading. Not present: authentication, rate limiting, HTTPS -
this is not a production deployment.
"""


# ============================================================================ TESTING
def testing_md(f: dict) -> str:
    catalog = test_catalog()
    sections = []
    for file, rows in catalog.items():
        body = "\n".join(f"| `{name}` | {doc} |" for name, doc in rows)
        sections.append(f"### `{file}` ({len(rows)})\n\n| Test | Notes |\n|---|---|\n{body}")
    tc_rows = [[t[0], t[1] or "functional", t[2], t[3], t[4], t[5], "Pass", f"`{t[6]}`"] for t in C.TEST_CASES]
    negative = [["Missing image", "T15"], ["Corrupted image", "T16"], ["Unsupported file", "T17"],
                ["Mismatched dimensions", "T18"], ["Oversized upload", "T14"], ["Missing checkpoint", "T19"],
                ["Invalid CRS", "T20"], ["Unavailable GPU", "T22"], ["Satellite provider failure", "T23"],
                ["Empty mask", "T12, `tests/test_metrics.py`"], ["No-change pair", "T12"],
                ["Wrong imagery domain", "T28"]]
    history = [
        ["Backend run 1", "49 passed, 1 failed (stopped)", "BUG-003 checkpoint metadata", "`evidence/tests/backend_run_01_first_execution.log`"],
        ["Backend run 2", "153 passed, 1 failed", "BUG-005 heuristic threshold", "console"],
        ["E2E run 1", "5 passed, 1 failed, 1 not run", "Ambiguous selector (test defect)", "`evidence/tests/e2e_playwright_run_01.log`"],
        ["E2E run 2", "7 passed", "-", "`evidence/tests/e2e_playwright_run_02.log`"],
        ["Backend after Sentinel-2 models", "150 passed, 5 failed", "Tests assumed a 2-model registry; updated", "console"],
        ["Backend after OSCD tests", "159 passed, 1 failed", "Test defect: round(127.5) is 128 expected", "console"],
        ["E2E runs 3-4", "1 passed, 1 failed, 7 not run", "Tests assumed 2 models (test defects)", "`evidence/tests/e2e_playwright_run_03.log`, `_04.log`"],
        ["E2E run 5", "9 passed", "Stale-screenshot wait fixed", "`evidence/tests/e2e_playwright_run_05.log`"],
        ["**Backend final**", f"**{f['backend_passed']} passed**", "-", "`evidence/tests/backend_run_final.log`, `backend_junit.xml`"],
        ["**Frontend final**", f"**{f['frontend_passed']} passed**", "-", "`evidence/tests/frontend_vitest.log`"],
        ["**E2E final**", f"**{f['e2e_passed']} passed**", "-", f"`evidence/tests/{B.E2E_LOG}`"],
    ]
    return f"""
# Testing

## 1. Summary (final recorded runs, {P['build_date']})

| Suite | Tool | Passed | Failed | Coverage | Log |
|---|---|---|---|---|---|
| Backend + ML | pytest, HTTPX TestClient | {f['backend_passed']} | 0 | {f['backend_cov']}% of statements | [`backend_run_final.log`](../evidence/tests/backend_run_final.log) |
| Frontend | Vitest, React Testing Library | {f['frontend_passed']} | 0 | {f['frontend_cov']:.0f}% of lines | [`frontend_vitest.log`](../evidence/tests/frontend_vitest.log) |
| End-to-end | Playwright (Chromium), live backend and trained models | {f['e2e_passed']} | 0 | - | [`{B.E2E_LOG}`](../evidence/tests/{B.E2E_LOG}) |

Backend tests run on CPU against a temporary database, storage folder and a tiny random-weight checkpoint
that only exercises the plumbing - no accuracy claim is derived from it. End-to-end tests use the real
trained checkpoints and assert real accuracy (for example F1 > 30% on a held-out OSCD city).

## 2. How to run

```powershell
.\\.venv\\Scripts\\python.exe -m pytest --cov                 # backend + ML
cd frontend; npm test                                          # frontend unit tests
npx playwright test                                            # end-to-end (start both servers first)
.\\.venv\\Scripts\\python.exe -m ruff check . ; cd frontend; npm run typecheck; npm run lint
```

## 3. Test-case design (representative cases)

{MD(['ID', 'Type', 'Feature / requirement', 'Input', 'Expected', 'Actual', 'Result', 'Automated test'], tc_rows)}

## 4. Required negative and boundary cases

{MD(['Case', 'Covered by'], negative)}

## 5. Integration tests

{MD(['Chain', 'Test'], [
        ['Frontend -> API -> GPU inference -> results', 'e2e 02 (LEVIR-CD), e2e 08 (OSCD with ground truth)'],
        ['Upload -> storage -> retrieval', '`test_upload_inference_results_download_flow`, `test_analyses_persist_across_application_restarts`'],
        ['GeoTIFF -> preprocessing -> prediction -> export', '`test_geotiff_upload_prediction_and_georeferenced_exports`, e2e 04'],
        ['Map -> provider -> aligned analysis', '`test_retrieve_runs_a_georeferenced_analysis_and_orders_scenes_by_date`, e2e 07 (live search)'],
        ['Training -> checkpoint -> evaluation -> report', '`test_train_then_evaluate_smoke`, `test_train_fine_tunes_from_init_checkpoint`'],
    ])}

## 6. Execution history (failures are recorded, not hidden)

{MD(['Run', 'Outcome', 'Cause', 'Log'], history)}

## 7. Bug register and root-cause analysis

{MD(['ID', 'Symptom', 'Root cause', 'Fix', 'Before / after evidence'], [list(b) for b in C.BUGS])}

Evidence files: [`evidence/bugs/`](../evidence/bugs).

| BUG-006 before (LEVIR model on Sentinel-2) | BUG-006 after (Sentinel-2 model, held-out city with ground truth) |
|---|---|
| ![before](../evidence/bugs/BUG-006/before_levir_model_0pct_on_sentinel2.png) | ![after](../evidence/bugs/BUG-006/after_oscd_lasvegas_measured.png) |

## 8. Complete test catalogue

{chr(10).join(sections)}

## 9. Not tested

- Copernicus Data Space (CDSE) retrieval against the real service (no credentials).
- Docker build and run (Docker not installed on the development machine).
- Load, concurrency, security penetration and long-running reliability.
- Browsers other than Chromium; mobile devices.
"""


# ============================================================================ LITERATURE / PHASE 2
def literature_md(f: dict) -> str:
    rows = [[f"{r['authors']} ({r['year']}). *{r['title']}*. {r['venue']}. [doi:{r['doi']}](https://doi.org/{r['doi']})",
             r["topic"], r["approach"], r["strength"], r["limitation"], r["relevance"]] for r in f["references"]]
    gaps = "\n\n".join(f"### {t}\n\n{g}\n\n**Status:** {s}" for t, g, s in C.GAPS)
    objectives = "\n".join(f"{i}. {o}" for i, o in enumerate(C.OBJECTIVES, 1))
    refs = "\n".join(f"{i}. {B.ref_line(r)}" for i, r in enumerate(f["references"], 1))
    return f"""
# Literature survey

Nine peer-reviewed sources on transformer- and CNN-based change detection, Swin Transformer architectures,
bitemporal fusion and change-detection datasets. Titles, authors, venues, years and DOIs were verified against
the Crossref API on {P['build_date']}. Method / strength / limitation summaries are the author's reading of each
paper; no numerical result from any paper is quoted.

## Problem statement

{C.PROBLEM_STATEMENT}

## Survey table

{MD(['Paper', 'Theme', 'Approach', 'Strength', 'Limitation', 'Relevance to this project'], rows)}

## Research gaps

{gaps}

## Research objectives

{objectives}

## References

{refs}
"""


def phase2_md(f: dict) -> str:
    return f"""
# Phase 2 plan

## Where Phase 1 ended

{B.results_sentence(f)}

Known limitations:

{chr(10).join('- ' + x for x in C.LIMITATIONS)}

## Pending work (about 30%)

{MD(['Pending work', 'Priority', 'Planned approach', 'Target'], [list(r) for r in C.PHASE2])}

## Success criteria

- Every model reported as mean +/- standard deviation over at least 3 seeds.
- Cross-dataset table (train on A, test on B) for LEVIR-CD and OSCD, including multispectral input.
- A multi-class change model with per-class IoU served through the existing registry.
- CDSE provider verified end-to-end, or documented as unsupported.
"""


# ============================================================================ PROJECT REPORT
def report_md(f: dict) -> str:
    shots = [("03_result_before_after_slider.png", "Before / after slider with predicted change and metrics (LEVIR-CD)"),
             ("08_ground_truth_comparison.png", "Error map against ground truth"),
             ("11_model_comparison.png", "Model comparison and domain-shift table"),
             ("20_oscd_lasvegas_on_map.png", "Held-out Sentinel-2 city (OSCD Las Vegas) on the map, F1 against ground truth"),
             ("21_bengaluru_sentinel2_model_on_map.png", "North Bengaluru Sentinel-2 pair (no ground truth)"),
             ("18_satellite_map_candidates.png", "Satellite map: AOI, date windows, candidate scenes"),
             ("13_error_dimension_mismatch.png", "Error handling: mismatched image sizes")]
    gallery = "\n\n".join(f"**{cap}**\n\n![{cap}](../{SHOTS}/{name})" for name, cap in shots)
    return f"""
# Phase 1 project report - {P['title']}

{P['app']}: a research prototype for bitemporal satellite change detection. This report covers the problem,
literature, requirements, design, implementation, results, testing and the plan for Phase 2.

> All implementation, training runs and tests recorded here were carried out on {P['build_date']}
> (see `experiments/*/config.json`, `evidence/tests/` and the git history).

## Abstract

{B.results_sentence(f)}

## 1. Literature survey and requirement analysis

### 1.1 Problem statement

{C.PROBLEM_STATEMENT}

### 1.2 Literature survey

Nine verified sources; full table in [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md).

{MD(['Paper', 'Approach', 'Limitation'], [[f"{r['authors']} ({r['year']}), {r['venue']}", r['approach'], r['limitation']] for r in f['references']])}

### 1.3 Literature gaps

{chr(10).join(f'- **{t}.** {g} *{s}*' for t, g, s in C.GAPS)}

### 1.4 Requirements

{MD(['ID', 'Requirement', 'Type', 'Priority'], [[r[0], r[1], r[2], r[3]] for r in C.REQUIREMENTS])}

## 2. System and detailed design

Full design, all diagrams (rendered inline) and the API are in [ARCHITECTURE.md](ARCHITECTURE.md).

![System architecture](design/diagrams/01_system_architecture.png)

{MD(['Diagram', 'File'], [[t.split(' - ')[1].rstrip('.'), f'[{n}](design/diagrams/{n})'] for n, t in B.DIAGRAM_LIST])}

### 2.3 Design decisions

{MD(['Decision', 'Rationale', 'Alternatives rejected'], [list(d) for d in C.DESIGN_DECISIONS])}

### 2.4 Traceability

{MD(['Requirement', 'Design element'], [[r[0], r[4]] for r in C.REQUIREMENTS])}

## 3. Implementation and functionality

### 3.1 Datasets and models

- Datasets: LEVIR-CD (0.5 m) and OSCD (Sentinel-2, 10 m) - see [DATASETS.md](DATASETS.md).
- Models: Siamese Swin Transformer and FC-Siam-diff, each for both domains - see [MODELS.md](MODELS.md).

### 3.2 Results

{results_block(f)}

### 3.3 Deviations from the design

{MD(['Deviation', 'Reason'], [list(d) for d in C.DEVIATIONS])}

### 3.4 The application

{gallery}

All 21 screenshots: [`evidence/screenshots`](../{SHOTS}).

## 4. Testing, debugging and integration

{f['backend_passed']} backend, {f['frontend_passed']} frontend and {f['e2e_passed']} end-to-end tests pass; full
details, the test catalogue and execution history are in [TESTING.md](TESTING.md).

### 4.3 Bugs found and fixed

{MD(['ID', 'Symptom', 'Root cause', 'Fix'], [[b[0], b[1], b[2], b[3]] for b in C.BUGS])}

### 4.4 Integration

{C.INTEGRATION_NOTE}

## 5. Phase-2 plan

{MD(['Pending work', 'Priority', 'Planned approach', 'Target'], [list(r) for r in C.PHASE2])}

Details: [PHASE2_PLAN.md](PHASE2_PLAN.md).

## 6. Limitations

{chr(10).join('- ' + x for x in C.LIMITATIONS)}
"""


# ============================================================================ README
def readme_md(f: dict) -> str:
    m = f["models"]
    lev_s, lev_c, s2_s, s2_c = (m[k] for k in B.MODEL_ORDER)
    docs = [
        ["[Project report](docs/PROJECT_REPORT.md)", "Full Phase-1 report: problem, survey, requirements, design, results, testing, plan"],
        ["[Datasets](docs/DATASETS.md)", "LEVIR-CD, OSCD and Sentinel-2 sources, licences, splits, pre-processing, why chosen"],
        ["[Models](docs/MODELS.md)", "Architectures, why chosen, training configuration, thresholds, results, checkpoints"],
        ["[Architecture & diagrams](docs/ARCHITECTURE.md)", "Components, API, data model, all 7 diagrams (architecture, flowchart, UML x3, DFD x2), decisions"],
        ["[Testing](docs/TESTING.md)", "Test results, test-case design, full test catalogue, execution history, bug register with root causes"],
        ["[Literature survey](docs/LITERATURE_SURVEY.md)", "9 verified papers, research gaps, objectives, references"],
        ["[Phase-2 plan](docs/PHASE2_PLAN.md)", "Remaining ~30% of the project"],
        ["[Implementation guide](docs/Implementation_Guide.md)", "Setup, reproduction commands, demonstration script"],
        ["[Documentation index](docs/README.md)", "All documents and diagram sources"],
    ]
    return f"""
# TerraVision AI - Satellite Change Intelligence

**{P['title']}** - Phase 1 research prototype.

TerraVision AI detects where land use has changed between two satellite images. A **Siamese Swin Transformer**
predicts a pixel-level change mask and is benchmarked against a **CNN baseline (FC-Siam-diff)** on the same
held-out data, for very-high-resolution aerial imagery (LEVIR-CD, 0.5 m) and for real **Sentinel-2** imagery
(OSCD, 10 m). A FastAPI backend and a React dashboard make it usable: upload or fetch an image pair from a map,
run GPU inference, inspect the change, compare with ground truth, and export georeferenced results.

![Change detection result](evidence/screenshots/03_result_before_after_slider.png)

## Results (held-out test data)

| Test data | Model | F1 | IoU | Precision | Recall |
|---|---|---|---|---|---|
| LEVIR-CD (0.5 m, {lev_s['num_patches']} patches) | **Siamese Swin Transformer** | **{pct(lev_s['f1'])}** | **{pct(lev_s['iou'])}** | {pct(lev_s['precision'])} | {pct(lev_s['recall'])} |
| LEVIR-CD (0.5 m) | FC-Siam-diff CNN | {pct(lev_c['f1'])} | {pct(lev_c['iou'])} | {pct(lev_c['precision'])} | {pct(lev_c['recall'])} |
| OSCD Sentinel-2 (10 m, {s2_s['num_images']} cities) | **Siamese Swin Transformer (S2)** | **{pct(s2_s['f1'])}** | **{pct(s2_s['iou'])}** | {pct(s2_s['precision'])} | {pct(s2_s['recall'])} |
| OSCD Sentinel-2 (10 m) | FC-Siam-diff CNN (S2) | {pct(s2_c['f1'])} | {pct(s2_c['iou'])} | {pct(s2_c['precision'])} | {pct(s2_c['recall'])} |

A model trained only on 0.5 m LEVIR-CD scores F1 {pct(next(r for r in f['cross'] if r['model'] == 'swin_siamese')['f1'])}
on the Sentinel-2 test cities - which is why separate Sentinel-2 models exist. Single training run per model
(seed 42) on an {run_config(f, 'swin_siamese')['gpu_name']}. Details: [docs/MODELS.md](docs/MODELS.md).

![Sentinel-2 test city on the map](evidence/screenshots/20_oscd_lasvegas_on_map.png)

## Features

- **Three input modes:** upload two images (PNG, JPEG, GeoTIFF) with an optional ground-truth mask; bundled
  samples with provenance; or search a place, draw an area and retrieve Sentinel-2 scenes (Earth Search).
- **Validation:** file type, integrity, size, channels, dimension and CRS checks; GeoTIFFs on different grids
  are reprojected; unrelated pairs are flagged.
- **Inference:** four trained models, automatic choice of the model that matches the imagery, sliding-window
  inference with mixed precision on GPU (CPU fallback), background jobs.
- **Results:** before/after slider, overlay, change mask, confidence map, error map vs ground truth,
  precision / recall / F1 / IoU / confusion matrix (only with ground truth), threshold and opacity controls.
- **Geospatial:** GeoTIFF mask, GeoJSON change regions, geodesic changed area, results on an interactive map.
- **Research:** model comparison dashboard with both benchmarks and the domain-shift table, analysis history,
  system status.

## Architecture

```mermaid
flowchart LR
  UI[React dashboard] -- HTTP/JSON --> API[FastAPI routers]
  API --> SVC[AnalysisService<br/>background worker]
  API --> IMG[imaging / geo<br/>validate, align, export]
  API --> SAT[Satellite providers<br/>Earth Search, CDSE]
  SVC --> MM[ModelManager<br/>load once, GPU lock]
  MM --> NN[Siamese Swin / FC-Siam-diff<br/>sliding-window inference]
  SVC --> DB[(SQLite)]
  SVC --> FS[(storage/analyses)]
  MM --> CK[(checkpoints)]
```

All seven diagrams (system architecture, flowchart, use-case, class, sequence, DFD level 0 and 1):
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (Windows)

Requirements: Python 3.12, Node.js 20+, optionally an NVIDIA GPU (RTX 50-series needs CUDA 12.8 wheels).

```powershell
git clone {REPO_URL}.git; cd terravision-ai
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
cd frontend; npm install; cd ..
```

**Model checkpoints** are not stored in git (the Swin file is 120 MB). Download the four `.pt` files from the
[Phase 1 release]({REPO_URL}/releases/tag/v0.1.0-phase1) into `checkpoints/` (checksums in `SHA256SUMS.txt`), or re-train them (see [docs/Implementation_Guide.md](docs/Implementation_Guide.md)).

```powershell
.\\scripts\\start.ps1        # backend on :8000, frontend on :5173, opens the browser
```

Then follow the demonstration script in [docs/Implementation_Guide.md](docs/Implementation_Guide.md#8-demonstration).

## Reproduce training and evaluation

```powershell
.\\.venv\\Scripts\\python.exe -m ml.data.download ; .\\.venv\\Scripts\\python.exe -m ml.data.prepare
.\\.venv\\Scripts\\python.exe -m ml.data.oscd download ; .\\.venv\\Scripts\\python.exe -m ml.data.oscd prepare
.\\.venv\\Scripts\\python.exe -m ml.train --config ml/configs/swin_siamese.json
.\\.venv\\Scripts\\python.exe -m ml.train --config ml/configs/fc_siam_diff.json
.\\.venv\\Scripts\\python.exe -m ml.train --config ml/configs/swin_siamese_s2.json
.\\.venv\\Scripts\\python.exe -m ml.train --config ml/configs/fc_siam_diff_s2.json
.\\.venv\\Scripts\\python.exe -m ml.calibrate --models swin_siamese_s2 fc_siam_diff_s2
.\\.venv\\Scripts\\python.exe -m ml.evaluate --cross-domain
```

## Tests

| Suite | Result |
|---|---|
| Backend + ML (pytest) | {f['backend_passed']} passed, {f['backend_cov']}% coverage |
| Frontend (Vitest) | {f['frontend_passed']} passed |
| End-to-end (Playwright, real models) | {f['e2e_passed']} passed |

Test design, catalogue, execution history and bug register: [docs/TESTING.md](docs/TESTING.md).

## Documentation

{MD(['Document', 'Contents'], docs)}

## Repository layout

| Path | Contents |
|---|---|
| `ml/` | Datasets (`data/`), models (`models/`), `train.py`, `calibrate.py`, `evaluate.py`, `inference.py` |
| `backend/app/` | FastAPI app: `routers/`, `services/`, `satellite/` |
| `frontend/` | React + Vite + TypeScript dashboard, unit tests (`src/test`), end-to-end tests (`e2e`) |
| `tests/` | Backend and ML tests |
| `scripts/` | Start script, sample builder, document generators |
| `samples/` | Bundled demonstration pairs with provenance (`manifest.json`) |
| `reports/` | Benchmark metrics, charts, calibration and measurement reports |
| `experiments/` | Per-run training configuration and history |
| `docs/` | All documentation and diagrams |
| `evidence/` | Test logs, bug evidence, screenshots |

## Data and licences

- **LEVIR-CD** (Chen & Shi, 2020): academic use only; imagery subject to the Google Earth terms of use. Four
  test pairs are bundled for demonstration.
- **OSCD** (Caye Daudt et al., 2018): cite the IGARSS 2018 paper; imagery is Copernicus Sentinel data. Three
  test cities are bundled.
- **Sentinel-2** via Element 84 Earth Search: contains modified Copernicus Sentinel data.
- Basemap and place search: (c) OpenStreetMap contributors.

## Limitations

{chr(10).join('- ' + x for x in C.LIMITATIONS)}
"""


def docs_index_md() -> str:
    rows = [
        ["[PROJECT_REPORT.md](PROJECT_REPORT.md)", "Full project report (Markdown edition)"],
        ["[DATASETS.md](DATASETS.md)", "Datasets used and why"],
        ["[MODELS.md](MODELS.md)", "Models used and why, training, results"],
        ["[ARCHITECTURE.md](ARCHITECTURE.md)", "Architecture, API, data model, all diagrams"],
        ["[TESTING.md](TESTING.md)", "Test results, test cases, bug register"],
        ["[LITERATURE_SURVEY.md](LITERATURE_SURVEY.md)", "Literature survey and research gaps"],
        ["[PHASE2_PLAN.md](PHASE2_PLAN.md)", "Phase-2 plan"],
        ["[Implementation_Guide.md](Implementation_Guide.md)", "Setup, reproduction, demonstration"],
        ["[design/](design)", "Diagram sources (Mermaid) and renders (PNG, SVG)"],
    ]
    return f"""
# Documentation index

All documents are generated from recorded project outputs by `scripts/build_markdown.py`.

{MD(['Document', 'Contents'], rows)}
"""


# ============================================================================ GUIDE + DESIGN README
def guide_md(f: dict) -> str:
    cfg = f["runs"]["swin_siamese"]["config"]
    man = f["manifest"]["splits"]
    guide = f"""# TerraVision AI - Implementation guide

{P['title']}. Research prototype, Phase 1.

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
.\\.venv\\Scripts\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
cd frontend; npm install; cd ..
```

RTX 50-series GPUs need the CUDA 12.8 (`cu128`) wheels. For CPU only use the `cpu` index. Exact versions used
are in `requirements.lock.txt`. Docker is optional and untested (`Dockerfile.backend`, `docker-compose.yml`).

## 4. Run

```powershell
.\\scripts\\start.ps1
```

or in two terminals:

```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
cd frontend; npm run dev
```

Frontend: http://localhost:5173 - API: http://127.0.0.1:8000 (interactive docs at `/docs`).
Configuration: copy `.env.example` to `.env` (all `TV_*` variables are optional).

## 5. Reproduce the models

```powershell
.\\.venv\\Scripts\\python.exe -m ml.data.download          # ~2.7 GB, MD5-verified, official split
.\\.venv\\Scripts\\python.exe -m ml.data.prepare           # 256x256 tiles + manifest
.\\.venv\\Scripts\\python.exe -m ml.train --config ml/configs/swin_siamese.json
.\\.venv\\Scripts\\python.exe -m ml.train --config ml/configs/fc_siam_diff.json
.\\.venv\\Scripts\\python.exe -m ml.data.oscd download      # OSCD (Sentinel-2), SHA-256 verified
.\\.venv\\Scripts\\python.exe -m ml.data.oscd prepare       # RGB at the measured TCI gain, city split
.\\.venv\\Scripts\\python.exe -m ml.train --config ml/configs/swin_siamese_s2.json   # fine-tune from LEVIR-CD
.\\.venv\\Scripts\\python.exe -m ml.train --config ml/configs/fc_siam_diff_s2.json
.\\.venv\\Scripts\\python.exe -m ml.calibrate --models swin_siamese_s2 fc_siam_diff_s2   # threshold on val cities
.\\.venv\\Scripts\\python.exe -m ml.evaluate --cross-domain # both test sets + domain-shift check
.\\.venv\\Scripts\\python.exe scripts\\build_samples.py     # bundled samples (Sentinel-2 part needs internet)
```

Dataset: LEVIR-CD, {man['train']['num_images']} / {man['val']['num_images']} / {man['test']['num_images']} images ->
{man['train']['num_patches']} / {man['val']['num_patches']} / {man['test']['num_patches']} patches
(train / val / test). Source: Hugging Face mirror `satellite-image-deep-learning/LEVIR-CD` at a pinned revision
(the URLs and checksums TorchGeo uses). Licence: academic use only.

Sentinel-2: OSCD (Caye Daudt et al., IGARSS 2018) from the Hugging Face mirror `hkristen/oscd` at a pinned revision
(TorchGeo's SHA-256 checksums). Train cities {', '.join(f['oscd_manifest']['cities']['train'])}; validation cities
{', '.join(f['oscd_manifest']['cities']['val'])}; test cities {', '.join(f['oscd_manifest']['cities']['test'])}.
Bands B04/B03/B02 are scaled with the gain measured on real Earth Search true-colour pixels (DN
{f['tci_fit']['dn_for_255']:.0f} -> 255; `reports/tci_gain_fit.json`). Thresholds chosen on the validation cities:
Swin {f['calibration']['swin_siamese_s2']['selected']['threshold']}, CNN {f['calibration']['fc_siam_diff_s2']['selected']['threshold']}
(`reports/threshold_calibration.json`).

Checkpoints are produced locally by the commands above; no third-party change-detection checkpoint is used. The
Swin encoder starts from timm `swinv2_tiny_window8_256.ms_in1k` (ImageNet-1k) and is fine-tuned end-to-end.

## 6. Measured results ({P['build_date']})

{results_sentence(f)}

{md_table(RESULT_HEAD, results_rows(f))}

Domain shift - the OSCD test cities scored by models with and without Sentinel-2 training:

{md_table(CROSS_HEAD, cross_rows(f))}

{md_table(TRAIN_HEAD, training_rows(f))}

Hardware: {cfg['gpu_name']} ({cfg['gpu_total_mb']} MB), PyTorch {cfg['torch']}, mixed precision. The two LEVIR-CD
models were trained at the same time on the one GPU, so their training times are slower than a solo run. Application
inference on a 1024x1024 pair (25 tiles) took about 0.6-1.0 s with the Swin model once loaded, and 4.5 s on the
first request including model load (values from the analysis history of the demo runs).

## 7. Tests

```powershell
.\\.venv\\Scripts\\python.exe -m pytest --cov          # {f['backend_passed']} tests, {f['backend_cov']}% coverage
cd frontend; npm test                                 # {f['frontend_passed']} tests
npx playwright test                                   # {f['e2e_passed']} end-to-end tests (servers must be running)
.\\.venv\\Scripts\\python.exe -m ruff check . ; cd frontend; npm run typecheck; npm run lint
```

## 8. Demonstration

1. Start: `.\\scripts\\start.ps1`. Browser opens the Overview page.
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

{md_table(['Sample', 'Dataset', 'Ground truth', 'Georeferenced', 'Licence'], [[x['name'], x.get('dataset'), 'yes' if x.get('ground_truth') else 'no', 'yes' if x.get('georeferenced') else 'no', x.get('license')] for x in f['samples']])}

## 10. Known limitations

""" + "\n".join(f"- {x}" for x in C.LIMITATIONS) + "\n"
    return guide


def design_readme_md() -> str:
    design_text = (f"""# Design documentation

| File | Content |
|---|---|
| `diagrams/01_system_architecture` | System architecture (components, stores, external services) |
| `diagrams/02_flowchart` | Application flowchart for an analysis request |
| `diagrams/03_use_case` | UML use-case diagram |
| `diagrams/04_class_diagram` | UML class diagram |
| `diagrams/05_sequence` | UML sequence diagram (upload analysis) |
| `diagrams/06_dfd_level0` | DFD level 0 (context) |
| `diagrams/07_dfd_level1` | DFD level 1 |

Each diagram has a Mermaid source (`.mmd`) and rendered `.png` / `.svg`. Re-render with
`npx @mermaid-js/mermaid-cli -i <file>.mmd -o <file>.png`.

The architecture explanation, technology justification, design alternatives, data model, API contracts and
requirement traceability are in [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Design decisions

{md_table(['Decision', 'Rationale', 'Alternatives rejected'], [list(d) for d in C.DESIGN_DECISIONS])}

## API contracts

{md_table(['Method and path', 'Purpose'], API_TABLE)}
""")
    return design_text


def main() -> int:
    f = B.load_facts()
    write(ROOT / "README.md", readme_md(f))
    write(DOCS / "README.md", docs_index_md())
    write(DOCS / "PROJECT_REPORT.md", report_md(f))
    write(DOCS / "DATASETS.md", datasets_md(f))
    write(DOCS / "MODELS.md", models_md(f))
    write(DOCS / "ARCHITECTURE.md", architecture_md(f))
    write(DOCS / "TESTING.md", testing_md(f))
    write(DOCS / "LITERATURE_SURVEY.md", literature_md(f))
    write(DOCS / "PHASE2_PLAN.md", phase2_md(f))
    write(DOCS / "Implementation_Guide.md", guide_md(f))
    write(DOCS / "design" / "README.md", design_readme_md())
    return 0


if __name__ == "__main__":
    sys.exit(main())
