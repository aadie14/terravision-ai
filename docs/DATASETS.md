# Datasets

This page documents every dataset used by Vision Transformers for Satellite Land-Use Monitoring: where it comes from, its licence, how it was split
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
- **Source used:** https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD at a pinned revision, with the MD5 checksums TorchGeo uses
  (`ml/data/download.py`). Official page: https://justchenhao.github.io/LEVIR/
- **Licence:** academic use only, commercial use prohibited; imagery subject to the Google Earth terms of use.
- **Content:** 637 pairs of 1024x1024 px images at 0.5 m/pixel over Texas, USA, with hand-drawn building-change masks.

| Split | Images | 256x256 patches | Changed pixels |
|---|---|---|---|
| train | 445 | 7120 | 4.59% |
| val | 64 | 1024 | 4.20% |
| test | 128 | 2048 | 5.09% |

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
- **Source used:** https://huggingface.co/datasets/hkristen/oscd (rev 4958d786c1389ede1511d91a6ecf1a75c4074933), with TorchGeo's SHA-256 checksums (`ml/data/oscd.py`).
  Project page: https://rcdaudt.github.io/oscd/
- **Licence:** imagery is Copernicus Sentinel data (free, full and open). The dataset README asks for the
  citation above and states no further licence terms.
- **Content:** 24 city pairs of Sentinel-2 images (2015-2018) with pixel-level urban-change labels.

| Split | Cities | Training patches | Changed pixels |
|---|---|---|---|
| train | abudhabi, aguasclaras, beirut, bercy, bordeaux, cupertino, mumbai, nantes, paris, pisa, saclay_e | 279 | 2.16% |
| val | beihai, hongkong, rennes | 70 | 2.82% |
| test | brasilia, chongqing, dubai, lasvegas, milano, montpellier, norcia, rio, saclay_w, valencia | whole images | 5.17% |

The 14 official training cities are split 11 / 3 into train and validation (deterministic, every 5th city
alphabetically from the third). The 10 official test cities are never used for training, early stopping
or threshold selection; they are evaluated whole with the same sliding-window inference the application uses.

**Pre-processing:**
- Only the 10 m true-colour bands B04, B03, B02 are used, because the application's imagery source
  (Earth Search) delivers true-colour images.
- **Radiometry matched to the application.** OSCD ships raw 16-bit digital numbers (DN). The mapping the
  app's Sentinel-2 imagery uses was measured on real Earth Search scenes (`scripts/fit_tci_gain.py`):

| Scene | Processing baseline | Red: DN for 255 | Green: DN for 255 | Blue: DN for 255 |
|---|---|---|---|---|
| S2A_43PGQ_20190315_0_L2A | 02.11 | 2510 (RMSE 4.5) | 2502 (RMSE 1.8) | 2499 (RMSE 3.7) |
| S2B_43PGQ_20250206_0_L2A | 05.11 | 2504 (RMSE 2.6) | 2501 (RMSE 1.4) | 2499 (RMSE 3.5) |

  One gain (DN 2503 -> 255) fits both an old and a new processing baseline, so OSCD bands are
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

| Sample | Dataset | Split | Imagery domain | Ground truth | Georeferenced | Licence |
|---|---|---|---|---|---|---|
| LEVIR-CD test_45 (dense change) | LEVIR-CD | test | vhr | yes | no | LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from Google Earth is subject to the Google Earth terms of use. |
| LEVIR-CD test_74 (moderate change) | LEVIR-CD | test | vhr | yes | no | LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from Google Earth is subject to the Google Earth terms of use. |
| LEVIR-CD test_101 (sparse change) | LEVIR-CD | test | vhr | yes | no | LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from Google Earth is subject to the Google Earth terms of use. |
| LEVIR-CD test_108 (no change) | LEVIR-CD | test | vhr | yes | no | LEVIR-CD (Chen & Shi, 2020): academic use only, commercial use prohibited; imagery from Google Earth is subject to the Google Earth terms of use. |
| OSCD lasvegas 2015 vs 2018 (Sentinel-2, ground truth) | OSCD | test | sentinel2 | yes | yes | Imagery: Copernicus Sentinel data (free, full and open). Labels: OSCD by Caye Daudt et al.; the dataset README requires citation and states no further licence terms. |
| OSCD dubai 2015 vs 2018 (Sentinel-2, ground truth) | OSCD | test | sentinel2 | yes | yes | Imagery: Copernicus Sentinel data (free, full and open). Labels: OSCD by Caye Daudt et al.; the dataset README requires citation and states no further licence terms. |
| OSCD chongqing 2017 vs 2018 (Sentinel-2, ground truth) | OSCD | test | sentinel2 | yes | yes | Imagery: Copernicus Sentinel data (free, full and open). Labels: OSCD by Caye Daudt et al.; the dataset README requires citation and states no further licence terms. |
| Sentinel-2 north Bengaluru 2019 vs 2025 (georeferenced) | Sentinel-2 L2A (Earth Search) | - | sentinel2 | no | yes | Copernicus Sentinel data: free, full and open access; attribution required. |

Built by `scripts/build_samples.py`; provenance is recorded in `samples/manifest.json`.

## 5. Not included in the repository

The full datasets are not committed (size and licence). Re-create them with:

```powershell
.\.venv\Scripts\python.exe -m ml.data.download      # LEVIR-CD
.\.venv\Scripts\python.exe -m ml.data.prepare
.\.venv\Scripts\python.exe -m ml.data.oscd download
.\.venv\Scripts\python.exe -m ml.data.oscd prepare
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
