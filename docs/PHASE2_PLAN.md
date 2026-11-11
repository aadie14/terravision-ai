# Phase 2 plan

## Where Phase 1 ended

On the held-out LEVIR-CD test split (2048 patches of 256x256 px at 0.5 m, threshold 0.5), the Siamese Swin Transformer reaches F1 91.35% and IoU 84.07%; the FC-Siam-diff baseline reaches F1 77.89% and IoU 63.79%. On real Sentinel-2 imagery (the 10 OSCD test cities at 10 m, whole images, thresholds selected on validation cities) the fine-tuned Swin model reaches F1 47.09% and IoU 30.80% against F1 39.70% and IoU 24.77% for the CNN, whereas the LEVIR-CD-only Swin model scores F1 0.00% on the same cities. The transformer costs 31x the parameters and 3.8x the per-tile latency (22 ms vs 6 ms on NVIDIA GeForce RTX 5050 Laptop GPU). All results come from a single training run per model (seed 42).

Known limitations:

- Two imagery domains are validated: 0.5 m LEVIR-CD-like aerial RGB and 10 m Sentinel-2 RGB (OSCD). Sentinel-2 accuracy is moderate (see the OSCD benchmark): small or thin changes are missed, and on the Bengaluru pair seasonal farmland and lakes that filled with water are also flagged. Deforestation and disaster damage are not supported.
- OSCD is small (11 training cities) and is Level-1C top-of-atmosphere data while the application receives Level-2A true colour; only the DN scale is matched, not the atmospheric correction. The CNN's selected threshold (0.05) lies at the edge of the search grid.
- Single training run per model (seed 42); no confidence intervals. The CNN baseline stopped early at epoch 25 and is weaker than figures reported in the literature for FC-Siam-diff trained longer.
- The Copernicus Data Space (CDSE) retrieval path is implemented from the documentation but has not been run against the real service (no credentials). Earth Search retrieval is verified.
- The pair-similarity warning is a calibrated heuristic; it can miss unrelated pairs or flag genuine pairs with very large change.
- Local research prototype: no authentication, single worker, SQLite. Docker files are untested. No load, security or reliability testing has been done.
- Frontend unit-test coverage is partial (map drawing and charts are covered by the Playwright run, not by unit tests).

## Pending work (about 30%)

| Pending work | Priority | Planned approach | Target |
|---|---|---|---|
| Strengthen the CNN baseline: longer schedule without early stop, learning-rate search, 3 seeds; report mean and standard deviation for both models | High | Reuse ml.train with new configs; add a seed sweep and aggregate in ml.evaluate | Phase-2 weeks 1-2 |
| Improve Sentinel-2 accuracy: all 13 OSCD bands, Level-1C vs Level-2A harmonisation, more training regions (e.g. Indian cities), 5-fold city cross-validation | High | Extend ml.data.oscd and the patch embedding to multispectral input; add an L2A band reader to the provider | Phase-2 weeks 2-5 |
| Multi-class (semantic) change: land-cover transitions rather than a binary mask | High | Adopt a semantic change dataset (e.g. SECOND); add a multi-class head and per-class IoU | Phase-2 weeks 5-8 |
| Robustness to clouds, seasons and water-level change for Sentinel-2 retrieval (the Bengaluru result flags filled lakes and farmland as change) | High | Use the L2A scene-classification layer to mask clouds; same-season scene pairing; photometric augmentation | Phase-2 weeks 6-8 |
| Wider transformer comparison: BIT, ChangeFormer, Swin-S, plus ablations of the fusion module | Medium | Implement behind the existing ModelSpec registry; same split, same metrics | Phase-2 weeks 8-10 |
| Explainability: attention / gradient-based maps and calibrated uncertainty | Medium | Grad-CAM on fused features, test-time augmentation variance, reliability diagrams | Phase-2 weeks 9-11 |
| Verify the Copernicus (CDSE) provider end-to-end; scheduled monitoring of saved areas of interest | Medium | Create an OAuth client; add a scheduler and change alerts | Phase-2 weeks 10-12 |
| Deployment hardening: authentication, Docker build verification, PostgreSQL option, load testing | Low | Build and test the provided Docker files; add Alembic migrations; Locust load test | Phase-2 weeks 12-14 |

## Success criteria

- Every model reported as mean +/- standard deviation over at least 3 seeds.
- Cross-dataset table (train on A, test on B) for LEVIR-CD and OSCD, including multispectral input.
- A multi-class change model with per-class IoU served through the existing registry.
- CDSE provider verified end-to-end, or documented as unsupported.
