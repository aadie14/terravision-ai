# Design documentation

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

## API contracts

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
