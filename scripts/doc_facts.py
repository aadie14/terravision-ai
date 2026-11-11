"""Facts shared by the documentation generators, read from recorded project outputs (no hand-typed numbers)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVID = ROOT / "evidence"
MODEL_ORDER = ("swin_siamese", "fc_siam_diff", "swin_siamese_s2", "fc_siam_diff_s2")
RUNS = {"swin_siamese": "swin_siamese_levir_v2", "fc_siam_diff": "fc_siam_diff_levir_v2",
        "swin_siamese_s2": "swin_siamese_oscd_v1", "fc_siam_diff_s2": "fc_siam_diff_oscd_v1"}
E2E_LOG = "e2e_playwright_final.log"


def load_facts() -> dict:
    ev = ROOT / "reports" / "evaluation"
    models = {k: json.loads((ev / f"{k}_metrics.json").read_text()) for k in MODEL_ORDER}
    runs = {}
    for key, run in RUNS.items():
        folder = ROOT / "experiments" / run
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        runs[key] = {
            "name": run, "config": json.loads((folder / "config.json").read_text()),
            "summary": json.loads((folder / "summary.json").read_text()), "history": history,
            "train_minutes": sum(h["epoch_seconds"] for h in history) / 60,
            "peak_gpu_mb": max(h["peak_gpu_mb"] or 0 for h in history),
        }
    manifest = json.loads((ROOT / "data/processed/levir256/manifest.json").read_text())
    backend_log = (EVID / "tests/backend_run_final.log").read_text(errors="replace")
    passed = re.search(r"(\d+) passed", backend_log)
    cov = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", backend_log)
    fe_log = (EVID / "tests/frontend_vitest.log").read_text(errors="replace")
    fe = re.search(r"Tests\s+(\d+) passed", fe_log)
    fe_cov = re.search(r"Lines\s+:\s+([\d.]+)%", fe_log)
    e2e_log = (EVID / "tests" / E2E_LOG).read_text(errors="replace")
    e2e = re.search(r"(\d+) passed", e2e_log)
    per_file = {}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        per_file[path.name] = len(re.findall(r"^def test_", path.read_text(encoding="utf-8"), flags=re.M))
    git = subprocess.run(["git", "log", "--pretty=%h %ad %s", "--date=short"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip().splitlines()
    return {
        "models": models, "runs": runs, "manifest": manifest,
        "backend_passed": int(passed.group(1)), "backend_cov": int(cov.group(3)), "backend_stmts": int(cov.group(1)),
        "frontend_passed": int(fe.group(1)), "frontend_cov": float(fe_cov.group(1)), "e2e_passed": int(e2e.group(1)),
        "per_file": per_file, "git": git,
        "references": json.loads((ROOT / "frontend/src/data/references.json").read_text(encoding="utf-8")),
        "samples": json.loads((ROOT / "samples/manifest.json").read_text(encoding="utf-8-sig"))["samples"],
        "cross": json.loads((ev / "cross_domain.json").read_text()),
        "oscd_manifest": json.loads((ROOT / "data/processed/oscd256/manifest.json").read_text()),
        "calibration": json.loads((ROOT / "reports/threshold_calibration.json").read_text()),
        "tci_fit": json.loads((ROOT / "reports/tci_gain_fit.json").read_text()),
    }


def pct(v: float | None, d: int = 2) -> str:
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def results_rows(f: dict) -> list[list[str]]:
    rows = []
    for key in MODEL_ORDER:
        m = f["models"][key]
        rows.append([m["dataset"], m["display_name"], pct(m["precision"]), pct(m["recall"]), pct(m["f1"]),
                     pct(m["iou"]), pct(m["accuracy"]), f"{m['threshold']:g}", f"{m['parameters'] / 1e6:.2f} M",
                     f"{m['latency_ms_median']:.1f} ms"])
    return rows


RESULT_HEAD = ["Test data", "Model", "Precision", "Recall", "F1", "IoU", "Accuracy", "Threshold", "Params",
               "Latency / tile"]


def cross_rows(f: dict) -> list[list[str]]:
    rows = [[r["display_name"], "LEVIR-CD only", pct(r["precision"]), pct(r["recall"]), pct(r["f1"]), pct(r["iou"])]
            for r in f["cross"]]
    for key in ("swin_siamese_s2", "fc_siam_diff_s2"):
        m = f["models"][key]
        rows.append([m["display_name"], "LEVIR-CD, then OSCD fine-tune", pct(m["precision"]), pct(m["recall"]),
                     pct(m["f1"]), pct(m["iou"])])
    return rows


CROSS_HEAD = ["Model (scored on OSCD test cities)", "Trained on", "Precision", "Recall", "F1", "IoU"]


def results_sentence(f: dict) -> str:
    s, c = f["models"]["swin_siamese"], f["models"]["fc_siam_diff"]
    s2, c2 = f["models"]["swin_siamese_s2"], f["models"]["fc_siam_diff_s2"]
    levir_swin_on_oscd = next(r for r in f["cross"] if r["model"] == "swin_siamese")
    return (
        f"On the held-out LEVIR-CD test split ({s['num_patches']} patches of 256x256 px at 0.5 m, threshold "
        f"{s['threshold']}), the Siamese Swin Transformer reaches F1 {pct(s['f1'])} and IoU {pct(s['iou'])}; the "
        f"FC-Siam-diff baseline reaches F1 {pct(c['f1'])} and IoU {pct(c['iou'])}. On real Sentinel-2 imagery (the "
        f"{s2['num_images']} OSCD test cities at 10 m, whole images, thresholds selected on validation cities) the "
        f"fine-tuned Swin model reaches F1 {pct(s2['f1'])} and IoU {pct(s2['iou'])} against F1 {pct(c2['f1'])} and IoU "
        f"{pct(c2['iou'])} for the CNN, whereas the LEVIR-CD-only Swin model scores F1 "
        f"{pct(levir_swin_on_oscd['f1'])} on the same cities. The transformer costs "
        f"{s['parameters'] / c['parameters']:.0f}x the parameters and {s['latency_ms_median'] / c['latency_ms_median']:.1f}x "
        f"the per-tile latency ({s['latency_ms_median']:.0f} ms vs {c['latency_ms_median']:.0f} ms on {s['gpu_name']}). "
        "All results come from a single training run per model (seed 42)."
    )


def training_rows(f: dict) -> list[list[str]]:
    rows = []
    for key in MODEL_ORDER:
        r = f["runs"][key]
        cfg, summ = r["config"], r["summary"]
        rows.append([f["models"][key]["display_name"], f"{r['name']} ({r['config']['dataset']})", str(cfg["batch_size"]), str(cfg["grad_accum"]),
                     f"{cfg['lr']:g}", f"{summ['epochs_run']} (best {summ['best_epoch']})", pct(summ["best_val_f1"]),
                     f"{r['train_minutes']:.0f} min", f"{r['peak_gpu_mb']} MB"])
    return rows


TRAIN_HEAD = ["Model", "Run", "Batch", "Grad. accum.", "LR", "Epochs", "Best val F1", "Train time", "Peak VRAM (train)"]



def ref_line(r: dict) -> str:
    return f"{r['authors']} ({r['year']}). {r['title']}. {r['venue']}. https://doi.org/{r['doi']}"



def md_table(head: list[str], rows: list[list]) -> str:
    def esc(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)



API_TABLE = [
    ["GET /health", "Liveness, database and model availability"],
    ["GET /api/system", "Compute device, GPU memory, storage, versions"],
    ["GET /api/models", "Model registry with checkpoint and benchmark information"],
    ["GET /api/config", "Public limits and defaults"],
    ["GET /api/samples, /api/samples/{id}/{role}", "Bundled samples and their images"],
    ["POST /api/analyses/upload", "Create an analysis from uploaded T1, T2 (+ mask) - multipart"],
    ["POST /api/analyses", "Create an analysis from a bundled sample, or re-run stored inputs with another model"],
    ["GET /api/analyses, /api/analyses/stats", "History (paginated, filterable) and overview statistics"],
    ["GET /api/analyses/{id}", "Status and details of one analysis"],
    ["GET /api/analyses/{id}/results", "Full results and download links (409 until completed)"],
    ["POST /api/analyses/{id}/threshold", "Re-threshold the stored probability map"],
    ["GET /api/analyses/{id}/downloads/{artifact}", "Whitelisted artefacts: images, JSON, GeoTIFF, GeoJSON"],
    ["DELETE /api/analyses/{id}", "Delete the record and its files"],
    ["POST /api/evaluation", "Evaluate available models on labelled bundled samples"],
    ["GET /api/evaluation/summary, /reports/{name}", "Offline test-split benchmark and figures"],
    ["GET /api/satellite/providers, /geocode", "Imagery providers; place search"],
    ["POST /api/satellite/search", "Candidate Sentinel-2 scenes for an AOI and date window"],
    ["POST /api/satellite/retrieve", "Fetch two scenes for the AOI and start an analysis"],
]



DIAGRAM_LIST = [
    ("01_system_architecture.png", "Figure A.1 - System architecture."),
    ("02_flowchart.png", "Figure A.2 - Application flowchart (analysis request)."),
    ("03_use_case.png", "Figure A.3 - UML use-case diagram."),
    ("04_class_diagram.png", "Figure A.4 - UML class diagram."),
    ("05_sequence.png", "Figure A.5 - UML sequence diagram (upload analysis)."),
    ("06_dfd_level0.png", "Figure A.6 - DFD level 0 (context)."),
    ("07_dfd_level1.png", "Figure A.7 - DFD level 1."),
]
