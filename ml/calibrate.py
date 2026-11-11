"""Choose a model's decision threshold on its VALIDATION split and store it in the checkpoint.

Change pixels are rare (2-3% on OSCD), so 0.5 is not necessarily the best operating point. The threshold
that maximises validation F1 is written to the checkpoint metadata (``threshold`` plus
``threshold_selection``); ``ml.evaluate`` and the application then use it. The test split is never read.

Usage:  python -m ml.calibrate --models swin_siamese_s2 fc_siam_diff_s2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.checkpoint import load_model, read_checkpoint
from ml.data.dataset import PatchPairDataset, normalize
from ml.metrics import ConfusionCounts, compute_metrics
from ml.models.registry import MODEL_SPECS
from ml.train import IGNORE_INDEX

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
DATA_ROOTS = {"vhr": Path("data/processed/levir256"), "sentinel2": Path("data/processed/oscd256")}


@torch.inference_mode()
def sweep(key: str, ckpt_dir: Path, device: torch.device) -> dict:
    spec = MODEL_SPECS[key]
    path = ckpt_dir / spec.default_checkpoint
    model, _ = load_model(path, expected_key=key)
    model.to(device)
    amp = device.type == "cuda"
    loader = DataLoader(PatchPairDataset(DATA_ROOTS[spec.domain], "val"), batch_size=16, shuffle=False)
    counts = {t: ConfusionCounts() for t in GRID}
    for a, b, m in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            prob = torch.sigmoid(model(normalize(a.to(device)), normalize(b.to(device))).float())[:, 0].cpu().numpy()
        m = m.numpy()
        valid = m != IGNORE_INDEX
        truth = m == 1
        for t in GRID:
            counts[t].update((prob > t)[valid], truth[valid])
    rows = [{"threshold": float(t), **{k: v for k, v in compute_metrics(c).to_dict().items()
                                       if k in ("precision", "recall", "f1", "iou")}} for t, c in counts.items()]
    best = max(rows, key=lambda r: r["f1"] or 0)
    default = next(r for r in rows if r["threshold"] == 0.5)

    payload = read_checkpoint(path)
    payload["meta"]["threshold"] = best["threshold"]
    payload["meta"]["threshold_selection"] = {
        "method": "maximum F1 on the validation split (test split not used)",
        "val_f1_at_0.5": default["f1"], "val_f1_at_selected": best["f1"],
        "grid": [float(GRID[0]), float(GRID[-1]), 0.05],
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return {"model": key, "selected": best, "at_0.5": default, "sweep": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("reports/threshold_calibration.json"))
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    existing = json.loads(args.out.read_text()) if args.out.is_file() else {}
    for key in args.models:
        result = sweep(key, args.checkpoint_dir, device)
        existing[key] = result
        s, d = result["selected"], result["at_0.5"]
        print(f"[{key}] threshold {s['threshold']:.2f}: val F1 {s['f1']:.4f} (at 0.50: {d['f1']:.4f})", flush=True)
    args.out.write_text(json.dumps(existing, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
