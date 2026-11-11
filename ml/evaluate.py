"""Evaluate trained checkpoints on the held-out LEVIR-CD test split and build comparison reports.

Usage:
    python -m ml.evaluate                       # evaluate every model that has a checkpoint
    python -m ml.evaluate --models swin_siamese --split test

Outputs under ``reports/evaluation/``:
    <model>_metrics.json           full metrics, confusion matrix, latency, provenance
    <model>_confusion_matrix.png
    <model>_examples.png           T1 / T2 / ground truth / prediction panels
    metrics.csv                    one row per model
    summary.json                   machine-readable comparison (served by the API)
    model_comparison.png           grouped bar chart
The decision threshold is the fixed training value (0.5); it is never tuned on the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.checkpoint import CheckpointError, load_model
from ml.data.dataset import PatchPairDataset, normalize
from ml.data.oscd import load_test_cities
from ml.inference import predict_pair
from ml.metrics import ConfusionCounts, compute_metrics
from ml.models.registry import MODEL_SPECS, count_parameters

METRIC_KEYS = ("precision", "recall", "f1", "iou", "accuracy")


@torch.inference_mode()
def measure_latency(model, device: torch.device, tile: int, amp: bool, runs: int = 30) -> dict:
    """Median forward latency for one bitemporal tile (batch 1), after warm-up."""
    x = torch.randn(1, 3, tile, tile, device=device)
    timings = []
    for i in range(runs + 5):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp):
            model(x, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 5:
            timings.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_ms_median": float(np.median(timings)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
        "latency_tile": tile,
        "latency_batch": 1,
        "latency_runs": runs,
    }


@torch.inference_mode()
def evaluate_model(key: str, ckpt_dir: Path, data_root: Path, split: str, out: Path,
                   batch_size: int, limit: int | None, device: torch.device) -> dict:
    spec = MODEL_SPECS[key]
    model, info = load_model(ckpt_dir / spec.default_checkpoint, expected_key=key)
    model.to(device)
    amp = device.type == "cuda"
    threshold = float(info["meta"].get("threshold", 0.5))
    ds = PatchPairDataset(data_root, split, limit)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    counts = ConfusionCounts()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for a, b, m in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(normalize(a.to(device)), normalize(b.to(device)))
        pred = (torch.sigmoid(logits.float())[:, 0] > threshold).cpu().numpy()
        counts.update(pred, m.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(counts)

    result = {
        "model": key,
        "display_name": spec.display_name,
        "family": spec.family,
        "architecture": spec.architecture,
        "parameters": count_parameters(model),
        "checkpoint": info["file"],
        "checkpoint_sha256": info["sha256"],
        "checkpoint_meta": info["meta"],
        "dataset": "LEVIR-CD",
        "domain": spec.domain,
        "evaluation": "256x256 test patches",
        "split": split,
        "num_patches": len(ds),
        "patch_size": spec.tile_size,
        "threshold": threshold,
        **metrics.to_dict(),
        "eval_seconds": elapsed,
        "throughput_patches_per_s": len(ds) / elapsed if elapsed > 0 else None,
        **measure_latency(model, device, spec.tile_size, amp),
        "peak_gpu_mb": torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else None,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": amp,
        "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / f"{key}_metrics.json").write_text(json.dumps(result, indent=2))
    plot_confusion(result, out / f"{key}_confusion_matrix.png")
    plot_examples(model, ds, device, amp, threshold, spec.display_name, out / f"{key}_examples.png")
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def evaluate_full_images(key: str, ckpt_dir: Path, oscd_root: Path, out: Path, device: torch.device,
                         write: bool = True) -> dict:
    """Evaluate on whole OSCD test-city images with the application's sliding-window inference."""
    spec = MODEL_SPECS[key]
    model, info = load_model(ckpt_dir / spec.default_checkpoint, expected_key=spec.key)
    model.to(device)
    amp = device.type == "cuda"
    threshold = float(info["meta"].get("threshold", 0.5))
    cities = load_test_cities(oscd_root)
    if not cities:
        raise FileNotFoundError(f"No OSCD test cities in {oscd_root / 'test'} - run `python -m ml.data.oscd prepare`")
    total, per_city, seconds, preds = ConfusionCounts(), [], 0.0, {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for city, a, b, m in cities:
        result = predict_pair(model, a, b, device=device, tile=spec.tile_size, overlap=64, batch_size=8, amp=amp)
        pred = result.probability > threshold
        preds[city] = pred
        counts = ConfusionCounts()
        counts.update(pred, m)
        total = total + counts
        seconds += result.seconds
        cm = compute_metrics(counts)
        per_city.append({"city": city, "height": int(m.shape[0]), "width": int(m.shape[1]), "f1": cm.f1,
                         "iou": cm.iou, "precision": cm.precision, "recall": cm.recall,
                         "changed_fraction_truth": float(m.mean()), "changed_fraction_pred": float(pred.mean()),
                         "seconds": result.seconds})
    metrics = compute_metrics(total)
    out_result = {
        "model": key, "display_name": spec.display_name, "family": spec.family, "architecture": spec.architecture,
        "parameters": count_parameters(model), "checkpoint": info["file"], "checkpoint_sha256": info["sha256"],
        "checkpoint_meta": info["meta"], "dataset": "OSCD", "domain": "sentinel2",
        "evaluation": "whole test-city images, 256 px sliding window (overlap 64)", "split": "test",
        "num_images": len(cities), "num_pixels": total.total, "threshold": threshold, **metrics.to_dict(),
        "per_city": per_city, "eval_seconds": seconds,
        **measure_latency(model, device, spec.tile_size, amp),
        "peak_gpu_mb": torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else None,
        "device": device.type, "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": amp, "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if write:
        (out / f"{key}_metrics.json").write_text(json.dumps(out_result, indent=2))
        plot_confusion(out_result, out / f"{key}_confusion_matrix.png")
        plot_city_examples(cities, preds, spec.display_name, out / f"{key}_examples.png")
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_result


def plot_city_examples(cities, preds, title: str, path: Path, n: int = 5) -> None:
    ranked = sorted(cities, key=lambda c: -c[3].mean())[:n]
    fig, axes = plt.subplots(len(ranked), 4, figsize=(11, 2.8 * len(ranked)), squeeze=False)
    for row, (city, a, b, m) in enumerate(ranked):
        panels = [(a, "T1 (before)"), (b, "T2 (after)"), (m, "Ground truth"), (preds[city], "Prediction")]
        for col, (img, name) in enumerate(panels):
            axes[row, col].imshow(img, cmap=None if col < 2 else "gray", vmin=0, vmax=None if col < 2 else 1)
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{city} - {name}" if col == 0 else name, fontsize=9)
    fig.suptitle(f"{title} - OSCD test cities (Sentinel-2, 10 m)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_confusion(result: dict, path: Path) -> None:
    cm = np.array(result["confusion_matrix"]["matrix"], dtype=np.float64)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = ["no change", "change"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{result['display_name']}\n{result['dataset']} {result['split']} split", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}\n{norm[i, j] * 100:.2f}%", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def plot_examples(model, ds, device, amp, threshold, title, path: Path, n: int = 6) -> None:
    # Deterministic selection: the n patches with most changed pixels spread across the split.
    fractions = np.asarray(ds.m[: len(ds)].reshape(len(ds), -1).mean(axis=1))
    candidates = np.flatnonzero(fractions > 0.03)
    if len(candidates) == 0:
        candidates = np.arange(len(ds))
    idx = candidates[np.linspace(0, len(candidates) - 1, min(n, len(candidates))).astype(int)]
    fig, axes = plt.subplots(len(idx), 4, figsize=(10, 2.5 * len(idx)), squeeze=False)
    for row, i in enumerate(idx):
        a, b, m = ds[int(i)]
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(normalize(a[None].to(device)), normalize(b[None].to(device)))
        pred = (torch.sigmoid(logits.float())[0, 0] > threshold).cpu().numpy()
        for col, (img, name) in enumerate(
            [(a.numpy(), "T1 (before)"), (b.numpy(), "T2 (after)"), (m.numpy(), "Ground truth"), (pred, "Prediction")]
        ):
            axes[row, col].imshow(img, cmap=None if col < 2 else "gray", vmin=0, vmax=None if col < 2 else 1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(name, fontsize=10)
    fig.suptitle(f"{title} - LEVIR-CD test patches", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_comparison(out: Path) -> list[dict]:
    """(Re)build metrics.csv, summary.json and one chart per dataset from every <model>_metrics.json in ``out``."""
    results = [json.loads(p.read_text()) for p in sorted(out.glob("*_metrics.json"))]
    fields = ["model", "display_name", "family", "parameters", "checkpoint", "dataset", "evaluation", "split",
              "num_patches", "num_images", "threshold", "tp", "fp", "tn", "fn", *METRIC_KEYS, "latency_ms_median",
              "latency_ms_p95", "throughput_patches_per_s", "peak_gpu_mb", "device"]
    with (out / "metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    by_dataset: dict[str, list[dict]] = {}
    for r in results:
        by_dataset.setdefault(r.get("dataset", "LEVIR-CD"), []).append(r)
    cross = out / "cross_domain.json"
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "Within each dataset all models are evaluated on the same held-out test data with a fixed 0.5 "
                "threshold. The test data is never used for training or model selection.",
        "datasets": {name: {"split": "test", "evaluation": rows[0].get("evaluation"), "models": rows}
                     for name, rows in by_dataset.items()},
        "cross_domain": json.loads(cross.read_text()) if cross.is_file() else [],
        "models": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, rows in by_dataset.items():
        fig, ax = plt.subplots(figsize=(8, 4.2))
        width = 0.8 / len(rows)
        x = np.arange(len(METRIC_KEYS))
        for i, r in enumerate(rows):
            vals = [(r[k] or 0) * 100 for k in METRIC_KEYS]
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=r["display_name"])
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax.set_xticks(x, [k.upper() if k in {"f1", "iou"} else k.capitalize() for k in METRIC_KEYS])
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 108)
        ax.set_title(f"Change-class metrics on the {name} test split")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        suffix = "" if name == "LEVIR-CD" else "_" + name.lower().replace("-", "")
        fig.savefig(out / f"model_comparison{suffix}.png", dpi=150)
        plt.close(fig)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=sorted(MODEL_SPECS))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/levir256"))
    parser.add_argument("--oscd-root", type=Path, default=Path("data/processed/oscd256"))
    parser.add_argument("--cross-domain", action="store_true",
                        help="also score the LEVIR-CD models on the OSCD test cities (domain-shift check)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for key in args.models:
        try:
            if MODEL_SPECS[key].domain == "sentinel2":
                result = evaluate_full_images(key, args.checkpoint_dir, args.oscd_root, args.out, device)
            else:
                result = evaluate_model(key, args.checkpoint_dir, args.data_root, args.split, args.out,
                                        args.batch_size, args.limit, device)
        except (CheckpointError, FileNotFoundError) as exc:
            print(f"[{key}] skipped: {exc}", flush=True)
            continue
        results.append(result)
        print(f"[{key}] " + "  ".join(f"{k}={result[k]:.4f}" for k in METRIC_KEYS if result[k] is not None)
              + f"  latency={result['latency_ms_median']:.1f}ms", flush=True)
    if args.cross_domain:
        cross = []
        for key in [k for k, spec in MODEL_SPECS.items() if spec.domain == "vhr"]:
            try:
                r = evaluate_full_images(key, args.checkpoint_dir, args.oscd_root, args.out, device, write=False)
            except (CheckpointError, FileNotFoundError) as exc:
                print(f"[{key} on OSCD] skipped: {exc}", flush=True)
                continue
            keep = ("model", "display_name", "dataset", "precision", "recall", "f1", "iou", "accuracy", "tp", "fp",
                    "tn", "fn", "num_images", "per_city")
            cross.append({k: r[k] for k in keep} | {"trained_on": "LEVIR-CD"})
            print(f"[{key} on OSCD] f1={r['f1']} iou={r['iou']} recall={r['recall']}", flush=True)
        (args.out / "cross_domain.json").write_text(json.dumps(cross, indent=2))
    write_comparison(args.out)
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
