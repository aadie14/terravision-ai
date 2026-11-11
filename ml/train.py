"""Train a change-detection model on the LEVIR-CD 256x256 patch dataset.

Usage:
    python -m ml.train --config ml/configs/swin_siamese.json
    python -m ml.train --config ml/configs/fc_siam_diff.json --epochs 5 --batch-size 4

Every run writes to ``experiments/<run_name>/``:
    config.json      resolved configuration (incl. seed, dataset manifest hash, environment)
    history.jsonl    one JSON record per epoch (losses, val F1/IoU, duration, peak GPU memory)
    summary.json     best epoch and checkpoint path
The best-validation-F1 weights are saved to ``checkpoints/<checkpoint_name>``.
The test split is never touched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.checkpoint import load_model, save_checkpoint
from ml.data.dataset import PatchPairDataset, augment, normalize
from ml.losses import BCEDiceLoss
from ml.metrics import ConfusionCounts, compute_metrics
from ml.models.registry import MODEL_SPECS, build_model, count_parameters

DEFAULTS: dict = {
    "model": "swin_siamese",
    "model_kwargs": {},
    "init_checkpoint": None,  # fine-tune from these weights (e.g. the LEVIR-CD model) instead of ImageNet
    "data_root": "data/processed/levir256",
    "epochs": 30,
    "batch_size": 2,
    "grad_accum": 4,
    "lr": 1e-4,
    "backbone_lr_mult": 0.5,
    "weight_decay": 0.01,
    "warmup_epochs": 1,
    "amp": True,
    "patience": 8,
    "seed": 42,
    "threshold": 0.5,
    "num_workers": 0,
    "limit_train": None,
    "limit_val": None,
    "run_name": None,
    "checkpoint_dir": "checkpoints",
    "experiment_dir": "experiments",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark picks conv algorithms non-deterministically; keep it off for repeatability.
    torch.backends.cudnn.benchmark = False


def adjust_for_gpu(cfg: dict, device: torch.device) -> dict:
    """Shrink the batch (keeping the effective batch) when little VRAM is free."""
    notes = {}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        notes = {"gpu_free_mb": free // 2**20, "gpu_total_mb": total // 2**20}
        effective = cfg["batch_size"] * cfg["grad_accum"]
        budget_mb_per_sample = 900 if cfg["model"] == "swin_siamese" else 250
        max_batch = max(1, int(free / 2**20 * 0.7 // budget_mb_per_sample))
        if cfg["batch_size"] > max_batch:
            cfg["batch_size"] = max_batch
            cfg["grad_accum"] = max(1, math.ceil(effective / max_batch))
            notes["batch_size_reduced_to"] = max_batch
    else:
        cfg["amp"] = False
    return notes


IGNORE_INDEX = 255  # mask value for padded pixels that must not count in loss or metrics


def split_target(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """{0, 1, 255} mask -> (binary target, validity weight)."""
    valid = (mask != IGNORE_INDEX).float()
    return (mask == 1).float(), valid


@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool, threshold: float) -> dict:
    model.eval()
    counts = ConfusionCounts()
    loss_sum, n = 0.0, 0
    for a, b, m in loader:
        a, b = normalize(a.to(device)), normalize(b.to(device))
        target, valid = split_target(m.to(device).unsqueeze(1).float())
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(a, b)
        loss_sum += criterion(logits, target, valid).item() * a.shape[0]
        n += a.shape[0]
        pred = torch.sigmoid(logits.float()) > threshold
        t = target > 0.5
        v = valid > 0.5
        counts.tp += int((pred & t & v).sum())
        counts.fp += int((pred & ~t & v).sum())
        counts.fn += int((~pred & t & v).sum())
        counts.tn += int((~pred & ~t & v).sum())
    metrics = compute_metrics(counts)
    return {"val_loss": loss_sum / max(n, 1), **metrics.to_dict()}


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_notes = adjust_for_gpu(cfg, device)
    spec = MODEL_SPECS[cfg["model"]]

    run_name = cfg["run_name"] or f"{cfg['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg["experiment_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg["checkpoint_dir"]) / spec.default_checkpoint

    data_root = Path(cfg["data_root"])
    manifest_text = (data_root / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    train_ds = PatchPairDataset(data_root, "train", cfg["limit_train"])
    val_ds = PatchPairDataset(data_root, "val", cfg["limit_val"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=cfg["num_workers"], generator=generator, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=max(cfg["batch_size"], 8), shuffle=False,
                            num_workers=cfg["num_workers"])

    model = build_model(cfg["model"], **cfg["model_kwargs"])
    init_info = None
    if cfg["init_checkpoint"]:
        init_model, init_info = load_model(Path(cfg["init_checkpoint"]))
        if type(init_model) is not type(model):
            raise ValueError("init_checkpoint holds a different architecture than the model being trained")
        model.load_state_dict(init_model.state_dict(), strict=True)
        init_info = {"file": init_info["file"], "sha256": init_info["sha256"],
                     "run_name": init_info["meta"].get("run_name"), "dataset": init_info["meta"].get("dataset")}
        del init_model
    model = model.to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg["lr"], cfg["backbone_lr_mult"]), weight_decay=cfg["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = steps_per_epoch * cfg["warmup_epochs"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"])
    aug_gen = torch.Generator().manual_seed(cfg["seed"] + 1)

    resolved = {
        **cfg,
        "run_name": run_name,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        **gpu_notes,
        "parameters": count_parameters(model),
        "train_patches": len(train_ds),
        "val_patches": len(val_ds),
        "dataset": manifest["dataset"],
        "initialised_from": init_info,
        "dataset_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2))
    print(json.dumps({k: v for k, v in resolved.items() if k not in {"files"}}, indent=2), flush=True)

    best_f1, best_epoch, stale = -1.0, 0, 0
    history_path = run_dir / "history.jsonl"
    history_path.write_text("")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss_sum, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (a, b, m) in enumerate(train_loader, start=1):
            a = normalize(a.to(device, non_blocking=True))
            b = normalize(b.to(device, non_blocking=True))
            raw = m.to(device, non_blocking=True).unsqueeze(1).float()
            a, b, raw = augment(a, b, raw, aug_gen)
            target, valid = split_target(raw)
            with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                logits = model(a, b)
            loss = criterion(logits, target, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}")
            scaler.scale(loss / cfg["grad_accum"]).backward()
            if step % cfg["grad_accum"] == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_sum += loss.item() * a.shape[0]
            seen += a.shape[0]
        val = validate(model, val_loader, criterion, device, cfg["amp"], cfg["threshold"])
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "val_loss": val["val_loss"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_iou": val["iou"],
            "lr": optimizer.param_groups[-1]["lr"],
            "epoch_seconds": round(time.perf_counter() - t0, 1),
            "peak_gpu_mb": torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else None,
        }
        f1 = val["f1"] or 0.0
        improved = f1 > best_f1
        if improved:
            best_f1, best_epoch, stale = f1, epoch, 0
            save_checkpoint(
                ckpt_path, cfg["model"], model, dict(cfg["model_kwargs"]),
                meta={
                    "run_name": run_name,
                    "epoch": epoch,
                    "val_f1": val["f1"],
                    "val_iou": val["iou"],
                    "threshold": cfg["threshold"],
                    "dataset": manifest["dataset"],
                    "initialised_from": init_info,
                    "dataset_manifest_sha256": resolved["dataset_manifest_sha256"],
                    "seed": cfg["seed"],
                    "tile_size": spec.tile_size,
                    "parameters": resolved["parameters"],
                    "torch": str(torch.__version__),
                    "saved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
        else:
            stale += 1
        record["checkpoint_saved"] = improved
        with history_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if stale >= cfg["patience"]:
            print(f"Early stopping: no val F1 improvement for {stale} epochs", flush=True)
            break

    summary = {
        "run_name": run_name,
        "model": cfg["model"],
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "epochs_run": epoch,
        "checkpoint": str(ckpt_path).replace("\\", "/"),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--experiment-dir")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)

    cfg = dict(DEFAULTS)
    if args.config:
        # utf-8-sig: tolerate the byte-order mark Windows editors / PowerShell add to JSON files
        cfg.update(json.loads(args.config.read_text(encoding="utf-8-sig")))
    for key, value in vars(args).items():
        if key in {"config", "no_amp"} or value is None:
            continue
        cfg[key] = value
    if args.no_amp:
        cfg["amp"] = False
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        parser.error(f"Unknown config keys: {sorted(unknown)}")
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
