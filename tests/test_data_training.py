"""Dataset preparation, augmentation, and an end-to-end train -> evaluate smoke test on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml import evaluate as evaluate_module
from ml import train as train_module
from ml.checkpoint import load_model
from ml.data.dataset import PatchPairDataset, augment, normalize
from ml.data.download import _safe_extract, md5sum
from ml.data.prepare import DatasetError, check_no_leakage, list_split, prepare, tile
from tests.conftest import scene, with_building


def make_raw_dataset(root: Path, counts=(("train", 3), ("val", 1), ("test", 1)), size: int = 512) -> None:
    for split, n in counts:
        for sub in ("A", "B", "label"):
            (root / split / sub).mkdir(parents=True)
        for i in range(n):
            a = scene(size, size, seed=hash((split, i)) % 1000)
            b, mask = with_building(a, (40 + 30 * i, 60, 200 + 30 * i, 260))
            name = f"{split}_{i + 1}.png"
            Image.fromarray(a).save(root / split / "A" / name)
            Image.fromarray(b).save(root / split / "B" / name)
            Image.fromarray(mask.astype(np.uint8) * 255).save(root / split / "label" / name)


def test_tile_cuts_non_overlapping_patches_in_row_major_order():
    arr = np.arange(8 * 8).reshape(8, 8)
    patches = tile(arr, patch=4)
    assert patches.shape == (4, 4, 4)
    assert np.array_equal(patches[0], arr[:4, :4]) and np.array_equal(patches[1], arr[:4, 4:])
    assert np.array_equal(patches[3], arr[4:, 4:])
    rgb = np.random.default_rng(0).integers(0, 255, (512, 768, 3), dtype=np.uint8)
    out = tile(rgb, patch=256)
    assert out.shape == (6, 256, 256, 3) and np.array_equal(out[4], rgb[256:, 256:512])
    assert tile(np.zeros((300, 300)), patch=256).shape == (1, 256, 256)  # ragged border dropped


def test_prepare_writes_memmaps_and_manifest(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "proc"
    make_raw_dataset(raw)
    manifest = prepare(raw, out)
    assert manifest["splits"]["train"]["num_images"] == 3 and manifest["splits"]["train"]["num_patches"] == 12
    assert manifest["splits"]["test"]["num_patches"] == 4
    assert 0 < manifest["splits"]["train"]["changed_pixel_fraction"] < 0.5
    assert json.loads((out / "manifest.json").read_text())["patch_size"] == 256
    ds = PatchPairDataset(out, "train")
    a, _b, m = ds[0]
    assert len(ds) == 12 and a.shape == (256, 256, 3) and a.dtype == torch.uint8
    assert m.shape == (256, 256) and set(m.unique().tolist()) <= {0, 1}
    assert len(PatchPairDataset(out, "train", limit=5)) == 5


def test_prepare_detects_unpaired_files_missing_folders_and_leakage(tmp_path):
    raw = tmp_path / "raw"
    make_raw_dataset(raw)
    (raw / "train" / "B" / "train_1.png").unlink()
    with pytest.raises(DatasetError, match="Unpaired files"):
        list_split(raw, "train")
    with pytest.raises(DatasetError, match="Missing folder"):
        list_split(tmp_path / "nowhere", "train")
    with pytest.raises(DatasetError, match="Leakage"):
        check_no_leakage({"train": ["a.png", "b.png"], "val": ["c.png"], "test": ["b.png"]})
    check_no_leakage({"train": ["a.png"], "val": ["b.png"], "test": ["c.png"]})


def test_missing_processed_dataset_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"ml\.data\.prepare"):
        PatchPairDataset(tmp_path, "train")


def test_normalize_uses_imagenet_statistics():
    batch = torch.full((1, 4, 4, 3), 255, dtype=torch.uint8)
    out = normalize(batch)
    assert out.shape == (1, 3, 4, 4)
    assert out[0, :, 0, 0].tolist() == pytest.approx([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225])


def test_augmentation_keeps_images_and_mask_geometrically_consistent():
    generator = torch.Generator().manual_seed(3)
    for _ in range(20):
        mask = torch.zeros(2, 1, 32, 32)
        mask[:, :, 2:9, 5:20] = 1
        # encode the mask position inside the images so the transforms can be compared
        a = mask.repeat(1, 3, 1, 1) * 5.0
        b = mask.repeat(1, 3, 1, 1) * 5.0
        a2, b2, m2 = augment(a.clone(), b.clone(), mask.clone(), generator)
        assert m2.sum() == mask.sum() and set(m2.unique().tolist()) == {0.0, 1.0}
        for img in (a2, b2):
            inside, outside = img[:, 0][m2[:, 0] == 1], img[:, 0][m2[:, 0] == 0]
            assert inside.min() > outside.max()  # the bright block still coincides with the mask


def test_safe_extract_blocks_path_traversal(tmp_path):
    import zipfile

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError, match="Unsafe path"):
        _safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
    assert md5sum(archive) == md5sum(archive) and len(md5sum(archive)) == 32


def test_train_then_evaluate_smoke(tmp_path, monkeypatch):
    """One CPU epoch on synthetic data must produce a loadable trained checkpoint, logs and reports."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    make_raw_dataset(raw, counts=(("train", 2), ("val", 1), ("test", 1)))
    prepare(raw, proc)
    cfg = dict(train_module.DEFAULTS)
    cfg.update({"model": "fc_siam_diff", "model_kwargs": {"base": 4, "dropout": 0.0}, "data_root": str(proc),
                "epochs": 2, "batch_size": 4, "grad_accum": 2, "lr": 1e-3, "run_name": "smoke",
                "checkpoint_dir": str(tmp_path / "ckpt"), "experiment_dir": str(tmp_path / "exp"), "patience": 5})
    summary = train_module.train(cfg)

    run_dir = tmp_path / "exp" / "smoke"
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert len(history) == summary["epochs_run"] == 2
    for record in history:
        assert {"train_loss", "val_loss", "val_f1", "val_iou", "epoch_seconds", "peak_gpu_mb", "lr"} <= set(record)
        assert np.isfinite(record["train_loss"]) and np.isfinite(record["val_loss"])
    config = json.loads((run_dir / "config.json").read_text())
    assert config["seed"] == 42 and config["device"] == "cpu" and config["amp"] is False
    assert len(config["dataset_manifest_sha256"]) == 64 and config["train_patches"] == 8

    ckpt = tmp_path / "ckpt" / "fc_siam_diff_levir.pt"
    model, info = load_model(ckpt, expected_key="fc_siam_diff")
    assert info["meta"]["run_name"] == "smoke" and info["meta"]["dataset"] == "LEVIR-CD"
    assert info["meta"]["parameters"] == sum(p.numel() for p in model.parameters())

    out = tmp_path / "reports"
    code = evaluate_module.main(["--models", "fc_siam_diff", "swin_siamese", "--data-root", str(proc),
                                 "--checkpoint-dir", str(tmp_path / "ckpt"), "--out", str(out), "--device", "cpu",
                                 "--batch-size", "4"])
    assert code == 0
    report = json.loads((out / "fc_siam_diff_metrics.json").read_text())
    assert report["split"] == "test" and report["num_patches"] == 4
    assert report["tp"] + report["fp"] + report["tn"] + report["fn"] == 4 * 256 * 256
    assert report["latency_ms_median"] > 0 and report["checkpoint_sha256"] == info["sha256"]
    summary_json = json.loads((out / "summary.json").read_text())
    assert [m["model"] for m in summary_json["models"]] == ["fc_siam_diff"]  # swin skipped: no checkpoint
    for name in ("metrics.csv", "model_comparison.png", "fc_siam_diff_confusion_matrix.png",
                 "fc_siam_diff_examples.png"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0


def test_train_cli_rejects_unknown_config_keys(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text('﻿{"model": "fc_siam_diff", "not_a_real_option": 1}', encoding="utf-8")  # with BOM
    with pytest.raises(SystemExit):
        train_module.main(["--config", str(cfg)])


# ------------------------------------------------------------------ OSCD / Sentinel-2 pipeline
def test_oscd_tiles_pad_small_images_and_mark_padding_as_ignore():
    from ml.data.oscd import IGNORE, PATCH, _tiles

    img = np.arange(200 * 300 * 3, dtype=np.uint32).reshape(200, 300, 3).astype(np.uint8)
    mask = np.ones((200, 300), np.uint8)
    tiles, masks = _tiles(img, None), _tiles(mask, IGNORE)
    assert len(tiles) == len(masks) == 2  # x positions 0 and 300-256=44
    assert all(t.shape == (PATCH, PATCH, 3) for t in tiles)
    assert (masks[0][200:] == IGNORE).all() and (masks[0][:200] == 1).all()


def test_oscd_rgb_gain_matches_measured_tci_scale():
    """DN 2500 maps to 255, as measured on Earth Search TCI (reports/tci_gain_fit.json)."""
    from ml.data.oscd import RGB_GAIN

    assert round(2500 * RGB_GAIN) == 255 and pytest.approx(127.5) == 1250 * RGB_GAIN


def test_ignore_pixels_do_not_contribute_to_loss_or_validation_counts():
    from ml.losses import BCEDiceLoss
    from ml.train import IGNORE_INDEX, split_target

    raw = torch.zeros(1, 1, 8, 8)
    raw[..., :4] = 1
    raw[..., 6:] = IGNORE_INDEX
    target, valid = split_target(raw)
    assert target.sum() == 32 and valid.sum() == 48
    logits = (target * 2 - 1) * 10.0
    garbage = logits.clone()
    garbage[..., 6:] = 10.0  # confidently wrong only where the pixels are ignored
    loss = BCEDiceLoss()
    assert torch.isclose(loss(logits, target, valid), loss(garbage, target, valid), atol=1e-6)
    assert loss(garbage, target) > loss(garbage, target, valid)  # without the mask the padding would be penalised


def test_train_fine_tunes_from_init_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    make_raw_dataset(raw, counts=(("train", 1), ("val", 1), ("test", 1)))
    prepare(raw, proc)
    from tests.conftest import write_tiny_checkpoint

    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init)
    cfg = dict(train_module.DEFAULTS)
    cfg.update({"model": "fc_siam_diff_s2", "model_kwargs": {"base": 4, "dropout": 0.0}, "data_root": str(proc),
                "init_checkpoint": str(init), "epochs": 1, "batch_size": 2, "grad_accum": 1, "run_name": "ft",
                "checkpoint_dir": str(tmp_path / "ckpt"), "experiment_dir": str(tmp_path / "exp")})
    train_module.train(cfg)
    _, info = load_model(tmp_path / "ckpt" / "fc_siam_diff_oscd.pt", expected_key="fc_siam_diff_s2")
    assert info["meta"]["initialised_from"]["run_name"] == "pytest-tiny"
    cfg["model"] = "swin_siamese_s2"
    with pytest.raises(ValueError, match="different architecture"):
        train_module.train({**cfg, "model_kwargs": {}})
