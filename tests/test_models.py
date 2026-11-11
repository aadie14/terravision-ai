"""Model architecture, loss, checkpoint and sliding-window inference tests (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.checkpoint import CheckpointError, load_model, read_checkpoint
from ml.inference import _positions, predict_pair, resolve_device
from ml.losses import BCEDiceLoss
from ml.models import MODEL_SPECS, FCSiamDiff, SiameseSwinCD, build_model, count_parameters
from tests.conftest import TINY_KWARGS, scene, write_tiny_checkpoint

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def swin() -> SiameseSwinCD:
    torch.manual_seed(0)
    return SiameseSwinCD(pretrained_backbone=False).eval()


def test_registry_lists_transformer_and_cnn():
    assert set(MODEL_SPECS) == {"swin_siamese", "fc_siam_diff", "swin_siamese_s2", "fc_siam_diff_s2"}
    assert {s.domain for s in MODEL_SPECS.values()} == {"vhr", "sentinel2"}
    assert MODEL_SPECS["swin_siamese"].family == "transformer"
    assert MODEL_SPECS["fc_siam_diff"].family == "cnn"
    with pytest.raises(KeyError):
        build_model("resnet")


def test_swin_forward_shape_and_parameter_count(swin):
    x1, x2 = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = swin(x1, x2)
    assert out.shape == (2, 1, 256, 256)
    assert torch.isfinite(out).all()
    assert 25e6 < count_parameters(swin) < 35e6  # Swin-V2-T encoder (~27.6 M) + decoder


def test_swin_encoder_is_shared_and_pyramid_has_four_scales(swin):
    feats = swin.encode(torch.randn(1, 3, 256, 256))
    assert [f.shape[1] for f in feats] == [96, 192, 384, 768]
    assert [f.shape[-1] for f in feats] == [64, 32, 16, 8]
    # one encoder module instance => weights are shared between dates by construction
    assert sum(1 for name, _ in swin.named_children() if name == "encoder") == 1


def test_swin_rejects_wrong_tile_size_and_mismatched_inputs(swin):
    with pytest.raises(ValueError, match="256x256"):
        swin(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    with pytest.raises(ValueError, match="identical shapes"):
        swin(torch.randn(1, 3, 256, 256), torch.randn(2, 3, 256, 256))


def test_swin_identical_inputs_still_produce_valid_logits(swin):
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        assert swin(x, x).shape == (1, 1, 256, 256)


@pytest.mark.parametrize("size", [(256, 256), (64, 96), (100, 132)])
def test_fc_siam_diff_forward_any_size(size):
    model = FCSiamDiff().eval()
    x = torch.randn(1, 3, *size)
    with torch.no_grad():
        out = model(x, x)
    assert out.shape == (1, 1, *size)


def test_fc_siam_diff_is_symmetric_in_time_for_difference_skips():
    # |F1 - F2| skips are symmetric; the bottleneck comes from T2, so outputs differ only via that path
    model = FCSiamDiff(dropout=0.0).eval()
    a, b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        s1, _ = model.encode(a)
        s2, _ = model.encode(b)
    for f1, f2 in zip(s1, s2, strict=True):
        assert torch.allclose((f1 - f2).abs(), (f2 - f1).abs())


def test_loss_is_finite_decreases_and_handles_empty_targets():
    loss = BCEDiceLoss()
    target = torch.zeros(2, 1, 32, 32)
    target[0, :, 8:16, 8:16] = 1
    good = (target * 2 - 1) * 8.0
    bad = -good
    assert loss(good, target) < loss(bad, target)
    empty = torch.zeros(2, 1, 32, 32)
    confident_empty = torch.full((2, 1, 32, 32), -12.0)
    value = loss(confident_empty, empty)
    assert torch.isfinite(value) and value < 0.05


def test_loss_gradient_flows():
    logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    target = torch.zeros(1, 1, 16, 16)
    target[..., :8] = 1
    BCEDiceLoss()(logits, target).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


# ------------------------------------------------------------------ checkpoints
def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "fc_siam_diff_levir.pt"
    write_tiny_checkpoint(path)
    model, info = load_model(path, expected_key="fc_siam_diff")
    assert not model.training
    assert info["model_key"] == "fc_siam_diff" and len(info["sha256"]) == 64
    reference = build_model("fc_siam_diff", **TINY_KWARGS)
    assert count_parameters(model) == count_parameters(reference)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_model(tmp_path / "nope.pt")


def test_corrupt_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"this is not a torch file")
    with pytest.raises(CheckpointError, match="could not be read"):
        read_checkpoint(path)


def test_foreign_checkpoint_without_metadata_is_rejected(tmp_path):
    path = tmp_path / "raw.pt"
    torch.save(build_model("fc_siam_diff", **TINY_KWARGS).state_dict(), path)
    with pytest.raises(CheckpointError, match="not a TerraVision checkpoint"):
        read_checkpoint(path)


def test_untrained_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "untrained.pt"
    write_tiny_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload["trained"] = False
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="not marked as trained"):
        read_checkpoint(path)


def test_checkpoint_for_other_model_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    write_tiny_checkpoint(path)
    with pytest.raises(CheckpointError, match="expected 'swin_siamese'"):
        load_model(path, expected_key="swin_siamese")


def test_architecture_mismatch_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    write_tiny_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload["model_kwargs"] = {"base": 8, "dropout": 0.0}
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="does not match"):
        load_model(path)


# ------------------------------------------------------------------- inference
@pytest.fixture(scope="module")
def tiny_cnn() -> FCSiamDiff:
    torch.manual_seed(0)
    return build_model("fc_siam_diff", **TINY_KWARGS).eval()


@pytest.mark.parametrize("shape", [(256, 256), (300, 420), (64, 64), (257, 100), (512, 512)])
def test_sliding_window_output_matches_input_size(tiny_cnn, shape):
    a, b = scene(*shape, seed=1), scene(*shape, seed=2)
    result = predict_pair(tiny_cnn, a, b, device=CPU, tile=256, overlap=32, batch_size=2)
    assert result.probability.shape == shape
    assert result.probability.dtype == np.float32
    assert result.probability.min() >= 0.0 and result.probability.max() <= 1.0
    assert result.device == "cpu" and result.amp is False and result.seconds > 0


def test_tile_positions_cover_image_exactly():
    assert _positions(256, 256, 192) == [0]
    assert _positions(100, 256, 192) == [0]
    pos = _positions(1024, 256, 192)
    assert pos[0] == 0 and pos[-1] == 1024 - 256
    covered = np.zeros(1024, bool)
    for p in pos:
        covered[p : p + 256] = True
    assert covered.all()


def test_inference_is_deterministic(tiny_cnn):
    a, b = scene(128, 128, seed=3), scene(128, 128, seed=4)
    r1 = predict_pair(tiny_cnn, a, b, device=CPU, overlap=0)
    r2 = predict_pair(tiny_cnn, a, b, device=CPU, overlap=0)
    assert np.array_equal(r1.probability, r2.probability)


def test_inference_input_validation(tiny_cnn):
    a = scene(64, 64)
    with pytest.raises(ValueError, match="shapes differ"):
        predict_pair(tiny_cnn, a, scene(64, 80), device=CPU)
    with pytest.raises(ValueError, match="uint8"):
        predict_pair(tiny_cnn, a.astype(np.float32), a.astype(np.float32), device=CPU)
    with pytest.raises(ValueError, match="overlap"):
        predict_pair(tiny_cnn, a, a, device=CPU, overlap=256)


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")
