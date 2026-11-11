"""Checkpoint save / load with structural validation.

A TerraVision checkpoint is a plain dict (safe to load with ``weights_only=True``):
    format_version, model_key, model_kwargs, state_dict, trained (bool), meta (dict)
``trained`` is only ever set by the training script after at least one optimisation epoch;
the application refuses to serve a checkpoint that is missing or not marked as trained.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.torch_version import TorchVersion

from ml.models.registry import MODEL_SPECS, build_model

FORMAT_VERSION = 1
REQUIRED_KEYS = {"format_version", "model_key", "model_kwargs", "state_dict", "trained", "meta"}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt or incompatible with the architecture."""


def save_checkpoint(path: Path, model_key: str, model: nn.Module, model_kwargs: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_key": model_key,
        "model_kwargs": model_kwargs,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "trained": True,
        "meta": meta,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path.name}")
    try:
        # TorchVersion is a plain ``str`` subclass; allow it so checkpoints whose metadata holds
        # ``torch.__version__`` still load under the safe weights-only unpickler (BUG-003).
        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Checkpoint {path.name} could not be read: {exc}") from exc
    if not isinstance(payload, dict) or not payload.keys() >= REQUIRED_KEYS:
        raise CheckpointError(f"Checkpoint {path.name} is not a TerraVision checkpoint (missing keys)")
    if payload["format_version"] != FORMAT_VERSION:
        raise CheckpointError(f"Unsupported checkpoint format version {payload['format_version']}")
    if payload["model_key"] not in MODEL_SPECS:
        raise CheckpointError(f"Checkpoint is for unknown model '{payload['model_key']}'")
    if not payload["trained"]:
        raise CheckpointError(f"Checkpoint {path.name} is not marked as trained")
    payload["meta"] = {k: str(v) if isinstance(v, TorchVersion) else v for k, v in payload["meta"].items()}
    return payload


def load_model(path: Path, expected_key: str | None = None) -> tuple[nn.Module, dict]:
    """Build the architecture named in the checkpoint and load its weights strictly."""
    payload = read_checkpoint(path)
    key = payload["model_key"]
    if expected_key is not None and key != expected_key:
        raise CheckpointError(f"Checkpoint {path.name} holds '{key}' weights, expected '{expected_key}'")
    kwargs = dict(payload["model_kwargs"])
    if key == "swin_siamese":
        kwargs["pretrained_backbone"] = False  # all weights come from the checkpoint
    model = build_model(key, **kwargs)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint {path.name} does not match the {key} architecture: {exc}") from exc
    model.eval()
    info = {
        "model_key": key,
        "meta": payload["meta"],
        "sha256": file_sha256(path),
        "file": path.name,
    }
    return model, info
