"""Model registry: the single place that knows which change-detection architectures exist."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from torch import nn

from ml.models.fc_siam_diff import FCSiamDiff
from ml.models.swin_siamese import SiameseSwinCD


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str  # "transformer" | "cnn"
    architecture: str
    reference: str
    tile_size: int
    builder: Callable[..., nn.Module]
    default_kwargs: dict = field(default_factory=dict)
    default_checkpoint: str = ""
    domain: str = "vhr"  # "vhr": LEVIR-CD 0.5 m aerial RGB; "sentinel2": 10 m Sentinel-2 RGB


MODEL_SPECS: dict[str, ModelSpec] = {
    "swin_siamese": ModelSpec(
        key="swin_siamese",
        display_name="Siamese Swin Transformer",
        family="transformer",
        architecture=(
            "Shared Swin-V2-Tiny encoder (4 stages) -> per-scale [F1, F2, |F1-F2|] fusion -> "
            "FPN decoder -> 1-channel change logits"
        ),
        reference="Liu et al., Swin Transformer V2 (CVPR 2022) backbone; Siamese fusion design after "
        "Zhang et al., SwinSUNet (IEEE TGRS 2022) and Bandara & Patel, ChangeFormer (IGARSS 2022)",
        tile_size=256,
        builder=SiameseSwinCD,
        default_checkpoint="swin_siamese_levir.pt",
    ),
    "swin_siamese_s2": ModelSpec(
        key="swin_siamese_s2",
        display_name="Siamese Swin Transformer (Sentinel-2)",
        family="transformer",
        architecture="Same Siamese Swin-V2-T network, fine-tuned on OSCD Sentinel-2 RGB (10 m) urban change",
        reference="Architecture as swin_siamese; data: Daudt et al., OSCD (IGARSS 2018)",
        tile_size=256,
        builder=SiameseSwinCD,
        default_checkpoint="swin_siamese_oscd.pt",
        domain="sentinel2",
    ),
    "fc_siam_diff": ModelSpec(
        key="fc_siam_diff",
        display_name="FC-Siam-diff (CNN baseline)",
        family="cnn",
        architecture="Siamese U-Net encoder (16-32-64-128) with |difference| skip connections",
        reference="Daudt, Le Saux, Boulch, Fully Convolutional Siamese Networks for Change Detection "
        "(IEEE ICIP 2018)",
        tile_size=256,
        builder=FCSiamDiff,
        default_checkpoint="fc_siam_diff_levir.pt",
    ),
    "fc_siam_diff_s2": ModelSpec(
        key="fc_siam_diff_s2",
        display_name="FC-Siam-diff CNN (Sentinel-2)",
        family="cnn",
        architecture="Same FC-Siam-diff network, fine-tuned on OSCD Sentinel-2 RGB (10 m) urban change",
        reference="Architecture as fc_siam_diff; data: Daudt et al., OSCD (IGARSS 2018)",
        tile_size=256,
        builder=FCSiamDiff,
        default_checkpoint="fc_siam_diff_oscd.pt",
        domain="sentinel2",
    ),
}


def build_model(key: str, **overrides) -> nn.Module:
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[key]
    return spec.builder(**{**spec.default_kwargs, **overrides})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
