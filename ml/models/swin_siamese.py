"""Siamese Swin Transformer change-detection network.

    T1 --> shared Swin encoder --> {F1_s} s=1..4 --+
                                                   +--> per-scale fusion --> FPN decoder --> logits
    T2 --> shared Swin encoder --> {F2_s} s=1..4 --+

The encoder is a hierarchical Swin Transformer V2-Tiny (timm ``swinv2_tiny_window8_256``,
ImageNet-1k pretrained) whose weights are shared between the two dates. At every one of
the four pyramid levels the two feature maps are fused with ``[F1, F2, |F1 - F2|]`` followed
by a 1x1 projection, and a lightweight top-down FPN decoder turns the fused pyramid into
full-resolution change logits.

The network operates on fixed ``img_size`` tiles (default 256); larger rasters are handled
by sliding-window inference in ``ml.inference``.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class TemporalFusion(nn.Module):
    """Fuse bitemporal features of one scale: concat(F1, F2, |F1 - F2|) -> 1x1 -> 3x3."""

    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = ConvBNReLU(channels * 3, out_channels, kernel=1)
        self.refine = ConvBNReLU(out_channels, out_channels, kernel=3)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        return self.refine(self.project(torch.cat([f1, f2, (f1 - f2).abs()], dim=1)))


class FPNDecoder(nn.Module):
    """Top-down pathway with lateral additions, then multi-scale aggregation."""

    def __init__(self, channels: int, num_levels: int) -> None:
        super().__init__()
        self.smooth = nn.ModuleList(ConvBNReLU(channels, channels) for _ in range(num_levels))
        self.aggregate = ConvBNReLU(channels * num_levels, channels)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats are ordered fine -> coarse
        outs = [feats[-1]]
        for lateral in reversed(feats[:-1]):
            up = F.interpolate(outs[0], size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            outs.insert(0, lateral + up)
        outs = [smooth(o) for smooth, o in zip(self.smooth, outs, strict=True)]
        size = outs[0].shape[-2:]
        merged = [outs[0]] + [
            F.interpolate(o, size=size, mode="bilinear", align_corners=False) for o in outs[1:]
        ]
        return self.aggregate(torch.cat(merged, dim=1))


class SiameseSwinCD(nn.Module):
    """Siamese Swin-V2-T encoder + difference fusion + FPN decoder -> 1-channel change logits."""

    def __init__(
        self,
        backbone: str = "swinv2_tiny_window8_256",
        pretrained_backbone: bool = True,
        decoder_channels: int = 128,
        img_size: int = 256,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.backbone_name = backbone
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
        )
        enc_channels: list[int] = self.encoder.feature_info.channels()
        self.fusions = nn.ModuleList(TemporalFusion(c, decoder_channels) for c in enc_channels)
        self.decoder = FPNDecoder(decoder_channels, len(enc_channels))
        self.head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        # timm Swin feature maps are channels-last (B, H, W, C); convert to (B, C, H, W).
        return [f.permute(0, 3, 1, 2).contiguous() for f in self.encoder(x)]

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        if t1.shape != t2.shape:
            raise ValueError(f"T1 and T2 must have identical shapes, got {tuple(t1.shape)} vs {tuple(t2.shape)}")
        if t1.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"Expected {self.img_size}x{self.img_size} tiles, got {tuple(t1.shape[-2:])}")
        batch = t1.shape[0]
        # One encoder pass over both dates (weights shared by construction).
        feats = self.encode(torch.cat([t1, t2], dim=0))
        fused = [fuse(f[:batch], f[batch:]) for fuse, f in zip(self.fusions, feats, strict=True)]
        decoded = self.decoder(fused)
        logits = self.head(decoded)
        return F.interpolate(logits, size=t1.shape[-2:], mode="bilinear", align_corners=False)

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.5) -> list[dict]:
        backbone = list(self.encoder.parameters())
        backbone_ids = {id(p) for p in backbone}
        rest = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone, "lr": lr * backbone_lr_mult},
            {"params": rest, "lr": lr},
        ]
