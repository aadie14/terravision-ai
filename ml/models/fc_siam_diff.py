"""FC-Siam-diff CNN baseline.

Re-implementation of the fully-convolutional Siamese-difference network from
R. C. Daudt, B. Le Saux, A. Boulch, "Fully Convolutional Siamese Networks for Change
Detection", IEEE ICIP 2018. A U-Net style encoder with shared weights processes both
dates; skip connections carry the absolute difference of the two encoder feature maps
into the decoder. Layer widths (16-32-64-128) follow the paper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def _block(in_ch: int, out_ch: int, n: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        ]
    return nn.Sequential(*layers)


class _Up(nn.Module):
    """Transposed-conv upsample, concat with difference skip, then conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2, padding=1, output_padding=1)
        self.block = _block(in_ch + skip_ch, out_ch, n, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dh, dw = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, (0, dw, 0, dh), mode="replicate")
        return self.block(torch.cat([x, skip], dim=1))


class FCSiamDiff(nn.Module):
    """FC-Siam-diff: Siamese encoder, |difference| skip connections, 1-channel logits."""

    def __init__(self, in_channels: int = 3, base: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = _block(in_channels, c1, 2, dropout)
        self.enc2 = _block(c1, c2, 2, dropout)
        self.enc3 = _block(c2, c3, 3, dropout)
        self.enc4 = _block(c3, c4, 3, dropout)
        self.up4 = _Up(c4, c4, c3, 3, dropout)
        self.up3 = _Up(c3, c3, c2, 3, dropout)
        self.up2 = _Up(c2, c2, c1, 2, dropout)
        self.up1 = _Up(c1, c1, c1, 1, dropout)
        self.head = nn.Conv2d(c1, 1, 3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = []
        for enc in (self.enc1, self.enc2, self.enc3, self.enc4):
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        return skips, x

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        if t1.shape != t2.shape:
            raise ValueError(f"T1 and T2 must have identical shapes, got {tuple(t1.shape)} vs {tuple(t2.shape)}")
        s1, _ = self.encode(t1)
        s2, bottom = self.encode(t2)
        diffs = [(a - b).abs() for a, b in zip(s1, s2, strict=True)]
        x = self.up4(bottom, diffs[3])
        x = self.up3(x, diffs[2])
        x = self.up2(x, diffs[1])
        x = self.up1(x, diffs[0])
        return self.head(x)

    def param_groups(self, lr: float, backbone_lr_mult: float = 1.0) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr}]
