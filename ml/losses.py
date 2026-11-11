"""BCE + soft-Dice loss for binary change segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class BCEDiceLoss(nn.Module):
    """``bce_weight * BCE + dice_weight * (1 - Dice)``.

    Dice is computed over the *whole batch* rather than per tile. Roughly 55% of LEVIR-CD
    256x256 tiles contain no change at all; a per-tile Dice gives each of those tiles a loss
    of ~1 for any non-zero probability mass (1 - smooth / (sum(p) + smooth)), which swamps the
    gradient and drives a from-scratch network to predict "no change" everywhere (BUG-002).
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """``valid`` (optional, same shape) marks pixels that count; padded pixels are excluded."""
        logits = logits.float()
        target = target.float()
        weight = torch.ones_like(target) if valid is None else valid.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * weight).sum() / weight.sum().clamp_min(1)
        prob = torch.sigmoid(logits) * weight
        target = target * weight
        intersection = (prob * target).sum()
        denom = prob.sum() + target.sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice
