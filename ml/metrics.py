"""Binary change-detection metrics with explicit handling of zero denominators.

A metric whose denominator is zero is *undefined*, not 0 or 1: e.g. precision when the
model predicts no change pixels at all. Such metrics are returned as ``None`` and listed
in ``undefined`` so callers can display "n/a" instead of a misleading number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate counts from boolean / {0,1} arrays of identical shape."""
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: prediction {pred.shape} vs ground truth {target.shape}")
        p = pred.astype(bool)
        t = target.astype(bool)
        self.tp += int(np.count_nonzero(p & t))
        self.fp += int(np.count_nonzero(p & ~t))
        self.fn += int(np.count_nonzero(~p & t))
        self.tn += int(np.count_nonzero(~p & ~t))

    def __add__(self, other: ConfusionCounts) -> ConfusionCounts:
        return ConfusionCounts(self.tp + other.tp, self.fp + other.fp, self.tn + other.tn, self.fn + other.fn)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


@dataclass
class BinaryMetrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    iou: float | None
    accuracy: float | None
    undefined: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["confusion_matrix"] = {
            "labels": ["no_change", "change"],
            "layout": "rows = ground truth, columns = prediction",
            "matrix": [[self.tn, self.fp], [self.fn, self.tp]],
        }
        return data


def _ratio(num: int, den: int) -> float | None:
    return num / den if den > 0 else None


def compute_metrics(counts: ConfusionCounts) -> BinaryMetrics:
    """Derive precision, recall, F1, IoU (change class) and pixel accuracy from counts."""
    tp, fp, tn, fn = counts.tp, counts.fp, counts.tn, counts.fn
    values = {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
    }
    undefined = [name for name, value in values.items() if value is None]
    return BinaryMetrics(tp=tp, fp=fp, tn=tn, fn=fn, undefined=undefined, **values)


def metrics_from_masks(pred: np.ndarray, target: np.ndarray) -> BinaryMetrics:
    counts = ConfusionCounts()
    counts.update(pred, target)
    return compute_metrics(counts)
