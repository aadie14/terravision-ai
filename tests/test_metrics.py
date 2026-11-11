"""Metric correctness: cross-checked against scikit-learn, plus zero-denominator safeguards."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score, precision_score, recall_score

from ml.metrics import ConfusionCounts, compute_metrics, metrics_from_masks


@pytest.mark.parametrize("seed,p_pred,p_true", [(0, 0.3, 0.3), (1, 0.05, 0.5), (2, 0.9, 0.1), (3, 0.5, 0.02)])
def test_metrics_match_sklearn(seed, p_pred, p_true):
    rng = np.random.default_rng(seed)
    pred = rng.random((64, 80)) < p_pred
    truth = rng.random((64, 80)) < p_true
    m = metrics_from_masks(pred, truth)
    yp, yt = pred.ravel(), truth.ravel()
    assert m.precision == pytest.approx(precision_score(yt, yp))
    assert m.recall == pytest.approx(recall_score(yt, yp))
    assert m.f1 == pytest.approx(f1_score(yt, yp))
    assert m.iou == pytest.approx(jaccard_score(yt, yp))
    assert m.accuracy == pytest.approx(accuracy_score(yt, yp))
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[False, True]).ravel()
    assert (m.tn, m.fp, m.fn, m.tp) == (tn, fp, fn, tp)
    assert m.to_dict()["confusion_matrix"]["matrix"] == [[tn, fp], [fn, tp]]


def test_hand_computed_example():
    pred = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    truth = np.array([[1, 0, 1, 0], [1, 0, 0, 0]])
    m = metrics_from_masks(pred, truth)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 4)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.iou == pytest.approx(0.5)
    assert m.accuracy == pytest.approx(0.75)
    assert m.undefined == []


def test_perfect_prediction():
    mask = np.zeros((10, 10), bool)
    mask[2:5, 2:5] = True
    m = metrics_from_masks(mask, mask)
    assert m.precision == m.recall == m.f1 == m.iou == m.accuracy == 1.0


def test_empty_prediction_makes_precision_undefined_not_zero():
    truth = np.zeros((8, 8), bool)
    truth[0, 0] = True
    m = metrics_from_masks(np.zeros((8, 8), bool), truth)
    assert m.precision is None and "precision" in m.undefined
    assert m.recall == 0.0 and m.f1 == 0.0 and m.iou == 0.0


def test_empty_ground_truth_makes_recall_undefined():
    pred = np.zeros((8, 8), bool)
    pred[0, 0] = True
    m = metrics_from_masks(pred, np.zeros((8, 8), bool))
    assert m.recall is None and "recall" in m.undefined
    assert m.precision == 0.0


def test_no_change_pair_all_change_metrics_undefined_accuracy_one():
    empty = np.zeros((16, 16), bool)
    m = metrics_from_masks(empty, empty)
    assert m.accuracy == 1.0
    assert m.precision is None and m.recall is None and m.f1 is None and m.iou is None
    assert set(m.undefined) == {"precision", "recall", "f1", "iou"}


def test_zero_pixels_everything_undefined():
    m = compute_metrics(ConfusionCounts())
    assert m.accuracy is None and len(m.undefined) == 5


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics_from_masks(np.zeros((4, 4), bool), np.zeros((4, 5), bool))


def test_counts_accumulate_and_add():
    a, b = ConfusionCounts(), ConfusionCounts()
    a.update(np.array([1, 0]), np.array([1, 1]))
    b.update(np.array([0, 1]), np.array([0, 0]))
    total = a + b
    assert (total.tp, total.fn, total.tn, total.fp) == (1, 1, 1, 1)
    assert total.total == 4
