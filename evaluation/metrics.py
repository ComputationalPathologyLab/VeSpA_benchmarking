from __future__ import annotations

import numpy as np


def _to_binary(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim > 2:
        array = array[..., 0]
    return array > 0


def _confusion(pred_mask: np.ndarray, true_mask: np.ndarray) -> tuple[int, int, int, int]:
    pred = _to_binary(pred_mask)
    true = _to_binary(true_mask)

    if pred.shape != true.shape:
        raise ValueError(f"Mask shape mismatch: predicted {pred.shape}, ground truth {true.shape}")

    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, np.logical_not(true)).sum())
    fn = int(np.logical_and(np.logical_not(pred), true).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(true)).sum())
    return tp, fp, fn, tn


def dice_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    tp, fp, fn, _ = _confusion(pred_mask, true_mask)
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 1.0
    return (2 * tp) / denom


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    tp, fp, fn, _ = _confusion(pred_mask, true_mask)
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def precision_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    tp, fp, _, _ = _confusion(pred_mask, true_mask)
    denom = tp + fp
    if denom == 0:
        return 1.0
    return tp / denom


def recall_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    tp, _, fn, _ = _confusion(pred_mask, true_mask)
    denom = tp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def compute_all_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    return {
        "dice": dice_coefficient(pred_mask, true_mask),
        "iou": iou_score(pred_mask, true_mask),
        "precision": precision_score(pred_mask, true_mask),
        "recall": recall_score(pred_mask, true_mask),
    }
