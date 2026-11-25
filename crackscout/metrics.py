"""Pixel scores for binary crack masks.

Foreground is 255 and background is 0, the same encoding as DeepCrack.
These functions do not change the arrays they receive.

Zero-denominator convention:

- If the ground truth and the prediction are both empty (tp = fp = fn = 0),
  precision, recall, F1, and IoU are 1. The masks agree that there is no crack.
  The row is marked both_empty.
- Otherwise a denominator of zero leaves that score undefined (None).
  An empty prediction with cracks in the ground truth leaves precision and F1
  undefined, and sets recall and IoU to 0.
  A prediction with no cracks in the ground truth leaves recall and F1
  undefined, and sets precision and IoU to 0.
- If precision and recall are both defined and both 0, F1 is 0.
- Undefined scores are left blank in the CSV. Means skip them and the summary
  counts how many images were skipped. They are not replaced with 0.
"""

import numpy as np

FOREGROUND = 255


def binary_counts(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    """Count true positives, false positives, and false negatives.

    Both arrays must be single-channel uint8 masks of equal shape, containing
    only 0 and 255. The arrays are not resized or inverted.
    """
    _require_binary_mask(ground_truth, "ground truth")
    _require_binary_mask(prediction, "prediction")
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            "Ground truth and prediction dimensions differ: "
            f"ground truth {ground_truth.shape}, prediction {prediction.shape}"
        )
    gt_crack = ground_truth == FOREGROUND
    pred_crack = prediction == FOREGROUND
    tp = int(np.count_nonzero(gt_crack & pred_crack))
    fp = int(np.count_nonzero(~gt_crack & pred_crack))
    fn = int(np.count_nonzero(gt_crack & ~pred_crack))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_crack_pixels": tp + fn,
        "pred_crack_pixels": tp + fp,
    }


def scores_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | str | None]:
    """Turn pixel counts into precision, recall, F1, and IoU."""
    if tp == 0 and fp == 0 and fn == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "zero_denominator": "both_empty",
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None

    undefined = []
    if precision is None:
        undefined.append("precision")
    if recall is None:
        undefined.append("recall")
    if f1 is None:
        undefined.append("f1")
    if iou is None:
        undefined.append("iou")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "zero_denominator": "none" if not undefined else "undefined:" + "+".join(undefined),
    }


def _require_binary_mask(mask: np.ndarray, name: str) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError(f"Expected a single-channel uint8 {name} mask")
    values = np.unique(mask)
    if not set(int(value) for value in values.tolist()).issubset({0, FOREGROUND}):
        shown = ", ".join(str(int(value)) for value in values.tolist())
        raise ValueError(f"{name.capitalize()} mask contains values other than 0 and 255: {shown}")
