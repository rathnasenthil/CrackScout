"""Image statistics for CrackScout, computed without ground-truth masks.

The definitions below are fixed constants. They are not adjusted from test-set
scores. A correlation in this module is not a claim that a feature predicts
or causes segmentation error.
"""

from pathlib import Path

import cv2
import numpy as np

from crackscout.deepcrack_dataset import list_pairs
from crackscout.failure_analysis import linear_percentile
from crackscout.preprocess import to_grayscale

# Pixels strictly below this grayscale value are counted as dark.
# The cutoff is a fixed point on the 0-255 scale, not a value fit to labels.
DARK_INTENSITY_THRESHOLD = 50
# Odd window for the local-mean contrast statistic.
LOCAL_CONTRAST_WINDOW = 15
LAPLACIAN_KSIZE = 3
FEATURE_NAMES = (
    "grayscale_mean",
    "grayscale_std",
    "percentile_range_p90_p10",
    "laplacian_variance",
    "dark_pixel_proportion",
    "local_contrast",
)

FEATURE_DEFINITIONS = """\
Feature definitions, computed from the original image after grayscale conversion only.
The segmentation blur is not applied. Ground-truth masks are not read.
- grayscale_mean: population mean of grayscale intensity.
- grayscale_std: population standard deviation (divisor n, not n-1).
- percentile_range_p90_p10: linear 90th percentile minus linear 10th percentile.
- laplacian_variance: variance of a 3x3 Laplacian response.
- dark_pixel_proportion: share of pixels with grayscale intensity strictly below 50.
- local_contrast: mean absolute deviation from a 15x15 box average.
"""


def extract_quality_features(image: np.ndarray) -> dict[str, float]:
    """Return quality features for one BGR or grayscale image.

    The argument is the image only. This function does not accept a mask.
    """
    gray = to_grayscale(image).astype(np.float64)
    low, high = np.percentile(gray, [10, 90], method="linear")
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=LAPLACIAN_KSIZE)
    local_mean = cv2.blur(gray, (LOCAL_CONTRAST_WINDOW, LOCAL_CONTRAST_WINDOW))
    return {
        "grayscale_mean": float(np.mean(gray)),
        "grayscale_std": float(np.std(gray)),
        "percentile_range_p90_p10": float(high - low),
        "laplacian_variance": float(np.var(laplacian)),
        "dark_pixel_proportion": float(np.mean(gray < DARK_INTENSITY_THRESHOLD)),
        "local_contrast": float(np.mean(np.abs(gray - local_mean))),
    }


def pearson_correlation(
    features: list[float],
    scores: list[float | None],
) -> tuple[float | None, int]:
    """Correlate paired values, skipping rows whose score is missing.

    Returns (coefficient, number of rows used). The coefficient is None when
    fewer than three scores are defined or either series has no variation.
    Missing scores are omitted. They are not replaced by 0 or 1.
    """
    paired = [
        (float(feature), float(score))
        for feature, score in zip(features, scores)
        if score is not None
    ]
    count = len(paired)
    if count < 3:
        return None, count
    xs = np.array([item[0] for item in paired], dtype=np.float64)
    ys = np.array([item[1] for item in paired], dtype=np.float64)
    x_center = xs - xs.mean()
    y_center = ys - ys.mean()
    denominator = float(np.sqrt(np.sum(x_center * x_center) * np.sum(y_center * y_center)))
    if denominator == 0.0:
        return None, count
    return float(np.sum(x_center * y_center) / denominator), count


def training_f1_cutoffs(
    rows: list[dict[str, object]],
    method: str,
) -> tuple[float, float, int] | None:
    """Return training Q1 and Q3 of defined F1 for one method.

    Test rows are ignored. None means fewer than four defined training scores.
    """
    values = [
        float(row["f1"])
        for row in rows
        if row["split"] == "train" and row["method"] == method and row["f1"] is not None
    ]
    if len(values) < 4:
        return None
    return linear_percentile(values, 25), linear_percentile(values, 75), len(values)


def error_group_means(
    rows: list[dict[str, object]],
    *,
    split: str,
    method: str,
    feature: str,
    low_f1_max: float,
    high_f1_min: float,
) -> dict[str, float | int | None]:
    """Mean feature value in higher-error and lower-error groups.

    Higher error means defined F1 at or below the training first quartile.
    Lower error means defined F1 at or above the training third quartile.
    Undefined F1 values are excluded from both groups.
    """
    higher: list[float] = []
    lower: list[float] = []
    excluded = 0
    for row in rows:
        if row["split"] != split or row["method"] != method:
            continue
        if row["f1"] is None:
            excluded += 1
            continue
        score = float(row["f1"])
        if score <= low_f1_max:
            higher.append(float(row[feature]))
        if score >= high_f1_min:
            lower.append(float(row[feature]))
    return {
        "higher_error_n": len(higher),
        "lower_error_n": len(lower),
        "undefined_f1_excluded": excluded,
        "higher_error_mean": (sum(higher) / len(higher)) if higher else None,
        "lower_error_mean": (sum(lower) / len(lower)) if lower else None,
    }


def join_features(
    evaluation_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Attach image features by split and stem.

    The same stem in train and test stays on separate rows. A missing image
    in the requested split is an error, not a lookup in the other split.
    """
    features_by_key = {
        (str(row["split"]), str(row["stem"])): row for row in feature_rows
    }
    joined: list[dict[str, object]] = []
    missing: list[str] = []
    for row in evaluation_rows:
        key = (str(row["split"]), str(row["stem"]))
        features = features_by_key.get(key)
        if features is None:
            missing.append(f"{key[0]}/{key[1]}")
            continue
        combined = dict(row)
        for name in FEATURE_NAMES:
            combined[name] = features[name]
        joined.append(combined)
    if missing:
        unique = ", ".join(sorted(set(missing)))
        raise ValueError(
            "No image features for these split/stem pairs. "
            f"The other split was not used as a substitute: {unique}"
        )
    return joined


def _feature_distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "min": linear_percentile(values, 0) if values else None,
        "q1": linear_percentile(values, 25) if values else None,
        "median": linear_percentile(values, 50) if values else None,
        "q3": linear_percentile(values, 75) if values else None,
        "max": linear_percentile(values, 100) if values else None,
    }


def _format_number(value: object) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.4f}"


def miss_rate(row: dict[str, object]) -> float | None:
    """False negatives divided by ground-truth crack pixels.

    The rate is undefined when the ground-truth crack count is missing or zero.
    It is not filled with 0 or 1.
    """
    gt = row.get("gt_crack_pixels")
    false_negatives = row.get("fn")
    true_positives = row.get("tp")
    if gt is None or false_negatives is None or true_positives is None:
        return None
    if int(gt) <= 0:
        return None
    return int(false_negatives) / int(gt)


def build_quality_report(joined_rows: list[dict[str, object]]) -> str:
    """Describe distributions and associations separately for each split."""
    lines = [
        "Exploratory image-quality notes",
        FEATURE_DEFINITIONS.rstrip(),
        "",
        "Correlation does not establish that a feature predicts errors or causes them.",
        "Undefined F1 values are omitted from F1 correlations and from error groups.",
        "They are not replaced with 0 or 1.",
        "Low and high F1 cutoffs are calculated from the training split of each method,",
        "then applied unchanged to that method's test rows.",
        "Test group comparisons use those frozen cutoffs.",
        "Test correlations were not used to choose the features or the cutoffs.",
        "",
    ]
    image_keys = sorted({(str(row["split"]), str(row["stem"])) for row in joined_rows})
    lines.append("Image-level feature distributions (one row per split and stem):")
    for split in ("train", "test"):
        keys = [key for key in image_keys if key[0] == split]
        lines.append(f"  {split}: {len(keys)} images")
        split_rows = []
        seen: set[str] = set()
        for row in joined_rows:
            if row["split"] != split or str(row["stem"]) in seen:
                continue
            seen.add(str(row["stem"]))
            split_rows.append(row)
        for feature in FEATURE_NAMES:
            stats = _feature_distribution([float(row[feature]) for row in split_rows])
            lines.append(
                f"    {feature}: n {stats['n']}, mean {_format_number(stats['mean'])}, "
                f"std {_format_number(stats['std'])}, "
                f"min {_format_number(stats['min'])}, q1 {_format_number(stats['q1'])}, "
                f"median {_format_number(stats['median'])}, q3 {_format_number(stats['q3'])}, "
                f"max {_format_number(stats['max'])}"
            )
    lines.append("")
    lines.append("Associations with defined per-image F1, by split and method:")
    for split in ("train", "test"):
        for method in ("otsu", "adaptive", "blackhat"):
            group = [
                row for row in joined_rows if row["split"] == split and row["method"] == method
            ]
            if not group:
                continue
            defined = sum(1 for row in group if row["f1"] is not None)
            lines.append(
                f"  {split} {method}: rows {len(group)}, defined F1 {defined}, "
                f"undefined F1 {len(group) - defined}"
            )
            for feature in FEATURE_NAMES:
                coefficient, count = pearson_correlation(
                    [float(row[feature]) for row in group],
                    [None if row["f1"] is None else float(row["f1"]) for row in group],
                )
                lines.append(
                    f"    {feature} vs F1: r {_format_number(coefficient)}, n {count}"
                )
    lines.append("")
    lines.append(
        "Associations with defined IoU and with miss rate "
        "(false negatives divided by ground-truth crack pixels):"
    )
    lines.append("Undefined IoU values are omitted. Miss rate is undefined when the ground truth has no crack pixels.")
    lines.append(
        "A row can contribute to IoU or miss rate while its F1 stays undefined. "
        "The empty training black-hat prediction has IoU 0 and miss rate 1, so those "
        "correlations include it and the F1 correlations do not."
    )
    for split in ("train", "test"):
        for method in ("otsu", "adaptive", "blackhat"):
            group = [
                row for row in joined_rows if row["split"] == split and row["method"] == method
            ]
            if not group:
                continue
            lines.append(f"  {split} {method}:")
            for feature in FEATURE_NAMES:
                iou_r, iou_n = pearson_correlation(
                    [float(row[feature]) for row in group],
                    [None if row.get("iou") is None else float(row["iou"]) for row in group],
                )
                miss_r, miss_n = pearson_correlation(
                    [float(row[feature]) for row in group],
                    [miss_rate(row) for row in group],
                )
                lines.append(
                    f"    {feature} vs IoU: r {_format_number(iou_r)}, n {iou_n}; "
                    f"vs miss rate: r {_format_number(miss_r)}, n {miss_n}"
                )
    lines.append("")
    lines.append("Higher-error and lower-error feature means:")
    lines.append("Higher error: defined F1 at or below the training first quartile.")
    lines.append("Lower error: defined F1 at or above the training third quartile.")
    for method in ("otsu", "adaptive", "blackhat"):
        cutoffs = training_f1_cutoffs(joined_rows, method)
        if cutoffs is None:
            lines.append(f"  {method}: not enough defined training F1 scores to set quartiles.")
            continue
        low_max, high_min, train_n = cutoffs
        lines.append(
            f"  {method}: training cutoff n {train_n}, "
            f"higher error F1 <= {_format_number(low_max)}, "
            f"lower error F1 >= {_format_number(high_min)}"
        )
        for split in ("train", "test"):
            role = "cutoff source" if split == "train" else "frozen-cutoff comparison"
            lines.append(f"    {split} ({role})")
            for feature in FEATURE_NAMES:
                means = error_group_means(
                    joined_rows,
                    split=split,
                    method=method,
                    feature=feature,
                    low_f1_max=low_max,
                    high_f1_min=high_min,
                )
                lines.append(
                    f"      {feature}: higher-error n {means['higher_error_n']} "
                    f"mean {_format_number(means['higher_error_mean'])}; "
                    f"lower-error n {means['lower_error_n']} "
                    f"mean {_format_number(means['lower_error_mean'])}; "
                    f"undefined F1 excluded {means['undefined_f1_excluded']}"
                )
    lines.append("")
    return "\n".join(lines)


def features_for_split(
    dataset_root: str | Path,
    split: str,
) -> list[dict[str, object]]:
    """Compute one feature row per image in a split, without reading masks."""
    rows: list[dict[str, object]] = []
    for pair in list_pairs(dataset_root, split):
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {pair.image_path}")
        features = extract_quality_features(image)
        rows.append({"split": split, "stem": pair.stem, **features})
    return rows
