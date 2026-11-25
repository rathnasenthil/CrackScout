"""Score the current segmentation defaults on DeepCrack.

Train and test are scored separately. Defaults are not changed or selected
here. Scores on the test split are exploratory until a training-only
configuration protocol and a held-out evaluation are written down.
"""

import csv
import inspect
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crackscout.deepcrack_dataset import (
    DEEPCRACK_CITATION,
    DeepCrackPair,
    list_pairs,
    load_image_and_mask,
)
from crackscout.metrics import binary_counts, scores_from_counts
from crackscout.preprocess import denoise, to_grayscale
from crackscout.segment import available_methods, segment, segment_adaptive, segment_blackhat, segment_otsu

EVALUATION_METHODS = ("otsu", "adaptive", "blackhat")
METRIC_NAMES = ("precision", "recall", "f1", "iou")

METRIC_NOTES = """\
Metric definitions (foreground = 255, background = 0):
- tp: pixels where the ground truth and the prediction are both 255
- fp: pixels predicted 255 where the ground truth is 0
- fn: pixels left 0 where the ground truth is 255
- precision = tp / (tp + fp)
- recall = tp / (tp + fn)
- F1 = 2 * precision * recall / (precision + recall)
- IoU = tp / (tp + fp + fn)
Zero denominators:
- both_empty (tp = fp = fn = 0): precision, recall, F1, and IoU are 1
- otherwise a zero denominator leaves that score blank, not 0
- precision and recall both defined and both 0: F1 is 0
Macro means skip blank scores and count those images as undefined.
Micro scores pool tp, fp, and fn across images with status ok, then use the same rules.
Ground-truth masks are not resized, inverted, or rewritten.
Prediction foreground is 255, matching DeepCrack. Predictions are not inverted to improve a score.
"""


def method_parameters(method: str) -> str:
    """Describe the shared preprocessing and the method's current defaults."""
    if method not in EVALUATION_METHODS:
        raise ValueError(f"Unknown evaluation method '{method}'.")
    functions = {
        "otsu": segment_otsu,
        "adaptive": segment_adaptive,
        "blackhat": segment_blackhat,
    }
    blur = inspect.signature(denoise).parameters["kernel_size"].default
    pieces = [f"grayscale", f"denoise kernel_size={blur}"]
    for name, param in inspect.signature(functions[method]).parameters.items():
        if name == "gray":
            continue
        pieces.append(f"{name}={param.default}")
    pieces.append("foreground=255")
    return "; ".join(pieces)


def predict_mask(image: np.ndarray, method: str) -> np.ndarray:
    """Run one default segmentation method. The result is not resized."""
    if method not in available_methods():
        raise ValueError(f"Method '{method}' is not registered.")
    mask = segment(denoise(to_grayscale(image)), method=method)
    if mask.shape != image.shape[:2]:
        raise ValueError(
            "Prediction and image dimensions differ: "
            f"prediction {mask.shape}, image {image.shape[:2]}"
        )
    return mask


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Pool successful rows and average only scores that are defined."""
    scored = [row for row in rows if row["status"] == "ok"]
    tp = sum(int(row["tp"]) for row in scored)
    fp = sum(int(row["fp"]) for row in scored)
    fn = sum(int(row["fn"]) for row in scored)
    if scored:
        micro = scores_from_counts(tp, fp, fn)
    else:
        micro = {name: None for name in METRIC_NAMES}
    summary: dict[str, object] = {
        "images": len(rows),
        "scored": len(scored),
        "errors": len(rows) - len(scored),
        "gt_crack_pixels": sum(int(row["gt_crack_pixels"]) for row in scored),
        "pred_crack_pixels": sum(int(row["pred_crack_pixels"]) for row in scored if row["pred_crack_pixels"] != ""),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    for name in METRIC_NAMES:
        summary[f"micro_{name}"] = micro[name]
        values = [row[name] for row in scored if row[name] is not None]
        summary[f"macro_{name}"] = (sum(values) / len(values)) if values else None
        summary[f"undefined_{name}"] = len(scored) - len(values)
    return summary


def _preview_path(preview_dir: Path, split: str, stem: str, method: str) -> Path:
    return preview_dir / f"{split}_{stem}_{method}_comparison.png"


def _planned_paths(
    output_dir: Path,
    pairs_by_split: dict[str, list[DeepCrackPair]],
    methods: tuple[str, ...],
    preview_count: int,
) -> list[Path]:
    paths = [
        output_dir / "per_image.csv",
        output_dir / "summary.txt",
        output_dir / "summary.csv",
    ]
    preview_dir = output_dir / "previews"
    for split, pairs in pairs_by_split.items():
        for pair in pairs[:preview_count]:
            for method in methods:
                paths.append(_preview_path(preview_dir, split, pair.stem, method))
    return paths


def _refuse_existing(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    listed = "\n".join(f"  {path}" for path in existing)
    raise FileExistsError(
        "Refusing to overwrite an existing evaluation:\n"
        f"{listed}\n"
        "Pass a new --output-dir. This command does not replace previous results."
    )


def _save_preview(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    path: Path,
    split: str,
    stem: str,
    method: str,
) -> None:
    """Save input, ground truth, prediction, and a color overlay."""
    gt_crack = ground_truth == 255
    pred_crack = prediction == 255
    overlay = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    overlay[gt_crack & pred_crack] = (0, 170, 0)
    overlay[~gt_crack & pred_crack] = (220, 30, 30)
    overlay[gt_crack & ~pred_crack] = (30, 80, 220)
    display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = (
        (display, None, "Input"),
        (ground_truth, "gray", "Ground truth (255 = crack)"),
        (prediction, "gray", "Prediction (255 = crack)"),
        (overlay, None, "Green TP, red FP, blue FN"),
    )
    for axis, (panel, cmap, title) in zip(axes, panels):
        axis.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.suptitle(
        f"{split} / {stem} / {method}. Exploratory comparison, not a final held-out result."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=100, bbox_inches="tight")
    finally:
        plt.close(figure)


def _format_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_per_image_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "stem",
        "method",
        "parameters",
        "height",
        "width",
        "status",
        "error",
        "gt_crack_pixels",
        "pred_crack_pixels",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "iou",
        "zero_denominator",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row[key]) for key in fieldnames})


def _format_metric(value: object) -> str:
    if value is None or value == "":
        return "blank"
    return f"{float(value):.4f}"


def build_summary(
    *,
    dataset_root: Path,
    output_dir: Path,
    methods: tuple[str, ...],
    pairs_available: dict[str, int],
    pairs_evaluated: dict[str, int],
    limit: int | None,
    rows: list[dict[str, object]],
    seconds_by_method: dict[tuple[str, str], float],
    total_seconds: float,
) -> str:
    """Explain the run and list aggregate scores without ranking a winner."""
    lines = [
        "DeepCrack exploratory segmentation scores",
        f"Dataset root: {dataset_root}",
        f"Output directory: {output_dir}",
        "Status: exploratory. Defaults were not chosen on the training split,",
        "and these test numbers are not a final held-out comparison.",
        "A higher score here does not mean a method is universally best.",
        "Citation:",
        DEEPCRACK_CITATION,
        "",
        METRIC_NOTES.rstrip(),
        "",
        f"Total runtime seconds: {total_seconds:.3f}",
    ]
    if limit is None:
        lines.append("Image limit: none (every pair in the requested splits).")
    else:
        lines.append(f"Image limit: first {limit} pairs in each requested split, in stem order.")
        lines.append("This limited run is a check. It is not the full-split result.")
    lines.append("")

    for split in pairs_available:
        lines.append(
            f"{split}: evaluated {pairs_evaluated[split]} of {pairs_available[split]} pairs"
        )
        for method in methods:
            group = [row for row in rows if row["split"] == split and row["method"] == method]
            stats = aggregate_rows(group)
            seconds = seconds_by_method.get((split, method), 0.0)
            parameters = method_parameters(method)
            lines.append(f"  {method}")
            lines.append(f"    parameters: {parameters}")
            lines.append(f"    segmentation seconds: {seconds:.3f}")
            lines.append(
                f"    images: {stats['images']}, scored: {stats['scored']}, errors: {stats['errors']}"
            )
            lines.append(
                "    crack pixels: "
                f"ground truth {stats['gt_crack_pixels']}, prediction {stats['pred_crack_pixels']}"
            )
            lines.append(f"    pooled counts: tp {stats['tp']}, fp {stats['fp']}, fn {stats['fn']}")
            lines.append(
                "    micro: "
                f"precision {_format_metric(stats['micro_precision'])}, "
                f"recall {_format_metric(stats['micro_recall'])}, "
                f"F1 {_format_metric(stats['micro_f1'])}, "
                f"IoU {_format_metric(stats['micro_iou'])}"
            )
            lines.append(
                "    macro: "
                f"precision {_format_metric(stats['macro_precision'])} "
                f"(undefined {stats['undefined_precision']}), "
                f"recall {_format_metric(stats['macro_recall'])} "
                f"(undefined {stats['undefined_recall']}), "
                f"F1 {_format_metric(stats['macro_f1'])} "
                f"(undefined {stats['undefined_f1']}), "
                f"IoU {_format_metric(stats['macro_iou'])} "
                f"(undefined {stats['undefined_iou']})"
            )
            for row in group:
                if row["status"] != "ok":
                    lines.append(f"    error: {row['stem']}: {row['error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_summary_csv(path: Path, summary_text_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "method",
        "parameters",
        "images",
        "scored",
        "errors",
        "gt_crack_pixels",
        "pred_crack_pixels",
        "tp",
        "fp",
        "fn",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_iou",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_iou",
        "undefined_precision",
        "undefined_recall",
        "undefined_f1",
        "undefined_iou",
        "segmentation_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_text_rows:
            writer.writerow({key: _format_cell(row.get(key, "")) for key in fieldnames})


def evaluate_deepcrack(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "test"),
    methods: tuple[str, ...] = EVALUATION_METHODS,
    limit: int | None = None,
    preview_count: int = 2,
) -> dict[str, Path]:
    """Score default methods and write CSV, a summary, and a few previews.

    Raises FileExistsError if any planned output file is already present.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if preview_count < 0:
        raise ValueError(f"preview_count must be at least 0, got {preview_count}")
    for method in methods:
        if method not in available_methods():
            raise ValueError(f"Method '{method}' is not registered.")

    root = Path(dataset_root)
    destination = Path(output_dir)
    pairs_by_split: dict[str, list[DeepCrackPair]] = {}
    pairs_available: dict[str, int] = {}
    for split in splits:
        found = list_pairs(root, split)
        pairs_available[split] = len(found)
        pairs_by_split[split] = found if limit is None else found[:limit]

    planned = _planned_paths(destination, pairs_by_split, methods, preview_count)
    _refuse_existing(planned)
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    seconds_by_method: dict[tuple[str, str], float] = {}
    started = time.perf_counter()
    parameters = {method: method_parameters(method) for method in methods}

    for split, pairs in pairs_by_split.items():
        loaded: list[tuple[DeepCrackPair, np.ndarray | None, np.ndarray | None, str]] = []
        for pair in pairs:
            try:
                image, ground_truth = load_image_and_mask(pair.image_path, pair.mask_path)
                loaded.append((pair, image, ground_truth, ""))
            except ValueError as exc:
                loaded.append((pair, None, None, str(exc)))

        for method in methods:
            method_started = time.perf_counter()
            method_rows: list[dict[str, object]] = []
            for index, (pair, image, ground_truth, load_error) in enumerate(loaded):
                if load_error:
                    row = {
                        "split": split,
                        "stem": pair.stem,
                        "method": method,
                        "parameters": parameters[method],
                        "height": "",
                        "width": "",
                        "status": "error",
                        "error": load_error,
                        "gt_crack_pixels": "",
                        "pred_crack_pixels": "",
                        "tp": "",
                        "fp": "",
                        "fn": "",
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "iou": None,
                        "zero_denominator": "",
                    }
                    method_rows.append(row)
                    continue
                assert image is not None and ground_truth is not None
                row = {
                    "split": split,
                    "stem": pair.stem,
                    "method": method,
                    "parameters": parameters[method],
                    "height": int(ground_truth.shape[0]),
                    "width": int(ground_truth.shape[1]),
                    "status": "error",
                    "error": "",
                    "gt_crack_pixels": int(np.count_nonzero(ground_truth == 255)),
                    "pred_crack_pixels": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "iou": None,
                    "zero_denominator": "",
                }
                try:
                    prediction = predict_mask(image, method)
                    counts = binary_counts(ground_truth, prediction)
                    scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
                except ValueError as exc:
                    row["error"] = str(exc)
                    method_rows.append(row)
                    continue
                row["status"] = "ok"
                row["pred_crack_pixels"] = counts["pred_crack_pixels"]
                row["tp"] = counts["tp"]
                row["fp"] = counts["fp"]
                row["fn"] = counts["fn"]
                row["precision"] = scores["precision"]
                row["recall"] = scores["recall"]
                row["f1"] = scores["f1"]
                row["iou"] = scores["iou"]
                row["zero_denominator"] = scores["zero_denominator"]
                method_rows.append(row)
                if index < preview_count:
                    _save_preview(
                        image,
                        ground_truth,
                        prediction,
                        _preview_path(destination / "previews", split, pair.stem, method),
                        split,
                        pair.stem,
                        method,
                    )
            seconds_by_method[(split, method)] = time.perf_counter() - method_started
            rows.extend(method_rows)
            stats = aggregate_rows(method_rows)
            summary_rows.append(
                {
                    "split": split,
                    "method": method,
                    "parameters": parameters[method],
                    "segmentation_seconds": seconds_by_method[(split, method)],
                    **stats,
                }
            )

    total_seconds = time.perf_counter() - started
    per_image_path = destination / "per_image.csv"
    summary_path = destination / "summary.txt"
    summary_csv_path = destination / "summary.csv"
    _write_per_image_csv(per_image_path, rows)
    summary_path.write_text(
        build_summary(
            dataset_root=root,
            output_dir=destination,
            methods=methods,
            pairs_available=pairs_available,
            pairs_evaluated={split: len(pairs) for split, pairs in pairs_by_split.items()},
            limit=limit,
            rows=rows,
            seconds_by_method=seconds_by_method,
            total_seconds=total_seconds,
        ),
        encoding="utf-8",
    )
    _write_summary_csv(summary_csv_path, summary_rows)
    return {
        "per_image_csv": per_image_path,
        "summary": summary_path,
        "summary_csv": summary_csv_path,
        "preview_dir": destination / "previews",
    }
