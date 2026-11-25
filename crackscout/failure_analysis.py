"""Read-only notes on an existing DeepCrack evaluation CSV.

This module does not change masks, predictions, or the evaluation files it reads.
Undefined precision and F1 stay undefined. They are not filled in with 0 or 1.
"""

import csv
from pathlib import Path

from crackscout.evaluate import aggregate_rows
from crackscout.metrics import scores_from_counts

EXPECTED_SPLITS = ("train", "test")
EXPECTED_METHODS = ("otsu", "adaptive", "blackhat")
EXPECTED_PAIR_COUNTS = {"train": 300, "test": 237}
METRIC_FIELDS = ("precision", "recall", "f1", "iou")
# Stored metrics were written with six decimal places.
METRIC_TOLERANCE = 1e-6


def linear_percentile(values: list[float], percent: float) -> float:
    """Linear percentile of a non-empty list.

    The position is (n - 1) * percent / 100, and the result blends the two
    nearest sorted values. Undefined scores must already have been removed.
    """
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percent / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def f1_distribution(values: list[float]) -> dict[str, float | int]:
    """Summarise defined F1 scores. The caller must not insert blanks as zeros."""
    return {
        "n": len(values),
        "min": linear_percentile(values, 0),
        "q1": linear_percentile(values, 25),
        "median": linear_percentile(values, 50),
        "q3": linear_percentile(values, 75),
        "max": linear_percentile(values, 100),
    }


def load_per_image_csv(path: str | Path) -> list[dict[str, object]]:
    """Read per-image evaluation rows. Blank metric cells become None."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Per-image CSV not found: {csv_path}")
    rows: list[dict[str, object]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Per-image CSV has no header: {csv_path}")
        required = {
            "split",
            "stem",
            "method",
            "status",
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
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Per-image CSV is missing columns: {', '.join(missing)}")
        for raw in reader:
            rows.append(_parse_row(raw))
    return rows


def _parse_optional_float(text: str) -> float | None:
    stripped = text.strip()
    if stripped == "":
        return None
    return float(stripped)


def _parse_optional_int(text: str) -> int | None:
    stripped = text.strip()
    if stripped == "":
        return None
    return int(stripped)


def _parse_row(raw: dict[str, str]) -> dict[str, object]:
    return {
        "split": raw["split"].strip(),
        "stem": raw["stem"].strip(),
        "method": raw["method"].strip(),
        "parameters": raw.get("parameters", "").strip(),
        "status": raw["status"].strip(),
        "error": raw.get("error", "").strip(),
        "gt_crack_pixels": _parse_optional_int(raw["gt_crack_pixels"]),
        "pred_crack_pixels": _parse_optional_int(raw["pred_crack_pixels"]),
        "tp": _parse_optional_int(raw["tp"]),
        "fp": _parse_optional_int(raw["fp"]),
        "fn": _parse_optional_int(raw["fn"]),
        "precision": _parse_optional_float(raw["precision"]),
        "recall": _parse_optional_float(raw["recall"]),
        "f1": _parse_optional_float(raw["f1"]),
        "iou": _parse_optional_float(raw["iou"]),
        "zero_denominator": raw["zero_denominator"].strip(),
    }


def validate_inventory(
    rows: list[dict[str, object]],
    *,
    splits: tuple[str, ...] = EXPECTED_SPLITS,
    methods: tuple[str, ...] = EXPECTED_METHODS,
    expected_pair_counts: dict[str, int] | None = EXPECTED_PAIR_COUNTS,
) -> list[str]:
    """Check split/method coverage, duplicates, and missing combinations."""
    problems: list[str] = []
    seen: dict[tuple[str, str, str], int] = {}
    stems_by_split_method: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = (str(row["split"]), str(row["stem"]), str(row["method"]))
        seen[key] = seen.get(key, 0) + 1
        stems_by_split_method.setdefault((key[0], key[2]), set()).add(key[1])
    for key, count in sorted(seen.items()):
        if count > 1:
            problems.append(
                f"Duplicate row: split {key[0]}, stem {key[1]}, method {key[2]} appears {count} times"
            )

    for split in splits:
        stem_sets = []
        for method in methods:
            stems = stems_by_split_method.get((split, method), set())
            stem_sets.append(stems)
            if expected_pair_counts is not None:
                expected = expected_pair_counts[split]
                if len(stems) != expected:
                    problems.append(
                        f"{split}/{method} has {len(stems)} stems; expected {expected}"
                    )
        if stem_sets:
            shared = set.intersection(*stem_sets) if stem_sets else set()
            union = set.union(*stem_sets) if stem_sets else set()
            for stem in sorted(union - shared):
                missing_methods = [
                    method
                    for method in methods
                    if stem not in stems_by_split_method.get((split, method), set())
                ]
                problems.append(
                    f"{split} stem {stem} is missing method(s): {', '.join(missing_methods)}"
                )
    return problems


def formula_mismatches(rows: list[dict[str, object]]) -> list[str]:
    """Recompute scores from tp, fp, and fn and compare them with the CSV.

    Rows without those three counts are reported as not verifiable. Stored
    blank metrics are compared with None, not with zero.
    """
    mismatches: list[str] = []
    for row in rows:
        label = f"{row['split']}/{row['stem']}/{row['method']}"
        if row["status"] != "ok":
            mismatches.append(f"{label}: status is {row['status']}; scores were not recomputed")
            continue
        if row["tp"] is None or row["fp"] is None or row["fn"] is None:
            mismatches.append(f"{label}: missing tp, fp, or fn, so the stored scores cannot be recomputed")
            continue
        tp = int(row["tp"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        if row["gt_crack_pixels"] != tp + fn:
            mismatches.append(
                f"{label}: gt_crack_pixels {row['gt_crack_pixels']} != tp + fn ({tp + fn})"
            )
        if row["pred_crack_pixels"] != tp + fp:
            mismatches.append(
                f"{label}: pred_crack_pixels {row['pred_crack_pixels']} != tp + fp ({tp + fp})"
            )
        expected = scores_from_counts(tp, fp, fn)
        for field in METRIC_FIELDS:
            stored = row[field]
            recomputed = expected[field]
            if stored is None and recomputed is None:
                continue
            if stored is None or recomputed is None or abs(float(stored) - float(recomputed)) > METRIC_TOLERANCE:
                mismatches.append(
                    f"{label}: {field} stored {stored} but tp/fp/fn imply {recomputed}"
                )
        if row["zero_denominator"] != expected["zero_denominator"]:
            mismatches.append(
                f"{label}: zero_denominator stored {row['zero_denominator']!r} "
                f"but tp/fp/fn imply {expected['zero_denominator']!r}"
            )
    return mismatches


def empty_predictions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rows where the ground truth has cracks and the prediction has none.

    Precision and F1 are undefined in this case. This list does not replace
    those blanks with a score.
    """
    found = []
    for row in rows:
        if row["status"] != "ok":
            continue
        gt_pixels = row["gt_crack_pixels"]
        pred_pixels = row["pred_crack_pixels"]
        if gt_pixels is None or pred_pixels is None:
            continue
        if int(gt_pixels) > 0 and int(pred_pixels) == 0:
            found.append(row)
    return found


def _group(rows: list[dict[str, object]], split: str, method: str) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["split"] == split and row["method"] == method and row["status"] == "ok"
    ]


def _lowest(rows: list[dict[str, object]], field: str) -> dict[str, object] | None:
    eligible = [row for row in rows if row[field] is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (float(row[field]), str(row["stem"])))


def _largest_gap(rows: list[dict[str, object]]) -> tuple[dict[str, object], float] | None:
    eligible: list[tuple[float, str, dict[str, object]]] = []
    for row in rows:
        if row["precision"] is None or row["recall"] is None:
            continue
        gap = abs(float(row["precision"]) - float(row["recall"]))
        eligible.append((gap, str(row["stem"]), row))
    if not eligible:
        return None
    gap, _stem, row = min(eligible, key=lambda item: (-item[0], item[1]))
    return row, gap


def representative_failures(
    rows: list[dict[str, object]],
    *,
    splits: tuple[str, ...] = EXPECTED_SPLITS,
    methods: tuple[str, ...] = EXPECTED_METHODS,
) -> list[dict[str, object]]:
    """Pick one transparent example per split, method, and criterion.

    Ties use the stem that comes first in alphabetical order. Images with an
    undefined score are skipped for that criterion rather than treated as zero.
    """
    examples: list[dict[str, object]] = []
    criteria = (
        ("lowest_defined_f1", "f1"),
        ("lowest_defined_recall", "recall"),
        ("lowest_defined_precision", "precision"),
    )
    for split in splits:
        for method in methods:
            group = _group(rows, split, method)
            for criterion, field in criteria:
                chosen = _lowest(group, field)
                if chosen is None:
                    continue
                examples.append(_example(criterion, chosen, None))
            gap_choice = _largest_gap(group)
            if gap_choice is not None:
                chosen, gap = gap_choice
                examples.append(_example("largest_precision_recall_gap", chosen, gap))
    return examples


def _example(criterion: str, row: dict[str, object], gap: float | None) -> dict[str, object]:
    precision = row["precision"]
    recall = row["recall"]
    signed_gap = None
    if precision is not None and recall is not None:
        signed_gap = float(precision) - float(recall)
    return {
        "criterion": criterion,
        "split": row["split"],
        "stem": row["stem"],
        "method": row["method"],
        "precision": precision,
        "recall": recall,
        "f1": row["f1"],
        "iou": row["iou"],
        "gt_crack_pixels": row["gt_crack_pixels"],
        "pred_crack_pixels": row["pred_crack_pixels"],
        "tp": row["tp"],
        "fp": row["fp"],
        "fn": row["fn"],
        "zero_denominator": row["zero_denominator"],
        "absolute_gap": gap if gap is not None else (abs(signed_gap) if signed_gap is not None else None),
        "precision_minus_recall": signed_gap,
    }


def compare_with_summary(
    rows: list[dict[str, object]],
    summary_csv: str | Path,
) -> list[str]:
    """Check pooled and mean scores in summary.csv against the per-image rows.

    Segmentation runtime in the summary is not checked, because the per-image
    CSV does not store it.
    """
    summary_path = Path(summary_csv)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    problems: list[str] = []
    with summary_path.open(encoding="utf-8", newline="") as handle:
        stored_rows = list(csv.DictReader(handle))
    for stored in stored_rows:
        split = stored["split"]
        method = stored["method"]
        group = [
            row for row in rows if row["split"] == split and row["method"] == method
        ]
        fresh = aggregate_rows(group)
        label = f"{split}/{method}"
        for field in ("images", "scored", "errors", "gt_crack_pixels", "pred_crack_pixels", "tp", "fp", "fn"):
            if int(stored[field]) != int(fresh[field]):
                problems.append(f"{label}: summary {field} {stored[field]} != recomputed {fresh[field]}")
        for name in METRIC_FIELDS:
            for kind in ("micro", "macro"):
                key = f"{kind}_{name}"
                recomputed = fresh[key]
                text = stored[key].strip()
                if text == "":
                    stored_value = None
                else:
                    stored_value = float(text)
                if stored_value is None and recomputed is None:
                    continue
                if (
                    stored_value is None
                    or recomputed is None
                    or abs(stored_value - float(recomputed)) > METRIC_TOLERANCE
                ):
                    problems.append(
                        f"{label}: summary {key} {stored_value} != recomputed {recomputed}"
                    )
            undefined_key = f"undefined_{name}"
            if int(stored[undefined_key]) != int(fresh[undefined_key]):
                problems.append(
                    f"{label}: summary {undefined_key} {stored[undefined_key]} "
                    f"!= recomputed {fresh[undefined_key]}"
                )
    return problems


def _format_metric(value: object) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.4f}"


def build_failure_report(
    rows: list[dict[str, object]],
    *,
    per_image_csv: Path,
    summary_problems: list[str] | None,
    preview_dir: Path | None,
    inventory_problems: list[str],
    metric_problems: list[str],
) -> str:
    """Write the text report. Macro and micro scores are labelled separately."""
    lines = [
        "DeepCrack failure notes from stored evaluation counts",
        f"Per-image CSV: {per_image_csv}",
        "This file only reads the evaluation. It does not retune methods or rewrite masks.",
        "Undefined precision and F1 are left undefined. They are not counted as success.",
        "Macro F1 is the unweighted mean of defined per-image F1 scores.",
        "Micro F1 is one F1 computed from summed tp, fp, and fn.",
        "Those two F1 numbers are not interchangeable, and neither names a universally best method.",
        "",
        f"Rows read: {len(rows)}",
    ]
    if inventory_problems:
        lines.append("Inventory problems:")
        lines.extend(f"  {problem}" for problem in inventory_problems)
    else:
        lines.append("Inventory: no duplicate or missing split/method rows against the expected 300/237 counts.")
    lines.append(
        f"Formula checks from tp, fp, and fn: {len(metric_problems)} mismatches."
    )
    if metric_problems:
        lines.extend(f"  {problem}" for problem in metric_problems[:20])
        if len(metric_problems) > 20:
            lines.append(f"  ... {len(metric_problems) - 20} more")
    else:
        lines.append("Stored precision, recall, F1, IoU, and pixel totals match the counts.")
    if summary_problems is None:
        lines.append("Summary CSV was not provided, so macro and micro totals were not compared with summary.csv.")
    elif summary_problems:
        lines.append("Summary CSV mismatches:")
        lines.extend(f"  {problem}" for problem in summary_problems)
    else:
        lines.append("Summary CSV macro and micro scores match the per-image counts.")
        lines.append("Summary runtime is not rechecked; the per-image CSV does not store it.")
    lines.append("")

    empties = empty_predictions(rows)
    lines.append(f"Empty predictions with cracks in the ground truth: {len(empties)}")
    if not empties:
        lines.append("  None.")
    for row in empties:
        lines.append(
            f"  {row['split']} {row['stem']} {row['method']}: "
            f"ground-truth crack pixels {row['gt_crack_pixels']}, predicted 0, "
            f"precision {_format_metric(row['precision'])}, "
            f"recall {_format_metric(row['recall'])}, "
            f"F1 {_format_metric(row['f1'])}, "
            f"IoU {_format_metric(row['iou'])}, "
            f"flag {row['zero_denominator']}"
        )
    lines.append(
        "An empty prediction does not receive precision or F1 of 1. "
        "Recall and IoU are 0 when the ground truth is non-empty."
    )
    lines.append("")

    lines.append("Defined per-image F1 distribution (blanks excluded, not treated as 0):")
    for split in EXPECTED_SPLITS:
        for method in EXPECTED_METHODS:
            group = _group(rows, split, method)
            defined = [float(row["f1"]) for row in group if row["f1"] is not None]
            undefined = len(group) - len(defined)
            if not defined:
                lines.append(f"  {split} {method}: no defined F1; undefined {undefined}")
                continue
            stats = f1_distribution(defined)
            fresh = aggregate_rows(group)
            lines.append(
                f"  {split} {method}: n {stats['n']}, undefined F1 {undefined}, "
                f"min {_format_metric(stats['min'])}, "
                f"q1 {_format_metric(stats['q1'])}, "
                f"median {_format_metric(stats['median'])}, "
                f"q3 {_format_metric(stats['q3'])}, "
                f"max {_format_metric(stats['max'])}"
            )
            lines.append(
                f"    macro F1 {_format_metric(fresh['macro_f1'])} "
                f"(mean of {stats['n']} defined image scores); "
                f"micro F1 {_format_metric(fresh['micro_f1'])} "
                f"(pooled pixels tp {fresh['tp']}, fp {fresh['fp']}, fn {fresh['fn']})"
            )
    lines.append("")

    lines.append("Representative rows, chosen only by the criterion named on each line:")
    for example in representative_failures(rows):
        preview = "not checked"
        if preview_dir is not None:
            path = preview_dir / f"{example['split']}_{example['stem']}_{example['method']}_comparison.png"
            preview = f"present at {path.name}" if path.is_file() else "no saved preview"
        lines.append(
            f"  {example['criterion']}: {example['split']} {example['stem']} {example['method']} "
            f"precision {_format_metric(example['precision'])}, "
            f"recall {_format_metric(example['recall'])}, "
            f"F1 {_format_metric(example['f1'])}, "
            f"IoU {_format_metric(example['iou'])}, "
            f"gt {example['gt_crack_pixels']}, pred {example['pred_crack_pixels']}, "
            f"preview {preview}"
        )
    lines.append("")
    lines.append(
        "The saved previews were not regenerated. A missing preview means that stem was outside the original preview sample."
    )
    lines.append(
        "No image-quality measure was tested as a predictor of these errors."
    )
    lines.append("")
    return "\n".join(lines)


def _refuse_existing(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    listed = "\n".join(f"  {path}" for path in existing)
    raise FileExistsError(
        "Refusing to overwrite an existing failure report:\n"
        f"{listed}\n"
        "Pass a new --output-dir. The evaluation files are left unchanged."
    )


def write_failure_analysis(
    per_image_csv: str | Path,
    output_dir: str | Path,
    *,
    summary_csv: str | Path | None = None,
    preview_dir: str | Path | None = None,
    expected_pair_counts: dict[str, int] | None = EXPECTED_PAIR_COUNTS,
) -> dict[str, object]:
    """Read the evaluation CSV and write a new report directory."""
    source = Path(per_image_csv)
    destination = Path(output_dir)
    report_path = destination / "report.txt"
    examples_path = destination / "examples.csv"
    _refuse_existing([report_path, examples_path])

    rows = load_per_image_csv(source)
    inventory_problems = validate_inventory(rows, expected_pair_counts=expected_pair_counts)
    metric_problems = formula_mismatches(rows)
    summary_problems = None
    if summary_csv is not None:
        summary_problems = compare_with_summary(rows, summary_csv)
    previews = Path(preview_dir) if preview_dir is not None else None
    text = build_failure_report(
        rows,
        per_image_csv=source,
        summary_problems=summary_problems,
        preview_dir=previews,
        inventory_problems=inventory_problems,
        metric_problems=metric_problems,
    )
    examples = representative_failures(rows)
    destination.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    _write_examples(examples_path, examples, previews)
    consistent = not inventory_problems and not metric_problems and not summary_problems
    return {
        "report": report_path,
        "examples": examples_path,
        "consistent": consistent,
        "empty_predictions": empty_predictions(rows),
        "examples_found": examples,
    }


def _write_examples(
    path: Path,
    examples: list[dict[str, object]],
    preview_dir: Path | None,
) -> None:
    fieldnames = [
        "criterion",
        "split",
        "stem",
        "method",
        "precision",
        "recall",
        "f1",
        "iou",
        "gt_crack_pixels",
        "pred_crack_pixels",
        "tp",
        "fp",
        "fn",
        "zero_denominator",
        "absolute_gap",
        "precision_minus_recall",
        "preview_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            preview_file = ""
            if preview_dir is not None:
                candidate = preview_dir / (
                    f"{example['split']}_{example['stem']}_{example['method']}_comparison.png"
                )
                preview_file = candidate.name if candidate.is_file() else ""
            stored = {key: example.get(key, "") for key in fieldnames}
            stored["preview_file"] = preview_file
            for key, value in list(stored.items()):
                if value is None:
                    stored[key] = ""
                elif isinstance(value, float):
                    stored[key] = f"{value:.6f}"
            writer.writerow(stored)
