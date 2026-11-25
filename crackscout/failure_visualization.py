"""Diagnostic figures for documented DeepCrack failure cases.

Predictions are made with the existing default segmentation functions.
Cases are resolved inside one split. A shared stem is not taken from the other split.
"""

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crackscout.deepcrack_dataset import list_pairs, load_image_and_mask
from crackscout.evaluate import method_parameters, predict_mask
from crackscout.failure_analysis import load_per_image_csv
from crackscout.metrics import binary_counts, scores_from_counts

# These cases are the named rows in the failure-analysis examples.
# 7Q3A9064-20 is included for both methods that the examples file records.
REQUIRED_CASES = (
    ("test", "IMG24-1", "otsu"),
    ("train", "11196", "otsu"),
    ("train", "IMG14-2", "adaptive"),
    ("train", "IMG74", "blackhat"),
    ("train", "7Q3A9064-20", "adaptive"),
    ("train", "7Q3A9064-20", "blackhat"),
    ("test", "11271-1", "blackhat"),
)


def load_example_cases(path: str | Path) -> list[dict[str, object]]:
    """Group examples.csv into one case per split, stem, and method."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Examples CSV not found: {csv_path}")
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"criterion", "split", "stem", "method", "f1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Examples CSV is missing required columns: {csv_path}")
        for raw in reader:
            key = (raw["split"].strip(), raw["stem"].strip(), raw["method"].strip())
            f1_text = raw["f1"].strip()
            stored_f1 = None if f1_text == "" else float(f1_text)
            if key not in grouped:
                grouped[key] = {
                    "split": key[0],
                    "stem": key[1],
                    "method": key[2],
                    "criteria": [raw["criterion"].strip()],
                    "example_f1": stored_f1,
                }
            else:
                grouped[key]["criteria"].append(raw["criterion"].strip())
    cases = list(grouped.values())
    present = {(case["split"], case["stem"], case["method"]) for case in cases}
    missing = [case for case in REQUIRED_CASES if case not in present]
    if missing:
        listed = ", ".join(f"{split}/{stem}/{method}" for split, stem, method in missing)
        raise ValueError(f"Examples CSV is missing documented failure cases: {listed}")
    return cases


def _index_evaluation(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    index: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        index[(str(row["split"]), str(row["stem"]), str(row["method"]))] = row
    return index


def _pairs_by_stem(dataset_root: Path, split: str) -> dict[str, object]:
    return {pair.stem: pair for pair in list_pairs(dataset_root, split)}


def resolve_pair(pairs_by_split: dict[str, dict[str, object]], split: str, stem: str):
    """Return the pair for this split only."""
    split_pairs = pairs_by_split.get(split)
    if split_pairs is None or stem not in split_pairs:
        raise ValueError(
            f"No {split} image for stem {stem}. The other split was not searched."
        )
    return split_pairs[stem]


def _save_diagnostic(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    path: Path,
    *,
    split: str,
    stem: str,
    method: str,
    criteria: list[str],
    parameters: str,
    stored_f1: float | None,
) -> None:
    gt_crack = ground_truth == 255
    pred_crack = prediction == 255
    comparison = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    comparison[gt_crack & pred_crack] = (0, 170, 0)
    comparison[~gt_crack & pred_crack] = (220, 30, 30)
    comparison[gt_crack & ~pred_crack] = (30, 80, 220)
    display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    f1_text = "undefined" if stored_f1 is None else f"{stored_f1:.4f}"

    figure, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    panels = (
        (display, None, "Original image"),
        (ground_truth, "gray", "Ground truth (255 = crack)"),
        (prediction, "gray", "Prediction (255 = crack)"),
        (comparison, None, "Green both, red prediction only, blue ground truth only"),
    )
    for axis, (panel, cmap, title) in zip(axes, panels):
        axis.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    figure.suptitle(
        f"{split} / {stem} / {method} / stored F1 {f1_text}\n"
        f"Criteria: {', '.join(criteria)}\n{parameters}",
        fontsize=9,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=100, bbox_inches="tight")
    finally:
        plt.close(figure)


def _scores_agree(stored_f1: float | None, recomputed_f1: float | None) -> bool:
    if stored_f1 is None or recomputed_f1 is None:
        return stored_f1 is None and recomputed_f1 is None
    return abs(stored_f1 - recomputed_f1) <= 1e-4


def render_failure_cases(
    examples_csv: str | Path,
    per_image_csv: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Write one diagnostic figure per documented failure case.

    Raises FileExistsError if a planned figure or the manifest already exists.
    """
    cases = load_example_cases(examples_csv)
    evaluation = _index_evaluation(load_per_image_csv(per_image_csv))
    root = Path(dataset_root)
    destination = Path(output_dir)
    pairs = {split: _pairs_by_stem(root, split) for split in ("train", "test")}

    planned: list[tuple[dict[str, object], object, Path]] = []
    for case in cases:
        key = (str(case["split"]), str(case["stem"]), str(case["method"]))
        if key not in evaluation:
            raise ValueError(f"No evaluation row for {key[0]}/{key[1]}/{key[2]}")
        pair = resolve_pair(pairs, key[0], key[1])
        figure_path = destination / f"{key[0]}_{key[1]}_{key[2]}_diagnostic.png"
        planned.append((case, pair, figure_path))

    manifest_path = destination / "manifest.csv"
    existing = [path for path in [manifest_path, *[item[2] for item in planned]] if path.exists()]
    if existing:
        listed = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite failure visualizations:\n"
            f"{listed}\n"
            "The evaluation files and dataset were not changed."
        )

    manifest_rows: list[dict[str, object]] = []
    rendered: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = []
    for case, pair, figure_path in planned:
        record = evaluation[(str(case["split"]), str(case["stem"]), str(case["method"]))]
        parameters = str(record["parameters"])
        current = method_parameters(str(case["method"]))
        if parameters != current:
            raise ValueError(
                f"Stored parameters for {case['split']}/{case['stem']}/{case['method']} "
                f"do not match the current defaults.\nStored: {parameters}\nCurrent: {current}"
            )
        image, ground_truth = load_image_and_mask(pair.image_path, pair.mask_path)
        prediction = predict_mask(image, str(case["method"]))
        counts = binary_counts(ground_truth, prediction)
        scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
        stored_f1 = record["f1"]
        if not _scores_agree(
            None if case["example_f1"] is None else float(case["example_f1"]),
            None if stored_f1 is None else float(stored_f1),
        ):
            raise ValueError(
                f"Examples F1 for {case['split']}/{case['stem']}/{case['method']} "
                f"is {case['example_f1']}, evaluation F1 is {stored_f1}."
            )
        if not _scores_agree(
            None if stored_f1 is None else float(stored_f1),
            None if scores["f1"] is None else float(scores["f1"]),
        ):
            raise ValueError(
                f"Reproduced F1 for {case['split']}/{case['stem']}/{case['method']} "
                f"is {scores['f1']}, stored F1 is {stored_f1}."
            )
        rendered.append((figure_path, image, ground_truth, prediction, {
            "split": case["split"],
            "stem": case["stem"],
            "method": case["method"],
            "criteria": "; ".join(case["criteria"]),
            "parameters": parameters,
            "stored_f1": "" if stored_f1 is None else f"{float(stored_f1):.6f}",
            "figure": figure_path.name,
        }))

    destination.mkdir(parents=True, exist_ok=True)
    for figure_path, image, ground_truth, prediction, metadata in rendered:
        _save_diagnostic(
            image,
            ground_truth,
            prediction,
            figure_path,
            split=str(metadata["split"]),
            stem=str(metadata["stem"]),
            method=str(metadata["method"]),
            criteria=str(metadata["criteria"]).split("; "),
            parameters=str(metadata["parameters"]),
            stored_f1=None if metadata["stored_f1"] == "" else float(metadata["stored_f1"]),
        )
        manifest_rows.append(metadata)

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "stem",
                "method",
                "criteria",
                "parameters",
                "stored_f1",
                "figure",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return [path for path, *_rest in rendered]
