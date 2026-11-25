"""Join fixed image-quality features to the DeepCrack evaluation CSV."""

import argparse
import csv
import sys
from pathlib import Path

from crackscout.deepcrack_dataset import default_dataset_root
from crackscout.failure_analysis import load_per_image_csv
from crackscout.image_quality import (
    FEATURE_NAMES,
    build_quality_report,
    features_for_split,
    join_features,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            stored = {}
            for field in fieldnames:
                value = row.get(field, "")
                if value is None:
                    stored[field] = ""
                elif isinstance(value, float):
                    stored[field] = f"{value:.6f}"
                else:
                    stored[field] = value
            writer.writerow(stored)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute image-quality features from DeepCrack photographs and "
            "compare them with stored F1 scores. Masks are not used to compute "
            "features. Existing evaluation files are not modified."
        )
    )
    parser.add_argument(
        "--per-image-csv",
        default="outputs/deepcrack_evaluation/per_image.csv",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root()),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deepcrack_quality_analysis",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = Path(args.output_dir)
    planned = [
        destination / "image_features.csv",
        destination / "joined_features.csv",
        destination / "report.txt",
    ]
    existing = [path for path in planned if path.exists()]
    if existing:
        listed = "\n".join(f"  {path}" for path in existing)
        print(
            "Error: Refusing to overwrite quality-analysis files:\n"
            f"{listed}",
            file=sys.stderr,
        )
        return 1
    try:
        evaluation_rows = load_per_image_csv(args.per_image_csv)
        feature_rows: list[dict[str, object]] = []
        for split in ("train", "test"):
            feature_rows.extend(features_for_split(args.dataset_root, split))
        joined = join_features(evaluation_rows, feature_rows)
        report = build_quality_report(joined)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(
        planned[0],
        feature_rows,
        ["split", "stem", *FEATURE_NAMES],
    )
    _write_csv(
        planned[1],
        joined,
        [
            "split",
            "stem",
            "method",
            "f1",
            "precision",
            "recall",
            "iou",
            "gt_crack_pixels",
            "pred_crack_pixels",
            "tp",
            "fp",
            "fn",
            *FEATURE_NAMES,
        ],
    )
    planned[2].write_text(report, encoding="utf-8")
    print(report)
    print(f"Image features: {planned[0]}")
    print(f"Joined table: {planned[1]}")
    print(f"Report: {planned[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
