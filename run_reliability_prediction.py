"""Predict poor classical segmentation from training-only image-quality features."""

import argparse
import hashlib
import sys
from pathlib import Path

from crackscout.reliability_prediction import load_joined_features, run_experiment, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit failure labels and quality-feature models on the DeepCrack "
            "training split, then score the test split once. Existing outputs "
            "are not modified."
        )
    )
    parser.add_argument(
        "--joined-csv",
        default="outputs/deepcrack_quality_analysis/joined_features.csv",
        help="Image-quality features already joined to per-image F1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deepcrack_reliability_prediction",
        help="New directory for the reliability-prediction outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.joined_csv)
    try:
        rows = load_joined_features(source)
        result = run_experiment(rows)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result["config"]["input_csv"] = str(source)
        result["config"]["input_sha256"] = digest
        write_outputs(args.output_dir, result)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    print(f"Report: {Path(args.output_dir) / 'report.txt'}")
    print(f"Predictions: {Path(args.output_dir) / 'predictions.csv'}")
    print(f"Metrics: {Path(args.output_dir) / 'metrics.csv'}")
    print(f"Configuration: {Path(args.output_dir) / 'config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
