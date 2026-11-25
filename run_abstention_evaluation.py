"""Evaluate a training-selected abstention rule on frozen reliability scores."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from crackscout.abstention_evaluation import load_failure_labels, load_predictions, run_abstention, validate_predictions, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score abstention from the saved reliability probabilities. "
            "The operating threshold is chosen on training scores only. "
            "Existing outputs are not modified."
        )
    )
    parser.add_argument(
        "--predictions",
        default="outputs/deepcrack_reliability_prediction/predictions.csv",
    )
    parser.add_argument(
        "--per-image-csv",
        default="outputs/deepcrack_evaluation/per_image.csv",
    )
    parser.add_argument(
        "--reliability-config",
        default="outputs/deepcrack_reliability_prediction/config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deepcrack_abstention_evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction_path = Path(args.predictions)
    try:
        rows = load_predictions(prediction_path)
        labels = load_failure_labels(args.per_image_csv)
        validate_predictions(rows, labels)
        reliability = json.loads(Path(args.reliability_config).read_text(encoding="utf-8"))
        result = run_abstention(rows)
        result["config"]["predictions_csv"] = str(prediction_path)
        result["config"]["predictions_sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        result["config"]["reliability_random_seed"] = reliability.get("random_seed")
        result["config"]["models_retrained"] = False
        write_outputs(args.output_dir, result)
    except (FileNotFoundError, FileExistsError, ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    destination = Path(args.output_dir)
    print(f"Report: {destination / 'report.txt'}")
    print(f"Operating points: {destination / 'operating_points.csv'}")
    print(f"Risk-coverage: {destination / 'risk_coverage.csv'}")
    print(f"Random baseline: {destination / 'random_baseline.csv'}")
    print(f"Configuration: {destination / 'config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
