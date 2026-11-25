"""Nested training validation and follow-up uncertainty for CS-006 abstention."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from crackscout.abstention_evaluation import load_predictions
from crackscout.reliability_prediction import METHODS, N_FOLDS, RANDOM_SEED, assign_folds
from crackscout.robustness_evaluation import (
    load_feature_table,
    run_robustness,
    training_model_rows,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the existing logistic specification inside training outer folds, "
            "then describe the already examined test split without choosing a new threshold."
        )
    )
    parser.add_argument(
        "--predictions",
        default="outputs/deepcrack_reliability_prediction/predictions.csv",
    )
    parser.add_argument(
        "--features",
        default="outputs/deepcrack_quality_analysis/joined_features.csv",
    )
    parser.add_argument(
        "--reliability-config",
        default="outputs/deepcrack_reliability_prediction/config.json",
    )
    parser.add_argument(
        "--abstention-config",
        default="outputs/deepcrack_abstention_evaluation/config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deepcrack_robustness_evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictions = load_predictions(args.predictions)
        features = load_feature_table(args.features)
        training_rows = training_model_rows(predictions, features)
        reliability = json.loads(Path(args.reliability_config).read_text(encoding="utf-8"))
        abstention = json.loads(Path(args.abstention_config).read_text(encoding="utf-8"))
        saved_folds = {str(stem): int(fold) for stem, fold in reliability["fold_by_training_stem"].items()}
        train_stems = [str(row["stem"]) for row in training_rows if row["method"] == METHODS[0]]
        recomputed = assign_folds(train_stems, N_FOLDS, RANDOM_SEED)
        if recomputed != saved_folds:
            raise ValueError("The saved CS-005 fold assignment does not match seed 0.")
        thresholds = {}
        for method in METHODS:
            entry = abstention["methods"][method]
            if not entry["justified"] or entry["threshold"] is None:
                raise ValueError(
                    f"CS-006 did not justify a {method} threshold. "
                    "No replacement was taken from the test split."
                )
            thresholds[method] = float(entry["threshold"])
        result = run_robustness(training_rows, predictions, thresholds, saved_folds)
        result_config_note = {
            "predictions_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
            "abstention_config_sha256": hashlib.sha256(Path(args.abstention_config).read_bytes()).hexdigest(),
        }
        write_outputs(args.output_dir, result, predictions)
        config_path = Path(args.output_dir) / "config.json"
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        stored.update(result_config_note)
        config_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (FileNotFoundError, FileExistsError, ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    destination = Path(args.output_dir)
    for name in (
        "report.txt",
        "outer_fold_results.csv",
        "operating_point_stability.csv",
        "risk_coverage.csv",
        "paired_bootstrap.csv",
        "sensitivity_analysis.csv",
        "config.json",
    ):
        print(f"{name}: {destination / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
