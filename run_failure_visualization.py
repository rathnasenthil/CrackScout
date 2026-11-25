"""Save diagnostic figures for documented DeepCrack failure cases."""

import argparse
import sys
from pathlib import Path

from crackscout.deepcrack_dataset import default_dataset_root
from crackscout.failure_visualization import render_failure_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce documented failure cases with the existing segmentation "
            "defaults and save diagnostic figures. Dataset files and earlier "
            "outputs are not modified."
        )
    )
    parser.add_argument(
        "--examples-csv",
        default="outputs/deepcrack_failure_analysis/examples.csv",
        help="Failure examples to visualize",
    )
    parser.add_argument(
        "--per-image-csv",
        default="outputs/deepcrack_evaluation/per_image.csv",
        help="Evaluation rows that supply stored F1 and parameters",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root()),
        help="Extracted DeepCrack dataset root",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deepcrack_failure_visualization",
        help="New directory for diagnostic figures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        figures = render_failure_cases(
            args.examples_csv,
            args.per_image_csv,
            args.dataset_root,
            args.output_dir,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(figures)} diagnostic figures in {Path(args.output_dir)}")
    for path in figures:
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
