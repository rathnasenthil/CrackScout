"""Write a read-only failure report from an existing DeepCrack evaluation CSV."""

import argparse
import sys
from pathlib import Path

from crackscout.failure_analysis import write_failure_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read outputs from the DeepCrack evaluation and write a failure "
            "report to a new directory. The dataset and the evaluation files "
            "are not modified. Undefined scores stay undefined."
        )
    )
    parser.add_argument(
        "--per-image-csv",
        default="outputs/deepcrack_evaluation/per_image.csv",
        help="Existing per-image evaluation CSV",
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/deepcrack_evaluation/summary.csv",
        help="Existing summary CSV used only for a consistency check",
    )
    parser.add_argument(
        "--preview-dir",
        default="outputs/deepcrack_evaluation/previews",
        help="Existing preview directory. Files are checked, not rewritten",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deepcrack_failure_analysis",
        help="New directory for the report. Existing report files are not replaced",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        saved = write_failure_analysis(
            args.per_image_csv,
            args.output_dir,
            summary_csv=args.summary_csv,
            preview_dir=args.preview_dir,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(Path(saved["report"]).read_text(encoding="utf-8"))
    print(f"Report: {saved['report']}")
    print(f"Examples: {saved['examples']}")
    return 0 if saved["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
