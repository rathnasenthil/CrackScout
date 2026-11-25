"""Score default CrackScout segmentations against DeepCrack masks.

The scores are exploratory. This script does not tune parameters, select a
method, or modify the dataset.
"""

import argparse
import sys

from crackscout.deepcrack_dataset import default_dataset_root
from crackscout.evaluate import EVALUATION_METHODS, evaluate_deepcrack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current Otsu, adaptive, and black-hat defaults on "
            "DeepCrack. Train and test stay separate. Results are exploratory "
            "and are not a final held-out comparison. Existing files in the "
            "output directory are left unchanged."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root()),
        help="Folder that directly contains train_img, train_lab, test_img, and test_lab",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deepcrack_evaluation",
        help="New directory for CSV files and a few comparison images",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Splits to score separately (default: train test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N pairs in each split, in stem order",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=2,
        help="Comparison figures to save for the first N pairs in each split (default: 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        saved = evaluate_deepcrack(
            args.dataset_root,
            args.output_dir,
            splits=tuple(args.splits),
            methods=EVALUATION_METHODS,
            limit=args.limit,
            preview_count=args.preview_count,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(saved["summary"].read_text(encoding="utf-8"))
    print(f"Per-image CSV: {saved['per_image_csv']}")
    print(f"Summary CSV: {saved['summary_csv']}")
    print(f"Previews: {saved['preview_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
