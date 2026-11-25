"""Run the first CrackScout segmentation baseline on one local image."""

import argparse
import sys
from pathlib import Path

from crackscout.pipeline import run_baseline
from crackscout.segment import available_methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment one concrete image with a registered classical OpenCV method. "
            "The default method is otsu. The mask is a starting point for later "
            "comparisons and is not an evaluated measure of crack-detection accuracy."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image, usually a file you placed in samples\\",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for the mask and comparison figure (default: outputs)",
    )
    parser.add_argument(
        "--method",
        default="otsu",
        choices=available_methods(),
        help="Registered segmentation method (default: otsu)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        saved = run_baseline(args.input, args.output_dir, method=args.method)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved mask: {Path(saved['mask'])}")
    print(f"Saved comparison: {Path(saved['comparison'])}")
    print(
        "This is an unevaluated classical segmentation. "
        "It does not measure crack-detection accuracy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
