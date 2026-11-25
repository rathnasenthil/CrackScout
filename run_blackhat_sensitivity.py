"""Run the fixed black-hat parameter grid on one local image."""

import argparse
import sys
from pathlib import Path

from crackscout.sensitivity import format_report, run_blackhat_sensitivity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare five fixed black-hat settings on one image. "
            "The settings are kernel/threshold pairs (15, 25), (9, 25), "
            "(21, 25), (15, 15), and (15, 35). "
            "The script records white-pixel percentage and 8-connected "
            "foreground components. Those counts describe the masks. "
            "They do not measure crack-detection accuracy, and the script "
            "does not select a parameter."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image, usually samples\\your_crack_image.jpeg",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for masks, figures, and the text report (default: outputs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = run_blackhat_sensitivity(args.input, args.output_dir)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_report(Path(args.input), rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
