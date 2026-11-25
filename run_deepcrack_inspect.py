"""Report DeepCrack pair counts and file problems without changing the dataset."""

import argparse
import sys

from crackscout.deepcrack_dataset import default_dataset_root, inspect_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the extracted DeepCrack folders and report pair counts, "
            "image orientations, missing pairs, invalid masks, and unreadable "
            "files. This command does not modify dataset files and does not "
            "compute detection scores."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root()),
        help="Folder that directly contains train_img, train_lab, test_img, and test_lab",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, matches_verified = inspect_dataset(args.dataset_root)
    print(report)
    return 0 if matches_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
