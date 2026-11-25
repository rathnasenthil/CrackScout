"""Read-only access to the extracted DeepCrack image and mask folders.

DeepCrack is restricted to non-commercial research and educational use.
If you use it in research, cite:

Liu, Yahui, Jian Yao, Xiaohu Lu, Renping Xie, and Li Li. 2019.
"DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack
Segmentation." Neurocomputing 338: 139-153. doi:10.1016/j.neucom.2019.01.036

This module only reads files. It does not resize, invert, or rewrite masks.
"""

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# Folders sit directly in the dataset directory. The zip does not add
# another DeepCrack folder around them.
SPLIT_FOLDERS = {
    "train": ("train_img", "train_lab"),
    "test": ("test_img", "test_lab"),
}

# Counts from the inspected DeepCrack.zip release. Used only to flag
# differences. They are not used to invent missing files.
VERIFIED_PAIR_COUNTS = {"train": 300, "test": 237}
VERIFIED_TEST_TALL_PAIRS = 47
VERIFIED_TALL_SHAPE = (544, 384)
VERIFIED_WIDE_SHAPE = (384, 544)

DEEPCRACK_CITATION = (
    "@article{liu2019deepcrack,\n"
    "  title={DeepCrack: A Deep Hierarchical Feature Learning Architecture "
    "for Crack Segmentation},\n"
    "  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping "
    "and Li, Li},\n"
    "  journal={Neurocomputing},\n"
    "  volume={338},\n"
    "  pages={139--153},\n"
    "  year={2019},\n"
    "  doi={10.1016/j.neucom.2019.01.036}\n"
    "}"
)


class DeepCrackPair(NamedTuple):
    """One image and its mask from the same split."""

    split: str
    stem: str
    image_path: Path
    mask_path: Path


def default_dataset_root() -> Path:
    """Return data/DeepCrack/dataset inside this project."""
    return Path(__file__).resolve().parents[1] / "data" / "DeepCrack" / "dataset"


def _check_split_name(split: str) -> None:
    if split not in SPLIT_FOLDERS:
        known = ", ".join(sorted(SPLIT_FOLDERS))
        raise ValueError(f"Unknown split '{split}'. Use one of: {known}.")


def scan_split(dataset_root: str | Path, split: str) -> dict[str, list]:
    """List pairs and pairing problems for one split.

    Stems are matched only inside this split. A stem that also exists in
    the other split is a different sample and is not linked here.
    """
    _check_split_name(split)
    root = Path(dataset_root)
    image_folder_name, mask_folder_name = SPLIT_FOLDERS[split]
    image_dir = root / image_folder_name
    mask_dir = root / mask_folder_name

    missing_directories = [
        str(path) for path in (image_dir, mask_dir) if not path.is_dir()
    ]
    if missing_directories:
        return {
            "pairs": [],
            "missing_directories": missing_directories,
            "images_without_masks": [],
            "masks_without_images": [],
            "unexpected_files": [],
        }

    image_stems: dict[str, Path] = {}
    mask_stems: dict[str, Path] = {}
    unexpected: list[str] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".jpg":
            image_stems[path.stem] = path
        else:
            unexpected.append(str(path))
    for path in sorted(mask_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".png":
            mask_stems[path.stem] = path
        else:
            unexpected.append(str(path))

    pairs = [
        DeepCrackPair(split, stem, image_stems[stem], mask_stems[stem])
        for stem in sorted(set(image_stems) & set(mask_stems))
    ]
    return {
        "pairs": pairs,
        "missing_directories": [],
        "images_without_masks": sorted(set(image_stems) - set(mask_stems)),
        "masks_without_images": sorted(set(mask_stems) - set(image_stems)),
        "unexpected_files": unexpected,
    }


def list_pairs(dataset_root: str | Path, split: str) -> list[DeepCrackPair]:
    """Return image/mask pairs for 'train' or 'test'.

    Raises FileNotFoundError when a split folder is missing, and ValueError
    when a file in that split has no partner or uses another extension.
    """
    scan = scan_split(dataset_root, split)
    if scan["missing_directories"]:
        root = Path(dataset_root)
        entries = sorted(path.name for path in root.iterdir()) if root.is_dir() else []
        zip_note = ""
        if (root / "DeepCrack.zip").is_file():
            zip_note = (
                f"\nFound {root / 'DeepCrack.zip'}, but the image folders "
                "are not extracted beside it."
            )
        listed = "\n".join(scan["missing_directories"])
        raise FileNotFoundError(
            "DeepCrack split folders were not found:\n"
            f"{listed}\n"
            "Extract DeepCrack.zip so train_img, train_lab, test_img, and "
            f"test_lab sit directly in {root}."
            f"{zip_note}\n"
            f"Entries in {root}: {entries or '(missing directory)'}"
        )

    problems: list[str] = []
    for stem in scan["images_without_masks"]:
        problems.append(f"{split}_img/{stem}.jpg has no {split}_lab/{stem}.png")
    for stem in scan["masks_without_images"]:
        problems.append(f"{split}_lab/{stem}.png has no {split}_img/{stem}.jpg")
    for path in scan["unexpected_files"]:
        problems.append(f"Unexpected file (expected .jpg images or .png masks): {path}")
    if problems:
        raise ValueError("DeepCrack pairing problems:\n" + "\n".join(problems))
    return scan["pairs"]


def load_image_and_mask(
    image_path: str | Path,
    mask_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one JPEG image and one PNG mask without changing either array.

    The image must be 3-channel uint8. The mask must be single-channel uint8
    and contain only 0 (background) and 255 (crack). Width and height must match.
    """
    image_file = Path(image_path)
    mask_file = Path(mask_path)
    image = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {image_file}")
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_file}")

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"Expected a 3-channel uint8 image, got shape {image.shape} "
            f"dtype {image.dtype}: {image_file}"
        )
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError(
            f"Expected a single-channel uint8 mask, got shape {mask.shape} "
            f"dtype {mask.dtype}: {mask_file}"
        )
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "Image and mask dimensions differ: "
            f"image {image.shape[:2]} ({image_file.name}), "
            f"mask {mask.shape[:2]} ({mask_file.name})"
        )
    values = np.unique(mask)
    if not set(values.tolist()).issubset({0, 255}):
        shown = ", ".join(str(int(value)) for value in values.tolist())
        raise ValueError(
            f"Mask contains values other than 0 and 255 ({shown}): {mask_file}"
        )
    return image, mask


def _load_problems(pairs: list[DeepCrackPair]) -> tuple[list[str], list[str], dict[tuple[int, int], int]]:
    """Load each pair. Return unreadable files, invalid masks, and shapes."""
    unreadable: list[str] = []
    invalid: list[str] = []
    shapes: dict[tuple[int, int], int] = {}
    for pair in pairs:
        try:
            image, _mask = load_image_and_mask(pair.image_path, pair.mask_path)
        except ValueError as exc:
            message = str(exc)
            if message.startswith("Could not read"):
                unreadable.append(message)
            else:
                invalid.append(message)
            continue
        shape = (int(image.shape[0]), int(image.shape[1]))
        shapes[shape] = shapes.get(shape, 0) + 1
    return unreadable, invalid, shapes


def inspect_dataset(dataset_root: str | Path) -> tuple[str, bool]:
    """Build a text report. The boolean is true only when the release checks pass.

    The report compares pair counts and the 47 tall test images with the
    inspected zip. It does not write or correct dataset files.
    """
    root = Path(dataset_root)
    lines = [
        "DeepCrack dataset inspection",
        f"Root: {root}",
        "Usage: non-commercial research and educational purposes.",
        "Citation:",
        DEEPCRACK_CITATION,
        "",
    ]
    ok = True
    tall_test_pairs = 0
    for split in ("train", "test"):
        scan = scan_split(root, split)
        pairs = scan["pairs"]
        unreadable, invalid, shapes = _load_problems(pairs)
        lines.append(f"{split}")
        lines.append(f"  paired files: {len(pairs)}")
        lines.append(f"  verified pair count: {VERIFIED_PAIR_COUNTS[split]}")
        if scan["missing_directories"]:
            ok = False
            for path in scan["missing_directories"]:
                lines.append(f"  missing directory: {path}")
        if len(pairs) != VERIFIED_PAIR_COUNTS[split]:
            ok = False
            lines.append(
                "  discrepancy: paired files "
                f"{len(pairs)} != verified {VERIFIED_PAIR_COUNTS[split]}"
            )
        for stem in scan["images_without_masks"]:
            ok = False
            lines.append(f"  missing mask for stem: {stem}")
        for stem in scan["masks_without_images"]:
            ok = False
            lines.append(f"  missing image for stem: {stem}")
        for path in scan["unexpected_files"]:
            ok = False
            lines.append(f"  unexpected file: {path}")
        if shapes:
            lines.append("  orientations (height x width):")
            for shape, count in sorted(shapes.items()):
                lines.append(f"    {shape[0]} x {shape[1]}: {count}")
        else:
            lines.append("  orientations: none loaded")
        if split == "test":
            tall_test_pairs = shapes.get(VERIFIED_TALL_SHAPE, 0)
            lines.append(
                f"  tall test pairs {VERIFIED_TALL_SHAPE[0]} x {VERIFIED_TALL_SHAPE[1]}: "
                f"{tall_test_pairs} (verified {VERIFIED_TEST_TALL_PAIRS})"
            )
            if tall_test_pairs != VERIFIED_TEST_TALL_PAIRS:
                ok = False
                lines.append(
                    "  discrepancy: tall test pairs "
                    f"{tall_test_pairs} != verified {VERIFIED_TEST_TALL_PAIRS}"
                )
        for message in unreadable:
            ok = False
            lines.append(f"  unreadable: {message}")
        for message in invalid:
            ok = False
            lines.append(f"  invalid: {message}")
        lines.append("")

    if (root / "DeepCrack.zip").is_file() and not (root / "train_img").is_dir():
        lines.append(
            f"Found {root / 'DeepCrack.zip'}, but train_img is not extracted beside it."
        )
        ok = False
    if ok:
        lines.append("No discrepancies against the verified DeepCrack counts.")
    return "\n".join(lines), ok
