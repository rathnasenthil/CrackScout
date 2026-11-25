"""Loader tests use temporary synthetic files, not the DeepCrack release."""

import inspect
import sys
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.deepcrack_dataset import (
    inspect_dataset,
    list_pairs,
    load_image_and_mask,
)
from crackscout.segment import available_methods, segment_blackhat
from run_deepcrack_inspect import main


def _color(height: int, width: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _mask(height: int, width: int, crack_column: int | None = None) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if crack_column is not None:
        mask[:, crack_column] = 255
    return mask


def _write_jpg(directory: Path, stem: str, image: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(directory / f"{stem}.jpg"), image):
        raise RuntimeError(f"Could not write test image {stem}")


def _write_png(directory: Path, stem: str, mask: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(directory / f"{stem}.png"), mask):
        raise RuntimeError(f"Could not write test mask {stem}")


def _make_split(root: Path, split: str, stem: str, image: np.ndarray, mask: np.ndarray) -> None:
    _write_jpg(root / f"{split}_img", stem, image)
    _write_png(root / f"{split}_lab", stem, mask)


class PairingTests(unittest.TestCase):
    def test_pairs_matching_stems_inside_one_split(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_split(root, "train", "crack-a", _color(8, 12, 40), _mask(8, 12, 3))
            _make_split(root, "train", "crack-b", _color(8, 12, 50), _mask(8, 12, 4))
            pairs = list_pairs(root, "train")
        self.assertEqual([pair.stem for pair in pairs], ["crack-a", "crack-b"])
        self.assertEqual(pairs[0].image_path.parent.name, "train_img")
        self.assertEqual(pairs[0].image_path.name, "crack-a.jpg")
        self.assertEqual(pairs[0].mask_path.parent.name, "train_lab")
        self.assertEqual(pairs[0].mask_path.name, "crack-a.png")

    def test_overlapping_stems_stay_in_their_own_split(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_split(root, "train", "11289-1", _color(6, 10, 20), _mask(6, 10, 1))
            _make_split(root, "test", "11289-1", _color(6, 10, 220), _mask(6, 10, 8))
            train_pairs = list_pairs(root, "train")
            test_pairs = list_pairs(root, "test")
            train_image, train_mask = load_image_and_mask(
                train_pairs[0].image_path, train_pairs[0].mask_path
            )
            test_image, test_mask = load_image_and_mask(
                test_pairs[0].image_path, test_pairs[0].mask_path
            )
        self.assertEqual(len(train_pairs), 1)
        self.assertEqual(len(test_pairs), 1)
        self.assertNotEqual(train_pairs[0].image_path, test_pairs[0].image_path)
        self.assertLess(float(train_image[:, :, 0].mean()), 40)
        self.assertGreater(float(test_image[:, :, 0].mean()), 200)
        self.assertEqual(int(train_mask[:, 1].max()), 255)
        self.assertEqual(int(test_mask[:, 1].max()), 0)
        self.assertEqual(int(test_mask[:, 8].max()), 255)

    def test_missing_mask_and_missing_image_are_reported(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_jpg(root / "train_img", "only-image", _color(4, 4, 10))
            _write_png(root / "train_lab", "only-mask", _mask(4, 4, 0))
            with self.assertRaises(ValueError) as caught:
                list_pairs(root, "train")
        message = str(caught.exception)
        self.assertIn("only-image", message)
        self.assertIn("only-mask", message)
        self.assertIn("has no", message)

    def test_missing_split_folder_names_the_expected_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError) as caught:
                list_pairs(root, "test")
        self.assertIn("test_img", str(caught.exception))
        self.assertIn("test_lab", str(caught.exception))


class LoadTests(unittest.TestCase):
    def test_valid_pair_keeps_mask_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = _mask(5, 7, 2)
            _make_split(root, "train", "kept", _color(5, 7, 80), original)
            pair = list_pairs(root, "train")[0]
            image, mask = load_image_and_mask(pair.image_path, pair.mask_path)
        self.assertEqual(image.shape, (5, 7, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(mask.shape, (5, 7))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask).tolist()).issubset({0, 255}))
        np.testing.assert_array_equal(mask, original)

    def test_invalid_mask_value_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bad = np.zeros((4, 4), dtype=np.uint8)
            bad[0, 0] = 128
            _make_split(root, "test", "bad-value", _color(4, 4, 30), bad)
            pair = list_pairs(root, "test")[0]
            with self.assertRaises(ValueError) as caught:
                load_image_and_mask(pair.image_path, pair.mask_path)
        self.assertIn("0 and 255", str(caught.exception))
        self.assertIn("128", str(caught.exception))

    def test_dimension_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_jpg(root / "train_img", "size", _color(6, 8, 15))
            _write_png(root / "train_lab", "size", _mask(4, 4, 1))
            pair = list_pairs(root, "train")[0]
            with self.assertRaises(ValueError) as caught:
                load_image_and_mask(pair.image_path, pair.mask_path)
        self.assertIn("dimensions differ", str(caught.exception))

    def test_unreadable_image_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "train_img"
            image_dir.mkdir()
            (image_dir / "notes.jpg").write_text("not a jpeg", encoding="utf-8")
            _write_png(root / "train_lab", "notes", _mask(3, 3, 0))
            pair = list_pairs(root, "train")[0]
            with self.assertRaises(ValueError) as caught:
                load_image_and_mask(pair.image_path, pair.mask_path)
        self.assertIn("Could not read image", str(caught.exception))


class InspectionTests(unittest.TestCase):
    def test_report_marks_a_small_folder_as_different_from_the_release(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_split(root, "train", "one", _color(4, 6, 90), _mask(4, 6, 1))
            _make_split(root, "test", "one", _color(6, 4, 90), _mask(6, 4, 1))
            report, matches = inspect_dataset(root)
        self.assertFalse(matches)
        self.assertIn("paired files: 1", report)
        self.assertIn("300", report)
        self.assertIn("discrepancy", report)
        self.assertIn("non-commercial", report)
        self.assertIn("liu2019deepcrack", report)

    def test_cli_missing_folders_return_an_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with redirect_stdout(StringIO()):
                code = main(["--dataset-root", temp_dir])
        self.assertEqual(code, 1)

    def test_existing_segmentation_defaults_are_unchanged(self) -> None:
        signature = inspect.signature(segment_blackhat)
        self.assertEqual(signature.parameters["kernel_size"].default, 15)
        self.assertEqual(signature.parameters["threshold"].default, 25)
        self.assertEqual(available_methods(), ["adaptive", "blackhat", "otsu"])


if __name__ == "__main__":
    unittest.main()
