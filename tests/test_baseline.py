"""Tests for the classical segmentation methods. They use synthetic images only."""

import sys
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr
from tempfile import TemporaryDirectory

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.io import load_image
from crackscout.preprocess import denoise, to_grayscale
from crackscout.pipeline import run_baseline
from crackscout.segment import available_methods, segment
from run_baseline import main


def _synthetic_crack_bgr() -> np.ndarray:
    """Light field with one dark horizontal band and mild noise."""
    image = np.full((200, 240, 3), 190, dtype=np.uint8)
    image[95:115, 20:220] = 25
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 12, size=image.shape, dtype=np.uint8)
    return cv2.subtract(image, noise)


def _thin_dark_line() -> np.ndarray:
    """Uneven light field with one dark one-pixel row."""
    rows = np.linspace(140, 220, 120, dtype=np.float32)[:, None]
    cols = np.linspace(0, 30, 160, dtype=np.float32)[None, :]
    gray = np.clip(rows + cols, 0, 255).astype(np.uint8)
    gray[60, 20:140] = 30
    return gray


def _assert_binary_mask(test: unittest.TestCase, mask: np.ndarray, shape: tuple[int, ...]) -> None:
    test.assertEqual(mask.shape, shape)
    test.assertEqual(mask.dtype, np.uint8)
    test.assertTrue(set(np.unique(mask)).issubset({0, 255}))


class LoadImageTests(unittest.TestCase):
    def test_missing_file_explains_where_to_put_an_image(self) -> None:
        missing = ROOT / "samples" / "this_file_is_not_in_the_repo.jpg"
        with self.assertRaises(FileNotFoundError) as caught:
            load_image(missing)
        message = str(caught.exception)
        self.assertIn("not found", message.lower())
        self.assertIn("samples", message)

    def test_unreadable_file_raises_value_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fake = Path(temp_dir) / "not-an-image.jpg"
            fake.write_text("this is not image data", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_image(fake)
        self.assertIn("Could not read image", str(caught.exception))


class PreprocessTests(unittest.TestCase):
    def test_grayscale_keeps_spatial_shape(self) -> None:
        color = _synthetic_crack_bgr()
        gray = to_grayscale(color)
        self.assertEqual(gray.shape, color.shape[:2])
        self.assertEqual(gray.dtype, np.uint8)

    def test_denoise_rejects_even_kernel(self) -> None:
        gray = to_grayscale(_synthetic_crack_bgr())
        with self.assertRaises(ValueError):
            denoise(gray, kernel_size=4)


class SegmentationTests(unittest.TestCase):
    def test_otsu_marks_dark_band_more_than_background(self) -> None:
        color = _synthetic_crack_bgr()
        gray = denoise(to_grayscale(color))
        mask = segment(gray, method="otsu")

        self.assertEqual(mask.shape, gray.shape)
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

        band = mask[95:115, 20:220]
        background = mask[0:40, 0:40]
        self.assertGreater(float(band.mean()), 200)
        self.assertLess(float(background.mean()), 20)

    def test_unknown_method_lists_available_names(self) -> None:
        gray = to_grayscale(_synthetic_crack_bgr())
        with self.assertRaises(ValueError) as caught:
            segment(gray, method="does-not-exist")
        message = str(caught.exception)
        self.assertEqual(available_methods(), ["adaptive", "blackhat", "otsu"])
        for name in available_methods():
            self.assertIn(name, message)

    def test_adaptive_returns_binary_mask_for_dark_line(self) -> None:
        gray = _thin_dark_line()
        mask = segment(gray, method="adaptive")
        _assert_binary_mask(self, mask, gray.shape)
        self.assertGreater(float(mask[60, 40:120].mean()), 200)
        self.assertLess(float(mask[20:40, 40:120].mean()), 20)

    def test_adaptive_mean_mode_stays_binary(self) -> None:
        gray = _thin_dark_line()
        mask = segment(gray, method="adaptive", adaptive_method="mean", C=-2)
        _assert_binary_mask(self, mask, gray.shape)

    def test_adaptive_rejects_invalid_block_size_and_method(self) -> None:
        gray = _thin_dark_line()
        for block_size in (1, 2, 4, 0, -3):
            with self.assertRaises(ValueError):
                segment(gray, method="adaptive", block_size=block_size)
        with self.assertRaises(ValueError):
            segment(gray, method="adaptive", adaptive_method="sobel")
        with self.assertRaises(ValueError):
            segment(gray, method="adaptive", C="5")

    def test_blackhat_returns_binary_mask_for_dark_line(self) -> None:
        gray = _thin_dark_line()
        mask = segment(gray, method="blackhat")
        _assert_binary_mask(self, mask, gray.shape)
        self.assertGreater(float(mask[60, 40:120].mean()), 200)
        self.assertLess(float(mask[20:40, 40:120].mean()), 20)

    def test_blackhat_rejects_invalid_kernel_and_threshold(self) -> None:
        gray = _thin_dark_line()
        for kernel_size in (0, 1, 2, 4, -5):
            with self.assertRaises(ValueError):
                segment(gray, method="blackhat", kernel_size=kernel_size)
        for threshold in (-1, 256, 25.5):
            with self.assertRaises(ValueError):
                segment(gray, method="blackhat", threshold=threshold)


class PipelineTests(unittest.TestCase):
    def test_run_baseline_writes_mask_and_comparison(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "synthetic_crack.png"
            self.assertTrue(cv2.imwrite(str(source), _synthetic_crack_bgr()))

            saved = run_baseline(source, temp / "out", method="otsu")
            self.assertTrue(saved["mask"].is_file())
            self.assertTrue(saved["comparison"].is_file())

            mask = cv2.imread(str(saved["mask"]), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(mask)
            self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_methods_write_distinct_filenames(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "synthetic_crack.png"
            self.assertTrue(cv2.imwrite(str(source), _synthetic_crack_bgr()))
            output_dir = temp / "out"
            saved = {
                method: run_baseline(source, output_dir, method=method)
                for method in ("otsu", "adaptive", "blackhat")
            }
            mask_names = {item["mask"].name for item in saved.values()}
            comparison_names = {item["comparison"].name for item in saved.values()}
            self.assertEqual(
                mask_names,
                {
                    "synthetic_crack_otsu_mask.png",
                    "synthetic_crack_adaptive_mask.png",
                    "synthetic_crack_blackhat_mask.png",
                },
            )
            self.assertEqual(len(comparison_names), 3)
            for item in saved.values():
                self.assertTrue(item["mask"].is_file())
                self.assertTrue(item["comparison"].is_file())
                mask = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(mask)
                self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_cli_missing_image_returns_error(self) -> None:
        missing = ROOT / "samples" / "this_file_is_not_in_the_repo.jpg"
        buffer = StringIO()
        with redirect_stderr(buffer):
            code = main(["--input", str(missing)])
        self.assertEqual(code, 1)
        self.assertIn("not found", buffer.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
