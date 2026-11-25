"""Tests for the fixed black-hat parameter grid. They use synthetic images."""

import inspect
import sys
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from tempfile import TemporaryDirectory

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.preprocess import denoise, to_grayscale
from crackscout.segment import available_methods, segment, segment_blackhat
from crackscout.sensitivity import (
    BLACKHAT_SENSITIVITY_CONFIGS,
    describe_mask,
    run_blackhat_sensitivity,
    summary_path,
)
from run_blackhat_sensitivity import main


def _synthetic_color_image() -> np.ndarray:
    """Light field with one thin dark row, saved as a color image."""
    gray = np.full((80, 100), 180, dtype=np.uint8)
    gray[40, 10:90] = 20
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class MeasurementTests(unittest.TestCase):
    def test_empty_and_separated_components(self) -> None:
        blank = np.zeros((10, 10), dtype=np.uint8)
        blank_stats = describe_mask(blank)
        self.assertEqual(blank_stats["white_pixels"], 0)
        self.assertEqual(blank_stats["white_percent"], 0.0)
        self.assertEqual(blank_stats["connected_components"], 0)

        separated = blank.copy()
        separated[1, 1] = 255
        separated[8, 8] = 255
        separated_stats = describe_mask(separated)
        self.assertEqual(separated_stats["white_pixels"], 2)
        self.assertEqual(separated_stats["white_percent"], 2.0)
        self.assertEqual(separated_stats["connected_components"], 2)

        diagonal = blank.copy()
        diagonal[1, 1] = 255
        diagonal[2, 2] = 255
        self.assertEqual(describe_mask(diagonal)["connected_components"], 1)

    def test_non_binary_mask_is_rejected(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 128
        with self.assertRaises(ValueError):
            describe_mask(mask)


class SensitivityExperimentTests(unittest.TestCase):
    def test_grid_and_blackhat_defaults_stay_fixed(self) -> None:
        self.assertEqual(
            BLACKHAT_SENSITIVITY_CONFIGS,
            ((15, 25), (9, 25), (21, 25), (15, 15), (15, 35)),
        )
        signature = inspect.signature(segment_blackhat)
        self.assertEqual(signature.parameters["kernel_size"].default, 15)
        self.assertEqual(signature.parameters["threshold"].default, 25)
        self.assertEqual(available_methods(), ["adaptive", "blackhat", "otsu"])

    def test_experiment_reuses_blackhat_and_writes_unique_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "synthetic_crack.png"
            self.assertTrue(cv2.imwrite(str(source), _synthetic_color_image()))
            output_dir = temp / "out"
            untouched = output_dir
            untouched.mkdir()
            prior = untouched / "synthetic_crack_blackhat_mask.png"
            prior.write_bytes(b"existing-blackhat-output")

            rows = run_blackhat_sensitivity(source, output_dir)
            self.assertEqual(len(rows), 5)
            self.assertEqual(prior.read_bytes(), b"existing-blackhat-output")

            names = {Path(row["mask"]).name for row in rows}
            self.assertEqual(
                names,
                {
                    "synthetic_crack_blackhat_k15_t25_mask.png",
                    "synthetic_crack_blackhat_k9_t25_mask.png",
                    "synthetic_crack_blackhat_k21_t25_mask.png",
                    "synthetic_crack_blackhat_k15_t15_mask.png",
                    "synthetic_crack_blackhat_k15_t35_mask.png",
                },
            )
            report = summary_path(source.stem, output_dir).read_text(encoding="utf-8")
            self.assertIn("do not measure detection accuracy", report)
            self.assertIn("do not show whether a shadow was removed", report)
            self.assertIn("does not rank them or select a parameter", report)

            color = cv2.imread(str(source), cv2.IMREAD_COLOR)
            gray = denoise(to_grayscale(color))
            for row in rows:
                self.assertTrue(Path(row["mask"]).is_file())
                self.assertTrue(Path(row["comparison"]).is_file())
                saved = cv2.imread(str(row["mask"]), cv2.IMREAD_GRAYSCALE)
                expected = segment(
                    gray,
                    method="blackhat",
                    kernel_size=row["kernel_size"],
                    threshold=row["threshold"],
                )
                self.assertIsNotNone(saved)
                np.testing.assert_array_equal(saved, expected)
                self.assertEqual(row["white_pixels"], int(np.count_nonzero(expected == 255)))

    def test_second_run_does_not_overwrite(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "synthetic_crack.png"
            self.assertTrue(cv2.imwrite(str(source), _synthetic_color_image()))
            output_dir = temp / "out"
            run_blackhat_sensitivity(source, output_dir)
            mask = output_dir / "synthetic_crack_blackhat_k15_t25_mask.png"
            original = mask.read_bytes()
            with self.assertRaises(FileExistsError):
                run_blackhat_sensitivity(source, output_dir)
            self.assertEqual(mask.read_bytes(), original)

    def test_cli_missing_image_returns_error(self) -> None:
        missing = ROOT / "samples" / "this_file_is_not_in_the_repo.jpg"
        buffer = StringIO()
        with redirect_stderr(buffer), redirect_stdout(StringIO()):
            code = main(["--input", str(missing), "--output-dir", "outputs"])
        self.assertEqual(code, 1)
        self.assertIn("not found", buffer.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
