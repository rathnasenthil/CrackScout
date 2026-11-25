"""Unit tests for image-quality features. They do not read DeepCrack."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.image_quality import (
    DARK_INTENSITY_THRESHOLD,
    build_quality_report,
    error_group_means,
    extract_quality_features,
    features_for_split,
    join_features,
    miss_rate,
    pearson_correlation,
    training_f1_cutoffs,
)


def _bgr(value: int, size: int = 20) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


class FeatureTests(unittest.TestCase):
    def test_constant_image_has_zero_variation(self) -> None:
        features = extract_quality_features(_bgr(100))
        self.assertAlmostEqual(features["grayscale_mean"], 100.0)
        self.assertAlmostEqual(features["grayscale_std"], 0.0)
        self.assertAlmostEqual(features["percentile_range_p90_p10"], 0.0)
        self.assertAlmostEqual(features["laplacian_variance"], 0.0)
        self.assertAlmostEqual(features["local_contrast"], 0.0)
        self.assertAlmostEqual(features["dark_pixel_proportion"], 0.0)

    def test_dark_proportion_uses_the_fixed_threshold(self) -> None:
        dark = _bgr(DARK_INTENSITY_THRESHOLD - 1)
        bright = _bgr(DARK_INTENSITY_THRESHOLD)
        self.assertAlmostEqual(extract_quality_features(dark)["dark_pixel_proportion"], 1.0)
        self.assertAlmostEqual(extract_quality_features(bright)["dark_pixel_proportion"], 0.0)

    def test_correlation_skips_undefined_scores(self) -> None:
        coefficient, count = pearson_correlation([1, 2, 3, 4], [1.0, 2.0, 3.0, None])
        self.assertEqual(count, 3)
        self.assertAlmostEqual(coefficient, 1.0)
        missing, missing_count = pearson_correlation([1, 2, 3], [None, None, 1.0])
        self.assertIsNone(missing)
        self.assertEqual(missing_count, 1)

    def test_training_cutoffs_ignore_the_test_split(self) -> None:
        rows = [
            {"split": "train", "method": "otsu", "f1": value}
            for value in (0.1, 0.2, 0.3, 0.4)
        ]
        rows.append({"split": "test", "method": "otsu", "f1": 0.99})
        rows.append({"split": "train", "method": "otsu", "f1": None})
        first = training_f1_cutoffs(rows, "otsu")
        rows[4]["f1"] = 0.0
        second = training_f1_cutoffs(rows, "otsu")
        self.assertEqual(first, second)
        self.assertEqual(first[2], 4)

    def test_error_groups_exclude_undefined_f1(self) -> None:
        rows = [
            {"split": "train", "method": "otsu", "f1": 0.1, "grayscale_mean": 10.0},
            {"split": "train", "method": "otsu", "f1": 0.2, "grayscale_mean": 20.0},
            {"split": "train", "method": "otsu", "f1": 0.8, "grayscale_mean": 80.0},
            {"split": "train", "method": "otsu", "f1": 0.9, "grayscale_mean": 90.0},
            {"split": "train", "method": "otsu", "f1": None, "grayscale_mean": 0.0},
        ]
        means = error_group_means(
            rows,
            split="train",
            method="otsu",
            feature="grayscale_mean",
            low_f1_max=0.2,
            high_f1_min=0.8,
        )
        self.assertEqual(means["higher_error_n"], 2)
        self.assertEqual(means["lower_error_n"], 2)
        self.assertEqual(means["undefined_f1_excluded"], 1)
        self.assertAlmostEqual(float(means["higher_error_mean"]), 15.0)
        self.assertAlmostEqual(float(means["lower_error_mean"]), 85.0)

    def test_miss_rate_stays_undefined_without_ground_truth_cracks(self) -> None:
        self.assertIsNone(miss_rate({"gt_crack_pixels": 0, "fn": 0, "tp": 0}))
        self.assertIsNone(miss_rate({"gt_crack_pixels": None, "fn": None, "tp": None}))
        self.assertAlmostEqual(
            miss_rate({"gt_crack_pixels": 10, "fn": 4, "tp": 6, "f1": None}),
            0.4,
        )


class PairingTests(unittest.TestCase):
    def test_same_stem_stays_inside_its_split(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split, value in (("train", 40), ("test", 180)):
                image_dir = root / f"{split}_img"
                mask_dir = root / f"{split}_lab"
                image_dir.mkdir()
                mask_dir.mkdir()
                self.assertTrue(cv2.imwrite(str(image_dir / "shared.jpg"), _bgr(value)))
                mask = np.zeros((20, 20), dtype=np.uint8)
                mask[0, 0] = 255
                self.assertTrue(cv2.imwrite(str(mask_dir / "shared.png"), mask))
            features = features_for_split(root, "train") + features_for_split(root, "test")
        self.assertAlmostEqual(float(features[0]["grayscale_mean"]), 40.0)
        self.assertAlmostEqual(float(features[1]["grayscale_mean"]), 180.0)
        evaluation = [
            {"split": "train", "stem": "shared", "method": "otsu", "f1": 0.2},
            {"split": "test", "stem": "shared", "method": "otsu", "f1": 0.8},
        ]
        joined = join_features(evaluation, features)
        self.assertAlmostEqual(float(joined[0]["grayscale_mean"]), 40.0)
        self.assertAlmostEqual(float(joined[1]["grayscale_mean"]), 180.0)

    def test_missing_split_stem_is_not_replaced(self) -> None:
        features = [{"split": "train", "stem": "only", "grayscale_mean": 1.0}]
        for name in (
            "grayscale_std",
            "percentile_range_p90_p10",
            "laplacian_variance",
            "dark_pixel_proportion",
            "local_contrast",
        ):
            features[0][name] = 0.0
        with self.assertRaises(ValueError) as caught:
            join_features(
                [{"split": "test", "stem": "only", "method": "otsu", "f1": None}],
                features,
            )
        self.assertIn("test/only", str(caught.exception))
        self.assertIn("not used as a substitute", str(caught.exception))

    def test_report_keeps_undefined_f1_blank(self) -> None:
        rows = []
        for index, score in enumerate((0.1, 0.2, 0.8, 0.9, None)):
            row = {
                "split": "train",
                "stem": f"image-{index}",
                "method": "otsu",
                "f1": score,
                "grayscale_mean": float(index + 1),
                "grayscale_std": 1.0,
                "percentile_range_p90_p10": 2.0,
                "laplacian_variance": 3.0,
                "dark_pixel_proportion": 0.1,
                "local_contrast": 0.2,
            }
            rows.append(row)
        text = build_quality_report(rows)
        self.assertIn("undefined F1 1", text)
        self.assertIn("not replaced with 0 or 1", text)
        self.assertIn("does not establish", text)


if __name__ == "__main__":
    unittest.main()
