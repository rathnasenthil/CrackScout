"""Metric tests use synthetic masks. They do not read the DeepCrack release."""

import csv
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

from crackscout.evaluate import aggregate_rows, evaluate_deepcrack
from crackscout.metrics import binary_counts, scores_from_counts
from crackscout.segment import segment_adaptive, segment_blackhat, segment_otsu
from run_deepcrack_evaluation import main


def _mask(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8)


class MetricTests(unittest.TestCase):
    def test_perfect_overlap(self) -> None:
        mask = _mask([[0, 255], [255, 0]])
        counts = binary_counts(mask, mask.copy())
        scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
        self.assertEqual(counts["tp"], 2)
        self.assertEqual(counts["fp"], 0)
        self.assertEqual(counts["fn"], 0)
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["f1"], 1.0)
        self.assertEqual(scores["iou"], 1.0)

    def test_partial_overlap(self) -> None:
        ground_truth = _mask([[255, 255], [255, 0]])
        prediction = _mask([[255, 0], [0, 255]])
        counts = binary_counts(ground_truth, prediction)
        scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 1, 2))
        self.assertAlmostEqual(float(scores["precision"]), 0.5)
        self.assertAlmostEqual(float(scores["recall"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(scores["f1"]), 0.4)
        self.assertAlmostEqual(float(scores["iou"]), 0.25)

    def test_no_overlap(self) -> None:
        ground_truth = _mask([[255, 255, 0, 0]])
        prediction = _mask([[0, 0, 255, 255]])
        counts = binary_counts(ground_truth, prediction)
        scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (0, 2, 2))
        self.assertEqual(scores["precision"], 0.0)
        self.assertEqual(scores["recall"], 0.0)
        self.assertEqual(scores["f1"], 0.0)
        self.assertEqual(scores["iou"], 0.0)

    def test_empty_prediction_leaves_precision_blank(self) -> None:
        ground_truth = _mask([[255, 0, 0, 0]])
        prediction = _mask([[0, 0, 0, 0]])
        counts = binary_counts(ground_truth, prediction)
        scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
        self.assertEqual(counts["gt_crack_pixels"], 1)
        self.assertEqual(counts["pred_crack_pixels"], 0)
        self.assertIsNone(scores["precision"])
        self.assertEqual(scores["recall"], 0.0)
        self.assertIsNone(scores["f1"])
        self.assertEqual(scores["iou"], 0.0)
        self.assertIn("precision", str(scores["zero_denominator"]))

    def test_both_empty_is_agreement(self) -> None:
        blank = np.zeros((3, 3), dtype=np.uint8)
        counts = binary_counts(blank, blank.copy())
        scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
        self.assertEqual(scores["zero_denominator"], "both_empty")
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["iou"], 1.0)

    def test_inverted_prediction_is_not_repaired(self) -> None:
        ground_truth = _mask([[255, 0], [0, 0]])
        inverted = _mask([[0, 255], [255, 255]])
        counts = binary_counts(ground_truth, inverted)
        scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
        self.assertEqual(counts["tp"], 0)
        self.assertEqual(scores["iou"], 0.0)

    def test_shape_mismatch_and_bad_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            binary_counts(_mask([[255, 0]]), _mask([[255], [0]]))
        bad = _mask([[128, 0]])
        with self.assertRaises(ValueError):
            binary_counts(bad, bad.copy())


class AggregateTests(unittest.TestCase):
    def test_macro_mean_skips_undefined_precision(self) -> None:
        rows = [
            {
                "status": "ok",
                "tp": 4,
                "fp": 0,
                "fn": 0,
                "gt_crack_pixels": 4,
                "pred_crack_pixels": 4,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 1.0,
            },
            {
                "status": "ok",
                "tp": 0,
                "fp": 0,
                "fn": 4,
                "gt_crack_pixels": 4,
                "pred_crack_pixels": 0,
                "precision": None,
                "recall": 0.0,
                "f1": None,
                "iou": 0.0,
            },
        ]
        stats = aggregate_rows(rows)
        self.assertEqual(stats["undefined_precision"], 1)
        self.assertEqual(stats["macro_precision"], 1.0)
        self.assertAlmostEqual(float(stats["macro_recall"]), 0.5)
        self.assertEqual(stats["macro_f1"], 1.0)
        self.assertAlmostEqual(float(stats["micro_precision"]), 1.0)
        self.assertAlmostEqual(float(stats["micro_recall"]), 0.5)
        self.assertAlmostEqual(float(stats["micro_iou"]), 0.5)


class EvaluationRunTests(unittest.TestCase):
    def test_synthetic_run_writes_results_and_refuses_a_second_write(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((24, 32, 3), 190, dtype=np.uint8)
            image[10:13, 4:28] = 20
            mask = np.zeros((24, 32), dtype=np.uint8)
            mask[10:13, 4:28] = 255
            for split in ("train", "test"):
                image_dir = root / f"{split}_img"
                mask_dir = root / f"{split}_lab"
                image_dir.mkdir()
                mask_dir.mkdir()
                self.assertTrue(cv2.imwrite(str(image_dir / "sample.jpg"), image))
                self.assertTrue(cv2.imwrite(str(mask_dir / "sample.png"), mask))
            output_dir = root / "eval"
            untouched = output_dir
            untouched.mkdir()
            sentinel = untouched / "keep.txt"
            sentinel.write_text("existing", encoding="utf-8")

            saved = evaluate_deepcrack(root, output_dir, preview_count=1)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing")
            self.assertTrue(saved["per_image_csv"].is_file())
            self.assertTrue(saved["summary"].is_file())
            summary = saved["summary"].read_text(encoding="utf-8")
            self.assertIn("exploratory", summary)
            self.assertIn("not a final held-out", summary)
            self.assertIn("both_empty", summary)
            self.assertIn("foreground = 255", summary)
            self.assertTrue(any((output_dir / "previews").glob("*.png")))

            with saved["per_image_csv"].open(encoding="utf-8", newline="") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), 6)
            self.assertEqual({row["method"] for row in table}, {"otsu", "adaptive", "blackhat"})
            self.assertEqual({row["split"] for row in table}, {"train", "test"})
            self.assertTrue(all(row["parameters"] for row in table))
            self.assertTrue(all(row["status"] == "ok" for row in table))
            otsu = next(row for row in table if row["split"] == "train" and row["method"] == "otsu")
            self.assertIn("open_kernel=3", otsu["parameters"])
            self.assertIn("close_kernel=5", otsu["parameters"])
            adaptive = next(row for row in table if row["method"] == "adaptive")
            self.assertIn("block_size=35", adaptive["parameters"])
            blackhat = next(row for row in table if row["method"] == "blackhat")
            self.assertIn("kernel_size=15", blackhat["parameters"])
            self.assertIn("threshold=25", blackhat["parameters"])

            with self.assertRaises(FileExistsError):
                evaluate_deepcrack(root, output_dir, preview_count=1)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing")

    def test_cli_missing_dataset_returns_an_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            buffer = StringIO()
            with redirect_stderr(buffer), redirect_stdout(StringIO()):
                code = main(["--dataset-root", temp_dir, "--output-dir", str(Path(temp_dir) / "out")])
        self.assertEqual(code, 1)
        self.assertIn("not found", buffer.getvalue().lower())

    def test_segmentation_defaults_remain_unchanged(self) -> None:
        self.assertEqual(inspect.signature(segment_otsu).parameters["open_kernel"].default, 3)
        self.assertEqual(inspect.signature(segment_otsu).parameters["close_kernel"].default, 5)
        self.assertEqual(inspect.signature(segment_adaptive).parameters["block_size"].default, 35)
        self.assertEqual(inspect.signature(segment_adaptive).parameters["C"].default, 5)
        self.assertEqual(inspect.signature(segment_blackhat).parameters["kernel_size"].default, 15)
        self.assertEqual(inspect.signature(segment_blackhat).parameters["threshold"].default, 25)


if __name__ == "__main__":
    unittest.main()
