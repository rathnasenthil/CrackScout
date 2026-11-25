"""Visualization tests use temporary images, not the DeepCrack release."""

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.evaluate import method_parameters, predict_mask
from crackscout.failure_visualization import (
    REQUIRED_CASES,
    load_example_cases,
    render_failure_cases,
    resolve_pair,
)
from crackscout.metrics import binary_counts, scores_from_counts


def _write_case(root: Path, split: str, stem: str, value: int) -> None:
    image_dir = root / f"{split}_img"
    mask_dir = root / f"{split}_lab"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 64, 3), value, dtype=np.uint8)
    image[16:24, 8:56] = 15
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[16:24, 8:56] = 255
    if not cv2.imwrite(str(image_dir / f"{stem}.jpg"), image):
        raise RuntimeError(stem)
    if not cv2.imwrite(str(mask_dir / f"{stem}.png"), mask):
        raise RuntimeError(stem)


class ResolveTests(unittest.TestCase):
    def test_same_stem_does_not_cross_splits(self) -> None:
        pairs = {
            "train": {"shared": "train-file"},
            "test": {"shared": "test-file"},
        }
        self.assertEqual(resolve_pair(pairs, "test", "shared"), "test-file")
        with self.assertRaises(ValueError) as caught:
            resolve_pair(pairs, "train", "missing")
        self.assertIn("other split was not searched", str(caught.exception))

    def test_missing_documented_case_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "examples.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["criterion", "split", "stem", "method", "f1"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "criterion": "lowest_defined_f1",
                        "split": "train",
                        "stem": "11196",
                        "method": "otsu",
                        "f1": "0.100000",
                    }
                )
            with self.assertRaises(ValueError) as caught:
                load_example_cases(path)
        self.assertIn("IMG24-1", str(caught.exception))


class RenderTests(unittest.TestCase):
    def test_render_uses_the_named_split_and_refuses_a_second_write(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split, stem, _method in REQUIRED_CASES:
                _write_case(root, split, stem, 180 if split == "test" else 30)
            _write_case(root, "train", "IMG24-1", 5)

            examples_path = root / "examples.csv"
            evaluation_path = root / "per_image.csv"
            example_rows = []
            evaluation_rows = []
            for split, stem, method in REQUIRED_CASES:
                image = cv2.imread(str(root / f"{split}_img" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(root / f"{split}_lab" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
                prediction = predict_mask(image, method)
                counts = binary_counts(mask, prediction)
                scores = scores_from_counts(counts["tp"], counts["fp"], counts["fn"])
                f1 = "" if scores["f1"] is None else f"{float(scores['f1']):.6f}"
                example_rows.append(
                    {
                        "criterion": "lowest_defined_f1",
                        "split": split,
                        "stem": stem,
                        "method": method,
                        "f1": f1,
                    }
                )
                evaluation_rows.append(
                    {
                        "split": split,
                        "stem": stem,
                        "method": method,
                        "parameters": method_parameters(method),
                        "status": "ok",
                        "error": "",
                        "gt_crack_pixels": str(counts["gt_crack_pixels"]),
                        "pred_crack_pixels": str(counts["pred_crack_pixels"]),
                        "tp": str(counts["tp"]),
                        "fp": str(counts["fp"]),
                        "fn": str(counts["fn"]),
                        "precision": "" if scores["precision"] is None else f"{float(scores['precision']):.6f}",
                        "recall": "" if scores["recall"] is None else f"{float(scores['recall']):.6f}",
                        "f1": f1,
                        "iou": "" if scores["iou"] is None else f"{float(scores['iou']):.6f}",
                        "zero_denominator": scores["zero_denominator"],
                    }
                )
            with examples_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["criterion", "split", "stem", "method", "f1"],
                )
                writer.writeheader()
                writer.writerows(example_rows)
            with evaluation_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(evaluation_rows[0]))
                writer.writeheader()
                writer.writerows(evaluation_rows)

            output_dir = root / "figures"
            figures = render_failure_cases(examples_path, evaluation_path, root, output_dir)
            self.assertEqual(len(figures), len(REQUIRED_CASES))
            self.assertTrue((output_dir / "test_IMG24-1_otsu_diagnostic.png").is_file())
            self.assertTrue((output_dir / "train_7Q3A9064-20_blackhat_diagnostic.png").is_file())
            with self.assertRaises(FileExistsError):
                render_failure_cases(examples_path, evaluation_path, root, output_dir)


if __name__ == "__main__":
    unittest.main()
