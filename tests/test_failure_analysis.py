"""Synthetic checks for failure-analysis calculations."""

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.failure_analysis import (
    empty_predictions,
    f1_distribution,
    formula_mismatches,
    linear_percentile,
    representative_failures,
    validate_inventory,
    write_failure_analysis,
)


def _row(
    split: str,
    stem: str,
    method: str,
    *,
    tp: int,
    fp: int,
    fn: int,
    precision: float | None,
    recall: float | None,
    f1: float | None,
    iou: float | None,
    zero_denominator: str,
) -> dict[str, object]:
    return {
        "split": split,
        "stem": stem,
        "method": method,
        "parameters": "defaults",
        "status": "ok",
        "error": "",
        "gt_crack_pixels": tp + fn,
        "pred_crack_pixels": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "zero_denominator": zero_denominator,
    }


class PercentileTests(unittest.TestCase):
    def test_linear_quartiles_skip_nothing_already_removed(self) -> None:
        self.assertEqual(linear_percentile([1, 2, 3, 4], 25), 1.75)
        self.assertEqual(linear_percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(linear_percentile([1, 2, 3, 4], 75), 3.25)
        stats = f1_distribution([0.2, 0.4])
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(float(stats["median"]), 0.3)
        self.assertAlmostEqual(float(stats["min"]), 0.2)
        self.assertAlmostEqual(float(stats["max"]), 0.4)


class InventoryAndFormulaTests(unittest.TestCase):
    def test_duplicate_and_missing_method_are_reported(self) -> None:
        rows = [
            _row("train", "a", "otsu", tp=1, fp=0, fn=0, precision=1, recall=1, f1=1, iou=1, zero_denominator="none"),
            _row("train", "a", "otsu", tp=1, fp=0, fn=0, precision=1, recall=1, f1=1, iou=1, zero_denominator="none"),
            _row("train", "a", "adaptive", tp=1, fp=0, fn=0, precision=1, recall=1, f1=1, iou=1, zero_denominator="none"),
        ]
        problems = validate_inventory(
            rows,
            splits=("train",),
            methods=("otsu", "adaptive", "blackhat"),
            expected_pair_counts={"train": 1},
        )
        self.assertTrue(any("Duplicate" in problem for problem in problems))
        self.assertTrue(any("blackhat" in problem and "missing" in problem for problem in problems))

    def test_wrong_precision_is_a_mismatch_and_blanks_stay_blank(self) -> None:
        consistent = _row(
            "train",
            "empty-pred",
            "blackhat",
            tp=0,
            fp=0,
            fn=4,
            precision=None,
            recall=0.0,
            f1=None,
            iou=0.0,
            zero_denominator="undefined:precision+f1",
        )
        altered = dict(consistent)
        altered["precision"] = 1.0
        altered["stem"] = "altered"
        self.assertEqual(formula_mismatches([consistent]), [])
        mismatches = formula_mismatches([altered])
        self.assertTrue(any("precision" in item for item in mismatches))
        found = empty_predictions([consistent])
        self.assertEqual(len(found), 1)
        self.assertIsNone(found[0]["precision"])
        self.assertIsNone(found[0]["f1"])
        self.assertEqual(found[0]["recall"], 0.0)

    def test_lowest_f1_ignores_undefined_scores(self) -> None:
        rows = [
            _row("test", "blank", "otsu", tp=0, fp=0, fn=3, precision=None, recall=0, f1=None, iou=0, zero_denominator="undefined:precision+f1"),
            _row("test", "mid", "otsu", tp=1, fp=1, fn=0, precision=0.5, recall=1, f1=2 / 3, iou=0.5, zero_denominator="none"),
            _row("test", "low", "otsu", tp=0, fp=2, fn=2, precision=0, recall=0, f1=0, iou=0, zero_denominator="none"),
        ]
        chosen = [
            example
            for example in representative_failures(rows, splits=("test",), methods=("otsu",))
            if example["criterion"] == "lowest_defined_f1"
        ]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["stem"], "low")
        self.assertEqual(chosen[0]["f1"], 0)


class ReportWriteTests(unittest.TestCase):
    def test_write_refuses_to_replace_an_existing_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "per_image.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "split",
                        "stem",
                        "method",
                        "parameters",
                        "status",
                        "error",
                        "gt_crack_pixels",
                        "pred_crack_pixels",
                        "tp",
                        "fp",
                        "fn",
                        "precision",
                        "recall",
                        "f1",
                        "iou",
                        "zero_denominator",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "train",
                        "stem": "only",
                        "method": "otsu",
                        "parameters": "defaults",
                        "status": "ok",
                        "error": "",
                        "gt_crack_pixels": "2",
                        "pred_crack_pixels": "2",
                        "tp": "2",
                        "fp": "0",
                        "fn": "0",
                        "precision": "1.000000",
                        "recall": "1.000000",
                        "f1": "1.000000",
                        "iou": "1.000000",
                        "zero_denominator": "none",
                    }
                )
            output_dir = root / "report"
            saved = write_failure_analysis(
                source,
                output_dir,
                summary_csv=None,
                preview_dir=None,
                expected_pair_counts=None,
            )
            text = Path(saved["report"]).read_text(encoding="utf-8")
            self.assertIn("macro F1", text)
            self.assertIn("micro F1", text)
            self.assertIn("undefined", text.lower())
            with self.assertRaises(FileExistsError):
                write_failure_analysis(
                    source,
                    output_dir,
                    summary_csv=None,
                    preview_dir=None,
                    expected_pair_counts=None,
                )


if __name__ == "__main__":
    unittest.main()
