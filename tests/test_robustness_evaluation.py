"""Robustness tests use synthetic rows and do not read DeepCrack."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.abstention_evaluation import select_operating_threshold
from crackscout.image_quality import FEATURE_NAMES
from crackscout.reliability_prediction import assign_folds
from crackscout.robustness_evaluation import (
    evaluate_outer_fold,
    lowest_risk_indexes,
    matched_acceptance_count,
    outer_fold_validation,
    paired_bootstrap,
    reduction_from_sample,
    run_robustness,
    write_outputs,
)


def _features(value: float) -> dict[str, float]:
    return {name: value + index / 10.0 for index, name in enumerate(FEATURE_NAMES)}


def _model_row(stem: str, method: str, value: float, failure: int, f1: float | None = 0.4) -> dict[str, object]:
    return {
        "split": "train",
        "stem": stem,
        "method": method,
        "f1": f1,
        "f1_undefined": 1 if f1 is None else 0,
        "failure": failure,
        "failure_label": failure if f1 is not None else 1,
        "failure_reason": "undefined_f1" if f1 is None else ("low_f1" if failure else "above_threshold"),
        **_features(value),
    }


def _prediction(split: str, stem: str, method: str, score: float, failure: int) -> dict[str, object]:
    return {
        "split": split,
        "stem": stem,
        "method": method,
        "f1": 0.2 if failure else 0.8,
        "f1_undefined": 0,
        "failure_label": failure,
        "failure_reason": "low_f1" if failure else "above_threshold",
        "logistic_probability": score,
        "internal_logistic_probability": score,
    }


class FoldAndSelectionTests(unittest.TestCase):
    def test_folds_are_deterministic_and_group_an_image_across_methods(self) -> None:
        stems = [f"image-{index:02d}" for index in range(8)]
        self.assertEqual(assign_folds(stems, 2, 0), assign_folds(list(reversed(stems)), 2, 0))
        longer = [f"stem-{index:02d}" for index in range(30)]
        self.assertNotEqual(assign_folds(longer, 5, 0), assign_folds(longer, 5, 1))
        folds = assign_folds(stems, 2, 0)
        rows = []
        for method in ("otsu", "blackhat"):
            for index, stem in enumerate(stems):
                rows.append(_model_row(stem, method, float(index), 1 if index > 5 else 0))
        records = outer_fold_validation(rows, folds, methods=("otsu", "blackhat"), n_inner_folds=2)
        sizes = {
            (record["method"], record["fold"]): record["n_outer"]
            for record in records
            if record["fold"] != "pooled"
        }
        self.assertEqual(sizes[("otsu", 0)], sizes[("blackhat", 0)])
        self.assertEqual(sizes[("otsu", 1)], sizes[("blackhat", 1)])

    def test_outer_labels_and_test_rows_do_not_select_the_threshold(self) -> None:
        inner = [_model_row(f"inner-{index}", "otsu", float(index), 1 if index >= 6 else 0) for index in range(8)]
        outer = [_model_row(f"outer-{index}", "otsu", float(index), 0) for index in range(4)]
        first = evaluate_outer_fold(inner, outer, n_inner_folds=2)
        changed = [dict(row) for row in outer]
        for row in changed:
            row["failure"] = 1
            row["failure_label"] = 1
            row["f1"] = None
            row["f1_undefined"] = 1
            row["failure_reason"] = "undefined_f1"
        second = evaluate_outer_fold(inner, changed, n_inner_folds=2)
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertTrue(set(first["selection_stems"]).isdisjoint({row["stem"] for row in outer}))
        with self.assertRaises(ValueError):
            outer_fold_validation(inner + [_model_row("held", "otsu", 0.0, 1) | {"split": "test"}], {}, methods=("otsu",))

    def test_ineligible_rule_is_not_relaxed(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "split": "train",
                    "stem": f"image-{index}",
                    "method": "otsu",
                    "f1": 0.1 if index < 2 else 0.8,
                    "f1_undefined": 0,
                    "failure_label": 1 if index < 2 else 0,
                    "failure_reason": "low_f1" if index < 2 else "above_threshold",
                    "internal_logistic_probability": float(index),
                }
            )
        selected = select_operating_threshold(rows)
        self.assertFalse(selected["justified"])
        self.assertIsNone(selected["threshold"])


class MetricTests(unittest.TestCase):
    def test_risk_reduction_counts_undefined_f1_and_empty_acceptance(self) -> None:
        result = reduction_from_sample(
            np.array([1, 0, 1]),
            np.array([0.2, 0.3, 0.8]),
            0.5,
        )
        self.assertEqual(result["n_accepted"], 2)
        self.assertEqual(result["n_accepted_failures"], 1)
        self.assertAlmostEqual(float(result["always_accept_risk"]), 2 / 3)
        self.assertAlmostEqual(float(result["risk_reduction"]), (2 / 3) - 0.5)
        empty = reduction_from_sample(np.array([1, 1]), np.array([0.9, 0.8]), 0.1)
        self.assertFalse(empty["defined"])
        self.assertIsNone(empty["risk_reduction"])
        undefined = _model_row("blank", "otsu", 1.0, 0, f1=None)
        self.assertEqual(undefined["failure_label"], 1)

    def test_matched_coverage_order_is_nested(self) -> None:
        stems = ["b", "a", "c", "d"]
        scores = np.array([0.4, 0.1, 0.2, 0.3])
        self.assertEqual(matched_acceptance_count(237, 0.5), 119)
        first = set(lowest_risk_indexes(stems, scores, 2))
        second = set(lowest_risk_indexes(stems, scores, 3))
        self.assertTrue(first.issubset(second))
        self.assertEqual(lowest_risk_indexes(stems, scores, 1), [1])

    def test_paired_bootstrap_reuses_image_identities_and_repeats(self) -> None:
        rows = {
            method: [
                _prediction("test", f"image-{index}", method, float(index) / 10.0, index % 2)
                for index in range(6)
            ]
            for method in ("otsu", "blackhat")
        }
        rows["blackhat"] = list(reversed(rows["blackhat"]))
        first, summary = paired_bootstrap(rows, {"otsu": 0.4, "blackhat": 0.4}, n_replicates=5, seed=3)
        second, _summary = paired_bootstrap(rows, {"otsu": 0.4, "blackhat": 0.4}, n_replicates=5, seed=3)
        self.assertEqual(
            [row["risk_reduction"] for row in first if row["comparison"] == "otsu"],
            [row["risk_reduction"] for row in second if row["comparison"] == "otsu"],
        )
        paired = [row for row in first if row["comparison"] == "blackhat_minus_otsu"]
        self.assertEqual(len(paired), 5)
        self.assertTrue(any(row["comparison"] == "otsu" for row in summary))


class OutputTests(unittest.TestCase):
    def test_existing_outputs_are_preserved(self) -> None:
        stems = [f"image-{index:02d}" for index in range(8)]
        folds = assign_folds(stems, 2, 0)
        training = [_model_row(stem, "otsu", float(index), 1 if index >= 6 else 0) for index, stem in enumerate(stems)]
        predictions = [
            _prediction("train", stem, "otsu", float(index) / 10.0, 1 if index >= 6 else 0)
            for index, stem in enumerate(stems)
        ]
        predictions.extend(
            _prediction("test", f"test-{index}", "otsu", float(index) / 10.0, index % 2) for index in range(4)
        )
        result = run_robustness(
            training,
            predictions,
            {"otsu": 0.4},
            folds,
            methods=("otsu",),
            n_inner_folds=2,
            n_bootstrap=4,
            n_random_trials=3,
        )
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "out"
            destination.mkdir()
            sentinel = Path(temp_dir) / "predictions.csv"
            sentinel.write_text("frozen", encoding="utf-8")
            blocked = destination / "report.txt"
            blocked.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_outputs(destination, result, predictions)
            self.assertEqual(blocked.read_text(encoding="utf-8"), "original")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "frozen")
            self.assertFalse((destination / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
