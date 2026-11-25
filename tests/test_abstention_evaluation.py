"""Abstention tests use synthetic rows and do not read DeepCrack."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.abstention_evaluation import (
    MINIMUM_ABSOLUTE_RISK_REDUCTION,
    MINIMUM_COVERAGE,
    SELECTION_RULE,
    bootstrap_outcome,
    build_curve,
    curve_is_nested,
    outcome_at_threshold,
    paired_bootstrap_draws,
    random_abstention_trials,
    run_abstention,
    select_operating_threshold,
    write_outputs,
)
from crackscout.reliability_prediction import METHODS


def _row(
    split: str,
    stem: str,
    method: str,
    score: float,
    failure: int,
    *,
    f1: float | None = 0.5,
    internal: float | None = None,
    feature_score: float | None = None,
) -> dict[str, object]:
    return {
        "split": split,
        "stem": stem,
        "method": method,
        "f1": f1,
        "f1_undefined": 1 if f1 is None else 0,
        "failure_threshold": 0.25,
        "failure_label": failure,
        "failure_reason": "undefined_f1" if f1 is None else ("low_f1" if failure else "above_threshold"),
        "logistic_probability": score,
        "internal_logistic_probability": score if internal is None else internal,
        "single_feature_score": score if feature_score is None else feature_score,
        "single_feature_predicted_failure": 0,
        "single_feature_threshold": 100.0,
    }


def _separable_training(method: str = "otsu") -> list[dict[str, object]]:
    rows = []
    for index in range(10):
        failure = 1 if index >= 7 else 0
        rows.append(
            _row("train", f"image-{index:02d}", method, float(index + 1), failure, f1=0.1 if failure else 0.8)
        )
    return rows


def _complete_methods(train_builder, test_builder) -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        rows.extend(train_builder(method))
        rows.extend(test_builder(method))
    return rows


class SelectionTests(unittest.TestCase):
    def test_rule_is_training_only_and_keeps_the_highest_eligible_coverage(self) -> None:
        self.assertIn("none is taken from the test split", SELECTION_RULE)
        self.assertEqual(MINIMUM_COVERAGE, 0.50)
        self.assertEqual(MINIMUM_ABSOLUTE_RISK_REDUCTION, 0.05)
        training = _separable_training()
        selected = select_operating_threshold(training)
        self.assertTrue(selected["justified"])
        self.assertEqual(selected["threshold"], 10.0)
        self.assertAlmostEqual(float(selected["training_outcome"]["coverage"]), 0.9)
        with self.assertRaises(ValueError):
            select_operating_threshold(training + [_row("test", "held", "otsu", 0.0, 1)])

    def test_test_labels_do_not_change_the_threshold_or_create_one(self) -> None:
        def train_rows(method: str) -> list[dict[str, object]]:
            rows = []
            for index in range(8):
                rows.append(_row("train", f"image-{index}", method, float(index), 1 if index < 2 else 0))
            return rows

        def test_rows(method: str, reverse: bool) -> list[dict[str, object]]:
            rows = []
            for index in range(8):
                failure = index >= 6 if reverse else index < 2
                rows.append(_row("test", f"test-{index}", method, float(index), 1 if failure else 0))
            return rows

        first = run_abstention(
            _complete_methods(train_rows, lambda method: test_rows(method, False)),
            n_bootstrap=20,
            n_random_trials=5,
        )
        second = run_abstention(
            _complete_methods(train_rows, lambda method: test_rows(method, True)),
            n_bootstrap=20,
            n_random_trials=5,
        )
        self.assertFalse(first["config"]["methods"]["otsu"]["justified"])
        self.assertEqual(
            first["config"]["methods"]["otsu"]["threshold"],
            second["config"]["methods"]["otsu"]["threshold"],
        )
        self.assertIsNone(first["config"]["methods"]["otsu"]["threshold"])

    def test_frozen_threshold_is_applied_to_test_scores(self) -> None:
        rows = []
        for method in METHODS:
            rows.extend(_separable_training(method))
            rows.append(_row("test", "low", method, 1.0, 0, internal=None))
            rows.append(_row("test", "high", method, 20.0, 1, internal=None))
        result = run_abstention(rows, n_bootstrap=15, n_random_trials=4)
        self.assertEqual(result["config"]["methods"]["otsu"]["threshold"], 10.0)
        test_point = next(
            point
            for point in result["operating_points"]
            if point["method"] == "otsu" and point["split"] == "test" and point["policy"] == "logistic"
        )
        self.assertEqual(test_point["threshold"], 10.0)
        self.assertEqual(test_point["n_accepted"], 1)
        self.assertEqual(test_point["accepted_failure"], 0)
        self.assertEqual(test_point["abstained_failure"], 1)


class MetricTests(unittest.TestCase):
    def test_coverage_risk_and_undefined_f1(self) -> None:
        rows = [
            _row("train", "a", "otsu", 0.1, 0, f1=0.9),
            _row("train", "b", "otsu", 0.2, 1, f1=0.1),
            _row("train", "c", "otsu", 0.9, 1, f1=None),
        ]
        scores = np.array([0.1, 0.2, 0.9])
        stats = outcome_at_threshold(rows, scores, 0.5)
        self.assertAlmostEqual(float(stats["coverage"]), 2 / 3)
        self.assertAlmostEqual(float(stats["accepted_failure_risk"]), 0.5)
        self.assertEqual(stats["accepted_failure"], 1)
        self.assertEqual(stats["n_accepted_undefined_f1"], 0)
        self.assertEqual(stats["abstained_failure"], 1)
        self.assertEqual(stats["n_abstained_undefined_f1"], 1)
        self.assertAlmostEqual(float(stats["failure_recall_among_abstained"]), 0.5)
        self.assertEqual(stats["defined_f1_n"], 2)
        accepted_undefined = outcome_at_threshold(
            [_row("train", "blank", "otsu", 0.1, 1, f1=None)],
            np.array([0.1]),
            0.5,
        )
        self.assertEqual(accepted_undefined["accepted_failure"], 1)
        self.assertEqual(accepted_undefined["defined_f1_n"], 0)
        self.assertIsNone(accepted_undefined["defined_f1_mean"])
        self.assertEqual(accepted_undefined["fraction_accepted_with_defined_f1"], 0.0)

    def test_saved_failure_bit_abstains_when_the_score_is_directed(self) -> None:
        rows = [
            _row("train", "low-std", "otsu", -30.0, 1),
            _row("train", "high-std", "otsu", -10.0, 0),
        ]
        rows[0]["single_feature_predicted_failure"] = 1
        rows[1]["single_feature_predicted_failure"] = 0
        stats = outcome_at_threshold(rows, np.array([1.0, 0.0]), 0.5)
        self.assertEqual(stats["n_abstained"], 1)
        self.assertEqual(stats["accepted_failure"], 0)
        self.assertEqual(stats["abstained_failure"], 1)

    def test_empty_accepted_and_empty_abstained_sets_stay_undefined(self) -> None:
        rows = [_row("train", "a", "otsu", 0.9, 1), _row("train", "b", "otsu", 0.8, 0)]
        empty_accept = outcome_at_threshold(rows, np.array([0.9, 0.8]), 0.1)
        self.assertEqual(empty_accept["n_accepted"], 0)
        self.assertIsNone(empty_accept["accepted_failure_risk"])
        self.assertIsNone(empty_accept["absolute_risk_reduction"])
        self.assertIsNone(empty_accept["defined_f1_mean"])
        empty_abstain = outcome_at_threshold(rows, np.array([0.2, 0.3]), 0.9)
        self.assertEqual(empty_abstain["n_abstained"], 0)
        self.assertIsNone(empty_abstain["failure_recall_among_abstained"])
        self.assertAlmostEqual(float(empty_abstain["absolute_risk_reduction"]), 0.0)

    def test_curve_coverage_is_nested_and_ordered(self) -> None:
        rows = [
            _row("train", "a", "otsu", 0.1, 0),
            _row("train", "b", "otsu", 0.3, 1),
            _row("train", "c", "otsu", 0.2, 0),
        ]
        self.assertTrue(curve_is_nested(rows, "logistic_probability"))
        curve = build_curve(rows, "logistic_probability", 0.25)
        coverages = [float(point["coverage"]) for point in curve]
        self.assertEqual(coverages, sorted(coverages))
        self.assertEqual(curve[0]["risk_defined"], 0)
        self.assertEqual(curve[-1]["is_always_accept"], 1)
        self.assertAlmostEqual(float(curve[-1]["accepted_failure_risk"]), 1 / 3)
        operating = [point for point in curve if point["is_operating_point"]]
        self.assertEqual(len(operating), 1)

    def test_random_trials_repeat_and_bootstrap_draws_are_paired(self) -> None:
        rows = [_row("train", f"image-{index}", "otsu", float(index), index % 2) for index in range(6)]
        first = random_abstention_trials(rows, 3, n_trials=5, seed=7)
        second = random_abstention_trials(rows, 3, n_trials=5, seed=7)
        self.assertEqual(
            [trial["n_accepted_failures"] for trial in first],
            [trial["n_accepted_failures"] for trial in second],
        )
        self.assertEqual(len({trial["n_accepted"] for trial in first}), 1)
        stems = ["b", "a", "c"]
        reordered = list(reversed(stems))
        draws = paired_bootstrap_draws(3, 4, 5)
        by_stem = {stem: index for index, stem in enumerate(sorted(stems))}
        other = {stem: index for index, stem in enumerate(sorted(reordered))}
        self.assertEqual(by_stem, other)
        sample = [sorted(stems)[index] for index in draws[0].tolist()]
        again = [sorted(reordered)[index] for index in draws[0].tolist()]
        self.assertEqual(sample, again)
        interval = bootstrap_outcome(rows, "logistic_probability", 3.0, paired_bootstrap_draws(6, 10, 1))
        self.assertEqual(interval["replicates"], 10)
        self.assertLessEqual(interval["replicates_undefined_risk"], 10)


class OutputTests(unittest.TestCase):
    def test_existing_files_are_preserved(self) -> None:
        rows = _complete_methods(
            _separable_training,
            lambda method: [_row("test", "only", method, 1.0, 0, internal=None)],
        )
        result = run_abstention(rows, n_bootstrap=8, n_random_trials=3)
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "out"
            destination.mkdir()
            sentinel = destination / "notes.txt"
            sentinel.write_text("keep", encoding="utf-8")
            blocked = destination / "report.txt"
            blocked.write_text("original", encoding="utf-8")
            source = Path(temp_dir) / "predictions.csv"
            source.write_text("frozen", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_outputs(destination, result)
            self.assertEqual(blocked.read_text(encoding="utf-8"), "original")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(source.read_text(encoding="utf-8"), "frozen")
            self.assertFalse((destination / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
