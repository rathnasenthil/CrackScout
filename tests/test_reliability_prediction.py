"""Reliability-prediction tests use synthetic rows, not the DeepCrack release."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackscout.failure_analysis import linear_percentile
from crackscout.image_quality import FEATURE_NAMES
from crackscout.reliability_prediction import (
    assign_folds,
    classification_metrics,
    failure_threshold,
    fit_logistic,
    fit_medians,
    fit_method_models,
    label_failure,
    load_joined_features,
    majority_class,
    predictor_matrix,
    run_experiment,
    select_single_feature,
    write_outputs,
)


def _features(seed: float) -> list[float]:
    return [seed, seed / 10.0, seed + 1.0, seed * 2.0, seed / 100.0, seed / 5.0]


def _row(split: str, stem: str, method: str, f1: float | None, seed: float) -> dict[str, object]:
    row: dict[str, object] = {"split": split, "stem": stem, "method": method, "f1": f1}
    for name, value in zip(FEATURE_NAMES, _features(seed)):
        row[name] = value
    return row


def _training_block(method: str = "otsu") -> list[dict[str, object]]:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rows = [
        _row("train", f"image-{index}", method, score, float(index + 1))
        for index, score in enumerate(scores)
    ]
    rows.append(_row("train", "image-missing", method, None, 9.0))
    return rows


class LabelTests(unittest.TestCase):
    def test_threshold_matches_the_linear_quartile_and_ignores_undefined_f1(self) -> None:
        rows = _training_block()
        rows.extend(_row("test", f"test-{index}", "otsu", 0.0, 50.0) for index in range(4))
        threshold, count = failure_threshold(rows, "otsu")
        defined = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.assertEqual(count, 8)
        self.assertEqual(threshold, linear_percentile(defined, 25))
        changed = [dict(row) for row in rows]
        for row in changed:
            if row["split"] == "test":
                row["f1"] = 1.0
        self.assertEqual(failure_threshold(changed, "otsu")[0], threshold)
        with_extra_blank = rows + [_row("train", "another-blank", "otsu", None, 3.0)]
        self.assertEqual(failure_threshold(with_extra_blank, "otsu")[0], threshold)

    def test_undefined_f1_is_a_failure_and_low_scores_use_the_threshold(self) -> None:
        threshold = linear_percentile([0.1, 0.2, 0.3, 0.4], 25)
        self.assertEqual(label_failure(None, threshold), (1, "undefined_f1"))
        self.assertEqual(label_failure(threshold, threshold), (1, "low_f1"))
        self.assertEqual(label_failure(threshold + 0.01, threshold), (0, "above_threshold"))


class LeakageAndGroupingTests(unittest.TestCase):
    def test_fitting_rejects_test_rows_and_test_features_do_not_change_coefficients(self) -> None:
        train = _training_block()
        labeled_train = []
        threshold, _count = failure_threshold(train, "otsu")
        for row in train:
            failure, reason = label_failure(None if row["f1"] is None else float(row["f1"]), threshold)
            labeled_train.append({**row, "failure": failure, "failure_reason": reason})
        with self.assertRaises(ValueError):
            fit_method_models(labeled_train + [_row("test", "held", "otsu", 0.0, 99.0)])
        first = fit_method_models(labeled_train)
        second_rows = [dict(row) for row in labeled_train]
        for row in second_rows:
            row["grayscale_mean"] = float(row["grayscale_mean"])
        # The fitting sample is unchanged; a test-only edit is a separate experiment below.
        second = fit_method_models(second_rows)
        self.assertTrue(
            np.allclose(first["logistic"]["coefficients"], second["logistic"]["coefficients"])
        )
        low_test = [_row("test", f"test-{index}", "otsu", 0.0, 100.0 + index) for index in range(4)]
        high_test = [_row("test", f"test-{index}", "otsu", 1.0, -100.0 - index) for index in range(4)]
        for extra in (low_test, high_test):
            extra[0]["stem"] = "image-0"
        one = run_experiment(train + low_test, methods=("otsu",), n_folds=2, seed=0)
        other = run_experiment(train + high_test, methods=("otsu",), n_folds=2, seed=0)
        self.assertEqual(
            one["config"]["method_fits"]["otsu"]["logistic_coefficients_intercept_then_features"],
            other["config"]["method_fits"]["otsu"]["logistic_coefficients_intercept_then_features"],
        )
        test_prediction = next(
            row for row in one["predictions"] if row["split"] == "test" and row["stem"] == "image-0"
        )
        self.assertEqual(test_prediction["internal_logistic_predicted_failure"], None)
        self.assertNotIn("image-missing-test", one["config"]["fold_by_training_stem"])

    def test_folds_group_duplicate_stems_and_repeat_with_the_same_seed(self) -> None:
        stems = ["b", "a", "b", "c", "d"]
        first = assign_folds(stems, 2, 0)
        second = assign_folds(stems, 2, 0)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotEqual(first, assign_folds(stems, 2, 1))
        with self.assertRaises(ValueError):
            assign_folds(["only"], 2, 0)

    def test_loader_drops_score_columns_other_than_f1(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "joined.csv"
            header = ["split", "stem", "method", "f1", "precision", "tp", *FEATURE_NAMES]
            path.write_text(
                ",".join(header) + "\ntrain,a,otsu,,0.2,5," + ",".join(["1"] * 6) + "\n",
                encoding="utf-8",
            )
            rows = load_joined_features(path)
        self.assertIsNone(rows[0]["f1"])
        self.assertNotIn("precision", rows[0])
        self.assertNotIn("tp", rows[0])
        matrix = predictor_matrix(rows)
        self.assertEqual(matrix.shape, (1, 6))
        self.assertTrue(np.allclose(matrix, np.ones((1, 6))))


class ModelAndMetricTests(unittest.TestCase):
    def test_baselines_and_single_feature_rule(self) -> None:
        self.assertEqual(majority_class(np.array([0, 0, 0, 1])), 0)
        self.assertEqual(majority_class(np.array([1, 1, 0, 0])), 0)
        self.assertEqual(majority_class(np.array([1, 1, 0])), 1)
        labels = np.array([0, 0, 1, 1], dtype=np.float64)
        imputed = np.column_stack(
            [
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([4.0, 3.0, 2.0, 1.0]),
                np.zeros(4),
                np.zeros(4),
                np.zeros(4),
                np.zeros(4),
            ]
        )
        rule = select_single_feature(imputed, labels)
        self.assertIsNotNone(rule)
        self.assertEqual(rule["feature"], "grayscale_mean")
        self.assertEqual(rule["direction"], "high_predicts_failure")
        self.assertAlmostEqual(float(rule["threshold"]), 2.5)

    def test_single_class_and_constant_scores_stay_undefined(self) -> None:
        empty = classification_metrics([0, 0, 0], [0, 0, 0], [0.2, 0.4, 0.6])
        self.assertIsNone(empty["precision"])
        self.assertIsNone(empty["recall"])
        self.assertIsNone(empty["f1"])
        self.assertIsNone(empty["pr_auc"])
        self.assertNotEqual(empty["precision"], 0)
        self.assertNotEqual(empty["pr_auc"], 0)
        constant = classification_metrics([1, 0, 1, 0], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertIsNone(constant["pr_auc"])
        self.assertIn("constant scores", constant["notes"])
        both_zero = classification_metrics([1, 0], [0, 1], [0.2, 0.8])
        self.assertEqual(both_zero["precision"], 0.0)
        self.assertEqual(both_zero["recall"], 0.0)
        self.assertEqual(both_zero["f1"], 0.0)

    def test_logistic_separation_is_reproducible(self) -> None:
        matrix = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
        # The public fitter scales inside fit_method_models. This checks the solver.
        padded = np.column_stack([matrix[:, 0], np.zeros((8, 5))])
        first = fit_logistic(padded, labels)
        second = fit_logistic(padded, labels)
        self.assertTrue(np.allclose(first["coefficients"], second["coefficients"]))
        from crackscout.reliability_prediction import predict_logistic

        probability = predict_logistic(first, padded)
        self.assertLess(float(probability[:4].mean()), 0.5)
        self.assertGreater(float(probability[4:].mean()), 0.5)

    def test_training_median_imputation_ignores_the_test_value(self) -> None:
        medians = fit_medians(np.array([[1.0, np.nan], [np.nan, 5.0], [3.0, 7.0]]))
        self.assertAlmostEqual(float(medians[0]), 2.0)
        self.assertAlmostEqual(float(medians[1]), 6.0)

    def test_repeated_experiment_matches_and_preserves_existing_files(self) -> None:
        rows = []
        for method in ("otsu", "adaptive"):
            rows.extend(_training_block(method))
            rows.extend(_row("test", f"test-{index}", method, 0.9, 20.0 + index) for index in range(4))
            rows.append(_row("test", "image-0", method, 0.01, 3.0))
        first = run_experiment(rows, methods=("otsu", "adaptive"), n_folds=2, seed=0)
        second = run_experiment(rows, methods=("otsu", "adaptive"), n_folds=2, seed=0)
        self.assertEqual(
            first["config"]["method_fits"]["otsu"]["logistic_coefficients_intercept_then_features"],
            second["config"]["method_fits"]["otsu"]["logistic_coefficients_intercept_then_features"],
        )
        self.assertEqual(
            first["config"]["fold_by_training_stem"],
            second["config"]["fold_by_training_stem"],
        )
        otsu_fold = first["config"]["fold_by_training_stem"]["image-0"]
        adaptive_fold = first["config"]["fold_by_training_stem"]["image-0"]
        self.assertEqual(otsu_fold, adaptive_fold)
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "out"
            destination.mkdir()
            sentinel = destination / "notes.txt"
            sentinel.write_text("keep", encoding="utf-8")
            blocked = destination / "predictions.csv"
            blocked.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_outputs(destination, first)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(blocked.read_text(encoding="utf-8"), "original")
            self.assertFalse((destination / "report.txt").exists())
            blocked.unlink()
            write_outputs(destination, first)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            stored = json.loads((destination / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["random_seed"], 0)
            self.assertEqual(stored["predictors"], list(FEATURE_NAMES))
            with self.assertRaises(FileExistsError):
                write_outputs(destination, first)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
