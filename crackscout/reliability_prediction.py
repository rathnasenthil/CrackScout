"""Training-only prediction of poor classical segmentation on DeepCrack.

Each method has its own failure label. The label threshold is the linear first
quartile of that method's defined training F1 scores. Undefined F1 is a failure
for the operational label and is counted separately. Test scores are not used
to choose features, fit models, or set thresholds.

The predictors are the six existing image-quality features. A result here is
evidence about this prediction task. It is not evidence of causation, deployment
reliability, or a reason to abstain.
"""

import csv
import json
from pathlib import Path

import numpy as np

from crackscout.image_quality import FEATURE_NAMES, pearson_correlation, training_f1_cutoffs

RANDOM_SEED = 0
N_FOLDS = 5
LOGISTIC_L2 = 1.0
LOGISTIC_DECISION_THRESHOLD = 0.5
LOGISTIC_MAX_ITERATIONS = 25
METHODS = ("otsu", "adaptive", "blackhat")
MODEL_NAMES = ("majority", "always_failure", "single_feature", "logistic_regression")
OUTPUT_NAMES = ("report.txt", "predictions.csv", "metrics.csv", "config.json")

QUARTILE_CONVENTION = (
    "The failure threshold is the linear 25th percentile of defined training F1 "
    "scores for that method. The position is (n - 1) * 0.25, and the threshold "
    "blends the two nearest sorted scores. Undefined F1 scores are excluded from "
    "n and from the percentile. A defined score at or below the threshold is a "
    "low-F1 failure. An undefined F1 is an operational failure as well and is "
    "reported separately. The threshold is applied unchanged to the test split."
)

DECISIONS_FIXED_BEFORE_TEST = (
    "Failure thresholds use defined training F1 only.",
    "Undefined F1 is an operational failure.",
    "Predictors are the six image-quality features.",
    "F1, precision, recall, IoU, confusion counts, masks, method, stem, and split are not predictors.",
    "The official training split fits imputation, scaling, the single-feature rule, and logistic regression.",
    "Internal validation is five folds. Unique training stems are sorted, shuffled with seed 0, and assigned by index modulo 5. Rows follow their stem. The same assignment is used for every method.",
    "The official test split is scored once after fitting and is not used to choose a feature, a threshold, or a model.",
    "Majority ties predict non-failure.",
    "The single-feature rule uses the largest absolute training Pearson correlation, with an earlier feature name winning a tie, and a training-median cutoff.",
    "Logistic regression uses median imputation, training-mean and population-standard-deviation scaling, balanced class weights from the fitting sample, L2 strength 1 on the mean loss, an unpenalized intercept, and probability cutoff 0.5.",
    "No abstention rule is applied.",
)


def load_joined_features(path: str | Path) -> list[dict[str, object]]:
    """Read split, stem, method, F1, and the six quality features.

    Score columns other than F1 are not loaded, so they cannot enter the model.
    A blank F1 stays missing. It is not replaced with 0 or 1.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Joined feature CSV not found: {csv_path}")
    required = {"split", "stem", "method", "f1", *FEATURE_NAMES}
    rows: list[dict[str, object]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Joined feature CSV has no header: {csv_path}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Joined feature CSV is missing columns: {', '.join(missing)}")
        for raw in reader:
            f1_text = raw["f1"].strip()
            row: dict[str, object] = {
                "split": raw["split"].strip(),
                "stem": raw["stem"].strip(),
                "method": raw["method"].strip(),
                "f1": None if f1_text == "" else float(f1_text),
            }
            for name in FEATURE_NAMES:
                text = raw[name].strip()
                row[name] = None if text == "" else float(text)
            rows.append(row)
    return rows


def failure_threshold(rows: list[dict[str, object]], method: str) -> tuple[float, int]:
    """Return the training-only linear first-quartile F1 and its sample size."""
    cutoffs = training_f1_cutoffs(rows, method)
    if cutoffs is None:
        raise ValueError(
            f"Method {method} needs at least four defined training F1 scores "
            "to set a failure threshold."
        )
    threshold, _third_quartile, count = cutoffs
    return float(threshold), int(count)


def label_failure(f1: float | None, threshold: float) -> tuple[int, str]:
    """Return the operational failure label and the reason for that label."""
    if f1 is None:
        return 1, "undefined_f1"
    if float(f1) <= threshold:
        return 1, "low_f1"
    return 0, "above_threshold"


def predictor_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    """Build a design matrix from the six quality features only."""
    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column, name in enumerate(FEATURE_NAMES):
            value = row[name]
            matrix[row_index, column] = np.nan if value is None else float(value)
    return matrix


def assign_folds(stems: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Assign each stem to one fold.

    Stems are sorted, shuffled with the seed, and dealt by index modulo the
    number of folds. Duplicate stems share a fold. The official test split is
    not part of this assignment.
    """
    if n_folds < 2:
        raise ValueError("Internal validation needs at least two folds.")
    unique = sorted(set(stems))
    if len(unique) < n_folds:
        raise ValueError(
            f"Cannot make {n_folds} image-level folds from {len(unique)} training stems."
        )
    order = np.array(unique)
    np.random.default_rng(seed).shuffle(order)
    return {str(stem): int(index % n_folds) for index, stem in enumerate(order)}


def majority_class(labels: np.ndarray) -> int:
    """Return the more common label. An equal count predicts non-failure."""
    failures = int(np.sum(labels == 1))
    non_failures = int(np.sum(labels == 0))
    if failures > non_failures:
        return 1
    return 0


def _require_training_rows(rows: list[dict[str, object]]) -> None:
    splits = {str(row["split"]) for row in rows}
    if splits != {"train"}:
        raise ValueError(
            "Model fitting received a row outside the training split: "
            + ", ".join(sorted(splits))
        )


def _sorted_training_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    _require_training_rows(rows)
    return sorted(rows, key=lambda row: str(row["stem"]))


def fit_medians(matrix: np.ndarray) -> np.ndarray:
    medians = np.zeros(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        finite = matrix[np.isfinite(matrix[:, column]), column]
        if len(finite):
            medians[column] = float(np.median(finite))
    return medians


def apply_medians(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    imputed = matrix.copy()
    missing = ~np.isfinite(imputed)
    imputed[missing] = np.take(medians, np.where(missing)[1])
    return imputed


def fit_scaler(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return means, scale


def apply_scaler(matrix: np.ndarray, means: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (matrix - means) / scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _mean_objective(
    design: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    coefficients: np.ndarray,
    l2: float,
) -> float:
    probability = np.clip(_sigmoid(design @ coefficients), 1e-12, 1.0 - 1e-12)
    loss = -(labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability))
    penalty = 0.5 * l2 * float(np.sum(coefficients[1:] ** 2))
    return float(np.sum(sample_weights * loss) / len(labels) + penalty)


def fit_logistic(matrix: np.ndarray, labels: np.ndarray, l2: float = LOGISTIC_L2) -> dict[str, object]:
    """Fit L2-regularized logistic regression to an already scaled matrix.

    The objective is the weighted mean negative log-likelihood plus
    (l2 / 2) times the sum of squared non-intercept coefficients. Class weights
    are n / (2 * class count) on the fitting sample. The intercept is not
    penalized. A one-class sample returns a constant probability instead of
    a coefficient vector.
    """
    labels = np.asarray(labels, dtype=np.float64)
    if len(labels) != len(matrix):
        raise ValueError("Logistic regression received a different number of rows and labels.")
    classes = set(int(value) for value in labels.tolist())
    if classes <= {0} or classes <= {1} or not classes:
        probability = 1.0 if classes == {1} else 0.0
        return {
            "kind": "constant",
            "probability": probability,
            "coefficients": None,
            "iterations": 0,
            "converged": True,
        }
    count = len(labels)
    class_counts = {label: max(int(np.sum(labels == label)), 1) for label in (0, 1)}
    sample_weights = np.where(
        labels == 1.0,
        count / (2.0 * class_counts[1]),
        count / (2.0 * class_counts[0]),
    )
    design = np.column_stack([np.ones(count), matrix])
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    converged = False
    iteration = 0
    for iteration in range(1, LOGISTIC_MAX_ITERATIONS + 1):
        probability = _sigmoid(design @ coefficients)
        residual = sample_weights * (probability - labels)
        gradient = design.T @ residual / count
        gradient[1:] += l2 * coefficients[1:]
        curvature = sample_weights * probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / count
        hessian[1:, 1:] += l2 * np.eye(matrix.shape[1])
        hessian = hessian + 1e-8 * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        current = _mean_objective(design, labels, sample_weights, coefficients, l2)
        accepted = coefficients
        for attempt in range(12):
            candidate = coefficients - step / (2.0 ** attempt)
            candidate_value = _mean_objective(design, labels, sample_weights, candidate, l2)
            if candidate_value <= current + 1e-12:
                accepted = candidate
                break
        coefficients = accepted
        if float(np.linalg.norm(step)) < 1e-8:
            converged = True
            break
    return {
        "kind": "logistic",
        "probability": None,
        "coefficients": coefficients,
        "iterations": iteration,
        "converged": converged,
    }


def predict_logistic(model: dict[str, object], matrix: np.ndarray) -> np.ndarray:
    if model["kind"] == "constant":
        return np.full(len(matrix), float(model["probability"]), dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    return _sigmoid(design @ coefficients)


def select_single_feature(
    imputed: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object] | None:
    """Choose one feature with training labels only.

    The selected feature has the largest absolute Pearson correlation with the
    failure label. An earlier feature in FEATURE_NAMES wins an equal absolute
    correlation. The cutoff is the training median. High values predict failure
    when the correlation is positive or zero, and low values predict failure
    when it is negative.
    """
    best: dict[str, object] | None = None
    best_key: tuple[float, int] | None = None
    for index, name in enumerate(FEATURE_NAMES):
        correlation, _count = pearson_correlation(
            imputed[:, index].tolist(),
            [float(label) for label in labels.tolist()],
        )
        if correlation is None:
            continue
        key = (abs(float(correlation)), -index)
        if best_key is None or key > best_key:
            best_key = key
            best = {"feature": name, "index": index, "correlation": float(correlation)}
    if best is None:
        return None
    column = imputed[:, int(best["index"])]
    correlation = float(best["correlation"])
    best["threshold"] = float(np.median(column))
    best["direction"] = "high_predicts_failure" if correlation >= 0.0 else "low_predicts_failure"
    return best


def single_feature_scores(
    imputed: np.ndarray,
    rule: dict[str, object] | None,
) -> np.ndarray | None:
    """Return a score that increases with predicted failure, or None."""
    if rule is None:
        return None
    column = imputed[:, int(rule["index"])]
    if rule["direction"] == "high_predicts_failure":
        return column.copy()
    return -column


def single_feature_predictions(
    imputed: np.ndarray,
    rule: dict[str, object] | None,
    fallback: int,
) -> np.ndarray:
    if rule is None:
        return np.full(len(imputed), fallback, dtype=np.int64)
    scores = single_feature_scores(imputed, rule)
    cutoff = float(rule["threshold"])
    if rule["direction"] == "low_predicts_failure":
        cutoff = -cutoff
    return (scores >= cutoff).astype(np.int64)


def classification_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    scores: list[float | None] | np.ndarray | None,
) -> dict[str, object]:
    """Score the failure class. Undefined rates stay missing.

    Precision is missing when no row is predicted to fail. Recall is missing
    when the evaluation rows contain no failures. F1 is missing when either
    of those rates is missing. When both rates are defined and both are zero,
    F1 is zero. PR-AUC is average precision. It is missing when the labels
    contain one class, any score is missing, or every score is equal.
    """
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if len(y_true) != len(y_pred):
        raise ValueError("Metrics received different numbers of labels and predictions.")
    count = len(y_true)
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else None
    )
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)
    notes: list[str] = []
    if true_positive + false_positive == 0:
        notes.append("no predicted failures; precision undefined")
    if true_positive + false_negative == 0:
        notes.append("no actual failures; recall undefined")
    if precision == 0.0 and recall == 0.0:
        notes.append("precision and recall are both zero, so failure F1 is zero")
    pr_auc = _average_precision(y_true, scores, notes)
    prevalence = (true_positive + false_negative) / count if count else None
    positive_rate = (true_positive + false_positive) / count if count else None
    return {
        "n": count,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "positive_prediction_rate": positive_rate,
        "notes": "; ".join(notes),
    }


def _average_precision(
    labels: np.ndarray,
    scores: list[float | None] | np.ndarray | None,
    notes: list[str],
) -> float | None:
    if len(set(int(label) for label in labels.tolist())) < 2:
        notes.append("single-class labels; PR-AUC undefined")
        return None
    if scores is None:
        notes.append("scores unavailable; PR-AUC undefined")
        return None
    values = [None if score is None else float(score) for score in list(scores)]
    if any(score is None or not np.isfinite(score) for score in values):
        notes.append("incomplete scores; PR-AUC undefined")
        return None
    numeric = [float(score) for score in values if score is not None]
    if len(set(numeric)) < 2:
        notes.append("constant scores; PR-AUC undefined")
        return None
    order = np.argsort(-np.asarray(numeric), kind="mergesort")
    ordered = labels[order]
    hits = np.cumsum(ordered == 1)
    examined = np.arange(1, len(ordered) + 1)
    precision = hits / examined
    total_hits = int(hits[-1])
    previous_recall = 0.0
    average = 0.0
    for index, label in enumerate(ordered.tolist()):
        if label != 1:
            continue
        recall = float(hits[index]) / total_hits
        average += (recall - previous_recall) * float(precision[index])
        previous_recall = recall
    return average


def _preprocess_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    medians = fit_medians(matrix)
    imputed = apply_medians(matrix, medians)
    means, scale = fit_scaler(imputed)
    return medians, means, scale, imputed


def fit_method_models(rows: list[dict[str, object]]) -> dict[str, object]:
    """Fit baselines and logistic regression on training rows for one method."""
    ordered = _sorted_training_rows(rows)
    labels = np.asarray([int(row["failure"]) for row in ordered], dtype=np.int64)
    matrix = predictor_matrix(ordered)
    medians, means, scale, imputed = _preprocess_fit(matrix)
    scaled = apply_scaler(imputed, means, scale)
    rule = select_single_feature(imputed, labels)
    logistic = fit_logistic(scaled, labels, LOGISTIC_L2)
    if logistic["kind"] == "logistic" and not np.all(np.isfinite(logistic["coefficients"])):
        raise ValueError("Logistic regression produced non-finite coefficients.")
    return {
        "majority": majority_class(labels),
        "failure_count": int(np.sum(labels == 1)),
        "nonfailure_count": int(np.sum(labels == 0)),
        "medians": medians,
        "means": means,
        "scale": scale,
        "single_feature": rule,
        "logistic": logistic,
    }


def _predict_frame(
    rows: list[dict[str, object]],
    models: dict[str, object],
) -> dict[str, np.ndarray | None]:
    matrix = predictor_matrix(rows)
    imputed = apply_medians(matrix, np.asarray(models["medians"]))
    scaled = apply_scaler(imputed, np.asarray(models["means"]), np.asarray(models["scale"]))
    count = len(rows)
    majority = np.full(count, int(models["majority"]), dtype=np.int64)
    always = np.ones(count, dtype=np.int64)
    rule = models["single_feature"]
    single_pred = single_feature_predictions(imputed, rule, int(models["majority"]))
    single_score = single_feature_scores(imputed, rule)
    probability = predict_logistic(models["logistic"], scaled)
    logistic_pred = (probability >= LOGISTIC_DECISION_THRESHOLD).astype(np.int64)
    return {
        "majority": majority,
        "always_failure": always,
        "single_feature": single_pred,
        "single_feature_score": single_score,
        "logistic_probability": probability,
        "logistic_regression": logistic_pred,
    }


def _metric_row(
    method: str,
    evaluation: str,
    model: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    decision_rule: str,
    decision_threshold: float | None,
    undefined_count: int,
    low_f1_count: int,
) -> dict[str, object]:
    metrics = classification_metrics(labels, predictions, scores)
    return {
        "method": method,
        "evaluation": evaluation,
        "model": model,
        "positive_class": "failure",
        "decision_rule": decision_rule,
        "decision_threshold": decision_threshold,
        "undefined_f1_count": undefined_count,
        "low_f1_failure_count": low_f1_count,
        **metrics,
    }


def _label_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        "n": len(rows),
        "failures": sum(int(row["failure"]) for row in rows),
        "nonfailures": sum(1 for row in rows if int(row["failure"]) == 0),
        "undefined_f1": sum(1 for row in rows if row["failure_reason"] == "undefined_f1"),
        "low_f1": sum(1 for row in rows if row["failure_reason"] == "low_f1"),
    }


def _decision_fields(
    models: dict[str, object],
    model: str,
    evaluation: str,
) -> tuple[str, float | None]:
    if model == "majority" and evaluation == "internal_validation":
        return "each training fold predicts its own majority class; ties predict non-failure", None
    if model == "majority":
        chosen = "failure" if int(models["majority"]) == 1 else "non-failure"
        return f"predict {chosen} for every row; ties predict non-failure", None
    if model == "always_failure":
        return "predict failure for every row", None
    if model == "single_feature" and evaluation == "internal_validation":
        return (
            "each training fold reselects its feature, direction, and median; "
            "the per-fold values are stored in config.json"
        ), None
    if model == "single_feature":
        rule = models["single_feature"]
        if rule is None:
            return "no training feature had a defined correlation; use the majority prediction", None
        return (
            f"{rule['feature']}; {rule['direction']}; "
            f"training median {float(rule['threshold']):.6f}; "
            "predict failure when the directed score meets or exceeds the directed median",
            float(rule["threshold"]),
        )
    if evaluation == "internal_validation":
        return (
            "predict failure when logistic probability >= 0.5; "
            "coefficients are re-fit inside each training fold"
        ), LOGISTIC_DECISION_THRESHOLD
    return "predict failure when logistic probability >= 0.5", LOGISTIC_DECISION_THRESHOLD


def _attach_labels(
    rows: list[dict[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    labeled = []
    for row in rows:
        failure, reason = label_failure(
            None if row["f1"] is None else float(row["f1"]),
            threshold,
        )
        labeled.append({**row, "failure": failure, "failure_reason": reason, "threshold": threshold})
    return labeled


def _internal_predictions(
    train_rows: list[dict[str, object]],
    folds: dict[str, int],
    n_folds: int,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    by_stem: dict[str, dict[str, object]] = {}
    fold_records: list[dict[str, object]] = []
    for fold in range(n_folds):
        fitting = [row for row in train_rows if folds[str(row["stem"])] != fold]
        held_out = [row for row in train_rows if folds[str(row["stem"])] == fold]
        models = fit_method_models(fitting)
        predicted = _predict_frame(held_out, models)
        rule = models["single_feature"]
        fold_records.append(
            {
                "fold": fold,
                "fit_rows": len(fitting),
                "held_out_rows": len(held_out),
                "majority": int(models["majority"]),
                "single_feature": None if rule is None else str(rule["feature"]),
                "single_feature_direction": None if rule is None else str(rule["direction"]),
                "single_feature_threshold": None if rule is None else float(rule["threshold"]),
                "single_feature_correlation": None if rule is None else float(rule["correlation"]),
                "logistic_iterations": int(models["logistic"]["iterations"]),
                "logistic_converged": bool(models["logistic"]["converged"]),
            }
        )
        for index, row in enumerate(held_out):
            score = predicted["single_feature_score"]
            by_stem[str(row["stem"])] = {
                "majority": int(predicted["majority"][index]),
                "single_feature": int(predicted["single_feature"][index]),
                "single_feature_name": "" if rule is None else str(rule["feature"]),
                "single_feature_score": None if score is None else float(score[index]),
                "logistic_probability": float(predicted["logistic_probability"][index]),
                "logistic_regression": int(predicted["logistic_regression"][index]),
            }
    return by_stem, fold_records


def run_experiment(
    rows: list[dict[str, object]],
    *,
    methods: tuple[str, ...] = METHODS,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Fit on the training split and score the test split once."""
    training_stems = [str(row["stem"]) for row in rows if row["split"] == "train"]
    folds = assign_folds(training_stems, n_folds, seed)
    predictions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    method_config: dict[str, object] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        train_rows = [row for row in method_rows if row["split"] == "train"]
        test_rows = [row for row in method_rows if row["split"] == "test"]
        if not train_rows or not test_rows:
            raise ValueError(f"Method {method} is missing training or test rows.")
        threshold, defined_count = failure_threshold(method_rows, method)
        train_rows = _attach_labels(train_rows, threshold)
        test_rows = _attach_labels(test_rows, threshold)
        models = fit_method_models(train_rows)
        internal, fold_records = _internal_predictions(train_rows, folds, n_folds)
        final_by_split = {
            "train": _predict_frame(train_rows, models),
            "test": _predict_frame(test_rows, models),
        }
        rule = models["single_feature"]
        for split_rows in (train_rows, test_rows):
            frame = final_by_split[str(split_rows[0]["split"])]
            score_values = frame["single_feature_score"]
            for index, row in enumerate(split_rows):
                internal_row = internal.get(str(row["stem"])) if row["split"] == "train" else None
                predictions.append(
                    {
                        "split": row["split"],
                        "stem": row["stem"],
                        "method": method,
                        "f1": row["f1"],
                        "f1_undefined": 1 if row["f1"] is None else 0,
                        "failure_threshold": threshold,
                        "failure_label": int(row["failure"]),
                        "failure_reason": row["failure_reason"],
                        "majority_predicted_failure": int(frame["majority"][index]),
                        "always_failure_predicted_failure": int(frame["always_failure"][index]),
                        "single_feature_name": "" if rule is None else str(rule["feature"]),
                        "single_feature_direction": "" if rule is None else str(rule["direction"]),
                        "single_feature_threshold": None if rule is None else float(rule["threshold"]),
                        "single_feature_score": None
                        if score_values is None
                        else float(score_values[index]),
                        "single_feature_predicted_failure": int(frame["single_feature"][index]),
                        "logistic_probability": float(frame["logistic_probability"][index]),
                        "logistic_decision_threshold": LOGISTIC_DECISION_THRESHOLD,
                        "logistic_predicted_failure": int(frame["logistic_regression"][index]),
                        "internal_majority_predicted_failure": None
                        if internal_row is None
                        else internal_row["majority"],
                        "internal_single_feature_name": None
                        if internal_row is None
                        else internal_row["single_feature_name"],
                        "internal_single_feature_score": None
                        if internal_row is None
                        else internal_row["single_feature_score"],
                        "internal_single_feature_predicted_failure": None
                        if internal_row is None
                        else internal_row["single_feature"],
                        "internal_logistic_probability": None
                        if internal_row is None
                        else internal_row["logistic_probability"],
                        "internal_logistic_predicted_failure": None
                        if internal_row is None
                        else internal_row["logistic_regression"],
                    }
                )
        method_predictions = [row for row in predictions if row["method"] == method]
        for evaluation, split_name, use_internal in (
            ("insample_training", "train", False),
            ("internal_validation", "train", True),
            ("test", "test", False),
        ):
            selected = [row for row in method_predictions if row["split"] == split_name]
            labels = np.asarray([int(row["failure_label"]) for row in selected], dtype=np.int64)
            undefined_count = sum(int(row["f1_undefined"]) for row in selected)
            low_f1_count = sum(1 for row in selected if row["failure_reason"] == "low_f1")
            for model in MODEL_NAMES:
                rule_text, rule_threshold = _decision_fields(models, model, evaluation)
                if use_internal and model == "always_failure":
                    predicted = np.ones(len(selected), dtype=np.int64)
                    scores = predicted.astype(np.float64)
                elif use_internal and model == "majority":
                    predicted = np.asarray(
                        [int(row["internal_majority_predicted_failure"]) for row in selected],
                        dtype=np.int64,
                    )
                    scores = predicted.astype(np.float64)
                elif use_internal and model == "single_feature":
                    predicted = np.asarray(
                        [int(row["internal_single_feature_predicted_failure"]) for row in selected],
                        dtype=np.int64,
                    )
                    raw_scores = [row["internal_single_feature_score"] for row in selected]
                    scores = None if any(score is None for score in raw_scores) else np.asarray(raw_scores)
                elif use_internal:
                    predicted = np.asarray(
                        [int(row["internal_logistic_predicted_failure"]) for row in selected],
                        dtype=np.int64,
                    )
                    scores = np.asarray(
                        [float(row["internal_logistic_probability"]) for row in selected]
                    )
                elif model == "majority":
                    predicted = np.asarray(
                        [int(row["majority_predicted_failure"]) for row in selected]
                    )
                    scores = predicted.astype(np.float64)
                elif model == "always_failure":
                    predicted = np.ones(len(selected), dtype=np.int64)
                    scores = predicted.astype(np.float64)
                elif model == "single_feature":
                    predicted = np.asarray(
                        [int(row["single_feature_predicted_failure"]) for row in selected]
                    )
                    raw_scores = [row["single_feature_score"] for row in selected]
                    scores = None if any(score is None for score in raw_scores) else np.asarray(raw_scores)
                else:
                    predicted = np.asarray(
                        [int(row["logistic_predicted_failure"]) for row in selected]
                    )
                    scores = np.asarray([float(row["logistic_probability"]) for row in selected])
                metrics.append(
                    _metric_row(
                        method,
                        evaluation,
                        model,
                        labels,
                        predicted,
                        scores,
                        rule_text,
                        rule_threshold,
                        undefined_count,
                        low_f1_count,
                    )
                )
        coefficients = models["logistic"]["coefficients"]
        method_config[method] = {
            "failure_threshold": threshold,
            "failure_threshold_rounded_4dp": round(threshold, 4),
            "defined_training_f1_n": defined_count,
            "quartile_percent": 25,
            "train_counts": _label_counts(train_rows),
            "test_counts": _label_counts(test_rows),
            "majority_class": int(models["majority"]),
            "single_feature": None
            if rule is None
            else {
                "feature": rule["feature"],
                "direction": rule["direction"],
                "threshold": float(rule["threshold"]),
                "correlation": float(rule["correlation"]),
            },
            "imputation_medians": [float(value) for value in models["medians"]],
            "scaling_means": [float(value) for value in models["means"]],
            "scaling_divisors": [float(value) for value in models["scale"]],
            "logistic_coefficients_intercept_then_features": None
            if coefficients is None
            else [float(value) for value in coefficients],
            "logistic_iterations": int(models["logistic"]["iterations"]),
            "logistic_converged": bool(models["logistic"]["converged"]),
            "internal_folds": fold_records,
        }
    config = {
        "random_seed": seed,
        "n_folds": n_folds,
        "predictors": list(FEATURE_NAMES),
        "methods": list(methods),
        "quartile_convention": QUARTILE_CONVENTION,
        "split_policy": (
            "Official training rows are the only rows used to calculate the F1 "
            "threshold, imputation medians, scaling, single-feature rule, class "
            "weights, and logistic coefficients. Fold ids are assigned from unique "
            "training stems. A stem that also occurs in the test split remains a "
            "different image and its test row is not placed in a training fold. "
            "Test labels and test features are read only to score the fitted models."
        ),
        "decisions_fixed_before_test": list(DECISIONS_FIXED_BEFORE_TEST),
        "logistic": {
            "l2": LOGISTIC_L2,
            "decision_threshold": LOGISTIC_DECISION_THRESHOLD,
            "max_iterations": LOGISTIC_MAX_ITERATIONS,
            "class_weight": "n / (2 * fitting-sample class count)",
            "hessian_stabilizer": 1e-8,
        },
        "fold_by_training_stem": folds,
        "method_fits": method_config,
    }
    report = build_report(metrics, config)
    return {"report": report, "predictions": predictions, "metrics": metrics, "config": config}


def _format_metric(value: object) -> str:
    if value is None or value == "":
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report(metrics: list[dict[str, object]], config: dict[str, object]) -> str:
    """Describe labels, fixed decisions, and separate training and test scores."""
    lines = [
        "Exploratory reliability-prediction experiment",
        "",
        "This experiment asks whether the six image-quality features can identify",
        "images on which an existing classical segmenter has poor F1. It does not",
        "establish that image quality causes segmentation error, that a predictor",
        "is reliable enough to deploy, or that abstaining on a predicted failure",
        "would help. No abstention policy is applied.",
        "",
        "Target",
        QUARTILE_CONVENTION,
        "The numeric threshold is the same linear training first quartile used for",
        "the higher-error group in the image-quality report. That earlier group",
        "excluded undefined F1. The operational label in this experiment counts",
        "undefined F1 as a failure and reports those rows separately.",
        "",
        "Decisions fixed before test evaluation",
    ]
    for decision in config["decisions_fixed_before_test"]:
        lines.append(f"- {decision}")
    lines.extend(
        [
            "",
            "Predictors: " + ", ".join(config["predictors"]),
            f"Random seed: {config['random_seed']}",
            f"Internal folds: {config['n_folds']}",
            config["split_policy"],
            "",
            "In-sample training scores reuse the rows that fitted the model.",
            "Internal-validation scores pool the out-of-fold predictions.",
            "Test scores use one model fitted on all training rows for that method.",
            "Precision, recall, and F1 below are for the failure class.",
            "A blank metric is undefined. It is not zero or one.",
            "",
        ]
    )
    lookup = {
        (row["method"], row["evaluation"], row["model"]): row for row in metrics
    }
    for method, fit in config["method_fits"].items():
        train_counts = fit["train_counts"]
        test_counts = fit["test_counts"]
        lines.append(method)
        lines.append(
            f"  Defined training F1 n {fit['defined_training_f1_n']}. "
            f"Threshold {float(fit['failure_threshold']):.10f} "
            f"(4 d.p. {float(fit['failure_threshold']):.4f})."
        )
        for split_name, counts in (("train", train_counts), ("test", test_counts)):
            prevalence = counts["failures"] / counts["n"] if counts["n"] else None
            lines.append(
                f"  {split_name}: n {counts['n']}, failures {counts['failures']} "
                f"(low F1 {counts['low_f1']}, undefined F1 {counts['undefined_f1']}), "
                f"non-failures {counts['nonfailures']}, "
                f"prevalence {_format_metric(prevalence)}"
            )
        rule = fit["single_feature"]
        if rule is None:
            lines.append("  Single-feature rule: undefined on the full training sample.")
        else:
            lines.append(
                f"  Single-feature rule fitted on all training rows: {rule['feature']}, "
                f"{rule['direction']}, median {float(rule['threshold']):.4f}, "
                f"training correlation {float(rule['correlation']):.4f}."
            )
        fold_choices = []
        for fold in fit["internal_folds"]:
            feature = fold["single_feature"] or "undefined"
            direction = fold["single_feature_direction"] or "undefined"
            fold_choices.append(f"fold {fold['fold']} {feature} ({direction})")
        lines.append("  Single-feature choices inside internal folds: " + "; ".join(fold_choices) + ".")
        lines.append(
            "  Logistic regression converged"
            if fit["logistic_converged"]
            else "  Logistic regression stopped at the iteration limit."
        )
        lines.append(f"  Logistic iterations: {fit['logistic_iterations']}.")
        for evaluation, title in (
            ("insample_training", "In-sample training"),
            ("internal_validation", "Internal validation"),
            ("test", "Untouched test"),
        ):
            lines.append(f"  {title}")
            for model in MODEL_NAMES:
                row = lookup[(method, evaluation, model)]
                lines.append(
                    f"    {model}: n {row['n']}, "
                    f"confusion tn {row['tn']} fp {row['fp']} fn {row['fn']} tp {row['tp']}, "
                    f"precision {_format_metric(row['precision'])}, "
                    f"recall {_format_metric(row['recall'])}, "
                    f"F1 {_format_metric(row['f1'])}, "
                    f"PR-AUC {_format_metric(row['pr_auc'])}, "
                    f"prevalence {_format_metric(row['prevalence'])}, "
                    f"positive prediction rate {_format_metric(row['positive_prediction_rate'])}"
                )
                if row["notes"]:
                    lines.append(f"      {row['notes']}")
                lines.append(f"      Decision: {row['decision_rule']}")
        lines.append("  " + _comparison_sentence(lookup, str(method)))
        lines.append("")
    lines.extend(
        [
            "Limitations",
            "The failure label is a relative training quartile, so about one quarter of",
            "defined training scores are failures by construction. Test prevalence can",
            "differ because the threshold is frozen.",
            "Internal validation re-fits imputation, the single-feature choice, and",
            "logistic regression inside each training fold. It does not choose the",
            "logistic penalty or the probability cutoff.",
            "Average precision uses a stable sort, so tied scores keep their input order.",
            "Constant scores and single-class labels leave PR-AUC undefined.",
            "A higher failure-class F1 or PR-AUC than a baseline describes this sample.",
            "It does not identify a cause and it does not justify an abstention policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_sentence(
    lookup: dict[tuple[object, object, object], dict[str, object]],
    method: str,
) -> str:
    logistic = lookup[(method, "test", "logistic_regression")]
    always = lookup[(method, "test", "always_failure")]
    single = lookup[(method, "test", "single_feature")]
    parts = [
        _higher_or_not(
            "logistic regression",
            logistic["f1"],
            "always-failure",
            always["f1"],
            "failure F1",
        ),
        _higher_or_not(
            "single-feature",
            single["f1"],
            "always-failure",
            always["f1"],
            "failure F1",
        ),
    ]
    if logistic["pr_auc"] is None or logistic["prevalence"] is None:
        parts.append("Logistic PR-AUC is undefined, so it is not compared with prevalence.")
    elif float(logistic["pr_auc"]) > float(logistic["prevalence"]):
        parts.append(
            "Logistic PR-AUC is above the test failure prevalence. "
            "That is a ranking comparison on this split, not a deployment result."
        )
    elif float(logistic["pr_auc"]) < float(logistic["prevalence"]):
        parts.append("Logistic PR-AUC is below the test failure prevalence.")
    else:
        parts.append("Logistic PR-AUC equals the test failure prevalence.")
    return " ".join(parts)


def _higher_or_not(
    left_name: str,
    left_value: object,
    right_name: str,
    right_value: object,
    metric_name: str,
) -> str:
    if left_value is None or right_value is None:
        return (
            f"The test {metric_name} comparison of {left_name} with {right_name} "
            "is undefined."
        )
    left = float(left_value)
    right = float(right_value)
    if left > right:
        relation = "higher than"
    elif left < right:
        relation = "lower than"
    else:
        relation = "equal to"
    return f"Test {left_name} {metric_name} is {relation} the {right_name} baseline."


def write_outputs(output_dir: str | Path, result: dict[str, object]) -> None:
    """Write the experiment files. Existing planned files are left unchanged."""
    destination = Path(output_dir)
    planned = [destination / name for name in OUTPUT_NAMES]
    existing = [path for path in planned if path.exists()]
    if existing:
        listed = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite reliability-prediction files:\n"
            f"{listed}\n"
            "Existing evaluation, failure-analysis, visualization, and quality files "
            "were not changed."
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.txt").write_text(str(result["report"]), encoding="utf-8")
    _write_csv(destination / "predictions.csv", list(result["predictions"]))
    _write_csv(destination / "metrics.csv", list(result["metrics"]))
    (destination / "config.json").write_text(
        json.dumps(result["config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stored = {}
            for field in fieldnames:
                value = row.get(field)
                if value is None:
                    stored[field] = ""
                elif isinstance(value, float) and field in {
                    "failure_threshold",
                    "single_feature_threshold",
                    "decision_threshold",
                    "logistic_decision_threshold",
                }:
                    stored[field] = f"{value:.10f}"
                elif isinstance(value, float):
                    stored[field] = f"{value:.6f}"
                else:
                    stored[field] = value
            writer.writerow(stored)
