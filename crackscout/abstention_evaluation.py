"""Evaluate abstention on frozen CrackScout failure probabilities.

The operating threshold is chosen from training out-of-fold probabilities
before any test outcome is consulted. Test rows are scored with that frozen
threshold. This module does not retrain the reliability model and does not
treat a lower accepted failure rate as structural safety.
"""

import csv
import json
from pathlib import Path

import numpy as np

from crackscout.failure_analysis import linear_percentile
from crackscout.reliability_prediction import METHODS

RANDOM_SEED = 0
N_BOOTSTRAP = 1000
N_RANDOM_TRIALS = 200
MINIMUM_COVERAGE = 0.50
MINIMUM_ABSOLUTE_RISK_REDUCTION = 0.05
OUTPUT_NAMES = (
    "report.txt",
    "operating_points.csv",
    "risk_coverage.csv",
    "random_baseline.csv",
    "config.json",
    "risk_coverage_otsu.png",
    "risk_coverage_adaptive.png",
    "risk_coverage_blackhat.png",
)

SELECTION_RULE = (
    "Use only training rows and the frozen out-of-fold logistic failure "
    "probability, internal_logistic_probability. Accept an image when that "
    "probability is strictly below a threshold. Candidate thresholds are the "
    "distinct training probabilities. A candidate is eligible when coverage is "
    "at least 0.50, at least one image is abstained, and the absolute reduction "
    "in operational failure risk versus accepting every training image is at "
    "least 0.05. Among eligible candidates, keep the highest coverage. Ties "
    "keep the lower accepted failure risk, then the lower threshold. If no "
    "candidate is eligible, no operating threshold is justified and none is "
    "taken from the test split. The same numeric threshold is then applied to "
    "the untouched test split using logistic_probability."
)

DECISIONS_FIXED_BEFORE_TEST = (
    SELECTION_RULE,
    "Undefined F1 remains an operational failure and is never counted as a successful segmentation.",
    "Risk-coverage curves on the test split are diagnostic. They do not choose the threshold.",
    "The single-feature comparison uses the frozen CS-005 median rule.",
    "Random abstention matches the realized coverage of a frozen policy and is repeated for 200 trials.",
    "Bootstrap intervals resample images within a split and reuse the same draws across methods.",
    "No abstention threshold is revised after the test outcomes are calculated.",
)


def _optional_float(text: str) -> float | None:
    stripped = text.strip()
    if stripped == "":
        return None
    return float(stripped)


def load_predictions(path: str | Path) -> list[dict[str, object]]:
    """Read frozen CS-005 prediction rows. Blank scores stay missing."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
    required = {
        "split",
        "stem",
        "method",
        "f1",
        "f1_undefined",
        "failure_threshold",
        "failure_label",
        "failure_reason",
        "logistic_probability",
        "internal_logistic_probability",
        "single_feature_score",
        "single_feature_direction",
        "single_feature_predicted_failure",
        "single_feature_threshold",
    }
    rows: list[dict[str, object]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Predictions CSV is missing columns: {', '.join(missing)}")
        for raw in reader:
            rows.append(
                {
                    "split": raw["split"].strip(),
                    "stem": raw["stem"].strip(),
                    "method": raw["method"].strip(),
                    "f1": _optional_float(raw["f1"]),
                    "f1_undefined": int(raw["f1_undefined"]),
                    "failure_threshold": float(raw["failure_threshold"]),
                    "failure_label": int(raw["failure_label"]),
                    "failure_reason": raw["failure_reason"].strip(),
                    "logistic_probability": _optional_float(raw["logistic_probability"]),
                    "internal_logistic_probability": _optional_float(
                        raw["internal_logistic_probability"]
                    ),
                    "single_feature_score": _optional_float(raw["single_feature_score"]),
                    "single_feature_direction": raw["single_feature_direction"].strip(),
                    "single_feature_predicted_failure": _optional_int(
                        raw["single_feature_predicted_failure"]
                    ),
                    "single_feature_threshold": _optional_float(raw["single_feature_threshold"]),
                }
            )
    return rows


def _optional_int(text: str) -> int | None:
    stripped = text.strip()
    if stripped == "":
        return None
    return int(stripped)


def load_failure_labels(path: str | Path) -> dict[tuple[str, str, str], float | None]:
    """Read stored per-image F1 values for a consistency check."""
    csv_path = Path(path)
    labels: dict[tuple[str, str, str], float | None] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "stem", "method", "f1"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Per-image CSV is missing required columns: {csv_path}")
        for raw in reader:
            labels[(raw["split"].strip(), raw["stem"].strip(), raw["method"].strip())] = (
                _optional_float(raw["f1"])
            )
    return labels


def validate_predictions(
    rows: list[dict[str, object]],
    evaluation_f1: dict[tuple[str, str, str], float | None] | None = None,
) -> None:
    """Check frozen labels. Undefined F1 must remain an operational failure."""
    for row in rows:
        undefined = row["f1"] is None or int(row["f1_undefined"]) == 1
        if undefined and int(row["failure_label"]) != 1:
            raise ValueError(
                f"{row['split']}/{row['stem']}/{row['method']} has undefined F1 "
                "and was not labeled as an operational failure."
            )
        if row["f1"] is not None and int(row["f1_undefined"]) == 0:
            distance = float(row["f1"]) - float(row["failure_threshold"])
            if abs(distance) > 1e-5:
                expected = 1 if float(row["f1"]) <= float(row["failure_threshold"]) else 0
                if int(row["failure_label"]) != expected:
                    raise ValueError(
                        f"Frozen failure label disagrees with the stored F1 threshold "
                        f"for {row['split']}/{row['stem']}/{row['method']}."
                    )
        if row["split"] == "train" and row["internal_logistic_probability"] is None:
            raise ValueError(
                f"Training row {row['stem']}/{row['method']} has no out-of-fold probability."
            )
        if row["logistic_probability"] is None:
            raise ValueError(f"{row['split']}/{row['stem']}/{row['method']} has no logistic probability.")
    if evaluation_f1 is None:
        return
    for row in rows:
        key = (str(row["split"]), str(row["stem"]), str(row["method"]))
        if key not in evaluation_f1:
            raise ValueError(f"Prediction row is absent from the evaluation CSV: {key}.")
        stored = evaluation_f1[key]
        if stored is None or row["f1"] is None:
            if stored is not None or row["f1"] is not None:
                raise ValueError(f"Undefined F1 does not match the evaluation CSV for {key}.")
        elif abs(float(stored) - float(row["f1"])) > 1e-6:
            raise ValueError(f"Stored F1 does not match the evaluation CSV for {key}.")


def _as_float_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row[key]
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"Missing risk score '{key}' for {row['split']}/{row['stem']}/{row['method']}.")
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def outcome_at_threshold(
    rows: list[dict[str, object]],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Accept rows whose risk score is strictly below the threshold."""
    if len(rows) != len(scores):
        raise ValueError("Outcome rows and scores have different lengths.")
    accepted = scores < float(threshold)
    failures = np.asarray([int(row["failure_label"]) for row in rows], dtype=np.int64)
    undefined = np.asarray([row["f1"] is None or int(row["f1_undefined"]) == 1 for row in rows])
    n = len(rows)
    n_accepted = int(np.sum(accepted))
    n_abstained = n - n_accepted
    n_failures = int(np.sum(failures))
    accepted_failure = int(np.sum(accepted & (failures == 1)))
    abstained_failure = int(np.sum(~accepted & (failures == 1)))
    accepted_nonfailure = int(np.sum(accepted & (failures == 0)))
    abstained_nonfailure = int(np.sum(~accepted & (failures == 0)))
    accepted_undefined = int(np.sum(accepted & undefined))
    abstained_undefined = int(np.sum(~accepted & undefined))
    coverage = n_accepted / n if n else None
    baseline = n_failures / n if n else None
    risk = accepted_failure / n_accepted if n_accepted else None
    if risk is None or baseline is None:
        absolute = None
        relative = None
    else:
        absolute = baseline - risk
        relative = absolute / baseline if baseline > 0 else None
    if n_abstained == 0 or n_failures == 0:
        abstain_recall = None
    else:
        abstain_recall = abstained_failure / n_failures
    defined_f1 = [
        float(row["f1"])
        for row, keep in zip(rows, accepted.tolist())
        if keep and row["f1"] is not None
    ]
    return {
        "n": n,
        "n_accepted": n_accepted,
        "n_abstained": n_abstained,
        "coverage": coverage,
        "n_failures": n_failures,
        "n_accepted_failures": accepted_failure,
        "n_abstained_failures": abstained_failure,
        "n_undefined_f1": int(np.sum(undefined)),
        "n_accepted_undefined_f1": accepted_undefined,
        "n_abstained_undefined_f1": abstained_undefined,
        "baseline_prevalence": baseline,
        "accepted_failure_risk": risk,
        "absolute_risk_reduction": absolute,
        "relative_risk_reduction": relative,
        "failure_recall_among_abstained": abstain_recall,
        "accepted_nonfailure": accepted_nonfailure,
        "accepted_failure": accepted_failure,
        "abstained_nonfailure": abstained_nonfailure,
        "abstained_failure": abstained_failure,
        "defined_f1_n": len(defined_f1),
        "defined_f1_mean": float(np.mean(defined_f1)) if defined_f1 else None,
        "defined_f1_min": linear_percentile(defined_f1, 0) if defined_f1 else None,
        "defined_f1_q1": linear_percentile(defined_f1, 25) if defined_f1 else None,
        "defined_f1_median": linear_percentile(defined_f1, 50) if defined_f1 else None,
        "defined_f1_q3": linear_percentile(defined_f1, 75) if defined_f1 else None,
        "defined_f1_max": linear_percentile(defined_f1, 100) if defined_f1 else None,
        "fraction_accepted_with_defined_f1": (len(defined_f1) / n_accepted) if n_accepted else None,
        "threshold": float(threshold),
    }


def select_operating_threshold(
    rows: list[dict[str, object]],
    score_key: str = "internal_logistic_probability",
) -> dict[str, object]:
    """Choose a training threshold, or report that none is justified."""
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Threshold selection received a row outside the training split.")
    scores = _as_float_array(rows, score_key)
    baseline_rows = outcome_at_threshold(rows, scores, float(np.max(scores)) + 1.0)
    baseline = float(baseline_rows["baseline_prevalence"])
    best: dict[str, object] | None = None
    best_key: tuple[float, float, float] | None = None
    examined = 0
    eligible = 0
    best_covered_reduction: float | None = None
    for threshold in sorted(set(scores.tolist())):
        examined += 1
        stats = outcome_at_threshold(rows, scores, float(threshold))
        if stats["coverage"] is None or float(stats["coverage"]) + 1e-15 < MINIMUM_COVERAGE:
            continue
        if int(stats["n_abstained"]) == 0 or stats["absolute_risk_reduction"] is None:
            continue
        reduction = float(stats["absolute_risk_reduction"])
        if best_covered_reduction is None or reduction > best_covered_reduction:
            best_covered_reduction = reduction
        if reduction + 1e-15 < MINIMUM_ABSOLUTE_RISK_REDUCTION:
            continue
        eligible += 1
        key = (
            float(stats["coverage"]),
            -float(stats["accepted_failure_risk"]),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {**stats, "threshold": float(threshold)}
    return {
        "justified": best is not None,
        "threshold": None if best is None else float(best["threshold"]),
        "training_outcome": best,
        "candidates_examined": examined,
        "candidates_eligible": eligible,
        "largest_reduction_at_minimum_coverage": best_covered_reduction,
        "training_prevalence": baseline,
        "score_key": score_key,
    }


def build_curve(
    rows: list[dict[str, object]],
    score_key: str,
    operating_threshold: float | None,
) -> list[dict[str, object]]:
    """Sweep thresholds. Coverage increases as the threshold increases."""
    scores = _as_float_array(rows, score_key)
    thresholds = sorted(set(scores.tolist()))
    if operating_threshold is not None and all(
        abs(float(operating_threshold) - threshold) > 1e-12 for threshold in thresholds
    ):
        thresholds.append(float(operating_threshold))
        thresholds = sorted(thresholds)
    points = [np.nextafter(float(np.min(scores)), -np.inf), *thresholds]
    points.append(np.nextafter(float(np.max(scores)), np.inf))
    curve = []
    for threshold in points:
        stats = outcome_at_threshold(rows, scores, float(threshold))
        accepts_all = int(stats["n_abstained"]) == 0
        is_operating = (
            operating_threshold is not None
            and abs(float(threshold) - float(operating_threshold)) <= 1e-12
        )
        curve.append(
            {
                **stats,
                "is_operating_point": int(is_operating),
                "is_always_accept": int(accepts_all),
                "risk_defined": int(stats["accepted_failure_risk"] is not None),
                "unique_scores": len(set(scores.tolist())),
            }
        )
    return curve


def curve_is_nested(rows: list[dict[str, object]], score_key: str) -> bool:
    """Return whether raising the threshold only adds accepted images."""
    scores = _as_float_array(rows, score_key)
    stems = [str(row["stem"]) for row in rows]
    order = np.argsort(scores, kind="mergesort")
    previous: set[str] = set()
    previous_coverage = -1.0
    for threshold in sorted(set(scores.tolist())):
        accepted = {
            stems[index]
            for index in order.tolist()
            if scores[index] < threshold
        }
        coverage = len(accepted) / len(rows)
        if not previous.issubset(accepted) or coverage < previous_coverage:
            return False
        previous = accepted
        previous_coverage = coverage
    return True


def random_abstention_trials(
    rows: list[dict[str, object]],
    n_accepted: int,
    *,
    n_trials: int,
    seed: int,
) -> list[dict[str, object]]:
    """Abstain uniformly at a fixed accepted count. Labels do not choose the images."""
    if n_accepted < 0 or n_accepted > len(rows):
        raise ValueError("Random abstention received an impossible accepted count.")
    generator = np.random.default_rng(seed)
    failures = np.asarray([int(row["failure_label"]) for row in rows], dtype=np.int64)
    n = len(rows)
    n_failures = int(np.sum(failures))
    baseline = n_failures / n if n else None
    trials = []
    for trial in range(n_trials):
        chosen = np.zeros(n, dtype=bool)
        if n_accepted:
            picked = generator.choice(n, size=n_accepted, replace=False)
            chosen[picked] = True
        accepted_failure = int(np.sum(chosen & (failures == 1)))
        abstained_failure = n_failures - accepted_failure
        risk = accepted_failure / n_accepted if n_accepted else None
        absolute = None if risk is None or baseline is None else baseline - risk
        recall = None if n_accepted == n or n_failures == 0 else abstained_failure / n_failures
        trials.append(
            {
                "trial": trial,
                "n_accepted": n_accepted,
                "coverage": n_accepted / n if n else None,
                "accepted_failure_risk": risk,
                "absolute_risk_reduction": absolute,
                "failure_recall_among_abstained": recall,
                "n_accepted_failures": accepted_failure,
                "seed": seed,
            }
        )
    return trials


def trial_seed(method: str, split: str, policy: str, base_seed: int = RANDOM_SEED) -> int:
    """Stable seed for one method, split, and reference policy."""
    method_index = METHODS.index(method)
    split_index = 0 if split == "train" else 1
    policy_index = 0 if policy == "logistic" else 1
    sequence = np.random.SeedSequence([base_seed, method_index, split_index, policy_index])
    return int(np.random.default_rng(sequence).integers(0, 2**31 - 1))


def summarize_trials(trials: list[dict[str, object]]) -> dict[str, float | int | None]:
    """Distribution of random-abstention risk. One draw is not the result."""
    risks = [float(trial["accepted_failure_risk"]) for trial in trials if trial["accepted_failure_risk"] is not None]
    if not risks:
        return {"n_trials": len(trials), "risk_mean": None, "risk_std": None, "risk_p2_5": None, "risk_median": None, "risk_p97_5": None}
    return {
        "n_trials": len(trials),
        "risk_mean": float(np.mean(risks)),
        "risk_std": float(np.std(risks, ddof=1)) if len(risks) > 1 else None,
        "risk_p2_5": linear_percentile(risks, 2.5),
        "risk_median": linear_percentile(risks, 50),
        "risk_p97_5": linear_percentile(risks, 97.5),
    }


def paired_bootstrap_draws(n_images: int, n_replicates: int, seed: int) -> np.ndarray:
    """Draw image indexes once so methods that share stems stay paired."""
    generator = np.random.default_rng(seed)
    return generator.integers(0, n_images, size=(n_replicates, n_images))


def bootstrap_outcome(
    rows: list[dict[str, object]],
    score_key: str,
    threshold: float,
    draws: np.ndarray,
) -> dict[str, object]:
    """Percentile interval for coverage and accepted risk at a frozen threshold."""
    ordered = sorted(rows, key=lambda row: str(row["stem"]))
    scores = _as_float_array(ordered, score_key)
    failures = np.asarray([int(row["failure_label"]) for row in ordered], dtype=np.int64)
    sample_scores = scores[draws]
    sample_failures = failures[draws]
    accepted = sample_scores < float(threshold)
    n_accepted = accepted.sum(axis=1)
    accepted_failures = (accepted & (sample_failures == 1)).sum(axis=1)
    defined = n_accepted > 0
    risks = np.full(len(draws), np.nan)
    risks[defined] = accepted_failures[defined] / n_accepted[defined]
    coverages = n_accepted / len(ordered)
    finite_risks = risks[np.isfinite(risks)].tolist()
    finite_coverages = coverages[defined].tolist()
    return {
        "replicates": int(len(draws)),
        "replicates_undefined_risk": int(np.sum(~defined)),
        "risk_p2_5": linear_percentile(finite_risks, 2.5) if finite_risks else None,
        "risk_p97_5": linear_percentile(finite_risks, 97.5) if finite_risks else None,
        "coverage_p2_5": linear_percentile(finite_coverages, 2.5) if finite_coverages else None,
        "coverage_p97_5": linear_percentile(finite_coverages, 97.5) if finite_coverages else None,
    }


def _split_rows(rows: list[dict[str, object]], method: str, split: str) -> list[dict[str, object]]:
    return [row for row in rows if row["method"] == method and row["split"] == split]


def _point(
    method: str,
    split: str,
    policy: str,
    score_source: str,
    role: str,
    stats: dict[str, object],
    justified: bool,
    notes: str,
) -> dict[str, object]:
    return {
        "method": method,
        "split": split,
        "policy": policy,
        "score_source": score_source,
        "role": role,
        "threshold_justified": int(justified),
        "notes": notes,
        **stats,
    }


def run_abstention(
    rows: list[dict[str, object]],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    n_random_trials: int = N_RANDOM_TRIALS,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Select training thresholds, then score the test split once."""
    selections = {}
    for method in METHODS:
        training = _split_rows(rows, method, "train")
        if not training:
            raise ValueError(f"No training rows for method {method}.")
        selections[method] = select_operating_threshold(training)

    operating_points: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    trials: list[dict[str, object]] = []
    bootstrap: dict[str, dict[str, object]] = {}
    for split in ("train", "test"):
        stem_sets = []
        for method in METHODS:
            stem_sets.append([str(row["stem"]) for row in _split_rows(rows, method, split)])
        if any(sorted(stems) != sorted(stem_sets[0]) for stems in stem_sets):
            raise ValueError(f"Methods do not share the same {split} image stems.")
        draws = paired_bootstrap_draws(len(stem_sets[0]), n_bootstrap, seed + (0 if split == "train" else 1))
        for method in METHODS:
            method_rows = sorted(_split_rows(rows, method, split), key=lambda row: str(row["stem"]))
            selection = selections[method]
            justified = bool(selection["justified"])
            threshold = selection["threshold"]
            score_key = (
                "internal_logistic_probability" if split == "train" else "logistic_probability"
            )
            if justified and threshold is not None:
                stats = outcome_at_threshold(
                    method_rows,
                    _as_float_array(method_rows, score_key),
                    float(threshold),
                )
                note = (
                    "Training out-of-fold scores. This is the selection sample."
                    if split == "train"
                    else "Untouched test scores. The threshold was not chosen here."
                )
                operating_points.append(
                    _point(
                        method,
                        split,
                        "logistic",
                        score_key,
                        "selected_operating_point",
                        stats,
                        True,
                        note,
                    )
                )
                interval = bootstrap_outcome(method_rows, score_key, float(threshold), draws)
                bootstrap[f"{method}:{split}:logistic"] = interval
                trial_rows = random_abstention_trials(
                    method_rows,
                    int(stats["n_accepted"]),
                    n_trials=n_random_trials,
                    seed=trial_seed(method, split, "logistic", seed),
                )
                for trial in trial_rows:
                    trials.append(
                        {
                            "method": method,
                            "split": split,
                            "reference_policy": "logistic",
                            **trial,
                        }
                    )
            else:
                operating_points.append(
                    {
                        "method": method,
                        "split": split,
                        "policy": "logistic",
                        "score_source": score_key,
                        "role": "no_justified_threshold",
                        "threshold_justified": 0,
                        "threshold": None,
                        "notes": (
                            "No training threshold met the coverage and risk-reduction rule. "
                            "No test threshold was substituted."
                        ),
                        "n": len(method_rows),
                    }
                )
            always = outcome_at_threshold(
                method_rows,
                _as_float_array(method_rows, score_key),
                float(np.max(_as_float_array(method_rows, score_key))) + 1.0,
            )
            operating_points.append(
                _point(
                    method,
                    split,
                    "always_accept",
                    score_key,
                    "baseline",
                    always,
                    True,
                    "Every image is accepted.",
                )
            )
            if all(row["single_feature_predicted_failure"] is not None for row in method_rows):
                # The saved bit already applies the training median and its direction.
                # Comparing the directed score with the raw median would accept every
                # low-direction image, so the bit is the frozen decision.
                decision = np.asarray(
                    [float(row["single_feature_predicted_failure"]) for row in method_rows],
                    dtype=np.float64,
                )
                feature_stats = outcome_at_threshold(method_rows, decision, 0.5)
                feature_stats["threshold"] = method_rows[0]["single_feature_threshold"]
                operating_points.append(
                    _point(
                        method,
                        split,
                        "single_feature",
                        "single_feature_predicted_failure",
                        "frozen_cs005_rule",
                        feature_stats,
                        True,
                        "Abstain when the saved CS-005 single-feature rule predicts failure. "
                        "The threshold column is that rule's training median, not a new cutoff.",
                    )
                )
                feature_trials = random_abstention_trials(
                    method_rows,
                    int(feature_stats["n_accepted"]),
                    n_trials=n_random_trials,
                    seed=trial_seed(method, split, "single_feature", seed),
                )
                for trial in feature_trials:
                    trials.append(
                        {
                            "method": method,
                            "split": split,
                            "reference_policy": "single_feature",
                            **trial,
                        }
                    )
            curve = build_curve(method_rows, score_key, None if threshold is None else float(threshold))
            for point in curve:
                curves.append(
                    {
                        "method": method,
                        "split": split,
                        "score_source": score_key,
                        "curve_role": "selection_curve" if split == "train" else "untouched_test_curve",
                        **point,
                    }
                )
            if split == "train" and justified and threshold is not None:
                insample = outcome_at_threshold(
                    method_rows,
                    _as_float_array(method_rows, "logistic_probability"),
                    float(threshold),
                )
                operating_points.append(
                    _point(
                        method,
                        "train",
                        "logistic_insample",
                        "logistic_probability",
                        "descriptive_insample",
                        insample,
                        True,
                        "Same frozen threshold on in-sample probabilities. Not used to select the threshold.",
                    )
                )
    config = {
        "random_seed": seed,
        "n_bootstrap": n_bootstrap,
        "n_random_trials": n_random_trials,
        "minimum_coverage": MINIMUM_COVERAGE,
        "minimum_absolute_risk_reduction": MINIMUM_ABSOLUTE_RISK_REDUCTION,
        "selection_rule": SELECTION_RULE,
        "decisions_fixed_before_test": list(DECISIONS_FIXED_BEFORE_TEST),
        "methods": {
            method: {
                "justified": bool(selection["justified"]),
                "threshold": selection["threshold"],
                "candidates_examined": selection["candidates_examined"],
                "candidates_eligible": selection["candidates_eligible"],
                "largest_reduction_at_minimum_coverage": selection[
                    "largest_reduction_at_minimum_coverage"
                ],
                "training_prevalence": selection["training_prevalence"],
                "score_key": selection["score_key"],
            }
            for method, selection in selections.items()
        },
        "bootstrap": bootstrap,
    }
    report = build_report(operating_points, trials, config)
    return {
        "report": report,
        "operating_points": operating_points,
        "curves": curves,
        "trials": trials,
        "config": config,
    }


def _format_metric(value: object) -> str:
    if value is None or value == "":
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _find_point(
    points: list[dict[str, object]],
    method: str,
    split: str,
    policy: str,
) -> dict[str, object] | None:
    matches = [
        point
        for point in points
        if point["method"] == method and point["split"] == split and point["policy"] == policy
    ]
    return matches[0] if matches else None


def build_report(
    points: list[dict[str, object]],
    trials: list[dict[str, object]],
    config: dict[str, object],
) -> str:
    """Describe the frozen rule and the separate training and test outcomes."""
    lines = [
        "Selective prediction and abstention evaluation",
        "",
        "This experiment asks whether abstaining on images with high predicted",
        "failure probability lowers the operational failure rate among the images",
        "that are still accepted, and how much coverage that choice removes.",
        "It does not establish structural safety, real-world crack-detection",
        "reliability, or a reason to deploy an abstention policy.",
        "",
        "Selection rule fixed before test evaluation",
        config["selection_rule"],
        f"Minimum coverage: {config['minimum_coverage']:.2f}.",
        f"Minimum absolute risk reduction: {config['minimum_absolute_risk_reduction']:.2f}.",
        f"Random trials: {config['n_random_trials']}. Bootstrap replicates: {config['n_bootstrap']}.",
        f"Random seed: {config['random_seed']}.",
        "",
    ]
    for decision in config["decisions_fixed_before_test"]:
        if decision != config["selection_rule"]:
            lines.append(f"- {decision}")
    lines.append("")
    for method, selection in config["methods"].items():
        lines.append(method)
        if selection["justified"]:
            lines.append(
                f"  Training-selected threshold {float(selection['threshold']):.6f}. "
                f"Eligible candidates {selection['candidates_eligible']} "
                f"of {selection['candidates_examined']}."
            )
        else:
            lines.append("  No operating threshold was justified on the training out-of-fold scores.")
            lines.append(
                "  Largest absolute risk reduction at coverage of at least "
                f"{config['minimum_coverage']:.2f}: "
                f"{_format_metric(selection['largest_reduction_at_minimum_coverage'])}."
            )
            lines.append("  The test curve was not used to replace this decision.")
        for split in ("train", "test"):
            logistic = _find_point(points, method, split, "logistic")
            always = _find_point(points, method, split, "always_accept")
            feature = _find_point(points, method, split, "single_feature")
            lines.append(f"  {split}")
            if always is not None:
                lines.append(
                    f"    Always accept: n {always['n']}, failures {always['n_failures']}, "
                    f"undefined F1 {always['n_undefined_f1']}, "
                    f"prevalence {_format_metric(always['baseline_prevalence'])}."
                )
            if logistic is None or logistic.get("role") == "no_justified_threshold":
                lines.append("    Logistic policy: no frozen threshold to apply.")
                continue
            lines.extend(_outcome_lines("Logistic policy", logistic))
            interval = config.get("bootstrap", {}).get(f"{method}:{split}:logistic")
            if interval is not None:
                lines.append(
                    "    Bootstrap 2.5-97.5 percentile interval, images resampled with the "
                    f"threshold held fixed: risk {_format_metric(interval['risk_p2_5'])} to "
                    f"{_format_metric(interval['risk_p97_5'])}; coverage "
                    f"{_format_metric(interval['coverage_p2_5'])} to "
                    f"{_format_metric(interval['coverage_p97_5'])}. "
                    f"Replicates with undefined risk: {interval['replicates_undefined_risk']} "
                    f"of {interval['replicates']}."
                )
                lines.append("    " + _interval_versus_baseline(logistic, interval))
            if int(logistic["n_failures"]) < 20 or int(logistic["n_accepted_failures"]) < 10:
                lines.append(
                    "    This estimate uses few failures "
                    f"({logistic['n_failures']} in the split, "
                    f"{logistic['n_accepted_failures']} among accepted images). "
                    "The bootstrap interval does not repair that."
                )
            policy_trials = [
                trial
                for trial in trials
                if trial["method"] == method
                and trial["split"] == split
                and trial["reference_policy"] == "logistic"
            ]
            lines.append("    " + _random_sentence(logistic, policy_trials))
            if feature is not None:
                lines.extend(_outcome_lines("Frozen single-feature rule", feature))
                feature_trials = [
                    trial
                    for trial in trials
                    if trial["method"] == method
                    and trial["split"] == split
                    and trial["reference_policy"] == "single_feature"
                ]
                lines.append("    " + _random_sentence(feature, feature_trials))
        insample = _find_point(points, method, "train", "logistic_insample")
        if insample is not None:
            lines.append(
                "  In-sample logistic probabilities at the same threshold are descriptive only: "
                f"coverage {_format_metric(insample['coverage'])}, "
                f"accepted failure risk {_format_metric(insample['accepted_failure_risk'])}."
            )
        lines.append("")
    lines.extend(
        [
            "Secondary F1 description",
            "Mean and median F1 use defined scores only. Undefined F1 stays a failure",
            "in the risk calculation and is excluded from the F1 average. A higher mean",
            "F1 among accepted images is not, by itself, evidence of improved reliability.",
            "",
            "Limitations",
            "The training operating point was selected on the same out-of-fold scores",
            "that are reported for training, so that training risk is not a confirmation sample.",
            "Test curves and test risks were calculated after the threshold was frozen.",
            "Bootstrap intervals condition on the chosen threshold. They do not repeat",
            "the threshold search, and they are unstable when few failures remain.",
            "Random abstention matches coverage. It is not a model of a deployment queue.",
            "Images that share a stem across the official train and test splits are",
            "different images and are resampled only inside their own split.",
            "No result here is a measure of structural safety.",
            "",
        ]
    )
    return "\n".join(lines)


def _outcome_lines(title: str, stats: dict[str, object]) -> list[str]:
    return [
        (
            f"    {title}: accepted {stats['n_accepted']}, abstained {stats['n_abstained']}, "
            f"coverage {_format_metric(stats['coverage'])}, "
            f"accepted failure risk {_format_metric(stats['accepted_failure_risk'])}, "
            f"absolute reduction {_format_metric(stats['absolute_risk_reduction'])}, "
            f"relative reduction {_format_metric(stats['relative_risk_reduction'])}, "
            f"failure recall among abstained {_format_metric(stats['failure_recall_among_abstained'])}."
        ),
        (
            f"    Acceptance counts: accepted non-failure {stats['accepted_nonfailure']}, "
            f"accepted failure {stats['accepted_failure']}, "
            f"abstained non-failure {stats['abstained_nonfailure']}, "
            f"abstained failure {stats['abstained_failure']}. "
            f"Undefined F1 accepted {stats['n_accepted_undefined_f1']}, "
            f"abstained {stats['n_abstained_undefined_f1']}."
        ),
        (
            f"    Defined F1 among accepted images: n {stats['defined_f1_n']}, "
            f"fraction {_format_metric(stats['fraction_accepted_with_defined_f1'])}, "
            f"mean {_format_metric(stats['defined_f1_mean'])}, "
            f"median {_format_metric(stats['defined_f1_median'])}, "
            f"min {_format_metric(stats['defined_f1_min'])}, "
            f"q1 {_format_metric(stats['defined_f1_q1'])}, "
            f"q3 {_format_metric(stats['defined_f1_q3'])}, "
            f"max {_format_metric(stats['defined_f1_max'])}."
        ),
    ]


def _interval_versus_baseline(stats: dict[str, object], interval: dict[str, object]) -> str:
    baseline = stats.get("baseline_prevalence")
    low = interval.get("risk_p2_5")
    high = interval.get("risk_p97_5")
    if baseline is None or low is None or high is None:
        return "The bootstrap interval cannot be compared with the always-accept prevalence."
    baseline = float(baseline)
    low = float(low)
    high = float(high)
    if low <= baseline <= high:
        relation = "includes"
    elif high < baseline:
        relation = "is below"
    else:
        relation = "is above"
    return (
        f"The bootstrap interval for accepted failure risk {relation} "
        "the always-accept prevalence."
    )


def _random_sentence(stats: dict[str, object], trials: list[dict[str, object]]) -> str:
    summary = summarize_trials(trials)
    if summary["risk_median"] is None or stats.get("accepted_failure_risk") is None:
        return "Random abstention at matched coverage is undefined for this policy."
    risks = [
        float(trial["accepted_failure_risk"])
        for trial in trials
        if trial["accepted_failure_risk"] is not None
    ]
    policy_risk = float(stats["accepted_failure_risk"])
    below = sum(1 for risk in risks if risk < policy_risk)
    return (
        f"Random abstention over {summary['n_trials']} trials at the same coverage: "
        f"mean risk {_format_metric(summary['risk_mean'])}, "
        f"sample std {_format_metric(summary['risk_std'])}, "
        f"2.5 percentile {_format_metric(summary['risk_p2_5'])}, "
        f"median {_format_metric(summary['risk_median'])}, "
        f"97.5 percentile {_format_metric(summary['risk_p97_5'])}. "
        f"Trials with lower risk than the policy: {below} of {len(risks)}. "
        f"Policy minus random mean: {_format_metric(policy_risk - float(summary['risk_mean']))}."
    )


def save_risk_coverage_plot(
    curves: list[dict[str, object]],
    method: str,
    path: str | Path,
    justified: bool,
) -> None:
    """Save training and test curves. The test panel does not select a threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, split, title in (
        (axes[0], "train", "Training out-of-fold selection curve"),
        (axes[1], "test", "Untouched test curve"),
    ):
        points = [
            point
            for point in curves
            if point["method"] == method
            and point["split"] == split
            and point["score_source"]
            in {"internal_logistic_probability", "logistic_probability"}
            and point["risk_defined"] == 1
        ]
        points = sorted(points, key=lambda point: float(point["coverage"]))
        if points:
            axis.plot(
                [float(point["coverage"]) for point in points],
                [float(point["accepted_failure_risk"]) for point in points],
                color="#1f4e79",
                linewidth=1.8,
            )
            prevalence = float(points[-1]["baseline_prevalence"])
            axis.axhline(prevalence, color="#666666", linestyle="--", linewidth=1, label="Always accept")
            chosen = [point for point in points if int(point["is_operating_point"]) == 1]
            if chosen:
                axis.scatter(
                    [float(chosen[0]["coverage"])],
                    [float(chosen[0]["accepted_failure_risk"])],
                    color="#b45309",
                    s=46,
                    zorder=3,
                    label="Training-selected threshold",
                )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Coverage")
        axis.set_title(title, fontsize=10)
        axis.grid(True, alpha=0.3)
        if not justified:
            axis.text(0.03, 0.95, "No training-justified threshold", transform=axis.transAxes, va="top")
        axis.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("Operational failure risk among accepted images")
    figure.suptitle(f"{method}: threshold selected from training data only", fontsize=12)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(destination, dpi=120, bbox_inches="tight")
    finally:
        plt.close(figure)


def write_outputs(output_dir: str | Path, result: dict[str, object]) -> None:
    """Write abstention files. Existing planned files are left unchanged."""
    destination = Path(output_dir)
    planned = [destination / name for name in OUTPUT_NAMES]
    existing = [path for path in planned if path.exists()]
    if existing:
        listed = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite abstention-evaluation files:\n"
            f"{listed}\n"
            "Reliability-prediction, evaluation, and earlier outputs were not changed."
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.txt").write_text(str(result["report"]), encoding="utf-8")
    _write_csv(destination / "operating_points.csv", list(result["operating_points"]))
    _write_csv(destination / "risk_coverage.csv", list(result["curves"]))
    _write_csv(destination / "random_baseline.csv", list(result["trials"]))
    (destination / "config.json").write_text(
        json.dumps(result["config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for method in METHODS:
        save_risk_coverage_plot(
            list(result["curves"]),
            method,
            destination / f"risk_coverage_{method}.png",
            bool(result["config"]["methods"][method]["justified"]),
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    if not fieldnames:
        fieldnames = ["method"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stored = {}
            for field in fieldnames:
                value = row.get(field)
                if value is None:
                    stored[field] = ""
                elif isinstance(value, float):
                    stored[field] = f"{value:.10f}" if "threshold" in field else f"{value:.6f}"
                else:
                    stored[field] = value
            writer.writerow(stored)
