"""Training-only nested checks and follow-up uncertainty for CS-006 abstention.

The held-out DeepCrack test split was already examined in CS-006. Test numbers
in this module are follow-up descriptions. They are not a pristine confirmation.

Saved CS-005 out-of-fold probabilities cannot validate the full threshold rule.
An image's out-of-fold score was fit without that image, but the scores of the
other folds were fit with that image still in the training set. Selecting a
threshold from those other scores would let the outer-fold labels influence
the selection. This module therefore refits the existing logistic specification
on training images only, inside image-level outer folds. It does not change the
CS-005 model file or its saved predictions.
"""

import csv
import json
from pathlib import Path

import numpy as np

from crackscout.abstention_evaluation import (
    MINIMUM_ABSOLUTE_RISK_REDUCTION,
    MINIMUM_COVERAGE,
    load_predictions,
    outcome_at_threshold,
    random_abstention_trials,
    select_operating_threshold,
    summarize_trials,
    trial_seed,
)
from crackscout.failure_analysis import linear_percentile
from crackscout.image_quality import FEATURE_NAMES
from crackscout.reliability_prediction import (
    METHODS,
    RANDOM_SEED,
    apply_medians,
    apply_scaler,
    assign_folds,
    fit_method_models,
    predict_logistic,
    predictor_matrix,
)

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
INNER_SEED = 0
N_BOOTSTRAP = 1000
N_RANDOM_TRIALS = 200
MATCHED_COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
SENSITIVITY_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)
PROTOCOL = (
    "Outer folds are the CS-005 image-level assignment: training stems are "
    "sorted, shuffled with seed 0, and dealt by index modulo 5. All three "
    "methods use that stem assignment. Inside each outer fold, the complement "
    "is split again with the same rule and seed 0. Inner out-of-fold logistic "
    "probabilities are produced by refitting the CS-005 logistic specification "
    "on the inner complement only. The CS-006 eligibility rule is applied to "
    "those inner probabilities and is not relaxed. The selected threshold is "
    "applied to the outer fold using a logistic model fit on the whole inner "
    "complement. Outer-fold labels are not used to choose that threshold. "
    "The official test split is not used in this procedure. "
    "Matched-coverage comparisons accept round-half-up of the target fraction, "
    "lowest predicted probability first, with stem order breaking ties, at "
    "coverages 0.50, 0.60, 0.70, 0.80, 0.90, and 1.00. "
    "Sensitivity adds -0.10, -0.05, 0, 0.05, and 0.10 to the frozen CS-006 "
    "threshold. "
    "The paired bootstrap resamples test image identities with seed 0 for "
    "1000 replicates, holds the CS-006 thresholds fixed, and uses percentile "
    "intervals at 2.5 and 97.5. A replicate with no accepted images has an "
    "undefined reduction and is omitted from the percentile. This bootstrap "
    "does not repeat threshold selection, and the test split was already "
    "examined in CS-006."
)


def _optional_float(text: str) -> float | None:
    stripped = text.strip()
    if stripped == "":
        return None
    return float(stripped)


def load_feature_table(path: str | Path) -> list[dict[str, object]]:
    """Read image-quality features. Score columns other than F1 are not kept."""
    csv_path = Path(path)
    required = {"split", "stem", "method", "f1", *FEATURE_NAMES}
    rows: list[dict[str, object]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Feature CSV is missing columns: {', '.join(missing)}")
        for raw in reader:
            row: dict[str, object] = {
                "split": raw["split"].strip(),
                "stem": raw["stem"].strip(),
                "method": raw["method"].strip(),
                "f1": _optional_float(raw["f1"]),
            }
            for name in FEATURE_NAMES:
                row[name] = _optional_float(raw[name])
            rows.append(row)
    return rows


def training_model_rows(
    predictions: list[dict[str, object]],
    features: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Join frozen training labels to features. Test rows are not returned."""
    by_key = {
        (str(row["split"]), str(row["stem"]), str(row["method"])): row
        for row in features
        if row["split"] == "train"
    }
    joined = []
    for prediction in predictions:
        if prediction["split"] != "train":
            continue
        key = (str(prediction["split"]), str(prediction["stem"]), str(prediction["method"]))
        feature = by_key.get(key)
        if feature is None:
            raise ValueError(f"No training features for {key}.")
        if prediction["f1"] is None or feature["f1"] is None:
            if prediction["f1"] is not None or feature["f1"] is not None:
                raise ValueError(f"Undefined F1 does not match features for {key}.")
        elif abs(float(prediction["f1"]) - float(feature["f1"])) > 1e-6:
            raise ValueError(f"Training F1 does not match features for {key}.")
        failure = int(prediction["failure_label"])
        if prediction["f1"] is None and failure != 1:
            raise ValueError(f"Undefined F1 was not labeled as a failure for {key}.")
        joined.append(
            {
                "split": "train",
                "stem": prediction["stem"],
                "method": prediction["method"],
                "f1": prediction["f1"],
                "f1_undefined": int(prediction["f1_undefined"]),
                "failure": failure,
                "failure_label": failure,
                "failure_reason": prediction["failure_reason"],
                **{name: feature[name] for name in FEATURE_NAMES},
            }
        )
    return joined


def logistic_probabilities(model: dict[str, object], rows: list[dict[str, object]]) -> np.ndarray:
    """Score rows with a fitted CS-005 logistic model."""
    matrix = predictor_matrix(rows)
    imputed = apply_medians(matrix, np.asarray(model["medians"]))
    scaled = apply_scaler(imputed, np.asarray(model["means"]), np.asarray(model["scale"]))
    return predict_logistic(model["logistic"], scaled)


def inner_oof_probabilities(
    rows: list[dict[str, object]],
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    """Refit logistic regression inside one outer-training complement."""
    folds = assign_folds([str(row["stem"]) for row in rows], n_folds, seed)
    scores: dict[str, float] = {}
    for fold in range(n_folds):
        fitting = [row for row in rows if folds[str(row["stem"])] != fold]
        held = [row for row in rows if folds[str(row["stem"])] == fold]
        model = fit_method_models(fitting)
        probabilities = logistic_probabilities(model, held)
        for row, probability in zip(held, probabilities.tolist()):
            scores[str(row["stem"])] = float(probability)
    return scores


def _with_scores(rows: list[dict[str, object]], scores: dict[str, float]) -> list[dict[str, object]]:
    scored = []
    for row in rows:
        if str(row["stem"]) not in scores:
            raise ValueError(f"No inner score for {row['stem']}.")
        scored.append({**row, "internal_logistic_probability": scores[str(row["stem"])]})
    return scored


def evaluate_outer_fold(
    inner_rows: list[dict[str, object]],
    outer_rows: list[dict[str, object]],
    *,
    n_inner_folds: int = N_INNER_FOLDS,
    inner_seed: int = INNER_SEED,
) -> dict[str, object]:
    """Select a threshold without the outer rows, then score those rows."""
    if any(row["split"] != "train" for row in [*inner_rows, *outer_rows]):
        raise ValueError("Outer-fold validation received a row outside the training split.")
    inner_stems = {str(row["stem"]) for row in inner_rows}
    outer_stems = {str(row["stem"]) for row in outer_rows}
    if inner_stems & outer_stems:
        raise ValueError("An image appears on both sides of an outer fold.")
    oof = inner_oof_probabilities(inner_rows, n_inner_folds, inner_seed)
    selection = select_operating_threshold(_with_scores(inner_rows, oof))
    always = outcome_at_threshold(
        outer_rows,
        np.ones(len(outer_rows)),
        2.0,
    )
    record: dict[str, object] = {
        "justified": bool(selection["justified"]),
        "threshold": selection["threshold"],
        "n_inner": len(inner_rows),
        "n_outer": len(outer_rows),
        "selection_stems": sorted(inner_stems),
        "candidates_eligible": selection["candidates_eligible"],
        "largest_reduction_at_minimum_coverage": selection["largest_reduction_at_minimum_coverage"],
        "always_accept_risk": always["baseline_prevalence"],
        "n_failures": always["n_failures"],
        "status": "selected" if selection["justified"] else "ineligible",
    }
    if not selection["justified"] or selection["threshold"] is None:
        record.update(
            {
                "coverage": None,
                "accepted_failure_risk": None,
                "absolute_risk_reduction": None,
                "n_accepted": None,
                "n_accepted_failures": None,
                "n_abstained_failures": None,
                "notes": "No eligible threshold. The coverage and reduction criteria were not relaxed.",
            }
        )
        return record
    model = fit_method_models(inner_rows)
    scores = logistic_probabilities(model, outer_rows)
    outcome = outcome_at_threshold(outer_rows, scores, float(selection["threshold"]))
    record.update(
        {
            "coverage": outcome["coverage"],
            "accepted_failure_risk": outcome["accepted_failure_risk"],
            "absolute_risk_reduction": outcome["absolute_risk_reduction"],
            "n_accepted": outcome["n_accepted"],
            "n_accepted_failures": outcome["n_accepted_failures"],
            "n_abstained_failures": outcome["n_abstained_failures"],
            "outer_scores": scores,
            "notes": "Threshold selected on the inner complement only.",
        }
    )
    return record


def outer_fold_validation(
    rows: list[dict[str, object]],
    fold_by_stem: dict[str, int],
    *,
    methods: tuple[str, ...] = METHODS,
    n_inner_folds: int = N_INNER_FOLDS,
    inner_seed: int = INNER_SEED,
) -> list[dict[str, object]]:
    """Run the nested threshold rule on training images only."""
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Outer-fold validation received a row outside the training split.")
    results = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        stems = {str(row["stem"]) for row in method_rows}
        missing = sorted(stems - set(fold_by_stem))
        if missing:
            raise ValueError(f"Training stems have no fold assignment: {', '.join(missing[:5])}")
        fold_ids = sorted(set(fold_by_stem[stem] for stem in stems))
        pooled_failures = []
        pooled_accepted = []
        for fold in fold_ids:
            inner = [row for row in method_rows if fold_by_stem[str(row["stem"])] != fold]
            outer = [row for row in method_rows if fold_by_stem[str(row["stem"])] == fold]
            evaluated = evaluate_outer_fold(
                inner,
                outer,
                n_inner_folds=n_inner_folds,
                inner_seed=inner_seed,
            )
            evaluated["method"] = method
            evaluated["fold"] = int(fold)
            results.append({key: value for key, value in evaluated.items() if key != "outer_scores"})
            if evaluated["justified"] and evaluated.get("outer_scores") is not None:
                scores = np.asarray(evaluated["outer_scores"], dtype=np.float64)
                accepted = scores < float(evaluated["threshold"])
                failures = np.asarray([int(row["failure_label"]) for row in outer], dtype=np.int64)
                pooled_failures.extend(failures.tolist())
                pooled_accepted.extend(accepted.tolist())
        results.append(_pooled_record(method, pooled_failures, pooled_accepted, len(method_rows)))
    return results


def _pooled_record(
    method: str,
    failures: list[int],
    accepted: list[bool],
    n_training: int,
) -> dict[str, object]:
    """Pool outer-fold decisions. Each selected image appears once."""
    if not failures:
        return {
            "method": method,
            "fold": "pooled",
            "justified": False,
            "status": "no_selected_folds",
            "n_outer": 0,
            "n_training_images": n_training,
            "coverage": None,
            "accepted_failure_risk": None,
            "absolute_risk_reduction": None,
            "always_accept_risk": None,
            "n_accepted": None,
            "n_accepted_failures": None,
            "n_abstained_failures": None,
            "n_failures": None,
            "notes": "No outer fold produced an eligible threshold.",
        }
    failure_array = np.asarray(failures, dtype=np.int64)
    accepted_array = np.asarray(accepted, dtype=bool)
    n = len(failure_array)
    n_accepted = int(np.sum(accepted_array))
    n_failures = int(np.sum(failure_array))
    accepted_failures = int(np.sum(accepted_array & (failure_array == 1)))
    abstained_failures = n_failures - accepted_failures
    baseline = n_failures / n
    risk = accepted_failures / n_accepted if n_accepted else None
    reduction = None if risk is None else baseline - risk
    return {
        "method": method,
        "fold": "pooled",
        "justified": True,
        "status": "pooled_selected_folds",
        "n_outer": n,
        "n_training_images": n_training,
        "coverage": n_accepted / n if n else None,
        "accepted_failure_risk": risk,
        "absolute_risk_reduction": reduction,
        "always_accept_risk": baseline,
        "n_accepted": n_accepted,
        "n_accepted_failures": accepted_failures,
        "n_abstained_failures": abstained_failures,
        "n_failures": n_failures,
        "notes": (
            "Each training image from an eligible outer fold is counted once. "
            "Fold-level training sets overlap, so the separate folds are not independent experiments."
        ),
    }


def fold_variability(records: list[dict[str, object]], method: str) -> dict[str, object]:
    """Range of outer-fold estimates. Overlapping folds are not treated as independent."""
    folds = [
        record
        for record in records
        if record["method"] == method and record["fold"] != "pooled"
    ]
    justified = [record for record in folds if record["justified"] and record["absolute_risk_reduction"] is not None]
    reductions = [float(record["absolute_risk_reduction"]) for record in justified]
    coverages = [float(record["coverage"]) for record in justified]
    risks = [float(record["accepted_failure_risk"]) for record in justified if record["accepted_failure_risk"] is not None]
    return {
        "method": method,
        "n_folds": len(folds),
        "n_justified": len(justified),
        "n_ineligible": len(folds) - len(justified),
        "reduction_min": min(reductions) if reductions else None,
        "reduction_median": linear_percentile(reductions, 50) if reductions else None,
        "reduction_max": max(reductions) if reductions else None,
        "coverage_min": min(coverages) if coverages else None,
        "coverage_median": linear_percentile(coverages, 50) if coverages else None,
        "coverage_max": max(coverages) if coverages else None,
        "risk_min": min(risks) if risks else None,
        "risk_median": linear_percentile(risks, 50) if risks else None,
        "risk_max": max(risks) if risks else None,
    }


def matched_acceptance_count(n_images: int, coverage: float) -> int:
    """Round half up to a number of images."""
    if n_images < 0:
        raise ValueError("Image count cannot be negative.")
    return int(np.floor(float(coverage) * n_images + 0.5))


def lowest_risk_indexes(stems: list[str], scores: np.ndarray, n_accept: int) -> list[int]:
    """Indexes of the lowest scores. Equal scores keep ascending stem order."""
    if n_accept < 0 or n_accept > len(stems):
        raise ValueError("Accepted count is outside the image list.")
    order = sorted(range(len(stems)), key=lambda index: (float(scores[index]), stems[index]))
    return order[:n_accept]


def matched_coverage_outcome(
    rows: list[dict[str, object]],
    scores: np.ndarray,
    coverage: float,
) -> dict[str, object]:
    """Accept a pre-specified fraction of the lowest-risk images."""
    n_accept = matched_acceptance_count(len(rows), coverage)
    if n_accept == 0:
        empty = outcome_at_threshold(rows, np.ones(len(rows)), 0.0)
        empty["target_coverage"] = coverage
        empty["notes"] = "No images accepted at this target. Accepted risk is undefined."
        return empty
    chosen = set(lowest_risk_indexes([str(row["stem"]) for row in rows], scores, n_accept))
    decision = np.asarray([0.0 if index in chosen else 1.0 for index in range(len(rows))])
    outcome = outcome_at_threshold(rows, decision, 0.5)
    outcome["target_coverage"] = coverage
    outcome["notes"] = "Lowest predicted probability first. Labels were not used to rank images."
    return outcome


def reduction_from_sample(
    failures: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Accepted-risk reduction for one sample. No accepted images leaves it undefined."""
    accepted = scores < float(threshold)
    n = len(failures)
    n_accepted = int(np.sum(accepted))
    n_failures = int(np.sum(failures == 1))
    accepted_failures = int(np.sum(accepted & (failures == 1)))
    baseline = n_failures / n if n else None
    risk = accepted_failures / n_accepted if n_accepted else None
    reduction = None if risk is None or baseline is None else baseline - risk
    return {
        "n": n,
        "n_accepted": n_accepted,
        "n_failures": n_failures,
        "n_accepted_failures": accepted_failures,
        "n_abstained_failures": n_failures - accepted_failures,
        "always_accept_risk": baseline,
        "accepted_failure_risk": risk,
        "risk_reduction": reduction,
        "defined": reduction is not None,
    }


def paired_bootstrap(
    rows_by_method: dict[str, list[dict[str, object]]],
    thresholds: dict[str, float],
    *,
    n_replicates: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Resample test images once and reuse those identities for every method."""
    methods = [method for method in METHODS if method in rows_by_method]
    stem_lists = []
    for method in methods:
        stem_lists.append([str(row["stem"]) for row in rows_by_method[method]])
    if any(sorted(stems) != sorted(stem_lists[0]) for stems in stem_lists):
        raise ValueError("Paired bootstrap requires the same image identities for every method.")
    stems = sorted(stem_lists[0])
    aligned = {}
    for method in methods:
        by_stem = {str(row["stem"]): row for row in rows_by_method[method]}
        aligned[method] = [by_stem[stem] for stem in stems]
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(stems), size=(n_replicates, len(stems)))
    replicates: list[dict[str, object]] = []
    reductions = {method: [] for method in methods}
    for replicate, indexes in enumerate(draws.tolist()):
        for method in methods:
            rows = aligned[method]
            sample_rows = [rows[index] for index in indexes]
            result = reduction_from_sample(
                np.asarray([int(row["failure_label"]) for row in sample_rows], dtype=np.int64),
                np.asarray([float(row["logistic_probability"]) for row in sample_rows], dtype=np.float64),
                float(thresholds[method]),
            )
            reductions[method].append(result["risk_reduction"])
            replicates.append({"replicate": replicate, "comparison": method, "method": method, **result})
        for left, right in (("blackhat", "otsu"), ("blackhat", "adaptive")):
            if left not in reductions or right not in reductions:
                continue
            first = reductions[left][-1]
            second = reductions[right][-1]
            difference = None if first is None or second is None else float(first) - float(second)
            replicates.append(
                {
                    "replicate": replicate,
                    "comparison": f"{left}_minus_{right}",
                    "method": "",
                    "n": len(stems),
                    "n_accepted": None,
                    "n_failures": None,
                    "n_accepted_failures": None,
                    "n_abstained_failures": None,
                    "always_accept_risk": None,
                    "accepted_failure_risk": None,
                    "risk_reduction": difference,
                    "defined": difference is not None,
                }
            )
    summaries = []
    for comparison in [method for method in methods] + [
        name
        for name in ("blackhat_minus_otsu", "blackhat_minus_adaptive")
        if "blackhat" in methods
    ]:
        values = [
            float(row["risk_reduction"])
            for row in replicates
            if row["comparison"] == comparison and row["risk_reduction"] is not None
        ]
        undefined = sum(1 for row in replicates if row["comparison"] == comparison and row["risk_reduction"] is None)
        summaries.append(
            {
                "comparison": comparison,
                "n_replicates": n_replicates,
                "n_defined": len(values),
                "n_undefined": undefined,
                "seed": seed,
                "interval": "percentile_2.5_97.5",
                "p2_5": linear_percentile(values, 2.5) if values else None,
                "median": linear_percentile(values, 50) if values else None,
                "p97_5": linear_percentile(values, 97.5) if values else None,
                "fraction_positive": (sum(1 for value in values if value > 0) / len(values)) if values else None,
            }
        )
    return replicates, summaries


def sensitivity_outcomes(
    rows: list[dict[str, object]],
    score_key: str,
    threshold: float,
    offsets: tuple[float, ...] = SENSITIVITY_OFFSETS,
) -> list[dict[str, object]]:
    """Apply fixed probability offsets. The test rows do not choose an offset."""
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    outcomes = []
    for offset in offsets:
        shifted = float(threshold) + float(offset)
        outcome = outcome_at_threshold(rows, scores, shifted)
        outcome["offset"] = float(offset)
        outcome["base_threshold"] = float(threshold)
        outcomes.append(outcome)
    return outcomes


def _format_metric(value: object) -> str:
    if value is None or value == "":
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report(result: dict[str, object]) -> str:
    """Separate the nested training check from follow-up descriptions of the test split."""
    lines = [
        "Robustness and statistical validation of selective segmentation",
        "",
        "The DeepCrack test split was already examined in CS-006. Comparisons",
        "that use test labels in this report are follow-up analyses. They are",
        "not a pristine confirmation and they are not a significance test.",
        "",
        "Protocol fixed before this run",
        str(result["protocol"]),
        "",
        "Saved CS-005 out-of-fold probabilities were not used to select the",
        "outer-fold thresholds. Those probabilities remain the CS-006 scores",
        "for the frozen-threshold bootstrap, the sensitivity grid, and the",
        "risk-coverage curves.",
        "",
        "Outer-fold threshold selection, training images only",
    ]
    for method in result["methods"]:
        summary = result["variability"][method]
        lines.append(
            f"  {method}: justified folds {summary['n_justified']} of {summary['n_folds']}, "
            f"ineligible {summary['n_ineligible']}."
        )
        lines.append(
            "    Reduction min/median/max "
            f"{_format_metric(summary['reduction_min'])} / "
            f"{_format_metric(summary['reduction_median'])} / "
            f"{_format_metric(summary['reduction_max'])}."
        )
        lines.append(
            "    Coverage min/median/max "
            f"{_format_metric(summary['coverage_min'])} / "
            f"{_format_metric(summary['coverage_median'])} / "
            f"{_format_metric(summary['coverage_max'])}."
        )
        lines.append(
            "    Accepted risk min/median/max "
            f"{_format_metric(summary['risk_min'])} / "
            f"{_format_metric(summary['risk_median'])} / "
            f"{_format_metric(summary['risk_max'])}."
        )
        for record in result["outer_folds"]:
            if record["method"] != method:
                continue
            lines.append(
                f"    fold {record['fold']}: status {record['status']}, "
                f"threshold {_format_metric(record.get('threshold'))}, "
                f"n {record.get('n_outer')}, coverage {_format_metric(record.get('coverage'))}, "
                f"accepted risk {_format_metric(record.get('accepted_failure_risk'))}, "
                f"always-accept risk {_format_metric(record.get('always_accept_risk'))}, "
                f"reduction {_format_metric(record.get('absolute_risk_reduction'))}, "
                f"accepted failures {_format_metric(record.get('n_accepted_failures'))}, "
                f"abstained failures {_format_metric(record.get('n_abstained_failures'))}."
            )
    lines.extend(
        [
            "",
            "The five fold reductions share training images, so their spread is a",
            "description of this partition. It is not five independent experiments",
            "and it is not a confidence interval.",
            "",
            "Matched coverage on frozen scores",
            "Targets were fixed at 0.50, 0.60, 0.70, 0.80, 0.90, and 1.00.",
            "Training rows use CS-005 out-of-fold probabilities. Test rows use the",
            "full-training logistic probabilities. Neither curve chooses a new threshold.",
            "",
        ]
    )
    for split in ("train", "test"):
        role = "selection-score description" if split == "train" else "follow-up on the previously examined test split"
        lines.append(f"  {split} ({role})")
        for coverage in MATCHED_COVERAGES:
            pieces = []
            for method in result["methods"]:
                point = _matched_point(result["matched"], method, split, coverage)
                if point is None:
                    continue
                pieces.append(
                    f"{method} risk {_format_metric(point['accepted_failure_risk'])} "
                    f"({point['n_accepted_failures']}/{point['n_accepted']} accepted failures, "
                    f"{point['n_failures']} failures in the split)"
                )
            lines.append(f"    coverage {coverage:.2f}: " + "; ".join(pieces))
        if split == "test":
            lines.append("    " + _coverage_ordering_sentence(result["matched"], "test"))
    lines.extend(["", "Fixed-threshold paired bootstrap on the previously examined test split"])
    for summary in result["bootstrap_summary"]:
        lines.append(
            f"  {summary['comparison']}: defined replicates {summary['n_defined']} of {summary['n_replicates']}, "
            f"undefined {summary['n_undefined']}, "
            f"2.5/median/97.5 {_format_metric(summary['p2_5'])} / {_format_metric(summary['median'])} / "
            f"{_format_metric(summary['p97_5'])}, "
            f"fraction of defined replicates above zero {_format_metric(summary['fraction_positive'])}."
        )
    lines.extend(
        [
            "  These percentile intervals hold the CS-006 thresholds fixed.",
            "  They do not include threshold-selection uncertainty.",
            "  A fraction of replicates is not a p-value, and separate intervals",
            "  are not a substitute for the paired differences above.",
            "  Otsu has 15 test failures, so its interval is wide.",
            "",
            "Sensitivity around the frozen CS-006 thresholds",
            "Offsets are -0.10, -0.05, 0, +0.05, and +0.10. No offset was chosen from the test results.",
        ]
    )
    for method in result["methods"]:
        lines.append(f"  {method}")
        for row in result["sensitivity"]:
            if row["method"] != method:
                continue
            lines.append(
                f"    {row['split']} offset {float(row['offset']):+.2f}: "
                f"threshold {_format_metric(row['threshold'])}, "
                f"coverage {_format_metric(row['coverage'])}, "
                f"accepted risk {_format_metric(row['accepted_failure_risk'])}, "
                f"reduction {_format_metric(row['absolute_risk_reduction'])}, "
                f"accepted failures {_format_metric(row['n_accepted_failures'])}."
            )
    lines.extend(["", "Random abstention at the frozen test coverage"])
    for method, summary in result["random_summary"].items():
        lines.append(
            f"  {method}: trials {summary['n_trials']}, mean risk {_format_metric(summary['risk_mean'])}, "
            f"2.5/median/97.5 {_format_metric(summary['risk_p2_5'])} / "
            f"{_format_metric(summary['risk_median'])} / {_format_metric(summary['risk_p97_5'])}, "
            f"frozen policy risk {_format_metric(summary.get('policy_accepted_risk'))}, "
            f"trials with lower risk than the policy {summary.get('n_trials_lower_risk_than_policy')}."
        )
    lines.append("  The trial counts are descriptions of these fixed-coverage draws. They are not significance tests.")
    lines.extend(
        [
            "",
            "Observations and interpretations",
            _stability_sentence(result),
            _outer_fold_comparison_sentence(result),
            _coverage_ordering_sentence(result["matched"], "test"),
            _accepted_risk_sentence(result["matched"], "test"),
            _bootstrap_sentence(result),
            "Test matched-coverage curves, the paired bootstrap, sensitivity on the test split,",
            "and the random-abstention comparison are exploratory follow-ups because CS-006",
            "already used this test split. The outer-fold procedure did not use the test split.",
            "A generalization claim would need a dataset or split that was not used to design",
            "or inspect this abstention rule, plus a validation that repeats threshold selection.",
            "",
            "Limitations",
            "The nested refit uses the CS-005 logistic specification. It is not a new model class.",
            "Inner and outer folds share images across different outer iterations, so fold",
            "estimates are dependent.",
            "The frozen-threshold bootstrap does not refit the reliability model or reselect",
            "the threshold inside each replicate.",
            "Undefined F1 remains an operational failure. An empty accepted set leaves risk undefined.",
            "No result here measures structural safety.",
            "",
        ]
    )
    return "\n".join(lines)


def _matched_point(
    rows: list[dict[str, object]],
    method: str,
    split: str,
    coverage: float,
) -> dict[str, object] | None:
    matches = [
        row
        for row in rows
        if row["method"] == method and row["split"] == split and abs(float(row["target_coverage"]) - coverage) < 1e-12
    ]
    return matches[0] if matches else None


def _coverage_ordering_sentence(rows: list[dict[str, object]], split: str) -> str:
    targets = [coverage for coverage in MATCHED_COVERAGES if coverage < 1.0]
    ahead = 0
    comparable = 0
    for coverage in targets:
        blackhat = _matched_point(rows, "blackhat", split, coverage)
        otsu = _matched_point(rows, "otsu", split, coverage)
        adaptive = _matched_point(rows, "adaptive", split, coverage)
        if blackhat is None or otsu is None or adaptive is None:
            continue
        if (
            blackhat["absolute_risk_reduction"] is None
            or otsu["absolute_risk_reduction"] is None
            or adaptive["absolute_risk_reduction"] is None
        ):
            continue
        comparable += 1
        if float(blackhat["absolute_risk_reduction"]) > float(otsu["absolute_risk_reduction"]) and float(
            blackhat["absolute_risk_reduction"]
        ) > float(adaptive["absolute_risk_reduction"]):
            ahead += 1
    return (
        f"On the {split} matched-coverage grid below full coverage, black-hat has a larger "
        f"risk reduction than both other methods at {ahead} of {comparable} comparable targets. "
        "This count is descriptive. It does not select an operating point."
    )


def _stability_sentence(result: dict[str, object]) -> str:
    pieces = []
    for method, summary in result["variability"].items():
        reductions = [
            float(record["absolute_risk_reduction"])
            for record in result["outer_folds"]
            if record["method"] == method
            and record["fold"] != "pooled"
            and record.get("absolute_risk_reduction") is not None
        ]
        below_rule = sum(1 for value in reductions if value < MINIMUM_ABSOLUTE_RISK_REDUCTION)
        pieces.append(
            f"{method} justified {summary['n_justified']}/{summary['n_folds']} folds, "
            f"reduction range {_format_metric(summary['reduction_min'])} to {_format_metric(summary['reduction_max'])}, "
            f"{below_rule} justified {'fold' if below_rule == 1 else 'folds'} below 0.05"
        )
    minima = [float(summary["reduction_min"]) for summary in result["variability"].values() if summary["reduction_min"] is not None]
    if minima and min(minima) > 0:
        closing = "Every justified outer-fold reduction in this run is positive, and several fall below the 0.05 rule used to select the threshold."
    else:
        closing = "At least one justified outer-fold reduction is zero or negative, or no justified reduction was available."
    return "Outer-fold selection stability: " + "; ".join(pieces) + ". " + closing


def _outer_fold_comparison_sentence(result: dict[str, object]) -> str:
    folds = sorted(
        {record["fold"] for record in result["outer_folds"] if record["fold"] != "pooled"},
        key=lambda fold: int(fold),
    )
    ahead = 0
    comparable = 0
    for fold in folds:
        by_method = {
            record["method"]: record
            for record in result["outer_folds"]
            if record["fold"] == fold
        }
        needed = ("otsu", "adaptive", "blackhat")
        if any(method not in by_method or by_method[method].get("absolute_risk_reduction") is None for method in needed):
            continue
        comparable += 1
        blackhat = float(by_method["blackhat"]["absolute_risk_reduction"])
        if blackhat > float(by_method["otsu"]["absolute_risk_reduction"]) and blackhat > float(
            by_method["adaptive"]["absolute_risk_reduction"]
        ):
            ahead += 1
    if comparable == 0:
        return "Outer folds do not contain a complete three-method reduction comparison."
    return (
        f"Within the same outer fold, black-hat's reduction exceeds both other methods in {ahead} of {comparable} folds. "
        "Those folds share training images, so the count is not an independent comparison."
    )


def _accepted_risk_sentence(rows: list[dict[str, object]], split: str) -> str:
    targets = [coverage for coverage in MATCHED_COVERAGES if coverage < 1.0]
    lower_risk = 0
    larger_reduction = 0
    comparable = 0
    for coverage in targets:
        blackhat = _matched_point(rows, "blackhat", split, coverage)
        otsu = _matched_point(rows, "otsu", split, coverage)
        if (
            blackhat is None
            or otsu is None
            or blackhat["accepted_failure_risk"] is None
            or otsu["accepted_failure_risk"] is None
            or blackhat["absolute_risk_reduction"] is None
            or otsu["absolute_risk_reduction"] is None
        ):
            continue
        comparable += 1
        if float(blackhat["accepted_failure_risk"]) < float(otsu["accepted_failure_risk"]):
            lower_risk += 1
        if float(blackhat["absolute_risk_reduction"]) > float(otsu["absolute_risk_reduction"]):
            larger_reduction += 1
    if comparable == 0:
        return f"The {split} matched-coverage grid does not support an Otsu versus black-hat comparison."
    return (
        f"On the {split} grid below full coverage, black-hat has a larger risk reduction than Otsu at "
        f"{larger_reduction} of {comparable} targets and a lower accepted-image risk than Otsu at "
        f"{lower_risk} of {comparable} targets. Reduction and accepted risk answer different questions "
        "because the methods do not have the same always-accept failure rate."
    )


def _bootstrap_sentence(result: dict[str, object]) -> str:
    by_name = {row["comparison"]: row for row in result["bootstrap_summary"]}
    pieces = []
    for name in ("otsu", "adaptive", "blackhat", "blackhat_minus_otsu", "blackhat_minus_adaptive"):
        row = by_name.get(name)
        if row is None or row["p2_5"] is None or row["p97_5"] is None:
            continue
        if float(row["p2_5"]) > 0:
            location = "lies above zero"
        elif float(row["p97_5"]) < 0:
            location = "lies below zero"
        else:
            location = "includes zero"
        pieces.append(f"{name} interval {location}")
    return (
        "Paired fixed-threshold bootstrap: "
        + "; ".join(pieces)
        + ". These statements describe the previously examined test split with thresholds held fixed."
    )


def run_robustness(
    training_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    thresholds: dict[str, float],
    fold_by_stem: dict[str, int],
    *,
    methods: tuple[str, ...] = METHODS,
    n_inner_folds: int = N_INNER_FOLDS,
    inner_seed: int = INNER_SEED,
    n_bootstrap: int = N_BOOTSTRAP,
    n_random_trials: int = N_RANDOM_TRIALS,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Run the nested training check before any follow-up test summary is built."""
    outer = outer_fold_validation(
        training_rows,
        fold_by_stem,
        methods=methods,
        n_inner_folds=n_inner_folds,
        inner_seed=inner_seed,
    )
    variability = {method: fold_variability(outer, method) for method in methods}
    matched: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    for method in methods:
        for split, score_key in (
            ("train", "internal_logistic_probability"),
            ("test", "logistic_probability"),
        ):
            rows = [row for row in prediction_rows if row["method"] == method and row["split"] == split]
            scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
            for coverage in MATCHED_COVERAGES:
                outcome = matched_coverage_outcome(rows, scores, coverage)
                matched.append({"method": method, "split": split, "score_source": score_key, **outcome})
            for outcome in sensitivity_outcomes(rows, score_key, float(thresholds[method])):
                sensitivity.append({"method": method, "split": split, "score_source": score_key, **outcome})
    test_by_method = {
        method: [row for row in prediction_rows if row["method"] == method and row["split"] == "test"]
        for method in methods
    }
    replicates, bootstrap_summary = paired_bootstrap(
        test_by_method,
        {method: float(thresholds[method]) for method in methods},
        n_replicates=n_bootstrap,
        seed=seed,
    )
    random_summary = {}
    for method in methods:
        test_rows = test_by_method[method]
        frozen = outcome_at_threshold(
            test_rows,
            np.asarray([float(row["logistic_probability"]) for row in test_rows], dtype=np.float64),
            float(thresholds[method]),
        )
        trials = random_abstention_trials(
            test_rows,
            int(frozen["n_accepted"]),
            n_trials=n_random_trials,
            seed=trial_seed(method, "test", "logistic", seed),
        )
        policy_risk = frozen["accepted_failure_risk"]
        random_summary[method] = {
            **summarize_trials(trials),
            "policy_accepted_risk": policy_risk,
            "n_trials_lower_risk_than_policy": (
                None
                if policy_risk is None
                else sum(
                    1
                    for trial in trials
                    if trial["accepted_failure_risk"] is not None
                    and float(trial["accepted_failure_risk"]) < float(policy_risk)
                )
            ),
        }
    result = {
        "protocol": PROTOCOL,
        "methods": list(methods),
        "outer_folds": outer,
        "variability": variability,
        "matched": matched,
        "sensitivity": sensitivity,
        "bootstrap_replicates": replicates,
        "bootstrap_summary": bootstrap_summary,
        "random_summary": random_summary,
        "thresholds": {method: float(thresholds[method]) for method in methods},
        "minimum_coverage": MINIMUM_COVERAGE,
        "minimum_absolute_risk_reduction": MINIMUM_ABSOLUTE_RISK_REDUCTION,
        "n_bootstrap": n_bootstrap,
        "n_random_trials": n_random_trials,
        "seed": seed,
    }
    result["report"] = build_report(result)
    return result


def save_plots(output_dir: Path, prediction_rows: list[dict[str, object]], result: dict[str, object]) -> None:
    """Draw follow-up curves and the five outer-fold reductions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crackscout.abstention_evaluation import build_curve

    for method in result["methods"]:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
        for axis, split, score_key, title in (
            (axes[0], "train", "internal_logistic_probability", "Training out-of-fold scores"),
            (axes[1], "test", "logistic_probability", "Previously examined test split"),
        ):
            rows = [row for row in prediction_rows if row["method"] == method and row["split"] == split]
            curve = build_curve(rows, score_key, float(result["thresholds"][method]))
            defined = [point for point in curve if int(point["risk_defined"]) == 1]
            defined = sorted(defined, key=lambda point: float(point["coverage"]))
            if defined:
                axis.plot(
                    [float(point["coverage"]) for point in defined],
                    [float(point["accepted_failure_risk"]) for point in defined],
                    color="#1f4e79",
                    linewidth=1.8,
                )
                axis.axhline(
                    float(defined[-1]["baseline_prevalence"]),
                    color="#666666",
                    linestyle="--",
                    linewidth=1,
                    label="Always accept",
                )
                chosen = [point for point in defined if int(point["is_operating_point"]) == 1]
                if chosen:
                    axis.scatter(
                        [float(chosen[0]["coverage"])],
                        [float(chosen[0]["accepted_failure_risk"])],
                        color="#b45309",
                        s=46,
                        zorder=3,
                        label="CS-006 threshold",
                    )
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_xlabel("Coverage")
            axis.set_title(title, fontsize=10)
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8, loc="best")
        axes[0].set_ylabel("Operational failure risk among accepted images")
        figure.suptitle(f"{method}: frozen scores, threshold not reselected", fontsize=12)
        figure.tight_layout()
        try:
            figure.savefig(output_dir / f"risk_coverage_{method}.png", dpi=120, bbox_inches="tight")
        finally:
            plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for index, method in enumerate(result["methods"]):
        reductions = [
            float(record["absolute_risk_reduction"])
            for record in result["outer_folds"]
            if record["method"] == method
            and record["fold"] != "pooled"
            and record.get("absolute_risk_reduction") is not None
        ]
        axis.scatter([index] * len(reductions), reductions, color="#1f4e79", s=36, zorder=3)
        ineligible = sum(
            1
            for record in result["outer_folds"]
            if record["method"] == method and record["fold"] != "pooled" and not record["justified"]
        )
        if ineligible:
            axis.scatter([index], [-0.02], marker="x", color="#9b1c1c", label="ineligible" if index == 0 else None)
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
    axis.axhline(MINIMUM_ABSOLUTE_RISK_REDUCTION, color="#b45309", linestyle=":", linewidth=1, label="0.05 training rule")
    axis.set_xticks(range(len(result["methods"])))
    axis.set_xticklabels(list(result["methods"]))
    axis.set_ylabel("Outer-fold absolute risk reduction")
    axis.set_title("Training outer folds; dependent estimates")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    try:
        figure.savefig(output_dir / "outer_fold_reductions.png", dpi=120, bbox_inches="tight")
    finally:
        plt.close(figure)


def write_outputs(
    output_dir: str | Path,
    result: dict[str, object],
    prediction_rows: list[dict[str, object]] | None = None,
) -> None:
    """Write the robustness files. Existing planned files are left unchanged."""
    destination = Path(output_dir)
    names = [
        "report.txt",
        "outer_fold_results.csv",
        "operating_point_stability.csv",
        "risk_coverage.csv",
        "paired_bootstrap.csv",
        "sensitivity_analysis.csv",
        "config.json",
        "outer_fold_reductions.png",
        *[f"risk_coverage_{method}.png" for method in result["methods"]],
    ]
    planned = [destination / name for name in names]
    existing = [path for path in planned if path.exists()]
    if existing:
        listed = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite robustness-evaluation files:\n"
            f"{listed}\n"
            "Earlier experiment outputs were not changed."
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.txt").write_text(str(result["report"]), encoding="utf-8")
    _write_csv(destination / "outer_fold_results.csv", [_public_record(row) for row in result["outer_folds"]])
    _write_csv(destination / "operating_point_stability.csv", list(result["variability"].values()))
    _write_csv(destination / "risk_coverage.csv", result["matched"])
    _write_csv(
        destination / "paired_bootstrap.csv",
        result["bootstrap_summary"] + result["bootstrap_replicates"],
    )
    _write_csv(destination / "sensitivity_analysis.csv", result["sensitivity"])
    config = {
        "protocol": result["protocol"],
        "methods": result["methods"],
        "frozen_cs006_thresholds": result["thresholds"],
        "minimum_coverage": result["minimum_coverage"],
        "minimum_absolute_risk_reduction": result["minimum_absolute_risk_reduction"],
        "matched_coverages": list(MATCHED_COVERAGES),
        "sensitivity_offsets": list(SENSITIVITY_OFFSETS),
        "n_bootstrap": result["n_bootstrap"],
        "n_random_trials": result["n_random_trials"],
        "bootstrap_interval": "percentile_2.5_97.5",
        "random_seed": result["seed"],
        "test_split_previously_examined_in_cs006": True,
        "cs005_outputs_modified": False,
        "nested_refit_uses_cs005_logistic_specification": True,
        "saved_oof_probabilities_used_for_threshold_selection": False,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if prediction_rows is not None:
        save_plots(destination, prediction_rows, result)


def _public_record(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"selection_stems", "outer_scores"}}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames and field not in {"selection_stems", "outer_scores"}:
                fieldnames.append(field)
    if not fieldnames:
        fieldnames = ["method"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
