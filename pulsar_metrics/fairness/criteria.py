"""Group fairness criteria: per-group statistics and the disparity of each group against a reference group."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Criterion:
    """A fairness criterion comparing one group with the reference group

    `statistic` computes the group statistic from the arrays of the group (y_pred, y_true, score);
    `compare` turns (group statistic, reference statistic) into the disparity; `acceptable` is the default
    acceptable zone [low, high] of the disparity.
    """

    name: str
    description: str
    statistic: Callable
    compare: Callable
    acceptable: tuple
    needs_labels: bool = False
    needs_scores: bool = False


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def selection_rate(y_pred, y_true=None, score=None) -> float:
    return _mean(y_pred)


def true_positive_rate(y_pred, y_true, score=None) -> float:
    positives = y_true == 1
    return _mean(y_pred[positives])


def false_positive_rate(y_pred, y_true, score=None) -> float:
    negatives = y_true == 0
    return _mean(y_pred[negatives])


def expected_calibration_error(y_pred, y_true, score, bins: int = 10) -> float:
    """Expected calibration error of the scores, on equal-width bins"""
    if len(score) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(score, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        in_bin = index == b
        if in_bin.any():
            ece += in_bin.mean() * abs(score[in_bin].mean() - y_true[in_bin].mean())
    return float(ece)


def mean_absolute_error(y_pred, y_true, score=None) -> float:
    return _mean(np.abs(y_pred - y_true))


def _ratio(group: float, reference: float) -> float:
    return group / reference if reference else float("nan")


def _difference(group: float, reference: float) -> float:
    return group - reference


CRITERIA = {
    c.name: c
    for c in [
        Criterion(
            "disparate_impact_ratio",
            "selection rate of the group divided by the selection rate of the reference group (4/5 rule)",
            selection_rate,
            _ratio,
            (0.8, 1.25),
        ),
        Criterion(
            "demographic_parity_difference",
            "selection rate of the group minus the selection rate of the reference group",
            selection_rate,
            _difference,
            (-0.1, 0.1),
        ),
        Criterion(
            "equal_opportunity_difference",
            "true positive rate of the group minus that of the reference group (needs labels)",
            true_positive_rate,
            _difference,
            (-0.1, 0.1),
            needs_labels=True,
        ),
        Criterion(
            "false_positive_rate_difference",
            "false positive rate of the group minus that of the reference group (needs labels)",
            false_positive_rate,
            _difference,
            (-0.1, 0.1),
            needs_labels=True,
        ),
        Criterion(
            "calibration_difference",
            "expected calibration error of the group minus that of the reference group (needs labels and scores)",
            expected_calibration_error,
            _difference,
            (-0.05, 0.05),
            needs_labels=True,
            needs_scores=True,
        ),
        Criterion(
            "error_rate_difference",
            "mean absolute error of the group minus that of the reference group (regression, needs labels)",
            mean_absolute_error,
            _difference,
            (-np.inf, np.inf),  # no sensible default: the unit is the target's, set it in the policy
            needs_labels=True,
        ),
    ]
}

# Equalized odds combines two criteria: max(|TPR difference|, |FPR difference|)
EQUALIZED_ODDS = "equalized_odds_difference"
DEFAULT_CRITERIA = ("disparate_impact_ratio", "demographic_parity_difference")


def verdict(ci_low: Optional[float], ci_high: Optional[float], value: float, acceptable: tuple) -> Optional[bool]:
    """True = disparity established (interval entirely outside the acceptable zone), False = within the zone,
    None = inconclusive (the interval straddles a bound) or not computable"""
    low, high = acceptable
    if value is None or np.isnan(value):
        return None
    lo, hi = (value, value) if ci_low is None or ci_high is None else (ci_low, ci_high)
    if hi < low or lo > high:
        return True
    if lo >= low and hi <= high:
        return False
    return None
