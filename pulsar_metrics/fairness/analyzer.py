"""Group fairness analysis with confidence intervals, small-cell suppression and intersectional groups."""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults
from .criteria import CRITERIA, DEFAULT_CRITERIA, EQUALIZED_ODDS, verdict


@dataclass
class FairnessPolicy:
    """Which criteria to evaluate and how to judge them

    Parameters
    ----------
    criteria : sequence of str
        Criteria to evaluate (see `CRITERIA`, plus "equalized_odds_difference"). Prediction-based criteria are the
        default; label-based ones (equal opportunity, false positive rate, calibration, error rate) must be chosen
        explicitly: labels arrive late and carry the biases of past decisions. Calibration and equalized odds cannot
        both hold when base rates differ: choose, and document the choice.
    acceptable : dict, optional
        Acceptable zone [low, high] per criterion, overriding the defaults (e.g. {"disparate_impact_ratio": (0.8, 1.25)})
    min_group_size : int, optional
        Groups with fewer rows are suppressed: no value, no count is published (default 30)
    confidence : float, optional
        Confidence level of the intervals (default 0.95)
    n_bootstrap : int, optional
        Bootstrap replicates (default 500; 0 disables the intervals and judges point estimates)
    simultaneous : bool, optional
        Simultaneous intervals over all groups of a criterion (max-t bootstrap band), so that the confidence level
        holds for all groups at once (default True)
    """

    criteria: Sequence[str] = DEFAULT_CRITERIA
    acceptable: dict = field(default_factory=dict)
    min_group_size: int = 30
    confidence: float = 0.95
    n_bootstrap: int = 500
    simultaneous: bool = True

    def __post_init__(self):
        unknown = [c for c in self.criteria if c not in CRITERIA and c != EQUALIZED_ODDS]
        if unknown:
            raise error_msg(
                value=unknown, message=f"unknown fairness criteria {unknown}; available: {sorted(CRITERIA) + [EQUALIZED_ODDS]}"
            )
        if self.min_group_size < 1 or not 0.5 < self.confidence < 1:
            raise error_msg(value=None, message="min_group_size >= 1 and 0.5 < confidence < 1 are required")

    def zone(self, criterion: str) -> tuple:
        if criterion in self.acceptable:
            return tuple(self.acceptable[criterion])
        return (0.0, 0.1) if criterion == EQUALIZED_ODDS else CRITERIA[criterion].acceptable


class FairnessAnalyzer:
    """Compare a decision rule across groups defined by protected attributes

    Parameters
    ----------
    attributes : sequence of str
        Protected attribute columns (e.g. ["sex", "age_group"])
    policy : FairnessPolicy, optional
    reference_groups : dict, optional
        Reference group per attribute (default: the largest group)
    intersections : bool, optional
        Also evaluate the groups formed by every combination of attributes (default False)
    y_pred, y_true, score : str, optional
        Columns of the decision (0/1, or the prediction for regression), of the observed outcome and of the score
    seed : int, optional
        Random seed of the bootstrap

    Protected attributes need not travel with the predictions: pass them to `run(attributes_data=..., on="id")`,
    the join happens in memory and only group-level results are returned.
    """

    def __init__(
        self,
        attributes: Sequence[str],
        policy: Optional[FairnessPolicy] = None,
        reference_groups: Optional[dict] = None,
        intersections: bool = False,
        y_pred: str = "y_pred",
        y_true: str = "y_true",
        score: str = "y_pred_proba",
        seed: int = 0,
    ):
        if not attributes:
            raise error_msg(value=None, message="at least one protected attribute is required")
        self.attributes = list(attributes)
        self.policy = policy or FairnessPolicy()
        self.reference_groups = reference_groups or {}
        self.intersections = intersections
        self.columns = {"y_pred": y_pred, "y_true": y_true, "score": score}
        self.seed = seed
        self._results: list[MetricResults] = []

    # ------------------------------------------------------------------ public API
    def run(self, data: pd.DataFrame, attributes_data: Optional[pd.DataFrame] = None, on: Optional[str] = None) -> list:
        if attributes_data is not None:
            if on is None:
                raise error_msg(value=None, message="`on` (pseudonymous id column) is required with attributes_data")
            data = data.merge(attributes_data[[on, *self.attributes]], on=on, how="inner", validate="one_to_one")
        missing = [c for c in [self.columns["y_pred"], *self.attributes] if c not in data.columns]
        if missing:
            raise error_msg(value=missing, message=f"missing columns {missing}")
        self._results = []
        rng = np.random.default_rng(self.seed)
        for attrs in self._attribute_sets():
            self._results += self._evaluate(data, list(attrs), rng)
        return self._results

    def get_result(self) -> list:
        return self._results

    def results_to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self._results])

    # ------------------------------------------------------------------ internals
    def _attribute_sets(self):
        sets = [(a,) for a in self.attributes]
        if self.intersections and len(self.attributes) > 1:
            sets += [c for k in range(2, len(self.attributes) + 1) for c in combinations(self.attributes, k)]
        return sets

    def _arrays(self, frame: pd.DataFrame) -> tuple:
        y_pred = frame[self.columns["y_pred"]].to_numpy(dtype=float)
        y_true = frame[self.columns["y_true"]].to_numpy(dtype=float) if self.columns["y_true"] in frame else None
        score = frame[self.columns["score"]].to_numpy(dtype=float) if self.columns["score"] in frame else None
        return y_pred, y_true, score

    def _evaluate(self, data: pd.DataFrame, attrs: list, rng) -> list:
        feature = "|".join(attrs)
        keys = data[attrs].astype(str).agg("|".join, axis=1) if len(attrs) > 1 else data[attrs[0]].astype(str)
        labels = {k: "|".join(f"{a}={v}" for a, v in zip(attrs, k.split("|"))) for k in keys.unique()}
        sizes = keys.value_counts()
        kept = sizes[sizes >= self.policy.min_group_size].index.tolist()
        suppressed = sizes[sizes < self.policy.min_group_size].index.tolist()

        reference = self._reference(feature, attrs, sizes, kept, labels)
        groups = {k: self._arrays(data[keys == k]) for k in kept}
        results = []
        for name in self.policy.criteria:
            if not self._computable(name, groups[reference]):
                results.append(self._skipped(name, feature, labels[reference]))
                continue
            estimates, intervals, method = self._estimate(name, groups, reference, rng)
            zone = self.policy.zone(name)
            for k in kept:
                if k == reference:
                    continue
                low, high = intervals.get(k, (None, None))
                results.append(
                    MetricResults(
                        metric_type="fairness",
                        metric_name=name,
                        feature_name=feature,
                        group=labels[k],
                        reference_group=labels[reference],
                        n=int(sizes[k]),
                        metric_value=estimates[k],
                        ci_low=low,
                        ci_high=high,
                        threshold=list(zone),
                        drift_status=verdict(low, high, estimates[k], zone),
                        method=method,
                        test_type="fixed",
                    )
                )
            for k in suppressed:
                results.append(
                    MetricResults(
                        metric_type="fairness",
                        metric_name=name,
                        feature_name=feature,
                        group=labels[k],
                        reference_group=labels[reference],
                        suppressed=True,
                        method=f"suppressed: fewer than {self.policy.min_group_size} rows",
                        test_type="fixed",
                    )
                )
        return results

    def _reference(self, feature, attrs, sizes, kept, labels):
        if not kept:
            raise error_msg(value=feature, message=f"every group of {feature} is below min_group_size")
        wanted = self.reference_groups.get(feature) or (self.reference_groups.get(attrs[0]) if len(attrs) == 1 else None)
        if wanted is None:
            return sizes[kept].idxmax()
        if str(wanted) not in kept:
            raise error_msg(value=wanted, message=f"reference group {wanted} of {feature} is missing or suppressed")
        return str(wanted)

    def _computable(self, name: str, arrays: tuple) -> bool:
        _, y_true, score = arrays
        needs_labels = name == EQUALIZED_ODDS or CRITERIA[name].needs_labels
        needs_scores = name != EQUALIZED_ODDS and CRITERIA[name].needs_scores
        return not ((needs_labels and y_true is None) or (needs_scores and score is None))

    def _skipped(self, name, feature, reference_label):
        return MetricResults(
            metric_type="fairness",
            metric_name=name,
            feature_name=feature,
            reference_group=reference_label,
            error=f"{name} needs {'labels and scores' if name in ('calibration_difference',) else 'labels'}",
            test_type="fixed",
        )

    def _disparity(self, name: str, group: tuple, reference: tuple) -> float:
        if name == EQUALIZED_ODDS:
            tpr = CRITERIA["equal_opportunity_difference"]
            fpr = CRITERIA["false_positive_rate_difference"]
            return max(
                abs(tpr.compare(tpr.statistic(*group), tpr.statistic(*reference))),
                abs(fpr.compare(fpr.statistic(*group), fpr.statistic(*reference))),
            )
        criterion = CRITERIA[name]
        return float(criterion.compare(criterion.statistic(*group), criterion.statistic(*reference)))

    def _estimate(self, name: str, groups: dict, reference, rng):
        others = [k for k in groups if k != reference]
        estimates = {k: self._disparity(name, groups[k], groups[reference]) for k in others}
        b = self.policy.n_bootstrap
        if b == 0 or not others:
            return estimates, {}, "point estimate (no interval)"

        def resample(arrays):
            idx = rng.integers(0, len(arrays[0]), len(arrays[0]))
            return tuple(None if a is None else a[idx] for a in arrays)

        draws = np.empty((b, len(others)))
        for i in range(b):
            ref = resample(groups[reference])  # stratified bootstrap: each group resampled within itself
            draws[i] = [self._disparity(name, resample(groups[k]), ref) for k in others]

        alpha = 1 - self.policy.confidence
        if self.policy.simultaneous and len(others) > 1:
            point = np.array([estimates[k] for k in others])
            se = np.nanstd(draws, axis=0, ddof=1)
            se[se == 0] = np.nan
            t_max = np.nanmax(np.abs(draws - point) / se, axis=1)
            q = float(np.nanquantile(t_max, 1 - alpha))
            intervals = {k: (float(point[j] - q * se[j]), float(point[j] + q * se[j])) for j, k in enumerate(others)}
            intervals = {k: (estimates[k], estimates[k]) if np.isnan(v[0]) else v for k, v in intervals.items()}
            method = f"stratified bootstrap ({b}), simultaneous {self.policy.confidence:.0%} max-t band over {len(others)} groups"
        else:
            low, high = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
            intervals = {k: (float(low[j]), float(high[j])) for j, k in enumerate(others)}
            method = f"stratified bootstrap ({b}), {self.policy.confidence:.0%} percentile interval"
        return estimates, intervals, method
