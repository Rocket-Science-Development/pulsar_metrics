"""Group fairness analysis with confidence intervals, small-cell suppression and intersectional groups."""

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults
from .criteria import CRITERIA, DEFAULT_CRITERIA, EQUALIZED_ODDS, equivalence_verdict
from .planning import rate_interval, required_group_size

# criteria whose group statistic is a rate of binary decisions: exact binomial bootstrap and score intervals apply
_RATES = {
    "disparate_impact_ratio": (None, "ratio"),
    "demographic_parity_difference": (None, "difference"),
    "equal_opportunity_difference": (1, "difference"),  # rate among the rows with y_true == 1
    "false_positive_rate_difference": (0, "difference"),  # rate among the rows with y_true == 0
}
_INTERVALS = ("auto", "bootstrap", "score")


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
        holds for all groups at once (default True). The band is about twice as wide as per-group intervals with
        hundreds of cells (lab FA-01: 2.2x over 609 cells); declare the cells of interest
        (`FairnessAnalyzer(declared_groups=...)`) to narrow it.
    interval : str, optional
        "auto" (default): score intervals (Wilson combined by Newcombe for differences, MOVER-R for the ratio) for
        groups under `small_group_size` rows on the rate criteria, the bootstrap otherwise; "bootstrap": always the
        bootstrap (the behaviour before 0.11, anti-conservative for small groups); "score": score intervals for every
        group on the rate criteria. Equalized odds, calibration and error rate always use the bootstrap.
    small_group_size : int, optional
        Rows under which "auto" uses score intervals (default 200)
    equivalence_confidence : float, optional
        Confidence level of the interval used to conclude "within the zone" (default: `confidence`). With 0.90, the
        "within" verdict is a TOST at 5 % per side, the usual equivalence test, while a disparity is still established
        at `confidence`. The default verdict is a TOST at 2.5 % per side.
    max_no_verdict_share : float, optional
        A warning is raised when more than this share of the rows of a feature (outside the reference group) lies in
        suppressed cells (default 0.05); `FairnessAnalyzer.coverage` reports it with the inconclusive groups
    """

    criteria: Sequence[str] = DEFAULT_CRITERIA
    acceptable: dict = field(default_factory=dict)
    min_group_size: int = 30
    confidence: float = 0.95
    n_bootstrap: int = 500
    simultaneous: bool = True
    interval: str = "auto"
    small_group_size: int = 200
    equivalence_confidence: Optional[float] = None
    max_no_verdict_share: float = 0.05

    def __post_init__(self):
        unknown = [c for c in self.criteria if c not in CRITERIA and c != EQUALIZED_ODDS]
        if unknown:
            raise error_msg(
                value=unknown, message=f"unknown fairness criteria {unknown}; available: {sorted(CRITERIA) + [EQUALIZED_ODDS]}"
            )
        if self.min_group_size < 1 or not 0.5 < self.confidence < 1:
            raise error_msg(value=None, message="min_group_size >= 1 and 0.5 < confidence < 1 are required")
        if self.interval not in _INTERVALS:
            raise error_msg(value=self.interval, message=f"interval must be one of {_INTERVALS}")
        if self.equivalence_confidence is not None and not 0.5 < self.equivalence_confidence < 1:
            raise error_msg(value=self.equivalence_confidence, message="0.5 < equivalence_confidence < 1 is required")

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
        Also evaluate the groups formed by combinations of attributes (default False)
    max_intersection_depth : int, optional
        Largest number of attributes combined (default 2: pairs). With k attributes there are 2^k - 1 combinations and
        the deeper cells are mostly suppressed (lab FA-01, 5 attributes: 65 to 75 % of the cells, 4 to 11 % of the
        rows, 30 to 80 times the run time). None combines all of them (the behaviour before 0.11).
    max_cells : int, optional
        Stop adding attribute combinations, with a warning, once this many cells have been evaluated (default None)
    declared_groups : sequence of str, optional
        Group labels declared before looking at the data (e.g. ["sex=F", "sex=F|region=North"]). Only they get a
        verdict and enter the simultaneous band, which is then narrower; the other groups are reported with their
        value and interval, drift_status None and "exploratory" in `method`
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
        max_intersection_depth: Optional[int] = 2,
        max_cells: Optional[int] = None,
        declared_groups: Optional[Sequence[str]] = None,
    ):
        if not attributes:
            raise error_msg(value=None, message="at least one protected attribute is required")
        if max_intersection_depth is not None and max_intersection_depth < 1:
            raise error_msg(value=max_intersection_depth, message="max_intersection_depth must be >= 1")
        self.attributes = list(attributes)
        self.policy = policy or FairnessPolicy()
        self.reference_groups = reference_groups or {}
        self.intersections = intersections
        self.max_intersection_depth = max_intersection_depth
        self.max_cells = max_cells
        self.declared_groups = None if declared_groups is None else set(declared_groups)
        self.columns = {"y_pred": y_pred, "y_true": y_true, "score": score}
        self.seed = seed
        self._results: list[MetricResults] = []
        self._coverage: list[MetricResults] = []

    # ------------------------------------------------------------------ public API
    def run(self, data: pd.DataFrame, attributes_data: Optional[pd.DataFrame] = None, on: Optional[str] = None) -> list:
        if attributes_data is not None:
            if on is None:
                raise error_msg(value=None, message="`on` (pseudonymous id column) is required with attributes_data")
            data = data.merge(attributes_data[[on, *self.attributes]], on=on, how="inner", validate="one_to_one")
        missing = [c for c in [self.columns["y_pred"], *self.attributes] if c not in data.columns]
        if missing:
            raise error_msg(value=missing, message=f"missing columns {missing}")
        self._results, self._coverage = [], []
        rng = np.random.default_rng(self.seed)
        cells = 0
        for attrs in self._attribute_sets():
            if self.max_cells is not None and cells >= self.max_cells:
                warnings.warn(f"max_cells ({self.max_cells}) reached: attribute combinations from {attrs} on are skipped")
                break
            try:
                results, n_cells = self._evaluate(data, list(attrs), rng)
            except error_msg:
                if len(attrs) == 1:
                    raise
                warnings.warn(f"every cell of {'|'.join(attrs)} is below min_group_size: combination skipped")
                continue
            self._results += results
            cells += n_cells
        return self._results

    def get_result(self) -> list:
        return self._results

    def coverage(self) -> list:
        """Share of the rows without verdict, per feature and criterion (metric_name "no_verdict_share")

        Rows of suppressed groups and of groups with an inconclusive verdict, over the rows of the feature outside the
        reference group. A fairness report silent on 10 % of the population must say so (lab FA-01)."""
        return list(self._coverage)

    def results_to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self._results])

    # ------------------------------------------------------------------ internals
    def _attribute_sets(self):
        sets = [(a,) for a in self.attributes]
        if self.intersections and len(self.attributes) > 1:
            depth = len(self.attributes) if self.max_intersection_depth is None else self.max_intersection_depth
            sets += [c for k in range(2, min(depth, len(self.attributes)) + 1) for c in combinations(self.attributes, k)]
        return sets

    def _arrays(self, frame: pd.DataFrame) -> tuple:
        y_pred = frame[self.columns["y_pred"]].to_numpy(dtype=float)
        y_true = frame[self.columns["y_true"]].to_numpy(dtype=float) if self.columns["y_true"] in frame else None
        score = frame[self.columns["score"]].to_numpy(dtype=float) if self.columns["score"] in frame else None
        return y_pred, y_true, score

    def _evaluate(self, data: pd.DataFrame, attrs: list, rng) -> tuple:
        feature = "|".join(attrs)
        keys = data[attrs[0]].astype(str)
        for a in attrs[1:]:  # vectorized concatenation (a row-wise join is slow on large frames)
            keys = keys + "|" + data[a].astype(str)
        labels = {k: "|".join(f"{a}={v}" for a, v in zip(attrs, k.split("|"))) for k in keys.unique()}
        sizes = keys.value_counts()
        kept = sizes[sizes >= self.policy.min_group_size].index.tolist()
        suppressed = sizes[sizes < self.policy.min_group_size].index.tolist()

        reference = self._reference(feature, attrs, sizes, kept, labels)
        rows = keys.reset_index(drop=True).groupby(keys.to_numpy()).indices
        columns = self._arrays(data)
        groups = {k: tuple(None if c is None else c[rows[k]] for c in columns) for k in kept}
        declared = None
        if self.declared_groups is not None:
            declared = {k for k in kept if labels[k] in self.declared_groups}
        results = []
        outside_reference = int(sizes.sum() - sizes[reference])
        hidden = int(sum(sizes[k] for k in suppressed))
        if outside_reference and hidden / outside_reference > self.policy.max_no_verdict_share:
            warnings.warn(
                f"{hidden / outside_reference:.1%} of the rows of {feature} outside the reference group are in suppressed "
                f"cells ({len(suppressed)} of {len(sizes)}), without value or verdict (see coverage())"
            )
        for name in self.policy.criteria:
            if not self._computable(name, groups[reference]):
                results.append(self._skipped(name, feature, labels[reference]))
                continue
            judged = [k for k in kept if k != reference and (declared is None or k in declared)]
            exploratory = [k for k in kept if k != reference and declared is not None and k not in declared]
            estimates, intervals, equivalence, method = self._estimate(name, groups, reference, judged, rng)
            if exploratory:
                e2, i2, _, m2 = self._estimate(name, groups, reference, exploratory, rng, simultaneous=False)
                estimates.update(e2)
                intervals.update(i2)
            zone = self.policy.zone(name)
            no_verdict = hidden
            for k in kept:
                if k == reference:
                    continue
                low, high = intervals.get(k, (None, None))
                if k in exploratory:
                    status = None
                    how = f"exploratory (not declared), no verdict; {m2}"
                else:
                    status = equivalence_verdict((low, high), equivalence.get(k, (low, high)), estimates[k], zone)
                    how = method[k] if isinstance(method, dict) else method
                    if status is None:
                        how += self._size_note(name, groups[k], groups[reference], zone, len(judged))
                if status is None:
                    no_verdict += int(sizes[k])
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
                        drift_status=status,
                        method=how,
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
            share = no_verdict / outside_reference if outside_reference else 0.0
            self._coverage.append(
                MetricResults(
                    metric_type="fairness",
                    metric_name="no_verdict_share",
                    feature_name=feature,
                    reference_group=labels[reference],
                    metric_value=share,
                    n=outside_reference,
                    threshold=self.policy.max_no_verdict_share,
                    method=f"{name}: share of the rows outside the reference group in suppressed or inconclusive "
                    f"groups; suppressed {hidden / outside_reference if outside_reference else 0:.1%} "
                    f"({len(suppressed)} cells)",
                    test_type="fixed",
                )
            )
        return results, len(sizes)

    def _size_note(self, name, group, reference, zone, n_groups) -> str:
        """For an inconclusive rate criterion: rows of the group needed for a verdict at the observed rates"""
        if name not in _RATES:
            return ""
        counts = self._counts(name, group), self._counts(name, reference)
        if counts[0] is None or counts[1] is None or counts[0][1] == 0 or counts[1][1] == 0:
            return ""
        (s1, n1), (s2, n2) = counts
        needed = required_group_size(
            s1 / n1,
            s2 / n2,
            zone,
            kind=_RATES[name][1],
            reference_size=n2,
            confidence=self.policy.equivalence_confidence or self.policy.confidence,
            n_groups=n_groups if self.policy.simultaneous else 1,
        )
        if needed is None:
            return "; inconclusive at any group size with this reference (the disparity sits near a bound of the zone)"
        return f"; inconclusive: about {needed} rows of the group for an 80 % chance of a verdict at the observed rates"

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

    # ------------------------------------------------------------------ binary rates
    @staticmethod
    def _binary(values) -> bool:
        return values is not None and bool(np.isin(values, (0.0, 1.0)).all())

    def _counts(self, name: str, arrays: tuple):
        """(successes, trials) of the rate behind a rate criterion, or None when the data is not binary"""
        y_pred, y_true, _ = arrays
        among, _ = _RATES[name]
        if not self._binary(y_pred) or (among is not None and not self._binary(y_true)):
            return None
        rows = np.ones(len(y_pred), dtype=bool) if among is None else y_true == among
        return float(y_pred[rows].sum()), int(rows.sum())

    def _binomial_rates(self, name: str, arrays: tuple, b: int, rng):
        """Bootstrap replicates of the group statistic(s) without resampling rows: for 0/1 data the row bootstrap of a
        rate is exactly binomial (counts of positives, then of decisions among them). None when not applicable."""
        y_pred, y_true, _ = arrays
        n = len(y_pred)
        if name == EQUALIZED_ODDS:
            if not (self._binary(y_pred) and self._binary(y_true)):
                return None
            positives = y_true == 1
            n1 = rng.binomial(n, positives.mean(), b)
            tpr = y_pred[positives].mean() if positives.any() else 0.0
            fpr = y_pred[~positives].mean() if (~positives).any() else 0.0
            tp, fp = rng.binomial(n1, tpr), rng.binomial(n - n1, fpr)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(n1 > 0, tp / n1, np.nan), np.where(n - n1 > 0, fp / (n - n1), np.nan)
        if name not in _RATES:
            return None
        among, _ = _RATES[name]
        if not self._binary(y_pred) or (among is not None and not self._binary(y_true)):
            return None
        if among is None:
            return (rng.binomial(n, y_pred.mean(), b) / n,)
        rows = y_true == among
        trials = rng.binomial(n, rows.mean(), b)
        rate = y_pred[rows].mean() if rows.any() else 0.0
        hits = rng.binomial(trials, rate)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.where(trials > 0, hits / trials, np.nan),)

    def _draws(self, name: str, groups: dict, reference, others: list, b: int, rng) -> np.ndarray:
        """b x len(others) bootstrap disparities: exact binomial replicates for 0/1 rates (vectorized), row resampling
        otherwise (stratified: each group resampled within itself)"""
        ref_rates = self._binomial_rates(name, groups[reference], b, rng)
        fast = ref_rates is not None and all(self._binomial_rates(name, groups[k], 1, rng) is not None for k in others)
        if fast:
            draws = np.empty((b, len(others)))
            with np.errstate(invalid="ignore", divide="ignore"):
                for j, k in enumerate(others):
                    g = self._binomial_rates(name, groups[k], b, rng)
                    if name == EQUALIZED_ODDS:
                        draws[:, j] = np.maximum(np.abs(g[0] - ref_rates[0]), np.abs(g[1] - ref_rates[1]))
                    elif _RATES[name][1] == "ratio":
                        draws[:, j] = np.where(ref_rates[0] > 0, g[0] / np.where(ref_rates[0] > 0, ref_rates[0], 1), np.nan)
                    else:
                        draws[:, j] = g[0] - ref_rates[0]
            return draws

        def resample(arrays):
            idx = rng.integers(0, len(arrays[0]), len(arrays[0]))
            return tuple(None if a is None else a[idx] for a in arrays)

        draws = np.empty((b, len(others)))
        for i in range(b):
            ref = resample(groups[reference])
            draws[i] = [self._disparity(name, resample(groups[k]), ref) for k in others]
        return draws

    def _score_interval(self, name: str, group: tuple, reference: tuple, level: float):
        counts = self._counts(name, group), self._counts(name, reference)
        if counts[0] is None or counts[1] is None or counts[0][1] == 0 or counts[1][1] == 0:
            return None
        (s1, n1), (s2, n2) = counts
        low, high = rate_interval(s1, n1, s2, n2, _RATES[name][1], level)
        if not (np.isfinite(low) and np.isfinite(high)):
            return None
        return float(low), float(high)

    def _estimate(self, name: str, groups: dict, reference, others: list, rng, simultaneous: Optional[bool] = None):
        """Point estimates, intervals at `confidence`, intervals at `equivalence_confidence`, and how they were made

        With several groups and `simultaneous`, the error rate is split by a union bound: each group under a score
        interval gets alpha / k (Bonferroni), the k_b bootstrap groups share a max-t band at alpha * k_b / k."""
        estimates = {k: self._disparity(name, groups[k], groups[reference]) for k in others}
        b = self.policy.n_bootstrap
        if b == 0 or not others:
            return estimates, {}, {}, "point estimate (no interval)"
        simultaneous = self.policy.simultaneous if simultaneous is None else simultaneous
        simultaneous = simultaneous and len(others) > 1
        alpha = 1 - self.policy.confidence
        eq_alpha = 1 - (self.policy.equivalence_confidence or self.policy.confidence)
        k_total = len(others)

        score_groups = []
        if name in _RATES and self.policy.interval != "bootstrap":
            for k in others:
                small = len(groups[k][0]) < self.policy.small_group_size
                if self.policy.interval == "score" or small:
                    score_groups.append(k)
        intervals, equivalence, method = {}, {}, {}
        split = k_total if simultaneous else 1
        for k in list(score_groups):
            main = self._score_interval(name, groups[k], groups[reference], 1 - alpha / split)
            eq = self._score_interval(name, groups[k], groups[reference], 1 - eq_alpha / split)
            if main is None or eq is None:
                score_groups.remove(k)
                continue
            intervals[k], equivalence[k] = main, eq
            kind = "Newcombe" if _RATES[name][1] == "difference" else "MOVER-R"
            level = f"{1 - alpha / split:.4g}" if simultaneous else f"{self.policy.confidence:.0%}"
            method[k] = f"Wilson score interval ({kind}), {level}" + (
                f" (Bonferroni over {k_total} groups)" if simultaneous else ""
            )
        boot = [k for k in others if k not in score_groups]
        if boot:
            draws = self._draws(name, groups, reference, boot, b, rng)
            share = len(boot) / k_total
            if simultaneous:
                point = np.array([estimates[k] for k in boot])
                se = np.nanstd(draws, axis=0, ddof=1)
                se[se == 0] = np.nan
                t_max = np.nanmax(np.abs(draws - point) / se, axis=1)
                for target, a in ((intervals, alpha * share), (equivalence, eq_alpha * share)):
                    q = float(np.nanquantile(t_max, 1 - a))
                    for j, k in enumerate(boot):
                        v = (float(point[j] - q * se[j]), float(point[j] + q * se[j]))
                        target[k] = (estimates[k], estimates[k]) if np.isnan(v[0]) else v
                text = f"stratified bootstrap ({b}), simultaneous {self.policy.confidence:.0%} max-t band over {k_total} groups"
                if share < 1:
                    text += f" (band at {1 - alpha * share:.4g} over {len(boot)}, union bound with the score intervals)"
            else:
                for target, a in ((intervals, alpha), (equivalence, eq_alpha)):
                    low, high = np.nanquantile(draws, [a / 2, 1 - a / 2], axis=0)
                    for j, k in enumerate(boot):
                        target[k] = (float(low[j]), float(high[j]))
                text = f"stratified bootstrap ({b}), {self.policy.confidence:.0%} percentile interval"
            for k in boot:
                method[k] = text
        if self.policy.equivalence_confidence is not None:
            note = f"; 'within the zone' judged at {self.policy.equivalence_confidence:.0%} (TOST)"
            method = {k: v + note for k, v in method.items()}
        return estimates, intervals, equivalence, method
