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
MISSING = "<missing>"  # value of a protected attribute that is missing (NaN, None): a group of its own


class AllCellsSuppressed(error_msg):
    """Every group of an attribute combination is below min_group_size"""


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
        "auto" (default): score intervals (Wilson combined by Newcombe for differences, MOVER on the log scale for the ratio) for
        groups under `small_group_size` rows on the rate criteria, the bootstrap otherwise; "bootstrap": always the
        bootstrap (the behaviour of 0.10 and before, anti-conservative for small groups); "score": score intervals for every
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
        rows, 30 to 80 times the run time). None combines all of them (the behaviour of 0.10 and before).
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
    weights : str, optional
        Column of row weights (e.g. "population" when each row is a community): every group rate and every interval
        is weighted, the score intervals and the binomial shortcut (which count independent rows) are replaced by the
        bootstrap, and the reference group defaults to the group with the largest weight. Rows are still the units
        resampled: a weight is a number of people sharing one decision, not that many independent observations.
    cluster : str, optional
        Column identifying the unit a row belongs to (e.g. "community_id"). The bootstrap then resamples units, not
        rows, jointly across groups (a unit can feed several groups), `n` counts the units of each group and
        `min_group_size` applies to units: the effective sample size is the number of units, however many rows each
        one has. Without it, rows are resampled as independent observations: rows that are copies of an aggregate
        unit (pseudo-individuals) then give intervals that shrink as 1/sqrt(copies) and verdicts that depend on that
        arbitrary choice (lab FA-03), which is why a warning is raised when most rows repeat each other exactly.
    unit : str, optional
        What a row (or a cluster) is, used in the results (e.g. "community": "n counts community units")
    bins : dict, optional
        Binning of numeric protected attributes, {attribute: number of quantile bins, or sorted bin edges}. The groups
        are then labelled "Q1 [low, high]", "Q2 (low, high]", ... (quantiles) or "B1 ..." (edges), the edges are kept
        in `bin_edges` and recorded in `method`, and a reference group can be named by its prefix ("Q1"). Quantiles
        are computed over rows, unweighted.
    numeric_levels : int, optional
        A numeric protected attribute with more distinct values than this, and no `bins`, raises a warning: each value
        would be a group of its own, most of them suppressed (lab FA-03, a share with 1110 distinct values: every group
        suppressed). Default 20.

    Protected attributes need not travel with the predictions: pass them to `run(attributes_data=..., on="id")`,
    the join happens in memory and only group-level results are returned. With aggregate data, `attributes_data` can
    be a composition table: several rows per unit (e.g. per community, one row per race) with the share or the count
    of each group (`run(..., share="share")` or `count="residents"`). Each prediction is then split across the groups
    of its unit with these weights (times `weights` for shares, e.g. the population), the bootstrap resamples the units
    (`cluster` defaults to `on`), and the results carry `status="ecological"`: they describe the people of each group
    through the decisions taken on their units, not decisions taken on individuals.

    Missing attribute values (NaN, None) form a group of their own, labelled "<missing>" (e.g. "sex=<missing>"),
    judged and suppressed like any other: the rows whose attribute is unknown are often the ones a report must not
    silently drop. Recode them beforehand to exclude them.
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
        weights: Optional[str] = None,
        cluster: Optional[str] = None,
        unit: Optional[str] = None,
        bins: Optional[dict] = None,
        numeric_levels: int = 20,
    ):
        if not attributes:
            raise error_msg(value=None, message="at least one protected attribute is required")
        if max_intersection_depth is not None and max_intersection_depth < 1:
            raise error_msg(value=max_intersection_depth, message="max_intersection_depth must be >= 1")
        unknown = sorted(set(bins or {}) - set(attributes))
        if unknown:
            raise error_msg(value=unknown, message=f"bins given for {unknown}, which are not protected attributes")
        self.attributes = list(attributes)
        self.policy = policy or FairnessPolicy()
        self.reference_groups = reference_groups or {}
        self.intersections = intersections
        self.max_intersection_depth = max_intersection_depth
        self.max_cells = max_cells
        self.declared_groups = None if declared_groups is None else set(declared_groups)
        self.columns = {"y_pred": y_pred, "y_true": y_true, "score": score}
        self.seed = seed
        self.weights = weights
        self.cluster = cluster
        self.unit = unit
        self.bins = dict(bins or {})
        self.numeric_levels = numeric_levels
        self.bin_edges: dict = {}
        self._results: list[MetricResults] = []
        self._coverage: list[MetricResults] = []
        self._reset_design()

    # ------------------------------------------------------------------ public API
    def run(
        self,
        data: pd.DataFrame,
        attributes_data: Optional[pd.DataFrame] = None,
        on: Optional[str] = None,
        share: Optional[str] = None,
        count: Optional[str] = None,
    ) -> list:
        """Evaluate the policy on `data`

        `attributes_data` (joined on `on`) holds the protected attributes when they do not travel with the
        predictions; with several rows per id it is a composition table and `share` or `count` names the column of
        each group's share (0 to 1) or count of people (see the class docstring)."""
        self._reset_design()
        factor = None
        if share is not None or count is not None:
            if attributes_data is None:
                raise error_msg(value=None, message="share / count describe a composition table: pass it as attributes_data")
            if share is not None and count is not None:
                raise error_msg(value=None, message="pass share or count, not both")
            if count is not None and self.weights is not None:
                raise error_msg(value=count, message="count already holds people: it cannot be multiplied by weights")
        if attributes_data is not None:
            if on is None:
                raise error_msg(value=None, message="`on` (pseudonymous id column) is required with attributes_data")
            data, factor = self._join(data, attributes_data, on, share or count, share is not None)
        missing = [c for c in [self.columns["y_pred"], *self.attributes] if c not in data.columns]
        missing += [c for c in (self.weights, self.cluster) if c is not None and c not in data.columns]
        if missing:
            raise error_msg(value=missing, message=f"missing columns {missing}")
        data = self._bin(data.reset_index(drop=True))
        self._design(data, factor, on)
        self._results, self._coverage = [], []
        rng = np.random.default_rng(self.seed)
        cells = 0
        for attrs in self._attribute_sets():
            if self.max_cells is not None and cells >= self.max_cells:
                warnings.warn(f"max_cells ({self.max_cells}) reached: attribute combinations from {attrs} on are skipped")
                break
            try:
                results, n_cells = self._evaluate(data, list(attrs), rng)
            except AllCellsSuppressed:
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

    # ------------------------------------------------------------------ internals: data design
    def _reset_design(self):
        self._w = None  # row weights, aligned with the evaluated frame (None: unweighted)
        self._codes = None  # unit code of each row (None: rows are the units)
        self._n_clusters = 0
        self._unit_name = self.unit or self.cluster
        self._ecological = False
        self._numeric = {}  # numeric attributes left unbinned: attribute -> distinct values

    def _join(self, data, attributes_data, on, column, is_share) -> tuple:
        """In-memory join with the attributes; with a composition column, (merged frame, weight factor per row)"""
        wanted = [on, *self.attributes] + ([column] if column else [])
        missing = [c for c in wanted if c not in attributes_data.columns] + ([on] if on not in data.columns else [])
        if missing:
            raise error_msg(value=missing, message=f"missing columns {missing} for the attributes_data join")
        right = attributes_data[wanted]
        repeated = int(data[on].duplicated().sum())
        if repeated:
            raise error_msg(
                value=on,
                message=f"{repeated} predictions repeat an `{on}` value already seen: attributes_data expects one prediction "
                "per id (aggregate first, or pass the attributes with the predictions and cluster= the id)",
            )
        several = int(right[on].duplicated().sum())
        if several and column is None:
            raise error_msg(
                value=on,
                message=f"attributes_data has several rows for {right[on][right[on].duplicated()].nunique()} `{on}` values: "
                "a composition table (one unit, several groups) needs share= or count=, the column of each group's share "
                "or number of people, which turns its rows into weights; otherwise keep one attribute row per id",
            )
        merged = data.merge(right, on=on, how="inner", validate="one_to_one" if column is None else "one_to_many")
        if column is None:
            return merged, None
        factor = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(factor).all() or (factor < 0).any():
            raise error_msg(value=column, message=f"{column} must hold finite values >= 0")
        if is_share:
            if (factor > 1 + 1e-9).any():
                raise error_msg(value=column, message=f"{column} holds shares above 1: pass it as count= if it counts people")
            total = pd.Series(factor).groupby(merged[on].to_numpy()).sum()
            if (total > 1.01).any():
                warnings.warn(f"the shares of {int((total > 1.01).sum())} units sum above 1: groups overlap or are miscoded")
        keep = factor > 0
        self._ecological = True
        if self.cluster is None:
            self._unit_name = self.unit or on
        return merged[keep].reset_index(drop=True), factor[keep]

    def _bin(self, data: pd.DataFrame) -> pd.DataFrame:
        """Bin the attributes listed in `bins`; warn on the numeric attributes left with many distinct values"""
        self.bin_edges = {}
        for a in self.attributes:
            values = data[a]
            numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
            if a not in self.bins:
                distinct = int(values.nunique())
                if numeric and distinct > self.numeric_levels:
                    self._numeric[a] = distinct
                    warnings.warn(
                        f"protected attribute {a} is numeric with {distinct} distinct values: each value is a group of its own "
                        f"and most are suppressed; bin it, e.g. bins={{'{a}': 3}} (quantiles) or bins={{'{a}': [edges]}}"
                    )
                continue
            if not numeric:
                raise error_msg(value=a, message=f"bins apply to numeric attributes; {a} is {values.dtype}")
            spec = self.bins[a]
            x = values.to_numpy(dtype=float)
            if isinstance(spec, (int, np.integer)):
                if spec < 2:
                    raise error_msg(value=spec, message="at least 2 quantile bins are required")
                edges, prefix = np.unique(np.nanquantile(x, np.linspace(0, 1, int(spec) + 1))), "Q"
            else:
                edges, prefix = np.asarray(spec, dtype=float), "B"
                if len(edges) < 2 or (np.diff(edges) <= 0).any():
                    raise error_msg(value=list(spec), message=f"bin edges of {a} must be increasing (at least 2)")
                outside = int(((x < edges[0]) | (x > edges[-1])).sum())
                if outside:
                    raise error_msg(value=a, message=f"{outside} values of {a} lie outside the bin edges")
            index = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)
            names = [
                f"{prefix}{i + 1} {'[' if i == 0 else '('}{edges[i]:.4g}, {edges[i + 1]:.4g}]" for i in range(len(edges) - 1)
            ]
            labels = np.array(names, dtype=object)[index]
            data = data.assign(**{a: pd.Series(labels, index=data.index).where(values.notna(), None)})
            self.bin_edges[a] = [float(e) for e in edges]
        return data

    def _design(self, data: pd.DataFrame, factor, on):
        """Row weights and unit codes of the evaluated frame; warning on pseudo-individual copies"""
        w = None
        if self.weights is not None:
            w = pd.to_numeric(data[self.weights], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(w).all() or (w < 0).any():
                raise error_msg(value=self.weights, message=f"weights {self.weights} must be finite and >= 0")
        if factor is not None:
            w = factor if w is None else w * factor
        self._w = w
        cluster = self.cluster or (on if self._ecological else None)
        if cluster is not None:
            codes, uniques = pd.factorize(data[cluster], use_na_sentinel=False)
            self._codes, self._n_clusters = codes.astype(np.int64), len(uniques)
            return
        if w is not None:
            return
        used = [c for c in [*self.columns.values(), *self.attributes] if c in data.columns]
        extra = [c for c in data.columns if c not in used]
        if not extra or len(data) < 2 * self.policy.min_group_size:
            return
        try:
            copies = float(data.duplicated(subset=used + extra).mean())
        except TypeError:  # unhashable cells
            return
        if copies > 0.5:
            warnings.warn(
                f"{copies:.0%} of the rows repeat another row exactly (attributes, decision and {extra[:3]}): if they are "
                "copies of aggregate units (pseudo-individuals), the intervals treat them as independent people and are "
                "too narrow; pass cluster=<unit column> so that the bootstrap resamples units (the effective sample size "
                "is the number of units), or one row per unit with weights="
            )

    def _notes(self, attrs) -> str:
        """Suffix of `method` describing the design (empty for unweighted rows without binning)"""
        notes = []
        if self._w is not None:
            notes.append("weighted rates" + (f" ({self.weights})" if self.weights else "") + ", no score interval")
        if self._unit_name:
            notes.append(f"n counts {self._unit_name} units")
        if self._ecological:
            notes.append("ecological: groups from the composition of each unit (aggregate attributes), not individual membership")
        for a in attrs:
            if a in self.bin_edges:
                notes.append(f"{a} binned, edges {[float(f'{e:.4g}') for e in self.bin_edges[a]]}")
        return "".join(f"; {n}" for n in notes)

    def _bootstrap_name(self) -> str:
        if self._codes is not None:
            return f"cluster bootstrap over {self._n_clusters} {self._unit_name} units"
        return "stratified bootstrap" + (" of weighted rows" if self._w is not None else "")

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
        return y_pred, y_true, score, self._w, self._codes

    def _sizes(self, keys: pd.Series, rows: dict) -> tuple:
        """(units, mass) per group: units are what min_group_size and n count (rows, or distinct clusters), mass what
        the coverage shares and the default reference weigh (rows, or the sum of the weights)"""
        if self._w is None and self._codes is None:
            sizes = keys.value_counts()
            return sizes, sizes
        mass = pd.Series({k: float(self._w[i].sum()) if self._w is not None else float(len(i)) for k, i in rows.items()})
        positive = {k: i if self._w is None else i[self._w[i] > 0] for k, i in rows.items()}
        units = pd.Series({k: len(np.unique(self._codes[i])) if self._codes is not None else len(i) for k, i in positive.items()})
        order = mass.sort_values(ascending=False, kind="stable").index
        return units[order], mass[order]

    def _evaluate(self, data: pd.DataFrame, attrs: list, rng) -> tuple:
        feature = "|".join(attrs)

        def column(a):  # missing values form their own group (pandas 3 keeps NaN through astype(str))
            values = data[a]
            return values.astype(object).where(values.notna(), MISSING).astype(str)

        keys = column(attrs[0])
        for a in attrs[1:]:  # vectorized concatenation (a row-wise join is slow on large frames)
            keys = keys + "|" + column(a)
        labels = {k: "|".join(f"{a}={v}" for a, v in zip(attrs, k.split("|"))) for k in keys.unique()}
        rows = keys.reset_index(drop=True).groupby(keys.to_numpy()).indices
        sizes, mass = self._sizes(keys, rows)
        kept = sizes[sizes >= self.policy.min_group_size].index.tolist()
        suppressed = sizes[sizes < self.policy.min_group_size].index.tolist()

        reference = self._reference(feature, attrs, mass, kept, labels)
        notes = self._notes(attrs)
        status_label = "ecological" if self._ecological else None
        columns = self._arrays(data)
        groups = {k: tuple(None if c is None else c[rows[k]] for c in columns) for k in kept}
        declared = None
        if self.declared_groups is not None:
            declared = {k for k in kept if labels[k] in self.declared_groups}
        results = []
        outside_reference = mass.sum() - mass[reference]
        hidden = sum(mass[k] for k in suppressed)
        if self._w is None:
            outside_reference, hidden = int(outside_reference), int(hidden)
        if outside_reference and hidden / outside_reference > self.policy.max_no_verdict_share:
            warnings.warn(
                f"{hidden / outside_reference:.1%} of the rows of {feature} outside the reference group are in suppressed "
                f"cells ({len(suppressed)} of {len(sizes)}), without value or verdict (see coverage())"
            )
        for name in self.policy.criteria:
            if not self._computable(name, groups[reference]):
                results.append(self._skipped(name, feature, labels[reference], groups[reference]))
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
                    how = f"exploratory (not declared), no verdict; {m2[k] if isinstance(m2, dict) else m2}"
                else:
                    status = equivalence_verdict((low, high), equivalence.get(k, (low, high)), estimates[k], zone)
                    how = method[k] if isinstance(method, dict) else method
                    if status is None:
                        how += self._size_note(name, groups[k], groups[reference], zone, len(judged))
                if status is None:
                    no_verdict += mass[k]
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
                        method=how + notes,
                        test_type="fixed",
                        status=status_label,
                    )
                )
            unit = f"{self._unit_name} units" if self._unit_name else "rows"
            for k in suppressed:
                results.append(
                    MetricResults(
                        metric_type="fairness",
                        metric_name=name,
                        feature_name=feature,
                        group=labels[k],
                        reference_group=labels[reference],
                        suppressed=True,
                        method=f"suppressed: fewer than {self.policy.min_group_size} {unit}",
                        test_type="fixed",
                        status=status_label,
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
                    n=int(sizes.sum() - sizes[reference]),
                    threshold=self.policy.max_no_verdict_share,
                    method=f"{name}: share of the {'weight' if self._w is not None else 'rows'} outside the reference group "
                    "in suppressed or inconclusive "
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
            message = f"every group of {feature} is below min_group_size"
            numeric = [a for a in attrs if a in self._numeric]
            if numeric:
                message += (
                    f": {', '.join(f'{a} is numeric with {self._numeric[a]} distinct values' for a in numeric)}, each value "
                    f"a group of its own; bin it, e.g. bins={{'{numeric[0]}': 3}} (quantiles)"
                )
            raise AllCellsSuppressed(value=feature, message=message)
        wanted = self.reference_groups.get(feature) or (self.reference_groups.get(attrs[0]) if len(attrs) == 1 else None)
        if wanted is None:
            return sizes[kept].idxmax()
        if len(attrs) == 1 and attrs[0] in self.bin_edges and str(wanted) not in kept:
            prefixed = [k for k in kept if k.split(" ", 1)[0] == str(wanted)]
            wanted = prefixed[0] if prefixed else wanted
        if str(wanted) not in kept:
            raise error_msg(value=wanted, message=f"reference group {wanted} of {feature} is missing or suppressed")
        return str(wanted)

    def _computable(self, name: str, arrays: tuple) -> bool:
        _, y_true, score = arrays[:3]
        needs_labels = name == EQUALIZED_ODDS or CRITERIA[name].needs_labels
        needs_scores = name != EQUALIZED_ODDS and CRITERIA[name].needs_scores
        return not ((needs_labels and y_true is None) or (needs_scores and score is None))

    def _skipped(self, name, feature, reference_label, arrays):
        """Criterion that cannot be computed: n_labelled=0 when it needs labels and there are none"""
        y_true = arrays[1]
        needs_labels = name == EQUALIZED_ODDS or CRITERIA[name].needs_labels
        return MetricResults(
            metric_type="fairness",
            metric_name=name,
            feature_name=feature,
            reference_group=reference_label,
            error=f"{name} needs {'labels and scores' if name in ('calibration_difference',) else 'labels'}",
            test_type="fixed",
            n_labelled=0 if needs_labels and y_true is None else None,
        )

    def _disparity(self, name: str, group: tuple, reference: tuple) -> float:
        if name == EQUALIZED_ODDS:
            tpr = CRITERIA["equal_opportunity_difference"]
            fpr = CRITERIA["false_positive_rate_difference"]
            return max(
                abs(tpr.compare(self._statistic(tpr, group), self._statistic(tpr, reference))),
                abs(fpr.compare(self._statistic(fpr, group), self._statistic(fpr, reference))),
            )
        criterion = CRITERIA[name]
        return float(criterion.compare(self._statistic(criterion, group), self._statistic(criterion, reference)))

    @staticmethod
    def _statistic(criterion, arrays: tuple) -> float:
        weights = arrays[3]
        return criterion.statistic(*arrays[:3]) if weights is None else criterion.statistic(*arrays[:3], weights=weights)

    # ------------------------------------------------------------------ binary rates
    @staticmethod
    def _binary(values) -> bool:
        return values is not None and bool(np.isin(values, (0.0, 1.0)).all())

    def _counts(self, name: str, arrays: tuple):
        """(successes, trials) of the rate behind a rate criterion, or None when the data is not binary"""
        y_pred, y_true, _, weights, codes = arrays
        if weights is not None or codes is not None:  # counts of independent rows only
            return None
        among, _ = _RATES[name]
        if not self._binary(y_pred) or (among is not None and not self._binary(y_true)):
            return None
        rows = np.ones(len(y_pred), dtype=bool) if among is None else y_true == among
        return float(y_pred[rows].sum()), int(rows.sum())

    def _binomial_rates(self, name: str, arrays: tuple, b: int, rng):
        """Bootstrap replicates of the group statistic(s) without resampling rows: for 0/1 data the row bootstrap of a
        rate is exactly binomial (counts of positives, then of decisions among them). None when not applicable."""
        y_pred, y_true, _, weights, codes = arrays
        if weights is not None or codes is not None:
            return None
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
        otherwise (stratified: each group resampled within itself); with clusters, units resampled across groups"""
        if self._codes is not None:
            return self._cluster_draws(name, groups, reference, others, b, rng)
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

    def _cluster_draws(self, name: str, groups: dict, reference, others: list, b: int, rng) -> np.ndarray:
        """b x len(others) bootstrap disparities resampling units: each replicate draws the units with replacement
        (multinomial multiplicities) and every row is weighted by the multiplicity of its unit, in all groups at once.
        Rate criteria are computed from per-unit sums (vectorized), the others replicate by replicate."""
        n_units = self._n_clusters
        members = [reference, *[k for k in others if k != reference]]
        vector = name == EQUALIZED_ODDS or name in _RATES
        sums = {}
        if vector:
            amongs = (1, 0) if name == EQUALIZED_ODDS else (_RATES[name][0],)
            for k in members:
                y_pred, y_true, _, w, codes = groups[k]
                w = np.ones(len(y_pred)) if w is None else w
                sums[k] = []
                for among in amongs:
                    rows = np.ones(len(y_pred)) if among is None else (y_true == among).astype(float)
                    sums[k].append(
                        (
                            np.bincount(codes, weights=w * y_pred * rows, minlength=n_units),
                            np.bincount(codes, weights=w * rows, minlength=n_units),
                        )
                    )

        def reweight(arrays, multiplicity):
            y_pred, y_true, score, w, codes = arrays
            return y_pred, y_true, score, multiplicity[codes] * (1.0 if w is None else w), codes

        draws = np.empty((b, len(others)))
        block = max(1, int(2e7 // max(n_units, 1)))  # bound the memory of the multiplicity matrix
        done = 0
        while done < b:
            m = min(block, b - done)
            counts = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=m).astype(float)
            if vector:
                with np.errstate(invalid="ignore", divide="ignore"):
                    rates = {k: [(counts @ num) / (counts @ den) for num, den in sums[k]] for k in members}
                    ref = rates[reference]
                    for j, k in enumerate(others):
                        g = rates[k]
                        if name == EQUALIZED_ODDS:
                            d = np.maximum(np.abs(g[0] - ref[0]), np.abs(g[1] - ref[1]))
                        elif _RATES[name][1] == "ratio":
                            d = np.where(ref[0] > 0, g[0] / np.where(ref[0] > 0, ref[0], 1), np.nan)
                        else:
                            d = g[0] - ref[0]
                        draws[done : done + m, j] = d
            else:
                for i in range(m):
                    ref = reweight(groups[reference], counts[i])
                    draws[done + i] = [self._disparity(name, reweight(groups[k], counts[i]), ref) for k in others]
            done += m
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
            kind = "Newcombe" if _RATES[name][1] == "difference" else "MOVER, log scale"
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
                text = (
                    f"{self._bootstrap_name()} ({b}), simultaneous {self.policy.confidence:.0%} max-t band over {k_total} groups"
                )
                if share < 1:
                    text += f" (band at {1 - alpha * share:.4g} over {len(boot)}, union bound with the score intervals)"
            else:
                for target, a in ((intervals, alpha), (equivalence, eq_alpha)):
                    low, high = np.nanquantile(draws, [a / 2, 1 - a / 2], axis=0)
                    for j, k in enumerate(boot):
                        target[k] = (float(low[j]), float(high[j]))
                text = f"{self._bootstrap_name()} ({b}), {self.policy.confidence:.0%} percentile interval"
            for k in boot:
                method[k] = text
        if self.policy.equivalence_confidence is not None:
            note = f"; 'within the zone' judged at {self.policy.equivalence_confidence:.0%} (TOST)"
            method = {k: v + note for k, v in method.items()}
        return estimates, intervals, equivalence, method
