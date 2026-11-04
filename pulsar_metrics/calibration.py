"""Calibration of drift thresholds on A/A draws, and empirical p-values.

A nominal threshold (p < 0.05, PSI > 0.2...) assumes independent samples and a deterministic system. Neither holds for
LLM and agent traffic: outputs vary between runs even at temperature 0, and successive requests are correlated. A
threshold applied naively then raises alerts without any real change.

AACalibrator builds, for each (metric, feature), the distribution of the statistic between two windows drawn from the
same reference ("A/A"), and derives:
- a calibrated threshold: the quantile of that null distribution (e.g. 95 %), usable as Analyzer options;
- an empirical p-value for any observed statistic, distance or test alike, on which multiple-testing correction applies.

By default an A/A draw compares one window of the reference with the rest of the reference, which is what
Analyzer.run does with a production window and the whole reference (compare_to="reference"). compare_to="window"
compares two disjoint windows instead, for a reference that is itself one window (e.g. the previous week).

The null distribution of a (metric, feature) depends only on the reference values of that feature and on the draw
settings, so it can be cached and reused across windows, runs and models (CalibrationCache, save / load).

On seasonal data, a reference spanning the whole year makes the null absorb the season: period_reference() selects the
same period of the reference (same weeks of the year +- k weeks) to calibrate on and to compare each window with.
"""

import contextlib
import hashlib
import io
import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import binom

from .exceptions import CustomExceptionPulsarMetric as error_msg
from .metrics.drift import (
    MISSINGNESS_METRICS,
    DriftMetric,
    DriftTestMetric,
    is_constant,
)
from .metrics.utils import EPSILON

logger = logging.getLogger(__name__)

MIN_WINDOWS = 10  # below, fit() refuses unless allow_short=True
MIN_CONTIGUOUS_WINDOWS = 4  # below, contiguous A/A draws overlap so much that the null is degenerate, even with allow_short
RELIABLE_WINDOWS = 30  # below, thresholds vary a lot between references (lab: KS alpha 0.02 to 0.15, p10-p90, at 10)
LARGE_INCREASE = 5  # an automatic increase of the draws beyond this factor is logged as a warning, with its cost
SECONDS_PER_EVALUATION = 0.0005  # order of magnitude of one scipy test on a window of a few hundred rows
FLOOR_MARGIN = 2  # the p-value floor 1/(n_draws+1) is kept at least this many times below the BH threshold alpha/m

STATUS_OK = "ok"
STATUS_CONSTANT = "constant_reference"  # the reference holds a single value: thresholds are degenerate
STATUS_DEGENERATE = "degenerate_null"  # the A/A statistic takes at most 2 distinct values
STATUS_SATURATED = "saturated"  # many A/A draws sit at the bound of the statistic: a drift beyond it cannot be seen

SATURATION_SHARE = 0.05  # above this share of A/A draws at the bound, the null is saturated
SATURATED_PVALUE = np.finfo(float).tiny  # a p-value at or below it has underflowed to the bound 0
# psi / kl at or above half their epsilon ceiling (2 ln(1/epsilon) = 27.6 for psi) only arise when bins are empty in one
# sample: the value then measures the smoothing, not the size of the shift
SATURATED_DIVERGENCE = 0.5 * math.log(1 / EPSILON)
BOUNDED_DISTANCES = ("psi", "kl")
# above this ratio of between-window to within-window variance, the A/A null is dominated by differences between
# periods of the reference (season, regimes) rather than by sampling noise
BETWEEN_WITHIN_WARNING = 10.0
COMPARE_TO = ("reference", "window")

CACHE_FORMAT = "pulsar-aa-null-v2"


@dataclass
class NullDistribution:
    """Distribution of a statistic between two windows of the same reference"""

    metric_name: str
    feature_name: str
    kind: str  # "distance" (drift when large) or "pvalue" (drift when small)
    values: np.ndarray = field(repr=False)
    quantile: float
    n_windows: Optional[float] = None  # reference rows / window_size
    n_effective: Optional[int] = None  # A/A draws that carry independent information (about distinct window pairs)
    status: str = STATUS_OK
    n_failed_draws: int = 0  # draws where the statistic could not be computed (left out)
    shrink_draws: int = 0  # pseudo-draws of the nominal alpha mixed into the calibrated one (p-values only)
    reliable_windows: int = RELIABLE_WINDOWS
    share_at_bound: float = 0.0  # A/A draws at the bound of the statistic (p-value 0, psi / kl at their epsilon ceiling)
    between_within_ratio: Optional[float] = None  # window_size x variance of window means / mean within-window variance
    window_size: Optional[int] = None  # size of the windows the null was drawn for

    @property
    def raw_threshold(self) -> float:
        """Distance above which, or p-value below which, only (1 - quantile) of A/A draws fall"""
        if self.kind == "distance":
            return float(np.quantile(self.values, self.quantile))
        return float(np.quantile(self.values, 1 - self.quantile))

    @property
    def shrinkage(self) -> float:
        """Weight of the nominal alpha (1 - quantile) in the threshold: shrink_draws / (shrink_draws + n_effective)"""
        if self.kind != "pvalue" or self.shrink_draws <= 0:
            return 0.0
        n_eff = self.n_effective or len(self.values)
        return self.shrink_draws / (self.shrink_draws + n_eff)

    @property
    def threshold(self) -> float:
        """Calibrated threshold, shrunk towards the nominal alpha when shrink_draws > 0 (p-values only)"""
        weight = self.shrinkage
        if weight == 0.0:
            return self.raw_threshold
        return float((1 - weight) * self.raw_threshold + weight * (1 - self.quantile))

    @property
    def pvalue_floor(self) -> float:
        """Smallest empirical p-value this distribution can give: 1 / (draws + 1)"""
        return 1.0 / (1 + len(self.values))

    @property
    def reliable(self) -> bool:
        """False for a constant reference, a degenerate or saturated null, or a reference shorter than reliable_windows"""
        if self.status != STATUS_OK:
            return False
        return self.n_windows is None or self.n_windows >= self.reliable_windows

    def threshold_interval(self, level: float = 0.90) -> tuple[float, float]:
        """Approximate confidence interval of the raw threshold (order statistics on n_effective draws)

        The A/A draws of one reference overlap: their information is that of about n_effective independent draws,
        which sets the width. A wide interval means the threshold would change a lot with another reference.
        """
        q = self.quantile if self.kind == "distance" else 1 - self.quantile
        n_eff = max(1, int(self.n_effective or len(self.values)))
        tail = (1 - level) / 2
        low = binom.ppf(tail, n_eff, q) / n_eff
        high = min(1.0, (binom.ppf(1 - tail, n_eff, q) + 1) / n_eff)
        return float(np.quantile(self.values, low)), float(np.quantile(self.values, high))

    def empirical_pvalue(self, observed: float) -> float:
        """Share of A/A draws at least as extreme as the observed statistic (with the +1 correction)"""
        if observed is None or np.isnan(observed):
            return float("nan")
        extreme = self.values >= observed if self.kind == "distance" else self.values <= observed
        return float((1 + extreme.sum()) / (1 + len(self.values)))


MetricSpec = Union[DriftMetric, DriftTestMetric]


class CalibrationCache:
    """Store of A/A null distributions, reused across windows, runs and models

    An entry is keyed by everything the A/A draws depend on: the metric and its missing-value settings, the values of
    the feature in the reference (hashed, in order), window_size, the number of draws, contiguous and seed. Two models
    that share the same reference values for a feature therefore share the entry, and a changed reference never
    reuses a stale one. The quantile is not part of the key: the threshold is recomputed from the stored values.

    Parameters
    ----------
    directory : str or Path, optional
        Folder where entries are stored as .npy files, to reuse them in other processes; None keeps them in memory
    """

    def __init__(self, directory: Union[str, Path, None] = None):
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        values = self._memory.get(key)
        if values is None and self.directory is not None:
            path = self.directory / f"{key}.npy"
            if path.exists():
                values = np.load(path, allow_pickle=False)
                self._memory[key] = values
        if values is None:
            self.misses += 1
        else:
            self.hits += 1
        return values

    def put(self, key: str, values: np.ndarray):
        values = np.asarray(values, dtype=float)
        self._memory[key] = values
        if self.directory is not None:
            np.save(self.directory / f"{key}.npy", values, allow_pickle=False)

    def __len__(self) -> int:
        return len(self._memory)


class AACalibrator:
    """Calibrate drift thresholds on A/A draws from a reference dataset

    Parameters
    ----------
    window_size : int
        Size of the windows compared in production; the null distribution depends strongly on it
    n_draws : int, optional
        Minimum number of A/A draws per (metric, feature) (default 200). Raised automatically, see auto_draws.
    quantile : float, optional
        Quantile of the null distribution used as threshold (default 0.95, i.e. 5 % false alarms per test)
    contiguous : bool, optional
        Draw contiguous windows from the reference, which must then be sorted by time (default True). Keeps the
        temporal dependence of real traffic; random rows (False) would hide it and under-estimate false alarms.
    seed : int, optional
        Random seed (default 0)
    allow_short : bool, optional
        Calibrate a reference shorter than 10 windows (default False: refused). Thresholds are then flagged
        reliable=False. Contiguous draws still need 4 windows.
    fdr_alpha : float, optional
        False discovery rate at which the empirical p-values will be corrected (default 0.05; None: no check). The
        empirical p-value cannot go below 1 / (n_draws + 1), while Benjamini-Hochberg needs alpha / m to reject one
        test out of m: with 200 draws, no single drift passes BH beyond about 10 to 30 tests per window.
    auto_draws : bool, optional
        Raise n_draws so that 1 / (n_draws + 1) <= fdr_alpha / (2 m), m = number of tests (default True). When False,
        a warning is emitted instead.
    max_draws : int, optional
        Upper bound of the automatic increase (default 10000); a warning says when it is not enough
    shrink_draws : int, optional
        Pseudo-draws of the nominal alpha (1 - quantile) mixed into calibrated alphas (default 0: none). Stabilises
        the thresholds of short references at the price of part of the calibration; distances are not shrunk.
    reliable_windows : int, optional
        Number of reference windows below which thresholds are flagged reliable=False (default 30)
    compare_to : str, optional
        What a window is compared with in an A/A draw. "reference" (default): the rest of the reference, as
        Analyzer.run compares a production window with the whole reference. "window": another window of the same size,
        for monitoring that compares each window with a reference of one window (e.g. the previous one). Calibrating
        with one and running with the other gives thresholds made for another comparison (lab TS-04: window vs window
        thresholds were 40 % above the statistics of reference weeks against the whole reference).
    """

    def __init__(
        self,
        window_size: int,
        n_draws: int = 200,
        quantile: float = 0.95,
        contiguous: bool = True,
        seed=0,
        allow_short: bool = False,
        fdr_alpha: Optional[float] = 0.05,
        auto_draws: bool = True,
        max_draws: int = 10000,
        shrink_draws: int = 0,
        reliable_windows: int = RELIABLE_WINDOWS,
        compare_to: str = "reference",
    ):
        if compare_to not in COMPARE_TO:
            raise error_msg(value=compare_to, message=f"compare_to must be one of {COMPARE_TO}")
        if window_size < 2 or n_draws < 20 or not 0.5 < quantile < 1:
            raise error_msg(value=None, message="window_size >= 2, n_draws >= 20 and 0.5 < quantile < 1 are required")
        if fdr_alpha is not None and not 0 < fdr_alpha < 1:
            raise error_msg(value=fdr_alpha, message="fdr_alpha must be in (0, 1) or None")
        if max_draws < n_draws or shrink_draws < 0:
            raise error_msg(value=None, message="max_draws >= n_draws and shrink_draws >= 0 are required")
        self.window_size = window_size
        self.n_draws = n_draws
        self.quantile = quantile
        self.contiguous = contiguous
        self.seed = seed
        self.allow_short = allow_short
        self.fdr_alpha = fdr_alpha
        self.auto_draws = auto_draws
        self.max_draws = max_draws
        self.shrink_draws = shrink_draws
        self.reliable_windows = reliable_windows
        self.compare_to = compare_to
        self.n_draws_used: Optional[int] = None
        self.nulls: dict[tuple, NullDistribution] = {}
        self.errors: dict[tuple, str] = {}

    # ------------------------------------------------------------------ fitting
    def _check_reference(self, n: int) -> float:
        w = self.window_size
        n_windows = n / w
        if n_windows < MIN_WINDOWS:
            if not self.allow_short:
                raise error_msg(
                    value=n,
                    message=(
                        f"reference of {n} rows holds {n_windows:.1f} windows of {w}: at least {MIN_WINDOWS} are needed "
                        f"({MIN_WINDOWS * w} rows; {self.reliable_windows} for stable thresholds). Pass allow_short=True "
                        "to calibrate anyway: thresholds are then flagged reliable=False"
                    ),
                )
            if self.contiguous and n_windows < MIN_CONTIGUOUS_WINDOWS:
                raise error_msg(
                    value=n,
                    message=(
                        f"contiguous A/A draws need at least {MIN_CONTIGUOUS_WINDOWS} windows ({MIN_CONTIGUOUS_WINDOWS * w} "
                        f"rows): with {n_windows:.1f}, the draws overlap and the null distribution is degenerate"
                    ),
                )
            warnings.warn(
                f"reference of {n} rows holds only {n_windows:.1f} windows of {w}: the calibrated thresholds are unreliable "
                f"and flagged reliable=False; use at least {self.reliable_windows * w} rows",
                UserWarning,
                stacklevel=3,
            )
        elif n_windows < self.reliable_windows:
            warnings.warn(
                f"reference of {n} rows holds {n_windows:.1f} windows of {w}: calibrated thresholds vary between references "
                f"(see threshold_interval()) and are flagged reliable=False; {self.reliable_windows} windows or more are "
                "recommended",
                UserWarning,
                stacklevel=3,
            )
        return n_windows

    def _required_draws(self, n_tests: int) -> Optional[int]:
        if self.fdr_alpha is None:
            return None
        return math.ceil(FLOOR_MARGIN * n_tests / self.fdr_alpha) - 1

    def _resolve_draws(self, n_tests: int, n_specs: int) -> int:
        required = self._required_draws(n_tests)
        if required is None or required <= self.n_draws:
            return self.n_draws
        floor = 1 / (self.n_draws + 1)
        if not self.auto_draws:
            warnings.warn(
                f"with {self.n_draws} draws the empirical p-value cannot go below {floor:.4f}, above the BH threshold "
                f"{self.fdr_alpha}/{n_tests} = {self.fdr_alpha / n_tests:.4f}: a single drift cannot pass the correction. "
                f"Use n_draws >= {required} (or auto_draws=True)",
                UserWarning,
                stacklevel=3,
            )
            return self.n_draws
        used = min(required, self.max_draws)
        if used < required:
            warnings.warn(
                f"{n_tests} tests at FDR {self.fdr_alpha} need {required} A/A draws; capped at max_draws={self.max_draws}, "
                f"so the empirical p-value floor {1 / (used + 1):.5f} stays close to the BH threshold "
                f"{self.fdr_alpha / n_tests:.5f}",
                UserWarning,
                stacklevel=3,
            )
        level = logging.WARNING if used > LARGE_INCREASE * self.n_draws else logging.INFO
        logger.log(
            level,
            "A/A draws raised from %d to %d (x%.0f) for %d tests at FDR %s: about %d evaluations, roughly %.0f s at "
            "%.1f ms each; reuse the null distributions with CalibrationCache or save() / load()",
            self.n_draws,
            used,
            used / self.n_draws,
            n_tests,
            self.fdr_alpha,
            used * n_specs,
            used * n_specs * SECONDS_PER_EVALUATION,
            SECONDS_PER_EVALUATION * 1000,
        )
        return used

    def _draw_positions(self, n: int, n_draws: int, rng: np.random.Generator) -> list:
        """Positions of the two samples of each A/A draw: slices (contiguous) or row indices

        With compare_to="reference", the second sample is None: the rest of the reference, outside the window.
        """
        w = self.window_size
        if self.compare_to == "reference":
            if self.contiguous:
                starts = rng.integers(0, n - w + 1, size=n_draws)
                return [(slice(int(i), int(i) + w), None) for i in starts]
            return [(np.sort(rng.choice(n, size=w, replace=False)), None) for _ in range(n_draws)]
        if self.contiguous:
            # uniform over ordered pairs of disjoint windows, without rejection sampling
            starts = np.arange(n - w + 1)
            left = np.clip(starts - w + 1, 0, None)  # b in [0, a - w]
            right = np.clip(len(starts) - starts - w, 0, None)  # b in [a + w, n - w]
            admissible = left + right
            a = rng.choice(len(starts), size=n_draws, p=admissible / admissible.sum())
            u = np.floor(rng.random(n_draws) * admissible[a]).astype(int)
            b = np.where(u < left[a], u, a + w + (u - left[a]))
            return [(slice(int(i), int(i) + w), slice(int(j), int(j) + w)) for i, j in zip(a, b)]
        replace = n < 2 * w
        draws = []
        for _ in range(n_draws):
            idx = rng.choice(n, size=2 * w, replace=replace)
            draws.append((idx[:w], idx[w:]))
        return draws

    @staticmethod
    def _rest(a, n: int) -> np.ndarray:
        """Row positions of the reference outside the window a (slice or sorted row indices)"""
        if isinstance(a, slice):
            return np.r_[0 : a.start, a.stop : n]
        mask = np.ones(n, dtype=bool)
        mask[a] = False
        return np.flatnonzero(mask)

    @classmethod
    def _null_values(cls, metric: MetricSpec, column, draws: list) -> np.ndarray:
        values = np.empty(len(draws))
        n = len(column)
        numeric = isinstance(column, pd.Series) and is_numeric_dtype(column) and not is_bool_dtype(column)
        if numeric:
            array = column.to_numpy(dtype=float)

            def evaluate(a, b):
                if b is None and isinstance(a, slice):
                    return metric._value_array(array[a], np.concatenate((array[: a.start], array[a.stop :])))
                return metric._value_array(array[a], array[cls._rest(a, n) if b is None else b])

        else:

            def evaluate(a, b):
                return metric._value(column.iloc[a], column.iloc[cls._rest(a, n) if b is None else b])

        for i, (a, b) in enumerate(draws):
            try:
                values[i] = evaluate(a, b)
            except Exception:
                values[i] = np.nan
        return values

    def _cache_key(self, metric: MetricSpec, column, n_draws: int) -> Optional[str]:
        if not isinstance(self.seed, (int, np.integer)):
            return None  # draws are not reproducible
        digest = hashlib.sha256()
        settings = [
            CACHE_FORMAT,
            type(metric).__name__,
            metric._name,
            repr(metric._missing_values),
            metric._nan_policy,
            self.window_size,
            n_draws,
            self.contiguous,
            self.compare_to,
            repr(sorted(getattr(metric, "_grouping", {}).items())),
            int(self.seed),
            len(column),
            repr(list(column.columns)) if isinstance(column, pd.DataFrame) else "",
            str(column.dtypes.to_dict() if isinstance(column, pd.DataFrame) else column.dtype),
        ]
        digest.update(json.dumps(settings, default=str).encode())
        digest.update(pd.util.hash_pandas_object(column, index=False).to_numpy().tobytes())
        return digest.hexdigest()

    def fit(
        self,
        reference: pd.DataFrame,
        metrics: Iterable[MetricSpec],
        n_tests: Optional[int] = None,
        cache: Optional[CalibrationCache] = None,
    ) -> "AACalibrator":
        """Build the null distribution of every drift metric (DriftMetric or DriftTestMetric) on the reference

        `metrics` can be `analyzer._metrics_list`; performance and custom metrics are ignored. A (metric, feature)
        that cannot be calibrated is recorded in `errors` (with a warning) and the others are calibrated.

        Parameters
        ----------
        reference : DataFrame
            Reference data, sorted by time when contiguous=True
        metrics : iterable
            Drift metrics and tests to calibrate (their missing_values / nan_policy settings apply)
        n_tests : int, optional
            Number of p-values corrected together in a window (default: the number of metrics calibrated), which
            sets the number of draws needed by Benjamini-Hochberg
        cache : CalibrationCache, optional
            Reuse null distributions already computed for the same feature values and settings
        """
        specs = [m for m in metrics if isinstance(m, (DriftMetric, DriftTestMetric))]
        if not specs:
            raise error_msg(value=None, message="no drift metric to calibrate")
        n = len(reference)
        n_windows = self._check_reference(n)
        n_draws = self._resolve_draws(n_tests or len(specs), len(specs))
        self.n_draws_used = n_draws
        whole_windows = int(n_windows)
        if self.compare_to == "reference":
            # one window against the rest: about one independent draw per disjoint window of the reference
            n_effective = max(1, min(n_draws, whole_windows))
        else:
            n_effective = max(1, min(n_draws, whole_windows * (whole_windows - 1) // 2))

        draws = None  # drawn on first need: a fully cached fit draws nothing
        fitted, failed = 0, {}
        with contextlib.redirect_stdout(io.StringIO()):
            for metric in specs:
                key = (metric._name, metric._feature_name)
                try:
                    column = metric._extract(reference)
                    cache_key = self._cache_key(metric, column, n_draws) if cache is not None else None
                    values = cache.get(cache_key) if cache_key is not None else None
                    if values is None:
                        if draws is None:
                            draws = self._draw_positions(n, n_draws, np.random.default_rng(self.seed))
                        values = self._null_values(metric, column, draws)
                        if cache_key is not None:
                            cache.put(cache_key, values)
                    self.nulls[key] = self._null_distribution(metric, column, values, n_windows, n_effective)
                    self.errors.pop(key, None)
                    fitted += 1
                except Exception as e:
                    failed[key] = str(e)
        for key, message in failed.items():
            self.errors[key] = message
            self.nulls.pop(key, None)
            warnings.warn(f"calibration of {key[0]} on {key[1]} failed: {message}", UserWarning, stacklevel=2)
        if fitted == 0:
            raise error_msg(value=None, message=f"no metric could be calibrated: {failed}")
        return self

    def _null_distribution(self, metric: MetricSpec, column, values: np.ndarray, n_windows, n_effective) -> NullDistribution:
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        n_failed = int(len(values) - len(finite))
        if len(finite) == 0:
            raise error_msg(value=metric._name, message="the statistic could not be computed on any A/A draw")
        if n_failed > len(values) / 2:
            raise error_msg(
                value=metric._name, message=f"the statistic could not be computed on {n_failed} of {len(values)} A/A draws"
            )
        status = STATUS_OK
        if metric._name not in MISSINGNESS_METRICS and is_constant(column):
            status = STATUS_CONSTANT
            warnings.warn(
                f"{metric._feature_name} is constant in the reference (imputed, or not yet in use): the calibrated "
                f"threshold of {metric._name} is degenerate and any variation will be flagged; status {STATUS_CONSTANT}",
                UserWarning,
                stacklevel=4,
            )
        share_at_bound = self._share_at_bound(metric, finite)
        if status == STATUS_OK and share_at_bound > SATURATION_SHARE:
            status = STATUS_SATURATED
            bound = "a p-value of 0" if isinstance(metric, DriftTestMetric) else "the epsilon ceiling"
            warnings.warn(
                f"{share_at_bound:.0%} of the A/A draws of {metric._name} on {metric._feature_name} sit at {bound}, the bound "
                f"of the statistic: a drift cannot go beyond the null, the calibrated verdict is blind; status "
                f"{STATUS_SATURATED}. The reference likely mixes regimes (e.g. monthly values): use wasserstein, which is "
                "unbounded, for this feature, or a reference per regime or period (period_reference)",
                UserWarning,
                stacklevel=4,
            )
        if status == STATUS_OK and metric._name not in MISSINGNESS_METRICS and len(np.unique(finite)) <= 2:
            status = STATUS_DEGENERATE
            warnings.warn(
                f"the A/A distribution of {metric._name} on {metric._feature_name} takes {len(np.unique(finite))} distinct "
                f"value(s): the threshold is degenerate; status {STATUS_DEGENERATE}",
                UserWarning,
                stacklevel=4,
            )
        return NullDistribution(
            metric_name=metric._name,
            feature_name=metric._feature_name,
            kind="pvalue" if isinstance(metric, DriftTestMetric) else "distance",
            values=finite,
            quantile=self.quantile,
            n_windows=n_windows,
            n_effective=min(n_effective, len(finite)),
            status=status,
            n_failed_draws=n_failed,
            shrink_draws=self.shrink_draws,
            reliable_windows=self.reliable_windows,
            share_at_bound=share_at_bound,
            between_within_ratio=self._between_within_ratio(metric, column),
            window_size=self.window_size,
        )

    @staticmethod
    def _share_at_bound(metric: MetricSpec, finite: np.ndarray) -> float:
        """Share of A/A draws at the critical bound of the statistic (p-value 0; psi / kl dominated by the epsilon)

        Ties at the other end (p-values of 1) leave the decision tail intact and are not counted.
        """
        if isinstance(metric, DriftTestMetric):
            return float((finite <= SATURATED_PVALUE).mean())
        if metric._name in BOUNDED_DISTANCES:
            return float((finite >= SATURATED_DIVERGENCE).mean())
        return 0.0  # wasserstein, mmd: unbounded

    def _between_within_ratio(self, metric: MetricSpec, column) -> Optional[float]:
        """window_size x variance of the means of consecutive windows / mean within-window variance (numeric features)

        About 1 when the windows of the reference differ by sampling noise only, (1 + phi) / (1 - phi) for AR(1)
        traffic, and far above for a reference spanning seasons or regimes: the A/A null then measures how different
        two periods of the reference are, and a window is only compared with that spread.
        """
        numeric = isinstance(column, pd.Series) and is_numeric_dtype(column) and not is_bool_dtype(column)
        if not numeric or not self.contiguous:
            return None
        w = self.window_size
        array = column.to_numpy(dtype=float)
        k = len(array) // w
        if k < 2:
            return None
        chunks = array[: k * w].reshape(k, w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(chunks, axis=1)
            within = np.nanmean(np.nanvar(chunks, axis=1, ddof=1))
            between = np.nanvar(means, ddof=1)
        if not np.isfinite(within) or within <= 0 or not np.isfinite(between):
            return None
        ratio = float(w * between / within)
        if ratio > BETWEEN_WITHIN_WARNING and metric._name not in MISSINGNESS_METRICS:
            warnings.warn(
                f"{metric._feature_name}: the windows of the reference differ far more than sampling noise (between / "
                f"within-window variance ratio {ratio:.0f}, about 1 for stationary data): the A/A null of {metric._name} "
                "absorbs the differences between periods (season, regimes), and a drift of that size goes unseen. On "
                "seasonal data, calibrate on and compare with the same period of the reference (period_reference)",
                UserWarning,
                stacklevel=4,
            )
        return ratio

    # ------------------------------------------------------------------ using the calibration
    def options(self, include_reliability: bool = True) -> dict:
        """Per-feature Analyzer options with the calibrated thresholds: {"metric|feature": {"threshold" or "alpha"}}

        With include_reliability (default), each entry also holds "reliable", "calibration_status" and "window_size",
        used by the drift metrics on their results: reliable=False marks a verdict as provisional (short reference,
        constant reference, degenerate or saturated null, current window more than 2x smaller or larger than
        window_size), with a reliability_reason.
        """
        options = {}
        for (metric_name, feature), null in self.nulls.items():
            key = "alpha" if null.kind == "pvalue" else "threshold"
            entry = {key: null.threshold}
            if include_reliability:
                entry["reliable"] = null.reliable
                entry["calibration_status"] = null.status
                entry["window_size"] = null.window_size or self.window_size
            options[f"{metric_name}|{feature}"] = entry
        return options

    @property
    def pvalue_floor(self) -> Optional[float]:
        """Largest empirical p-value floor over the fitted distributions (None before fit)"""
        return max((null.pvalue_floor for null in self.nulls.values()), default=None)

    def empirical_pvalues(self, results: pd.DataFrame, fdr_alpha: Optional[float] = None) -> pd.Series:
        """Empirical p-value of each row of Analyzer.results_to_pandas() (NaN for rows without null distribution)

        Warns when the p-value floor of the distributions is above the Benjamini-Hochberg threshold fdr_alpha / m
        for the m rows that get a p-value (default fdr_alpha: the one given to the calibrator).
        """

        def pvalue(row) -> float:
            null = self.nulls.get((row["metric_name"], row["feature_name"]))
            return null.empirical_pvalue(row["metric_value"]) if null else float("nan")

        pvalues = results.apply(pvalue, axis=1) if len(results) else pd.Series(dtype=float)
        alpha = self.fdr_alpha if fdr_alpha is None else fdr_alpha
        m = int(pvalues.notna().sum())
        floors = [
            self.nulls[(r["metric_name"], r["feature_name"])].pvalue_floor
            for _, r in results.iterrows()
            if (r["metric_name"], r["feature_name"]) in self.nulls
        ]
        if "n" in results.columns and len(results):
            sizes = pd.to_numeric(results["n"], errors="coerce")
            off = sizes.notna() & ((sizes * 2 < self.window_size) | (sizes > 2 * self.window_size)) & pvalues.notna()
            if off.any():
                warnings.warn(
                    f"{int(off.sum())} statistic(s) come from windows of {int(sizes[off].min())} to {int(sizes[off].max())} "
                    f"rows, more than 2x away from the calibrated window of {self.window_size}: their empirical p-values "
                    "do not follow the null and the BH verdicts are unreliable (reliable=False on the results). "
                    "Calibrate on the size of the windows analysed",
                    UserWarning,
                    stacklevel=2,
                )
        if alpha and m and max(floors) > alpha / m:
            warnings.warn(
                f"empirical p-values cannot go below {max(floors):.5f}, above the BH threshold {alpha}/{m} = {alpha / m:.5f}: "
                "a drift on a single feature cannot pass the correction. Refit with more draws (auto_draws, n_tests)",
                UserWarning,
                stacklevel=2,
            )
        return pvalues

    def false_alarm_rate(self, metric_name: str, feature_name: str) -> Optional[float]:
        """Share of A/A draws flagged by the nominal rule (p < 0.05 for tests); None for distances"""
        null = self.nulls[(metric_name, feature_name)]
        return float((null.values < 0.05).mean()) if null.kind == "pvalue" else None

    def summary(self, level: float = 0.90) -> pd.DataFrame:
        """One row per calibrated (metric, feature): threshold, its interval, reliability, status, p-value floor"""
        rows = []
        for null in self.nulls.values():
            low, high = null.threshold_interval(level)
            rows.append(
                {
                    "metric_name": null.metric_name,
                    "feature_name": null.feature_name,
                    "kind": null.kind,
                    "threshold": null.threshold,
                    "threshold_low": low,
                    "threshold_high": high,
                    "reliable": null.reliable,
                    "status": null.status,
                    "n_draws": len(null.values),
                    "n_failed_draws": null.n_failed_draws,
                    "n_effective": null.n_effective,
                    "n_windows": null.n_windows,
                    "pvalue_floor": null.pvalue_floor,
                    "share_at_bound": null.share_at_bound,
                    "between_within_ratio": null.between_within_ratio,
                }
            )
        for (metric_name, feature), message in self.errors.items():
            rows.append({"metric_name": metric_name, "feature_name": feature, "status": "error", "error": message})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ persistence
    _SETTINGS = (
        "window_size",
        "n_draws",
        "quantile",
        "contiguous",
        "seed",
        "allow_short",
        "fdr_alpha",
        "auto_draws",
        "max_draws",
        "shrink_draws",
        "reliable_windows",
        "compare_to",
    )

    def save(self, path: Union[str, Path]):
        """Write the calibration (settings and null distributions) to a JSON file, to reuse it in another process"""
        payload = {
            "format": "pulsar-aa-calibration",
            "version": 1,
            "settings": {name: getattr(self, name) for name in self._SETTINGS},
            "n_draws_used": self.n_draws_used,
            "nulls": [
                {
                    "metric_name": null.metric_name,
                    "feature_name": null.feature_name,
                    "kind": null.kind,
                    "values": null.values.tolist(),
                    "n_windows": null.n_windows,
                    "n_effective": null.n_effective,
                    "status": null.status,
                    "n_failed_draws": null.n_failed_draws,
                    "share_at_bound": null.share_at_bound,
                    "between_within_ratio": null.between_within_ratio,
                }
                for null in self.nulls.values()
            ],
            "errors": [[k[0], k[1], v] for k, v in self.errors.items()],
        }
        Path(path).write_text(json.dumps(payload, default=_json_default), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AACalibrator":
        """Read a calibration written by save()"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "pulsar-aa-calibration":
            raise error_msg(value=str(path), message="not an AACalibrator file")
        # files written before compare_to existed were calibrated window against window
        calibrator = cls(**{"compare_to": "window", **payload["settings"]})
        calibrator.n_draws_used = payload.get("n_draws_used")
        for item in payload["nulls"]:
            null = NullDistribution(
                metric_name=item["metric_name"],
                feature_name=item["feature_name"],
                kind=item["kind"],
                values=np.asarray(item["values"], dtype=float),
                quantile=calibrator.quantile,
                n_windows=item["n_windows"],
                n_effective=item["n_effective"],
                status=item["status"],
                n_failed_draws=item["n_failed_draws"],
                shrink_draws=calibrator.shrink_draws,
                reliable_windows=calibrator.reliable_windows,
                share_at_bound=item.get("share_at_bound", 0.0),
                between_within_ratio=item.get("between_within_ratio"),
                window_size=calibrator.window_size,
            )
            calibrator.nulls[(null.metric_name, null.feature_name)] = null
        calibrator.errors = {(m, f): message for m, f, message in payload.get("errors", [])}
        return calibrator


def period_reference(
    reference: pd.DataFrame,
    current: Union[pd.DataFrame, pd.Timestamp, str],
    weeks: int = 4,
    time_column: str = "pred_timestamp",
) -> pd.DataFrame:
    """Rows of the reference in the same period of the year as the current window, +- `weeks` weeks

    On seasonal data, a reference spanning a whole year makes the A/A null compare winter with summer: the calibrated
    thresholds absorb the season and a real change of that size goes unseen (lab TS-04, +63 % growth on bike rentals:
    confirmed in 0 % of the weeks with the whole of 2011 as reference, 76 % with the same weeks of 2011). Calibrate on
    the returned rows and compare the window with the same rows:

        ref = period_reference(reference, window, weeks=4)
        cal = AACalibrator(window_size=len(window), allow_short=True).fit(ref, metrics, cache=cache)
        analyzer.run(current=window, reference=ref, options=cal.options())

    Parameters
    ----------
    reference : DataFrame
        Reference data with a time column; may span several years (every year contributes its matching weeks)
    current : DataFrame or timestamp
        Current window (its time span is used) or one date
    weeks : int, optional
        Margin on each side of the window, in weeks (default 4)
    time_column : str, optional
        Name of the time column (default "pred_timestamp")

    Returns
    -------
    DataFrame
        the matching rows of the reference, in their original order
    """
    if weeks < 0:
        raise error_msg(value=weeks, message="weeks must be >= 0")
    if isinstance(current, pd.DataFrame):
        times = pd.to_datetime(current[time_column], format="mixed")
        start, end = times.min(), times.max()
    else:
        start = end = pd.Timestamp(current)
    ref_days = pd.to_datetime(reference[time_column], format="mixed").dt.day_of_year.to_numpy()
    margin = 7 * weeks
    low, high = start.day_of_year - margin, start.day_of_year + (end - start).days + margin
    if high - low >= 365:
        return reference
    # day of year on a circle: a window in early January matches late December of the reference
    days = ref_days.astype(float)
    selected = np.zeros(len(days), dtype=bool)
    for shift in (-366, -365, 0, 365, 366):
        selected |= (days + shift >= low) & (days + shift <= high)
    return reference.loc[selected]


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not serializable: {type(value)}")
