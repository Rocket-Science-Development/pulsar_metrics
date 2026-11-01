"""Calibration of drift thresholds on A/A draws, and empirical p-values.

A nominal threshold (p < 0.05, PSI > 0.2...) assumes independent samples and a deterministic system. Neither holds for
LLM and agent traffic: outputs vary between runs even at temperature 0, and successive requests are correlated. A
threshold applied naively then raises alerts without any real change.

AACalibrator builds, for each (metric, feature), the distribution of the statistic between two windows drawn from the
same reference ("A/A"), and derives:
- a calibrated threshold: the quantile of that null distribution (e.g. 95 %), usable as Analyzer options;
- an empirical p-value for any observed statistic, distance or test alike, on which multiple-testing correction applies.
"""

import contextlib
import io
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from .exceptions import CustomExceptionPulsarMetric as error_msg
from .metrics.drift import DriftMetric, DriftTestMetric


@dataclass
class NullDistribution:
    """Distribution of a statistic between two windows of the same reference"""

    metric_name: str
    feature_name: str
    kind: str  # "distance" (drift when large) or "pvalue" (drift when small)
    values: np.ndarray = field(repr=False)
    quantile: float

    @property
    def threshold(self) -> float:
        """Distance above which, or p-value below which, only (1 - quantile) of A/A draws fall"""
        if self.kind == "distance":
            return float(np.quantile(self.values, self.quantile))
        return float(np.quantile(self.values, 1 - self.quantile))

    def empirical_pvalue(self, observed: float) -> float:
        """Share of A/A draws at least as extreme as the observed statistic (with the +1 correction)"""
        if observed is None or np.isnan(observed):
            return float("nan")
        extreme = self.values >= observed if self.kind == "distance" else self.values <= observed
        return float((1 + extreme.sum()) / (1 + len(self.values)))


MetricSpec = Union[DriftMetric, DriftTestMetric]


class AACalibrator:
    """Calibrate drift thresholds on A/A draws from a reference dataset

    Parameters
    ----------
    window_size : int
        Size of the windows compared in production; the null distribution depends strongly on it
    n_draws : int, optional
        Number of A/A draws per (metric, feature) (default 200)
    quantile : float, optional
        Quantile of the null distribution used as threshold (default 0.95, i.e. 5 % false alarms per test)
    contiguous : bool, optional
        Draw contiguous windows from the reference, which must then be sorted by time (default True). Keeps the
        temporal dependence of real traffic; random rows (False) would hide it and under-estimate false alarms.
        The reference should hold at least 10 windows: with fewer, the A/A draws overlap too much to estimate the
        null distribution (a warning is emitted).
    seed : int, optional
        Random seed (default 0)
    """

    def __init__(self, window_size: int, n_draws: int = 200, quantile: float = 0.95, contiguous: bool = True, seed=0):
        if window_size < 2 or n_draws < 20 or not 0.5 < quantile < 1:
            raise error_msg(value=None, message="window_size >= 2, n_draws >= 20 and 0.5 < quantile < 1 are required")
        self.window_size = window_size
        self.n_draws = n_draws
        self.quantile = quantile
        self.contiguous = contiguous
        self.seed = seed
        self.nulls: dict[tuple, NullDistribution] = {}

    # ------------------------------------------------------------------ fitting
    def _windows(self, reference: pd.DataFrame, rng: np.random.Generator):
        n, w = len(reference), self.window_size
        if self.contiguous:
            if n < 2 * w:
                raise error_msg(value=n, message=f"contiguous A/A draws need at least 2 x window_size = {2 * w} rows")
            if n < 10 * w:
                warnings.warn(
                    f"reference of {n} rows holds only {n // w} windows of {w}: A/A draws overlap and the calibrated "
                    f"thresholds are unreliable; use at least {10 * w} rows",
                    UserWarning,
                    stacklevel=3,
                )
            for _ in range(self.n_draws):
                a, b = rng.choice(n - w + 1, size=2, replace=False)
                while abs(int(a) - int(b)) < w:  # disjoint windows
                    a, b = rng.choice(n - w + 1, size=2, replace=False)
                yield reference.iloc[a : a + w], reference.iloc[b : b + w]
        else:
            replace = n < 2 * w
            for _ in range(self.n_draws):
                idx = rng.choice(n, size=2 * w, replace=replace)
                yield reference.iloc[idx[:w]], reference.iloc[idx[w:]]

    @staticmethod
    def _evaluate(metric: MetricSpec, current: pd.DataFrame, reference: pd.DataFrame) -> float:
        fresh = type(metric)(metric_name=metric._name, feature_name=metric._feature_name)
        with contextlib.redirect_stdout(io.StringIO()):
            result = fresh.evaluate(current=current, reference=reference)
        return float(result.metric_value)

    def fit(self, reference: pd.DataFrame, metrics: Iterable[MetricSpec]) -> "AACalibrator":
        """Build the null distribution of every drift metric (DriftMetric or DriftTestMetric) on the reference

        `metrics` can be `analyzer._metrics_list`; performance and custom metrics are ignored.
        """
        rng = np.random.default_rng(self.seed)
        specs = [m for m in metrics if isinstance(m, (DriftMetric, DriftTestMetric))]
        if not specs:
            raise error_msg(value=None, message="no drift metric to calibrate")
        draws = list(self._windows(reference, rng))
        for metric in specs:
            values = np.array([self._evaluate(metric, a, b) for a, b in draws])
            kind = "pvalue" if isinstance(metric, DriftTestMetric) else "distance"
            self.nulls[(metric._name, metric._feature_name)] = NullDistribution(
                metric_name=metric._name,
                feature_name=metric._feature_name,
                kind=kind,
                values=values,
                quantile=self.quantile,
            )
        return self

    # ------------------------------------------------------------------ using the calibration
    def options(self) -> dict:
        """Per-feature Analyzer options with the calibrated thresholds: {"metric|feature": {"threshold" or "alpha"}}"""
        options = {}
        for (metric_name, feature), null in self.nulls.items():
            key = "alpha" if null.kind == "pvalue" else "threshold"
            options[f"{metric_name}|{feature}"] = {key: null.threshold}
        return options

    def empirical_pvalues(self, results: pd.DataFrame) -> pd.Series:
        """Empirical p-value of each row of Analyzer.results_to_pandas() (NaN for rows without null distribution)"""

        def pvalue(row) -> float:
            null = self.nulls.get((row["metric_name"], row["feature_name"]))
            return null.empirical_pvalue(row["metric_value"]) if null else float("nan")

        return results.apply(pvalue, axis=1)

    def false_alarm_rate(self, metric_name: str, feature_name: str) -> Optional[float]:
        """Share of A/A draws flagged by the nominal rule (p < 0.05 for tests); None for distances"""
        null = self.nulls[(metric_name, feature_name)]
        return float((null.values < 0.05).mean()) if null.kind == "pvalue" else None
