#  Copyright Développement Science Fusée inc.
#  SPDX-License-Identifier: Apache-2.0
import warnings
from numbers import Number
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..exceptions import MetricEvaluationError
from ..utils import drift_from_threshold
from . import constant
from .base import AbstractMetrics, MetricResults, MetricsType
from .enums import DriftMetricsFuncs, DriftTestMetricsFuncs
from .utils import get_population_counts, low_expected_share

NAN_POLICIES = ("omit", "propagate", "raise")
# Metrics about the missing values themselves: NaN are kept, and a reference without variation is normal
MISSINGNESS_METRICS = ("missing_rate",)
# Tests of the rate of a 0 / 1 indicator: a reference without any positive (zero base rate) is normal, not degenerate
RATE_METRICS = ("rate_increase",)
CONSTANT_REFERENCE = "constant_reference"
# Functions that take numpy arrays as well as pandas series (scipy statistics)
ARRAY_FUNCTIONS = ("wasserstein", "ttest", "manwu", "levene", "bftest", "ks_2samp", "CvM", "rate_increase")
# Metrics computed on a table of counts per level or bin: they accept level grouping and report low_expected_share
COUNT_METRICS = ("chi2", "psi", "kl")
SPARSE_SHARE = 0.2  # above this share of cells with expected count < 5, a warning suggests grouping the levels
WINDOW_RATIO = 2  # a window more than this factor away from the calibrated window size gets reliable=False

MissingValues = Union[None, object, Iterable]


def normalize_missing_values(missing_values: MissingValues) -> tuple:
    """Sentinel values declared as missing: None, one value or a list of values, returned as a tuple"""
    if missing_values is None:
        return ()
    if isinstance(missing_values, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
        return tuple(missing_values)
    return (missing_values,)


def mask_missing(data: Union[pd.Series, pd.DataFrame], missing_values: tuple):
    """Replace the declared sentinel values by NaN (no copy when there is none)"""
    if not missing_values:
        return data
    return data.mask(data.isin(list(missing_values)))


def missing_share(data: Union[pd.Series, pd.DataFrame]) -> Optional[float]:
    """Share of missing values; for a DataFrame, share of rows with at least one missing value"""
    if len(data) == 0:
        return None
    missing = data.isna().any(axis=1) if isinstance(data, pd.DataFrame) else data.isna()
    return float(missing.mean())


def is_constant(data: Union[pd.Series, pd.DataFrame]) -> bool:
    """True when the non-missing values hold a single distinct value (or none)"""
    if isinstance(data, pd.DataFrame):
        return all(data[c].nunique(dropna=True) <= 1 for c in data.columns)
    return data.nunique(dropna=True) <= 1


def _is_nan(value) -> bool:
    return value is None or (isinstance(value, Number) and not isinstance(value, bool) and bool(np.isnan(value)))


def _align_dtypes(new, reference):
    """Numeric series of different dtypes (int vs float, e.g. after masking a sentinel) are compared as float"""
    if (
        isinstance(new, pd.Series)
        and new.dtype != reference.dtype
        and is_numeric_dtype(new)
        and is_numeric_dtype(reference)
        and not is_bool_dtype(new)
        and not is_bool_dtype(reference)
    ):
        return new.astype(float), reference.astype(float)
    return new, reference


class _FeatureDriftMetric(AbstractMetrics):
    """Shared handling of declared missing values, NaN and constant references for drift metrics and tests"""

    _funcs = None  # enum of the metric functions
    _kind = "drift metric"

    def __init__(
        self,
        metric_name: str,
        feature_name: str,
        missing_values: MissingValues = None,
        nan_policy: str = "omit",
        min_expected: Optional[float] = None,
        max_levels: Optional[int] = None,
    ):
        super().__init__(metric_name)
        self._check_metrics_name(metric_name)
        self._feature_name = feature_name
        self._missing_values = normalize_missing_values(missing_values)
        self._check_nan_policy(nan_policy)
        self._nan_policy = nan_policy
        # level grouping of chi2 / psi / kl, applied alike in evaluate() and in the A/A draws of AACalibrator
        self._grouping = {}
        if metric_name in COUNT_METRICS:
            self._grouping = {k: v for k, v in (("min_expected", min_expected), ("max_levels", max_levels)) if v is not None}

    @staticmethod
    def _check_nan_policy(nan_policy: str):
        if nan_policy not in NAN_POLICIES:
            raise error_msg(value=nan_policy, message=f"nan_policy must be one of {NAN_POLICIES}")

    def _check_metrics_name(self, name: str):
        if name not in self._funcs._member_names_:
            raise error_msg(value=name, message=f"InvalidInput: unknown metric key {name} given.")

    def _extract(self, data: pd.DataFrame):
        """Column of the feature (whole frame when feature_name is None), declared sentinels turned into NaN"""
        column = data[self._feature_name] if self._feature_name is not None else data
        return mask_missing(column, self._missing_values)

    def _clean(self, new, reference, nan_policy: Optional[str] = None):
        """Apply the NaN policy; return (new, reference, missing shares)"""
        policy = nan_policy or self._nan_policy
        self._check_nan_policy(policy)
        info = {"missing_share_current": missing_share(new), "missing_share_reference": missing_share(reference)}
        if self._name in MISSINGNESS_METRICS:
            return new, reference, info
        if policy == "raise" and (info["missing_share_current"] or info["missing_share_reference"]):
            raise error_msg(value=self._feature_name, message="missing values found and nan_policy='raise'")
        if policy == "omit":
            new, reference = new.dropna(), reference.dropna()
            if len(new) == 0 or len(reference) == 0:
                raise error_msg(value=self._feature_name, message="no non-missing value left in one of the samples")
        new, reference = _align_dtypes(new, reference)
        return new, reference, info

    def _raw_value(self, result):
        return result

    def _value(self, new, reference, nan_policy: Optional[str] = None, **kwargs) -> float:
        """Statistic on columns already extracted with _extract() (fast path of AACalibrator); NaN when undefined"""
        new, reference, _ = self._clean(new, reference, nan_policy)
        value = self._raw_value(self._funcs[self._name].value(new, reference, **{**self._grouping, **kwargs}))
        return float("nan") if _is_nan(value) else float(value)

    def _value_array(self, new: np.ndarray, reference: np.ndarray) -> float:
        """Same as _value() on float arrays of a numeric feature (A/A draws: avoids the pandas overhead)"""
        if self._name not in MISSINGNESS_METRICS and self._nan_policy != "propagate":
            new_nan, ref_nan = np.isnan(new), np.isnan(reference)
            if new_nan.any() or ref_nan.any():
                if self._nan_policy == "raise":
                    return float("nan")
                new, reference = new[~new_nan], reference[~ref_nan]
            if len(new) == 0 or len(reference) == 0:
                return float("nan")
        if self._name in ARRAY_FUNCTIONS:
            raw = self._funcs[self._name].value(new, reference)
        else:
            raw = self._funcs[self._name].value(pd.Series(new), pd.Series(reference), **self._grouping)
        value = self._raw_value(raw)
        return float("nan") if _is_nan(value) else float(value)

    def _low_expected_share(self, new, ref, kwargs) -> Optional[float]:
        """Share of cells with expected count < 5 in the table chi2 / psi / kl were computed on (None otherwise)"""
        if self._name not in COUNT_METRICS or kwargs.get("binned"):
            return None
        grouping = {k: v for k, v in {**self._grouping, **kwargs}.items() if k in ("min_expected", "max_levels")}
        try:
            share = low_expected_share(get_population_counts(new, ref, **grouping))
        except Exception:
            return None
        categorical = isinstance(new, pd.Series) and not is_numeric_dtype(new)  # numeric bins: sparse tails are expected
        if share is not None and share > SPARSE_SHARE and not grouping and categorical:
            warnings.warn(
                f"{self._name} on {self._feature_name}: {share:.0%} of the cells have an expected count below 5 (many sparse "
                "levels): the test loses power. Group the levels with min_expected=5 or max_levels= (add_drift_metrics)",
                UserWarning,
                stacklevel=4,
            )
        return share

    def _evaluate(
        self, current, reference, verdict, threshold, nan_policy, reliable, kwargs, calibration_status=None, window_size=None
    ) -> MetricResults:
        try:
            new = self._extract(current)
            ref = self._extract(reference)
            self._column = new
            n_rows = len(new)  # rows of the window, missing values included, as in the A/A windows
            new, ref, info = self._clean(new, ref, nan_policy)
            value = self._raw_value(self._funcs[self._name].value(new, ref, **{**self._grouping, **kwargs}))
            constant_reference = self._name not in MISSINGNESS_METRICS + RATE_METRICS and is_constant(ref)
        except Exception as e:
            raise MetricEvaluationError(
                value=self._name, message=f"Cannot evaluate {self._kind} {self._name} on {self._feature_name}: {e}"
            ) from e

        status = CONSTANT_REFERENCE if constant_reference else None
        reasons = []
        if calibration_status is not None and calibration_status != "ok":
            # e.g. "saturated": the calibrated threshold cannot separate a drift from the null (AACalibrator)
            status = status or calibration_status
            reliable = False
            reasons.append(f"calibrated null distribution {calibration_status}")
        if window_size and (n_rows * WINDOW_RATIO < window_size or n_rows > WINDOW_RATIO * window_size):
            # the null distribution, hence the threshold, depends strongly on the window size (lab TC-04: 10 to 67 rows
            # judged with thresholds calibrated on 1000)
            reliable = False
            reasons.append(f"window of {n_rows} rows, threshold calibrated on windows of {window_size} rows")
        common = dict(
            metric_name=self._name,
            metric_type=MetricsType.drift.value,
            feature_name=self._feature_name,
            conf_int=None,
            threshold=threshold,
            status=status,
            reliable=reliable,
            reliability_reason="; ".join(reasons) or None,
            n=n_rows,
            low_expected_share=self._low_expected_share(new, ref, kwargs),
            **info,
        )
        if _is_nan(value):
            # NaN compared with a threshold gives an arbitrary verdict (True for distances, False for tests): withhold it
            self._result = MetricResults(
                metric_value=None,
                drift_status=None,
                error=f"the statistic of {self._name} on {self._feature_name} is NaN (missing or degenerate data): no verdict",
                **common,
            )
        else:
            self._result = MetricResults(metric_value=value, drift_status=verdict(value), **common)
        return self._result


class DriftMetric(_FeatureDriftMetric):
    """Distance between the current and reference distributions of a feature (verdict on `threshold`)

    Parameters
    ----------
    metric_name : str
        One of DriftMetricsFuncs (psi, kl, wasserstein, mmd)
    feature_name : str
        Feature analyzed (None: the whole frame, for mmd)
    missing_values : scalar or list, optional
        Sentinel values meaning "no value" (e.g. -200), turned into NaN before the computation
    nan_policy : str, optional
        "omit" (default): missing values are set aside and their share is recorded in `missing_share_current` /
        `missing_share_reference`; "propagate": passed to the function as is; "raise": error when any is found.
        Whatever the policy, a NaN statistic gives `drift_status=None` and an `error`.
    min_expected : float, optional
        psi and kl: merge the levels (or bins) whose expected count in the current window, computed from the
        reference shares, is below it into one "other" level (default None: no grouping), see utils.group_levels
    max_levels : int, optional
        psi and kl: keep at most this many levels, "other" included (default None)
    """

    _funcs = DriftMetricsFuncs
    _kind = "drift metric"

    def __init__(
        self,
        metric_name: str,
        feature_name: str,
        missing_values: MissingValues = None,
        nan_policy: str = "omit",
        min_expected: Optional[float] = None,
        max_levels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            metric_name,
            feature_name,
            missing_values=missing_values,
            nan_policy=nan_policy,
            min_expected=min_expected,
            max_levels=max_levels,
        )

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
        threshold: Union[list, float, int] = None,
        upper_bound: bool = True,
        nan_policy: Optional[str] = None,
        reliable: Optional[bool] = None,
        calibration_status: Optional[str] = None,
        window_size: Optional[int] = None,
        **kwargs,
    ) -> MetricResults:
        """Method evaluate() to evaluate the DriftMetric

        Parameters
        ----------
        current : DataFrame
                The input current (pandas DataFrame)
        reference : DataFrame
                The input reference (pandas DataFrame)
        threshold : Union[list, float, int]
                Threshold values to validate the input value
        upper_bound : bool, optional
                True (default) if the threshold is the maximum acceptable value: drift_status is True when
                the value exceeds it. A [Min, Max] threshold flags values outside the interval
        nan_policy : str, optional
                Overrides the policy given at construction ("omit", "propagate" or "raise")
        reliable : bool, optional
                Reliability of a calibrated threshold (AACalibrator.options()), copied to the result
        calibration_status : str, optional
                Status of the calibrated null (AACalibrator.options()); other than "ok" (e.g. "saturated"), it is
                copied to `status` and the result gets reliable=False
        window_size : int, optional
                Window size of the calibration (AACalibrator.options()); a current window more than 2x smaller or
                larger gets reliable=False and a `reliability_reason`
        kwargs :
                keyworded variable length of arguments to a function

        Returns
        -------
        MetricResults
                returns the result of the calculated DriftMetric
        """
        return self._evaluate(
            current,
            reference,
            lambda value: drift_from_threshold(value, threshold, upper_bound),
            threshold,
            nan_policy,
            reliable,
            kwargs,
            calibration_status=calibration_status,
            window_size=window_size,
        )


class DriftTestMetric(_FeatureDriftMetric):
    """Statistical test between the current and reference samples of a feature (verdict on `alpha`)

    `metric_value` is the p-value. Same `missing_values` and `nan_policy` parameters as DriftMetric, and the same
    `min_expected` / `max_levels` level grouping for chi2. The `missing_rate` test compares the share of missing values
    itself: its NaN are never set aside. The `rate_increase` test compares the share of positives of a 0 / 1 indicator,
    one-sided (alert when the window rate is higher); its `method` gives both rates ("window 3/8 = 0.375 vs reference
    0/80 = 0").
    """

    _funcs = DriftTestMetricsFuncs
    _kind = "drift test"

    def __init__(
        self,
        metric_name: str,
        feature_name: str,
        missing_values: MissingValues = None,
        nan_policy: str = "omit",
        min_expected: Optional[float] = None,
        max_levels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            metric_name,
            feature_name,
            missing_values=missing_values,
            nan_policy=nan_policy,
            min_expected=min_expected,
            max_levels=max_levels,
        )

    def _raw_value(self, result):
        self._raw = result  # rate_increase: the counts behind the p-value, for `method`
        return result.pvalue

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
        alpha: float = constant.SIGNIFICANCE_LEVEL,
        nan_policy: Optional[str] = None,
        reliable: Optional[bool] = None,
        calibration_status: Optional[str] = None,
        window_size: Optional[int] = None,
        **kwargs,
    ) -> MetricResults:
        """Method evaluate() to evaluate the DriftTestMetric

        Parameters
        ----------
        current : DataFrame
            The input current (pandas DataFrame)
        reference : DataFrame
            The input reference (pandas DataFrame)
        alpha : float
            Value to define significance level
        nan_policy : str, optional
            Overrides the policy given at construction ("omit", "propagate" or "raise")
        reliable : bool, optional
            Reliability of a calibrated alpha (AACalibrator.options()), copied to the result
        calibration_status, window_size : optional
            From AACalibrator.options(), see DriftMetric.evaluate
        kwargs :
            keyworded variable length of arguments to a function

        Returns
        -------
        MetricResults
            returns the result of the test; metric_value is the p-value
        """
        result = self._evaluate(
            current,
            reference,
            lambda pvalue: pvalue < alpha if isinstance(alpha, (int, float)) else None,
            alpha,
            nan_policy,
            reliable,
            kwargs,
            calibration_status=calibration_status,
            window_size=window_size,
        )
        if self._name in RATE_METRICS:
            result.method = rate_method(self._raw)
        return result


def rate_method(raw) -> str:
    """`method` of a rate_increase result: the two rates behind the p-value (read back by pulsar_metrics.evidence)"""
    current = raw.current_count / raw.current_n
    reference = raw.reference_count / raw.reference_n
    return (
        f"one-sided Fisher exact test, window rate higher: window {raw.current_count}/{raw.current_n} = {current:.4g} "
        f"vs reference {raw.reference_count}/{raw.reference_n} = {reference:.4g}"
    )


def CustomDriftMetric(func):
    """Decorator for custom metrics"""

    def inner(metric_name: str, feature_name: str) -> AbstractMetrics:
        class CustomClass(AbstractMetrics):
            def __init__(self, metric_name, feature_name):
                super().__init__(metric_name)
                self._feature_name = feature_name

            def evaluate(self, current: pd.DataFrame, reference: pd.DataFrame, **kwargs):
                value = func(current[self._feature_name], reference[self._feature_name], **kwargs)
                threshold = kwargs.get("threshold", None)
                upper_bound = kwargs.get("upper_bound", True)

                status = drift_from_threshold(value, threshold, upper_bound)

                self._result = MetricResults(
                    metric_name=self._name,
                    metric_type=MetricsType.custom.value,
                    feature_name=self._feature_name,
                    metric_value=value,
                    conf_int=None,
                    drift_status=status,
                    threshold=threshold,
                )

                return self._result

        return CustomClass(metric_name=metric_name, feature_name=feature_name)

    return inner
