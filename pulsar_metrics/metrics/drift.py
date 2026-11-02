#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>
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

NAN_POLICIES = ("omit", "propagate", "raise")
# Metrics about the missing values themselves: NaN are kept, and a reference without variation is normal
MISSINGNESS_METRICS = ("missing_rate",)
CONSTANT_REFERENCE = "constant_reference"
# Functions that take numpy arrays as well as pandas series (scipy statistics)
ARRAY_FUNCTIONS = ("wasserstein", "ttest", "manwu", "levene", "bftest", "ks_2samp", "CvM")

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

    def __init__(self, metric_name: str, feature_name: str, missing_values: MissingValues = None, nan_policy: str = "omit"):
        super().__init__(metric_name)
        self._check_metrics_name(metric_name)
        self._feature_name = feature_name
        self._missing_values = normalize_missing_values(missing_values)
        self._check_nan_policy(nan_policy)
        self._nan_policy = nan_policy

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
        value = self._raw_value(self._funcs[self._name].value(new, reference, **kwargs))
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
            raw = self._funcs[self._name].value(pd.Series(new), pd.Series(reference))
        value = self._raw_value(raw)
        return float("nan") if _is_nan(value) else float(value)

    def _evaluate(self, current, reference, verdict, threshold, nan_policy, reliable, kwargs) -> MetricResults:
        try:
            new = self._extract(current)
            ref = self._extract(reference)
            self._column = new
            new, ref, info = self._clean(new, ref, nan_policy)
            value = self._raw_value(self._funcs[self._name].value(new, ref, **kwargs))
            constant_reference = self._name not in MISSINGNESS_METRICS and is_constant(ref)
        except Exception as e:
            raise MetricEvaluationError(
                value=self._name, message=f"Cannot evaluate {self._kind} {self._name} on {self._feature_name}: {e}"
            ) from e

        common = dict(
            metric_name=self._name,
            metric_type=MetricsType.drift.value,
            feature_name=self._feature_name,
            conf_int=None,
            threshold=threshold,
            status=CONSTANT_REFERENCE if constant_reference else None,
            reliable=reliable,
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
    """

    _funcs = DriftMetricsFuncs
    _kind = "drift metric"

    def __init__(
        self, metric_name: str, feature_name: str, missing_values: MissingValues = None, nan_policy: str = "omit", **kwargs
    ):
        super().__init__(metric_name, feature_name, missing_values=missing_values, nan_policy=nan_policy)

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
        threshold: Union[list, float, int] = None,
        upper_bound: bool = True,
        nan_policy: Optional[str] = None,
        reliable: Optional[bool] = None,
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
        )


class DriftTestMetric(_FeatureDriftMetric):
    """Statistical test between the current and reference samples of a feature (verdict on `alpha`)

    `metric_value` is the p-value. Same `missing_values` and `nan_policy` parameters as DriftMetric. The
    `missing_rate` test compares the share of missing values itself: its NaN are never set aside.
    """

    _funcs = DriftTestMetricsFuncs
    _kind = "drift test"

    def __init__(
        self, metric_name: str, feature_name: str, missing_values: MissingValues = None, nan_policy: str = "omit", **kwargs
    ):
        super().__init__(metric_name, feature_name, missing_values=missing_values, nan_policy=nan_policy)

    def _raw_value(self, result):
        return result.pvalue

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
        alpha: float = constant.SIGNIFICANCE_LEVEL,
        nan_policy: Optional[str] = None,
        reliable: Optional[bool] = None,
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
        kwargs :
            keyworded variable length of arguments to a function

        Returns
        -------
        MetricResults
            returns the result of the test; metric_value is the p-value
        """
        return self._evaluate(
            current,
            reference,
            lambda pvalue: pvalue < alpha if isinstance(alpha, (int, float)) else None,
            alpha,
            nan_policy,
            reliable,
            kwargs,
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
