#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>
from typing import Union

import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..exceptions import MetricEvaluationError
from ..utils import drift_from_threshold
from . import constant
from .base import AbstractMetrics, MetricResults, MetricsType
from .enums import DriftMetricsFuncs, DriftTestMetricsFuncs


class DriftMetric(AbstractMetrics):
    def __init__(self, metric_name: str, feature_name: str, **kwargs):
        """Constructor of the DriftMetric class

        Parameters
        ----------
        metric_name : str
            The input value for metric_name
        feature_name : str
            The input value for feature_name
        kwargs :
            keyworded variable length of arguments to a function
        """
        # Call the constructor of the parent class
        super().__init__(metric_name)

        self._check_metrics_name(metric_name)
        self._feature_name = feature_name

    def _check_metrics_name(self, name: str):
        if name not in DriftMetricsFuncs._member_names_:
            raise error_msg(
                value=None,
                message=f'{"InvalidInput: unknown metric key {name} given."}',
            )

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
        threshold: Union[list, float, int] = None,
        upper_bound: bool = True,
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
        kwargs :
                keyworded variable length of arguments to a function

        Returns
        -------
        list
                returns the result of the calculated DriftMetric
        """
        try:
            ref_column = reference[self._feature_name] if self._feature_name is not None else reference
            self._column = current[self._feature_name] if self._feature_name is not None else current

            value = DriftMetricsFuncs[self._name].value(self._column, ref_column, **kwargs)

            status = drift_from_threshold(value, threshold, upper_bound)

            self._result = MetricResults(
                metric_name=self._name,
                metric_type=MetricsType.drift.value,
                feature_name=self._feature_name,
                metric_value=value,
                conf_int=None,
                drift_status=status,
                threshold=threshold,
            )

            return self._result

        except Exception as e:
            raise MetricEvaluationError(
                value=self._name, message=f"Cannot evaluate drift metric {self._name} on {self._feature_name}: {e}"
            ) from e


class DriftTestMetric(AbstractMetrics):
    def __init__(self, metric_name: str, feature_name: str, **kwargs):
        """Constructor of the DriftTestMetric class

        Parameters
        ----------
        metric_name : str
            The input value for metric_name
        feature_name : str
            The input value for feature_name
        kwargs :
            keyworded variable length of arguments to a function
        """
        # Call the constructor of the parent class
        super().__init__(metric_name)

        self._check_metrics_name(metric_name)

        self._feature_name = feature_name

    def _check_metrics_name(self, name: str):
        if name not in DriftTestMetricsFuncs._member_names_:
            raise error_msg(
                value=name,
                message=f'{"InvalidInput: unknown metric key {name} given."}',
            )

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
        alpha: float = constant.SIGNIFICANCE_LEVEL,
        **kwargs,
    ) -> MetricResults:
        """Method  evaluate() to evaluate in DriftTestMetric

                Parameters
                ----------
                current : DataFrame
                        The input current (pandas DataFrame)
                reference : DataFrame
                        The input reference (pandas DataFrame)
        alpha : float
            Value to define significance level
                kwargs :
                        keyworded variable length of arguments to a function

                Returns
                -------
                list
                         returns the result of the calculated DriftMetric
        """
        try:
            ref_column = reference[self._feature_name] if self._feature_name is not None else reference
            self._column = current[self._feature_name] if self._feature_name is not None else current

            test_result = DriftTestMetricsFuncs[self._name].value(self._column, ref_column, **kwargs)

            status = test_result.pvalue < alpha if isinstance(alpha, (int, float)) else None

            self._result = MetricResults(
                metric_name=self._name,
                metric_type=MetricsType.drift.value,
                feature_name=self._feature_name,
                metric_value=test_result.pvalue,
                conf_int=None,
                drift_status=status,
                threshold=alpha,
            )

            return self._result

        except Exception as e:
            raise MetricEvaluationError(
                value=self._name, message=f"Cannot evaluate drift test {self._name} on {self._feature_name}: {e}"
            ) from e


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
