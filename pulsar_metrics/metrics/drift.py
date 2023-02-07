#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from enum import Enum
from functools import partial
from typing import Union

import pandas as pd
from black import InvalidInput


from ..utils import compare_to_threshold
from .base import AbstractMetrics, MetricResults, MetricsType
from .enums import DriftMetricsFuncs, DriftTestMetricsFuncs

class DriftMetric(AbstractMetrics):
    def __init__(self, metric_name: str, feature_name: str, **kwargs):

        """Supercharged init method for drift metrics"""

        super().__init__(metric_name)

        self._check_metrics_name(metric_name)

        try:
            self._feature_name = feature_name
            #self._column = data[feature_name]

        except Exception as e:
            print(str(e))

    def _check_metrics_name(self, name: str):
        if name not in DriftMetricsFuncs._member_names_:
            raise InvalidInput(f"unknown metric key '{name}' given. " f"Should be one of {DriftMetricsFuncs._member_names_}.")

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
        threshold: Union[list, float, int] = None,
        upper_bound: bool = True,
        **kwargs,
    ) -> MetricResults:

        """Evaluation function for performance metrics

        Parameters
        ----------
        - bootstrap (bool): whether to bootstrap the metric for confidence interval calculation
        - n_bootstrap (int): number of bootstrapping samples
        - seed (int): seed for random number generator
        - alpha (float): significance level
        - **kwargs: parameters of the function used to calculate the metric
        """

        try:

            value = DriftMetricsFuncs[self._name].value(current[self._feature_name], reference[self._feature_name], **kwargs)

            status = compare_to_threshold(value, threshold, upper_bound)

            self._result = MetricResults(
                metric_name=self._name,
                metric_type=MetricsType.drift.value,
                #model_id=self._model_id,
                #model_version=self._model_version,
                feature_name=self._feature_name,
                metric_value=value,
                conf_int=None,
                drift_status=status,
                threshold=threshold,
                #period_start=self._period_start,
                #period_end=self._period_end,
            )

            return self._result

        except Exception as e:
            print(str(e))


class DriftTestMetric(AbstractMetrics):
    def __init__(self, metric_name: str, feature_name: str, **kwargs):

        """Supercharged init method for drift metrics"""

        super().__init__(metric_name)

        self._check_metrics_name(metric_name)

        try:
            self._feature_name = feature_name
            #self._column = data[feature_name]

        except Exception as e:
            print(str(e))

    def _check_metrics_name(self, name: str):
        if name not in DriftTestMetricsFuncs._member_names_:
            raise InvalidInput(f"unknown metric key '{name}' given. " f"Should be one of {DriftTestMetricsFuncs._member_names_}.")

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
        alpha: float = 0.05,
        **kwargs,
    ) -> MetricResults:

        """Evaluation function for performance metrics

        Parameters
        ----------
        - bootstrap (bool): whether to bootstrap the metric for confidence interval calculation
        - n_bootstrap (int): number of bootstrapping samples
        - seed (int): seed for random number generator
        - alpha (float): significance level
        - **kwargs: parameters of the function used to calculate the metric
        """

        try:

            if self._name != "CvM":
                _, pvalue = DriftTestMetricsFuncs[self._name].value(current[self._feature_name], reference[self._feature_name], **kwargs)
            else:
                res = DriftTestMetricsFuncs[self._name].value(current[self._feature_name], reference[self._feature_name], **kwargs)
                # statistic = res.statistic
                pvalue = res.pvalue

            if isinstance(alpha, (int, float)):
                status = pvalue < alpha
            else:
                status = None

            self._result = MetricResults(
                metric_name=self._name,
                metric_type=MetricsType.drift.value,
                #model_id=self._model_id,
                #model_version=self._model_version,
                feature_name=self._feature_name,
                metric_value=pvalue,
                conf_int=None,
                drift_status=status,
                threshold=alpha,
                #period_start=self._period_start,
                #period_end=self._period_end,
            )

            return self._result

        except Exception as e:
            print(str(e))

def CustomDriftMetric(func):
    """Decorator for custom metrics"""

    def inner(metric_name: str, feature_name:str) -> AbstractMetrics:
        class CustomClass(AbstractMetrics):
            def __init__(self, metric_name, feature_name):
                super().__init__(metric_name)
                self._feature_name = feature_name

            def evaluate(self, current: pd.DataFrame, reference: pd.DataFrame, **kwargs):

                value = func(current[self._feature_name], reference[self._feature_name], **kwargs)
                threshold = kwargs.get("threshold", None)
                upper_bound = kwargs.get("upper_bound", True)

                status = compare_to_threshold(value, threshold, upper_bound)

                self._result = MetricResults(
                    metric_name=self._name,
                    metric_type=MetricsType.custom.value,
                    feature_name = self._feature_name,
                    #model_id=self._model_id,
                    #model_version=self._model_version,
                    metric_value=value,
                    conf_int=None,
                    drift_status=status,
                    threshold=threshold,
                    #period_start=self._period_start,
                    #period_end=self._period_end,
                )

                return self._result
        return CustomClass(metric_name=metric_name, feature_name = feature_name)

    return inner
