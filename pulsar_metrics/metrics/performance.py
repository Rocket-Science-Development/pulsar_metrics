#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from enum import Enum
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from black import InvalidInput


from ..utils import compare_to_threshold
from .base import AbstractMetrics, MetricResults, MetricsType
from .enums import PerformanceMetricsFuncs


class PerformanceMetric(AbstractMetrics):
    def __init__(self, metric_name: str, **kwargs):

        """Supercharged init method for performance metrics"""

        super().__init__(metric_name)

        self._check_metrics_name(metric_name)

        try:
            self._y_name = kwargs.get("y_name", "y_true")
            self._pred_name = kwargs.get("pred_name", "y_pred")
            #self._y_true = data[y_name]
            #self._y_pred = data[pred_name]

        except Exception as e:
            print(str(e))

    def _check_metrics_name(self, metric_name: str):
        if metric_name not in PerformanceMetricsFuncs._member_names_:
            raise InvalidInput(
                f"unknown metric key '{metric_name}' given. " f"Should be one of {PerformanceMetricsFuncs._member_names_}."
            )

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        alpha: float = 0.05,
        seed: int = 123,
        threshold: Union[float, int, list] = None,
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

            self._n_sample = current.shape[0]

            value = PerformanceMetricsFuncs[self._name].value(current[self._y_name], current[self._pred_name], **kwargs)

            if bootstrap:
                conf_int = self._bootstrap(current = current, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, **kwargs)
            else:
                conf_int = None

            status = compare_to_threshold(value, threshold, upper_bound)

            self._result = MetricResults(
                metric_name=self._name,
                metric_type=MetricsType.performance.value,
                #model_id=self._model_id,
                #model_version=self._model_version,
                metric_value=value,
                feature_name="prediction",
                conf_int=conf_int,
                drift_status=status,
                threshold=threshold,
                #period_start=self._period_start,
                #period_end=self._period_end,
            )

            return self._result

        except Exception as e:
            print(str(e))

    def _bootstrap(self, current: pd.DataFrame, n_bootstrap: int = 100, seed: int = 123, alpha: float = 0.05, **kwargs):

        """Function to bootstrap the metrics for confidence interval evaluation

        Parameters
        ----------
        - n_bootstrap (int): number of bootstrapping samples
        - seed (int): seed for random number generator
        - alpha (float): significance level
        """

        rng = np.random.default_rng(seed)
        n = self._n_sample

        values = []

        for i in range(n_bootstrap):
            indices = rng.integers(low=0, high=n, size=n)
            values.append(
                PerformanceMetricsFuncs[self._name].value(current.iloc[indices][self._y_name], current.iloc[indices][self._pred_name], **kwargs)
            )
        return [np.quantile(values, alpha / 2), np.quantile(values, 1 - alpha / 2)]
