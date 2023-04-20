#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>
from typing import Union

import constant
import numpy as np
import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..utils import compare_to_threshold
from .base import AbstractMetrics, MetricResults, MetricsType
from .enums import PerformanceMetricsFuncs


class PerformanceMetric(AbstractMetrics):
    def __init__(self, metric_name: str, **kwargs):
        """Constructor of the PerformanceMetric class

        Parameters
        ----------
        feature_name : str
            The input feature_name for representing name of feature
        kwargs :
            keyworded variable length of arguments to a function

        Raises
        ------
        TypeError  if input value type are wrong
        """

        # Call the constructor of the parent class
        super().__init__(metric_name)

        self._check_metrics_name(metric_name)

        try:
            self._y_name = kwargs.get("y_name", "y_true")
            self._pred_name = kwargs.get("pred_name", "y_pred")

        except Exception as e:
            print(f"Exception in initializing __init__() in the PerformanceMetric class(performance): {str(e)}")

    def _check_metrics_name(self, metric_name: str):
        if metric_name not in PerformanceMetricsFuncs._member_names_:
            raise error_msg(
                value=None,
                message=f'{"unknown metric key {metric_name} given"}',
            )

    def evaluate(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame = None,
        bootstrap: bool = False,
        n_bootstrap: int = constant.BOOTSTRAP_SIZE,
        alpha: float = constant.SIGNIFICANCE_LEVEL,
        seed: int = constant.SEED_SIZE,
        threshold: Union[float, int, list] = None,
        upper_bound: bool = True,
        **kwargs,
    ) -> MetricResults:
        """Method evaluate() to evaluate the metrics performance

        Parameters
        ----------
        current : DataFrame
            The input current (pandas DataFrame)
        reference : DataFrame
            The input reference (pandas DataFrame)
        bootstrap : bool
            Boolean flag set to  bootstrap the metric for confidence interval calculation
        n_bootstrap : int
            Number of bootstrapping samples
        alpha : float
            Value to define significance level
        seed : int
            seed value for random number generator
        threshold : Union[list, float, int]
            Threshold values to validate the input value
        kwargs :
            keyworded variable length of arguments to a function

        Returns
        -------
        list
             returns the result of the calculated metric
        """

        try:
            self._n_sample = current.shape[0]
            value = PerformanceMetricsFuncs[self._name].value(current[self._y_name], current[self._pred_name], **kwargs)
            conf_int = None
            if bootstrap:
                conf_int = self._bootstrap(current=current, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, **kwargs)

            status = compare_to_threshold(value, threshold, upper_bound)

            self._result = MetricResults(
                metric_name=self._name,
                metric_type=MetricsType.performance.value,
                metric_value=value,
                feature_name="prediction",
                conf_int=conf_int,
                drift_status=status,
                threshold=threshold,
            )

            return self._result

        except Exception as e:
            print(f"Exception in evaluate() in the PerformanceMetric class (performance): {str(e)}")

    def _bootstrap(
        self,
        current: pd.DataFrame,
        n_bootstrap: int = constant.BOOTSTRAP_SIZE,
        seed: int = constant.SEED_SIZE,
        alpha: float = constant.SIGNIFICANCE_LEVEL,
        **kwargs,
    ):
        """Method to bootstrap the metrics for confidence interval evaluation

        Parameters
        ----------
        current : DataFrame
            The input data (pandas DataFrame)
        n_bootstrap : int
            Number of bootstrapping samples
        seed : int
            seed value for random number generator
        alpha : float
            value to define significance level
        kwargs :
            keyworded variable length of arguments to a function
        """

        values = []
        rng = np.random.default_rng(seed)
        for i in range(n_bootstrap):
            indices = rng.integers(low=0, high=self._n_sample, size=self._n_sample)
            values.append(
                PerformanceMetricsFuncs[self._name].value(
                    current.iloc[indices][self._y_name], current.iloc[indices][self._pred_name], **kwargs
                )
            )
        return [np.quantile(values, alpha / 2), np.quantile(values, 1 - alpha / 2)]
