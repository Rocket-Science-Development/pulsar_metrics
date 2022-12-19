#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
from black import InvalidInput
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .base import AbstractMetrics, MetricResults, MetricsType


class PerformanceMetricsFuncs(Enum):

    """Set of performance metrics functions"""

    # Classification Metrics
    accuracy = partial(accuracy_score)
    precision = partial(precision_score)
    recall = partial(recall_score)
    f1 = partial(f1_score)
    log_loss = partial(log_loss)
    # Area under ROC Curve
    auc = partial(roc_auc_score)
    # Area under PR Curve
    aucpr = partial(average_precision_score)
    brier = partial(brier_score_loss)

    # Regression metrics
    mse = partial(mean_squared_error)
    mae = partial(mean_absolute_error)
    mape = partial(mean_absolute_error)
    r2 = partial(r2_score)


class PerformanceMetric(AbstractMetrics):
    def __init__(self, name: str, data: pd.DataFrame, **kwargs):

        """Supercharged init method for performance metrics"""

        super().__init__(name, data)

        self._check_metrics_name(name)

        try:
            y_name = kwargs.get("y_name", "y_true")
            pred_name = kwargs.get("pred_name", "y_pred")
            self._y_true = data[y_name]
            self._y_pred = data[pred_name]

        except Exception as e:
            print(str(e))

    def _check_metrics_name(self, name: str):
        if name not in PerformanceMetricsFuncs._member_names_:
            raise InvalidInput(
                f"unknown metric key '{name}' given. " f"Should be one of {PerformanceMetricsFuncs._member_names_}."
            )

    def evaluate(
        self,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        alpha: float = 0.05,
        seed: int = 123,
        threshold: float = None,
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

            value = PerformanceMetricsFuncs[self._name].value(self._y_true, self._y_pred, **kwargs)

            if bootstrap:
                conf_int = self._bootstrap(n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, **kwargs)
            else:
                conf_int = None

            if isinstance(threshold, (int, float)):
                status = value < threshold
            else:
                status = None

            self._result = MetricResults(
                metric_name=self._name,
                type=MetricsType.performance.value,
                model_id=self._model_id,
                model_version=self._model_version,
                value=value,
                feature="prediction",
                conf_int=conf_int,
                status=status,
                threshold=threshold,
                period_start=self._period_start,
                period_end=self._period_end,
            )

            return self._result

        except Exception as e:
            print(str(e))

    def _bootstrap(self, n_bootstrap: int = 100, seed: int = 123, alpha: float = 0.05, **kwargs):

        """Function to bootstrap the metrics for confidence interval evaluation

        Parameters
        ----------
        - n_bootstrap (int): number of bootstrapping samples
        - seed (int): seed for random number generator
        - alpha (float): significance level
        """

        rng = np.random.default_rng(seed)
        n = self._data.shape[0]

        values = []

        for i in range(n_bootstrap):
            indices = rng.integers(low=0, high=n, size=n)
            values.append(
                PerformanceMetricsFuncs[self._name].value(self._y_true.loc[indices], self._y_pred.loc[indices], **kwargs)
            )
        return [np.quantile(values, alpha / 2), np.quantile(values, 1 - alpha / 2)]
