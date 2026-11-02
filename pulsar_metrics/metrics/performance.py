#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>
import warnings
from typing import Union

import numpy as np
import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..exceptions import MetricEvaluationError
from ..utils import drift_from_threshold
from . import constant
from .base import AbstractMetrics, MetricResults, MetricsType
from .drift import mask_missing, normalize_missing_values
from .enums import PerformanceMetricsFuncs

NO_GROUND_TRUTH = "Dataset has no ground truth for performance assessment"
LOW_COVERAGE = 0.5  # below, with no minimum set, a warning says the verdict rests on few labelled rows


class PerformanceMetric(AbstractMetrics):
    def __init__(self, metric_name: str, **kwargs):
        """Constructor of the PerformanceMetric class

        Parameters
        ----------
        metric_name : str
            One of PerformanceMetricsFuncs
        kwargs :
            y_name, pred_name : columns of the ground truth and the prediction (default "y_true", "y_pred")
            missing_values : sentinel value(s) of y_name and pred_name meaning "no value", turned into NaN
            min_label_coverage : minimum share of rows with ground truth (default 0.0: any labelled row is enough)
            min_labelled : minimum number of labelled rows (default 1)

        Raises
        ------
        TypeError  if input value type are wrong
        """

        # Call the constructor of the parent class
        super().__init__(metric_name)

        self._check_metrics_name(metric_name)

        self._y_name = kwargs.get("y_name", "y_true")
        self._pred_name = kwargs.get("pred_name", "y_pred")
        self._missing_values = normalize_missing_values(kwargs.get("missing_values"))
        self._min_label_coverage = kwargs.get("min_label_coverage", 0.0)
        self._min_labelled = kwargs.get("min_labelled", 1)

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
        upper_bound: bool = None,
        min_label_coverage: float = None,
        min_labelled: int = None,
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
        upper_bound : bool, optional
            True if the threshold is the maximum acceptable value (error metrics), False if it is the minimum
            (accuracy, precision, recall, f1, auc, aucpr, r2). Inferred from the metric when None.
            drift_status is True when the value violates the threshold.
        min_label_coverage : float, optional
            Minimum share of rows with both ground truth and prediction; below it the result carries an `error` and no
            value or verdict. Defaults to the value given at construction (0.0).
        min_labelled : int, optional
            Minimum number of such rows (default from the construction, 1)
        kwargs :
            keyworded variable length of arguments to a function

        Returns
        -------
        list
             returns the result of the calculated metric
        """

        min_label_coverage = self._min_label_coverage if min_label_coverage is None else min_label_coverage
        min_labelled = self._min_labelled if min_labelled is None else min_labelled
        try:
            # Ground truth often arrives late or partly: evaluate on the rows that have it, and say how many they are
            y_true = mask_missing(current[self._y_name], self._missing_values)
            y_pred = mask_missing(current[self._pred_name], self._missing_values)
            labelled = (y_true.notna() & y_pred.notna()).to_numpy()
            n_rows = int(current.shape[0])
            n_labelled = int(labelled.sum())
            coverage = n_labelled / n_rows if n_rows else 0.0
            coverage_fields = dict(
                metric_name=self._name,
                metric_type=MetricsType.performance.value,
                feature_name="prediction",
                threshold=threshold,
                n_labelled=n_labelled,
                label_coverage=coverage,
            )
            if n_labelled == 0:
                self._result = MetricResults(error=f"{NO_GROUND_TRUTH} (0 labelled rows of {n_rows})", **coverage_fields)
                return self._result
            if coverage < min_label_coverage or n_labelled < min_labelled:
                self._result = MetricResults(
                    error=(
                        f"label coverage too low: {n_labelled} labelled rows of {n_rows} ({coverage:.1%}), "
                        f"minimum {min_label_coverage:.1%} and {min_labelled} rows"
                    ),
                    **coverage_fields,
                )
                return self._result
            if n_labelled < n_rows:
                coverage_fields["method"] = f"labelled rows only: {n_labelled} of {n_rows} (coverage {coverage:.1%})"
                if min_label_coverage == 0 and coverage < LOW_COVERAGE:
                    warnings.warn(
                        f"{self._name} is evaluated on {n_labelled} labelled rows of {n_rows} ({coverage:.1%}) and "
                        "min_label_coverage is 0: the verdict rests on few rows. Set min_label_coverage to refuse such "
                        "windows",
                        UserWarning,
                        stacklevel=2,
                    )
            current = pd.DataFrame({self._y_name: y_true[labelled], self._pred_name: y_pred[labelled]})
            self._n_sample = n_labelled
            value = PerformanceMetricsFuncs[self._name].value(current[self._y_name], current[self._pred_name], **kwargs)
            conf_int = None
            if bootstrap:
                conf_int = self._bootstrap(current=current, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, **kwargs)

            if upper_bound is None:
                upper_bound = self._name not in constant.HIGHER_IS_BETTER_METRICS
            status = drift_from_threshold(value, threshold, upper_bound)

            self._result = MetricResults(metric_value=value, conf_int=conf_int, drift_status=status, **coverage_fields)

            return self._result

        except Exception as e:
            raise MetricEvaluationError(value=self._name, message=f"Cannot evaluate performance metric {self._name}: {e}") from e

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
