#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from enum import Enum
from functools import partial

import pandas as pd
from black import InvalidInput
from scipy.stats import (
    chisquare,
    cramervonmises_2samp,
    ks_2samp,
    levene,
    mannwhitneyu,
    ttest_ind,
    wasserstein_distance,
)

from .base import AbstractMetrics, MetricResults, MetricsType
from .utils import kl_divergence


class DriftMetricsFuncs(Enum):
    kl = partial(kl_divergence)
    wasserstein = partial(wasserstein_distance)


class DriftTestMetricsFuncs(Enum):
    ttest = partial(ttest_ind, equal_var=False)
    manwu = partial(mannwhitneyu)
    levene = partial(levene, center="mean")
    bftest = partial(levene, center="median")
    ks_2samp = partial(ks_2samp)
    CvM = partial(cramervonmises_2samp)
    chi2 = partial(chisquare)


class DriftMetric(AbstractMetrics):
    def __init__(self, name: str, data: pd.DataFrame, feature_name: str, **kwargs):

        """Supercharged init method for drift metrics"""

        super().__init__(name, data)

        self._check_metrics_name(name)

        try:
            self._feature_name = feature_name
            self._column = data[feature_name]

        except Exception as e:
            print(str(e))

    def _check_metrics_name(self, name: str):
        if name not in DriftMetricsFuncs._member_names_:
            raise InvalidInput(f"unknown metric key '{name}' given. " f"Should be one of {DriftMetricsFuncs._member_names_}.")

    def evaluate(
        self,
        reference: pd.Series,
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

            value = DriftMetricsFuncs[self._name].value(self._column, reference, **kwargs)

            if isinstance(threshold, (int, float)):
                status = value < threshold
            else:
                status = None

            self._result = MetricResults(
                metric_name=self._name,
                type=MetricsType.drift.value,
                model_id=self._model_id,
                model_version=self._model_version,
                feature=self._feature_name,
                value=value,
                conf_int=None,
                status=status,
                threshold=threshold,
                period_start=self._period_start,
                period_end=self._period_end,
            )

            return self._result

        except Exception as e:
            print(str(e))


class DriftTestMetric(AbstractMetrics):
    def __init__(self, name: str, data: pd.DataFrame, feature_name: str, **kwargs):

        """Supercharged init method for drift metrics"""

        super().__init__(name, data)

        self._check_metrics_name(name)

        try:
            self._feature_name = feature_name
            self._column = data[feature_name]

        except Exception as e:
            print(str(e))

    def _check_metrics_name(self, name: str):
        if name not in DriftTestMetricsFuncs._member_names_:
            raise InvalidInput(f"unknown metric key '{name}' given. " f"Should be one of {DriftTestMetricsFuncs._member_names_}.")

    def evaluate(
        self,
        reference: pd.Series,
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
                _, pvalue = DriftTestMetricsFuncs[self._name].value(self._column, reference, **kwargs)
            else:
                res = DriftTestMetricsFuncs[self._name].value(self._column, reference, **kwargs)
                # statistic = res.statistic
                pvalue = res.pvalue

            if isinstance(alpha, (int, float)):
                status = pvalue < alpha
            else:
                status = None

            self._result = MetricResults(
                metric_name=self._name,
                type=MetricsType.drift.value,
                model_id=self._model_id,
                model_version=self._model_version,
                feature=self._feature_name,
                value=pvalue,
                conf_int=None,
                status=status,
                threshold=alpha,
                period_start=self._period_start,
                period_end=self._period_end,
            )

            return self._result

        except Exception as e:
            print(str(e))
