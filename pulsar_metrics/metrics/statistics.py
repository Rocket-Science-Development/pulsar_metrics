from abc import ABC, abstractmethod
from typing import Sequence

import constant
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from scipy.stats import kurtosis, skew

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from .base import MetricResults

_numeric_dict = {"mean": np.mean, "median": np.median, "std": np.std, "skewness": skew, "kurtosis": kurtosis}


class FeatureSummaryAbstract(ABC):

    """Base abstract class for feature summary statistics"""

    def __init__(self, feature_name: str):
        """Parameters
        ----------
        - feature_name: name of the feature
        """

        self._feature_name = feature_name
        self._result = []

    @property
    @abstractmethod
    def evaluate(self) -> Sequence[MetricResults]:
        raise error_msg(
            value=None,
            message=f'{"NotImplementedError in evaluate() in FeatureSummaryAbstract class(statistics)"}',
        )

    def _check_feature_name(self, data: pd.DataFrame):
        try:
            if self._feature_name not in data.columns:
                raise error_msg(
                    value=self._feature_name,
                    message=f'{"InvalidInput in _check_feature_name() in FeatureSummaryAbstract class(statistics)"}',
                )
        except Exception as e:
            print(f"Exception in evaluate() in FeatureSummaryAbstract class: {str(e)}")

    def get_result(self):
        return self._result

    def results_to_pandas(self):
        if self._result is None:
            return None
        else:
            return pd.DataFrame.from_records([self._result[i].dict() for i in range(len(self._result))])


class FeatureSummary(FeatureSummaryAbstract):
    def __init__(self, feature_name: str):
        """Supercharged init method for feature summary statistics"""

        super().__init__(feature_name)

    def evaluate(
        self, current: pd.DataFrame, reference: pd.DataFrame = None, percentiles: list[float] = [0.25, 0.95]
    ) -> Sequence[MetricResults]:
        try:
            # Checking that the features exists in the current dataframe
            self._check_feature_name(current)

            if is_numeric_dtype(current[self._feature_name]):
                # Iterating through the list of functions for numerical features
                for name, func in _numeric_dict.items():
                    threshold = func(reference[self._feature_name]) if reference is not None else None
                    statistics = MetricResults(
                        metric_type="statistics",
                        metric_name=name,
                        feature_name=self._feature_name,
                        metric_value=func(current[self._feature_name]),
                        threshold=threshold,
                    )
                    self._result.append(statistics)

                # Adding quantiles
                for percentile in percentiles:
                    threshold = reference[self._feature_name].quantile(percentile) if reference is not None else None
                    statistics = MetricResults(
                        metric_type="statistics",
                        metric_name="P" + str(constant.HUNDRED * percentile),
                        feature_name=self._feature_name,
                        metric_value=current[self._feature_name].quantile(percentile),
                        threshold=threshold,
                    )
                    self._result.append(statistics)
            else:
                # For now only the most frequent category is calculated
                category_top = current[self._feature_name].value_counts().index[0]
                threshold = reference[self._feature_name].value_counts().index[0] if reference is not None else None
                statistics = MetricResults(
                    metric_type="statistics",
                    metric_name="top",
                    feature_name=self._feature_name,
                    metric_value=category_top,
                    threshold=threshold,
                )

            # Adding the count for all types of features
            count = MetricResults(
                metric_type="statistics",
                metric_name="count",
                feature_name=self._feature_name,
                metric_value=current[self._feature_name].count(),
            )
            self._result.append(count)
        except Exception as e:
            print(f"Exception in evaluate() in the FeatureSummary class( statistics): {str(e)}")
