from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd
from black import InvalidInput
from pandas.core.dtypes.common import is_numeric_dtype
from scipy.stats import kurtosis, skew

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
        raise NotImplementedError

    def _check_feature_name(self, data: pd.DataFrame):

        try:
            if self._feature_name not in data.columns:
                raise InvalidInput(f"Unknwon feature with name '{self._feature_name}'.")
        except Exception as e:
            print(str(e))

    def get_result(self):
        return self._result

    def results_to_pandas(self):
        if self._result is None:
            return None
        else:
            results = pd.DataFrame.from_records([self._result[i].dict() for i in range(len(self._result))])
            return results


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
                    if reference is not None:
                        threshold = func(reference[self._feature_name])
                    else:
                        threshold = None
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
                    if reference is not None:
                        threshold = reference[self._feature_name].quantile(percentile)
                    else:
                        threshold = None
                    statistics = MetricResults(
                        metric_type="statistics",
                        metric_name="P" + str(100 * percentile),
                        feature_name=self._feature_name,
                        metric_value=current[self._feature_name].quantile(percentile),
                        threshold=threshold,
                    )
                    self._result.append(statistics)
            else:
                # For now only the most frequent category is calculated
                category_top = current[self._feature_name].value_counts().index[0]
                if reference is not None:
                    threshold = reference[self._feature_name].value_counts().index[0]
                else:
                    threshold = None
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
            print(str(e))
