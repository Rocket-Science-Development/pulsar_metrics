#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from abc import ABC, abstractmethod

import pandas as pd
from black import InvalidInput

from ..metrics.drift import (
    DriftMetric,
    DriftMetricsFuncs,
    DriftTestMetric,
    DriftTestMetricsFuncs,
)
from ..metrics.performance import PerformanceMetric


class AbstractAnalyzer(ABC):

    """Base abstract class for analyzers"""

    def __init__(self, name: str, data: pd.DataFrame, description: str = None):

        """Parameters
        ----------
        - name: name of the analyzer
        - data: dataset from which the abnalyzer is defined
        - description (Optional): description for the analyzer
        """

        self._name = name
        self._data = data.copy(deep=True)  # Not sure I want to attach the data as an attribute ...
        self._description = description

        # TODO: validation on the dataset ?

        try:
            # TODO: better handling of date format
            data["pred_timestamp"] = pd.to_datetime(data["pred_timestamp"])
            self._model_id = str(data["model_id"].unique()[0])
            self._model_version = str(data["model_version"].unique()[0])
            self._period_start = data.pred_timestamp.min()
            self._period_end = data.pred_timestamp.max()
            self._metrics_list = []
            self._results = None
        except Exception as e:
            print(str(e))

    @property
    @abstractmethod
    def run(self, data_ref: pd.DataFrame):
        raise NotImplementedError

    def schedule(self):
        pass

    def add_performance_metrics(self, metrics_list: list):
        pass

    def add_drift_metrics(self, metrics_list: list):
        pass

    def add_drift_test_metrics(self, metrics_list: list):
        pass

    def get_result(self):
        return self._results

    def results_to_json(self):
        if self._results is None:
            return {}
        else:
            return [result.json() for result in self._results]

    def results_to_pandas(self):
        if self._results is None:
            return None
        else:
            return pd.DataFrame.from_records([self._results[i].dict() for i in range(len(self._results))])

    def log_results(self):
        pass


class Analyzer(AbstractAnalyzer):
    def __init__(self, name: str, data: pd.DataFrame, description: str = None, **kwargs):

        """Supercharged init method for performance metrics"""

        super().__init__(name, data, description)

    def add_performance_metrics(self, metrics_list: list, **kwargs):

        """Adding a list of performance metrics to the analyzer"""

        """Parameters
        ----------
        - metrics_list: list of performance metrics names
        """

        for metric_name in metrics_list:
            try:
                metric = PerformanceMetric(name=metric_name, data=self._data, **kwargs)
                if hasattr(metric, "_y_true"):
                    self._metrics_list.append(metric)
                    print("Performance metric '{}' added to the analyzer list".format(metric_name))
                else:
                    raise ValueError(
                        f"The dataset contains no ground truth for performance assessment. Metric '{metric_name}'"
                        f"was NOT added to the analyzer list"
                    )
            except Exception as e:
                print(str(e))

    def add_drift_metrics(self, metrics_list: list, features_list: list = None):

        """Adding a list of drift metrics to the analyzer"""

        """Parameters
        ----------
        - metrics_list: list of drift metric names
        """

        if features_list is None:
            features_list = self._data.select_dtypes("number").columns

        for metric_name in metrics_list:
            for feature in features_list:
                try:
                    if metric_name in DriftMetricsFuncs._member_names_:
                        metric = DriftMetric(name=metric_name, data=self._data, feature_name=feature)
                    elif metric_name in DriftTestMetricsFuncs._member_names_:
                        metric = DriftTestMetric(name=metric_name, data=self._data, feature_name=feature)
                    else:
                        metric = None
                        raise InvalidInput(f"unknown drift metric key '{metric_name}' given. ")
                    if metric is not None:
                        self._metrics_list.append(metric)
                        print("Drift metric '{}' for feature '{}' added to the analyzer list".format(metric_name, feature))
                except Exception as e:
                    print(str(e))

    def run(self, data_ref: pd.DataFrame, options: dict = {}):

        """Running the analyzer from the list of metrics"""

        if len(self._metrics_list) == 0:
            raise ValueError("The list of metrics for the analyzer is empty.")
        else:
            try:
                self._results = []
                for metric in self._metrics_list:
                    kwargs = options.get(metric._name, {})
                    if isinstance(metric, (DriftMetric, DriftTestMetric)):
                        metric.evaluate(reference=data_ref[metric._feature_name], **kwargs)
                    elif isinstance(metric, PerformanceMetric):
                        metric.evaluate(**kwargs)

                    self._results.append(metric._result)
            except Exception as e:
                print(str(e))
