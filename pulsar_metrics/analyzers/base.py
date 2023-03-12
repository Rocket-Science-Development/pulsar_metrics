#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import numpy as np
from black import InvalidInput

# mport warnings
from tqdm import tqdm

from ..metrics.drift import DriftMetric, DriftTestMetric
from ..metrics.enums import (  # MetricsType,
    DriftMetricsFuncs,
    DriftTestMetricsFuncs,
    PerformanceMetricsFuncs,
)
from ..metrics.performance import PerformanceMetric
from ..metrics.statistics import FeatureSummary


class AbstractAnalyzer(ABC):

    """Base abstract class for analyzers"""

    def __init__(self, name: str, model_id: str, model_version: str, description: str = None):
        """Parameters
        ----------
        - name: name of the analyzer
        - data: dataset from which the abnalyzer is defined
        - description (Optional): description for the analyzer
        """

        self._name = name
        # self._data = data.copy(deep=True)  # Not sure I want to attach the data as an attribute ...
        self._description = description

        # TODO: validation on the dataset ?

        try:
            # TODO: better handling of date format
            # data["pred_timestamp"] = pd.to_datetime(data["pred_timestamp"])
            self._model_id = model_id  # str(data["model_id"].unique()[0])
            self._model_version = model_version  # str(data["model_version"].unique()[0])
            # self._period_start = data.pred_timestamp.min()
            # self._period_end = data.pred_timestamp.max()
            self._metrics_list = []
            self._metadata = {"name": name, "description": description, "model_id": model_id, "model_version": model_version}
            self._results = None
        except Exception as e:
            print(str(e))

    @property
    @abstractmethod
    def run(self, current: pd.DataFrame, reference: pd.DataFrame):
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
            results = pd.DataFrame.from_records([self._results[i].dict() for i in range(len(self._results))])
            for key, value in self._metadata.items():
                if key not in ["name", "description"]:
                    results[key] = value
            return results

    def log_results(self):
        pass


class Analyzer(AbstractAnalyzer):
    def __init__(self, name: str, model_id: str, model_version: str, description: str = None, **kwargs):
        """Supercharged init method for performance metrics"""

        super().__init__(name, model_id, model_version, description)

    def add_performance_metrics(self, metrics_list: list, **kwargs):
        """Adding a list of performance metrics to the analyzer"""

        """Parameters
        ----------
        - metrics_list: list of performance metrics names
        """

        for metric_name in metrics_list:
            try:
                if metric_name in PerformanceMetricsFuncs._member_names_:
                    metric = PerformanceMetric(metric_name=metric_name, **kwargs)
                    self._metrics_list.append(metric)
                # if hasattr(metric, "_y_true"):
                #     self._metrics_list.append(metric)
                #     print("Performance metric '{}' added to the analyzer list".format(metric_name))
                # else:
                #     raise ValueError(
                #         f"The dataset contains no ground truth for performance assessment. Metric '{metric_name}'"
                #         f"was NOT added to the analyzer list"
                #     )
            except Exception as e:
                print(str(e))

    def add_drift_metrics(self, metrics_list: list, features_list: list = None):
        """Adding a list of drift metrics to the analyzer"""

        """Parameters
        ----------
        - metrics_list: list of drift metric names
        """

        # TODO: better handling of numeric vs categorical variables
        if features_list is None:
            features_list = np.setdiff1d(
                self._data.select_dtypes("number").columns, ["y_true", "y_pred", "y_pred_proba", "model_id", "model_version"]
            )

        for metric_name in metrics_list:
            for feature in features_list:
                try:
                    if metric_name in DriftMetricsFuncs._member_names_:
                        metric = DriftMetric(metric_name=metric_name, feature_name=feature)
                    elif metric_name in DriftTestMetricsFuncs._member_names_:
                        metric = DriftTestMetric(metric_name=metric_name, feature_name=feature)
                    else:
                        metric = None
                        raise InvalidInput(f"unknown drift metric key '{metric_name}' given. ")
                    if metric is not None:
                        self._metrics_list.append(metric)
                        print("Drift metric '{}' for feature '{}' added to the analyzer list".format(metric_name, feature))
                except Exception as e:
                    print(str(e))

    def run(self, current: pd.DataFrame, reference: pd.DataFrame, options: dict = {}):
        """Running the analyzer from the list of metrics"""

        df_reference = reference.loc[
            (reference.model_id == self._metadata["model_id"]) & ((reference.model_version == self._metadata["model_version"]))
        ]

        df_current = current.loc[
            (current.model_id == self._metadata["model_id"]) & ((current.model_version == self._metadata["model_version"]))
        ]

        df_current["pred_timestamp"] = pd.to_datetime(df_current["pred_timestamp"])

        self._metadata.update(
            {
                "period_start": df_current.pred_timestamp.min(),
                "period_end": df_current.pred_timestamp.max(),
                "eval_timestamp": datetime.now(),
                "options": options,
            }
        )

        if len(self._metrics_list) == 0:
            raise ValueError("The list of metrics for the analyzer is empty.")
        elif df_reference.shape[0] == 0:
            raise ValueError("Wrong model metadata for reference dataset")
        elif df_current.shape[0] == 0:
            raise ValueError("Wrong model metadata for current dataset")
        else:
            try:
                self._results = []
                # Summary statistics. is it better to do it on all features by default or allow the user to chose which features
                for feature_name in current.columns:
                    statistics = FeatureSummary(feature_name=feature_name)
                    statistics.evaluate(current, reference)
                    self._results += statistics._result
                for metric in tqdm(self._metrics_list):
                    kwargs = options.get(metric._name, {})
                    if isinstance(metric, (DriftMetric, DriftTestMetric)):
                        metric.evaluate(current=df_current, reference=df_reference, **kwargs)
                    elif isinstance(metric, PerformanceMetric):
                        if (metric._y_name in df_current.columns) and (df_current[metric._y_name].isnull().sum() == 0):
                            metric.evaluate(current=df_current, reference=df_reference, **kwargs)
                        else:
                            raise ValueError(
                                f"The dataset contains no ground truth for performance assessment. Metric '{metric._name}'"
                                f"was NOT calculated"
                            )

                    self._results.append(metric._result)
            except Exception as e:
                print(str(e))
