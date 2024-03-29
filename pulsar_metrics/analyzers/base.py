#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd

# import warnings
from tqdm import tqdm

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.drift import DriftMetric, DriftTestMetric
from ..metrics.enums import (  # MetricsType,
    DriftMetricsFuncs,
    DriftTestMetricsFuncs,
    PerformanceMetricsFuncs,
)
from ..metrics.performance import PerformanceMetric
from ..metrics.statistics import FeatureSummary


class AbstractAnalyzer(ABC):
    """AbstractAnalyzer class for for analyzers"""

    def __init__(self, name: str, model_id: str, model_version: str, description: str = None):
        """Constructor of the AbstractAnalyzer class

        Parameters
        ----------
        name : str
            The input value for name
        model_id : str
            The input value for model_id
        model_version : str
            The input value for model_version
        description : str, optional
            The input value for description for the analyzer
        """
        self._name = name
        self._model_id = model_id
        self._model_version = model_version
        self._description = description
        self._metrics_list = []
        self._metadata = {"name": name, "description": description, "model_id": model_id, "model_version": model_version}
        self._results = None

    @property
    @abstractmethod
    def run(self, current: pd.DataFrame, reference: pd.DataFrame):
        raise error_msg(
            value=None,
            message=f'{"NotImplemented Error in run() in analyzers base"}',
        )

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
        empty_dict = {}
        result = empty_dict if self._results is None else [result.json() for result in self._results]
        return result

    def results_to_pandas(self):
        result = None
        if self._results is not None:
            results = pd.DataFrame.from_records([self._results[i].dict() for i in range(len(self._results))])
            for key, value in self._metadata.items():
                if key not in ["name", "description"]:
                    results[key] = value

        return result

    def log_results(self):
        pass


class Analyzer(AbstractAnalyzer):
    def __init__(self, name: str, model_id: str, model_version: str, description: str = None, **kwargs):
        """Constructor of the Analyzer class

        Parameters
        ----------
        name : str
            The input value for name
        model_id : str
            The input value for model_id
        model_version : str
            The input value for model_version
        description : str, optional
            The input value for description for the analyzer
        kwargs :
            keyworded variable length of arguments to a function

        Returns
        -------
        None
           return type of None
        """
        # Call the constructor of the parent class
        super().__init__(name, model_id, model_version, description)

    def add_performance_metrics(self, metrics_list: list, **kwargs):
        """Method to add performance metrics list to the analyzer

        Parameters
        ----------
        metrics_list : list
            List of performance metrics names
        kwargs :
            keyworded variable length of arguments to a function
        """
        for metric_name in metrics_list:
            try:
                if metric_name in PerformanceMetricsFuncs._member_names_:
                    metric = PerformanceMetric(metric_name=metric_name, **kwargs)
                    self._metrics_list.append(metric)
            except Exception as e:
                print(f"Error in add_performance_metrics() in the analysers base: {str(e)}")

    def add_drift_metrics(self, metrics_list: list, features_list: list = None):
        """Method to add drift metrics list to the analyzer

        Parameters
        ----------
        metrics_list : list
            List of performance metrics names
        features_list : list
            List of features for drift metrics
        """

        # TODO: better handling of numeric vs categorical variables
        if features_list is None:
            input_array = self._data.select_dtypes("number").columns
            input_comparison_array = ["y_true", "y_pred", "y_pred_proba", "model_id", "model_version"]

            features_list = np.setdiff1d(input_array, input_comparison_array)

        for metric_name in metrics_list:
            for feature in features_list:
                try:
                    if metric_name in DriftMetricsFuncs._member_names_:
                        metric = DriftMetric(metric_name=metric_name, feature_name=feature)
                        self._metrics_list.append(metric)
                    elif metric_name in DriftTestMetricsFuncs._member_names_:
                        metric = DriftTestMetric(metric_name=metric_name, feature_name=feature)
                        self._metrics_list.append(metric)
                    else:
                        raise error_msg(
                            value=metric_name,
                            message=f'{"unknown drift metric key {metric_name} given."}',
                        )
                except Exception as e:
                    print(f"Error in add_drift_metrics() in the analysers base: {str(e)}")

    def run(self, current: pd.DataFrame, reference: pd.DataFrame, options: dict = {}):
        """Method run() in analyzer from the list of metrics

        Parameters
        ----------
        current : DataFrame
            The input current (pandas DataFrame)
        reference : DataFrame
            The input reference (pandas DataFrame)
        options : dict,optional
            List of performance metrics names
        """

        ref_model_id_validation = reference.model_id == self._metadata["model_id"]
        ref_model_version_validation = reference.model_version == self._metadata["model_version"]
        df_reference = reference.loc[ref_model_id_validation & ref_model_version_validation]

        cur_model_id_validation = current.model_id == self._metadata["model_id"]
        cur_model_version_validation = current.model_version == self._metadata["model_version"]
        df_current = current.loc[cur_model_id_validation & cur_model_version_validation]

        df_current["pred_timestamp"] = pd.to_datetime(df_current["pred_timestamp"])

        self._metadata.update(
            {
                "period_start": df_current.pred_timestamp.min(),
                "period_end": df_current.pred_timestamp.max(),
                "eval_timestamp": datetime.now(),
                "options": options,
            }
        )

        if not self._metrics_list:
            raise error_msg(
                value=None,
                message=f'{"The metrics list for the analyzer is empty."}',
            )
        elif df_reference.shape[0] == 0:
            raise error_msg(
                value=None,
                message=f'{"Wrong model metadata for reference dataset."}',
            )
        elif df_current.shape[0] == 0:
            raise error_msg(
                value=None,
                message=f'{"Wrong model metadata for current dataset."}',
            )
        else:
            try:
                self._results = []
                # Summary statistics. Recommended all features for users (by default) otherwise configurable based on perferences
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
                            raise error_msg(
                                value=None,
                                message=f'{"Dataset has no ground truth for performance assessment"}',
                            )

                    self._results.append(metric._result)
            except Exception as e:
                print(f"Exception in run() in the analyzers class (base): {str(e)}")
