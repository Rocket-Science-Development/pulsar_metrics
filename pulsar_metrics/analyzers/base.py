#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

# import warnings
from tqdm import tqdm

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults, MetricsType
from ..metrics.drift import DriftMetric, DriftTestMetric
from ..metrics.enums import (  # MetricsType,
    DriftMetricsFuncs,
    DriftTestMetricsFuncs,
    PerformanceMetricsFuncs,
)
from ..metrics.performance import PerformanceMetric
from ..metrics.statistics import FeatureSummary

logger = logging.getLogger(__name__)


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
        result = empty_dict if self._results is None else [result.model_dump_json() for result in self._results]
        return result

    def results_to_pandas(self):
        results = None
        if self._results is not None:
            results = pd.DataFrame.from_records([self._results[i].model_dump() for i in range(len(self._results))])
            for key, value in self._metadata.items():
                if key not in ["name", "description", "options"]:
                    results[key] = value

        return results

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
            if metric_name not in PerformanceMetricsFuncs._member_names_:
                raise error_msg(value=metric_name, message=f"unknown performance metric key {metric_name} given.")
            self._metrics_list.append(PerformanceMetric(metric_name=metric_name, **kwargs))

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
            raise error_msg(
                value=None,
                message=f'{"features_list is required: the analyzer holds no data before run()."}',
            )

        for metric_name in metrics_list:
            if metric_name in DriftMetricsFuncs._member_names_:
                metric_class = DriftMetric
            elif metric_name in DriftTestMetricsFuncs._member_names_:
                metric_class = DriftTestMetric
            else:
                raise error_msg(value=metric_name, message=f"unknown drift metric key {metric_name} given.")
            for feature in features_list:
                self._metrics_list.append(metric_class(metric_name=metric_name, feature_name=feature))

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
        df_current = current.loc[cur_model_id_validation & cur_model_version_validation].copy()

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
            # A failing metric does not stop the run: it is recorded with its error and no value or status
            self._results = []
            # Summary statistics. Recommended all features for users (by default) otherwise configurable based on perferences
            for feature_name in current.columns:
                statistics = FeatureSummary(feature_name=feature_name)
                try:
                    statistics.evaluate(current, reference)
                    self._results += statistics._result
                except Exception as e:
                    logger.warning("Summary statistics failed for %s: %s", feature_name, e)
                    self._results.append(
                        MetricResults(
                            metric_type=MetricsType.statistics.value,
                            metric_name="summary",
                            feature_name=feature_name,
                            error=str(e),
                        )
                    )
            for metric in tqdm(self._metrics_list):
                kwargs = options.get(metric._name, {})
                try:
                    if isinstance(metric, PerformanceMetric) and (
                        metric._y_name not in df_current.columns or df_current[metric._y_name].isnull().any()
                    ):
                        raise error_msg(value=None, message="Dataset has no ground truth for performance assessment")
                    self._results.append(metric.evaluate(current=df_current, reference=df_reference, **kwargs))
                except Exception as e:
                    logger.warning("Metric %s failed: %s", metric._name, e)
                    self._results.append(self._failed_result(metric, e))

    @staticmethod
    def _failed_result(metric, error: Exception) -> MetricResults:
        """Result recorded for a metric that could not be computed"""
        if isinstance(metric, PerformanceMetric):
            metric_type = MetricsType.performance.value
        elif isinstance(metric, (DriftMetric, DriftTestMetric)):
            metric_type = MetricsType.drift.value
        else:
            metric_type = MetricsType.custom.value
        return MetricResults(
            metric_type=metric_type,
            metric_name=metric._name,
            feature_name=getattr(metric, "_feature_name", None),
            error=str(error),
        )
