#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

# import warnings
from tqdm import tqdm

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults, MetricsType
from ..metrics.drift import (
    DriftMetric,
    DriftTestMetric,
    mask_missing,
    normalize_missing_values,
)
from ..metrics.enums import (  # MetricsType,
    DriftMetricsFuncs,
    DriftTestMetricsFuncs,
    PerformanceMetricsFuncs,
)
from ..metrics.performance import NO_GROUND_TRUTH, PerformanceMetric
from ..metrics.statistics import FeatureSummary

logger = logging.getLogger(__name__)


def _matches(column: pd.Series, expected) -> pd.Series:
    """Rows whose model_id / model_version equals the expected value, compared as text

    A round trip through a CSV file or a store turns "1.0" into the float 1.0 and "1" into the integer 1: text
    comparison keeps them, and numeric values are also compared numerically ("1" matches 1.0).
    """
    matched = column.astype(str) == str(expected)
    try:
        number = float(expected)
    except (TypeError, ValueError):
        return matched
    numeric = pd.to_numeric(column, errors="coerce")
    return matched | (numeric == number)


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
        self._missing_values: dict = {}  # column -> sentinel values turned into NaN before the analysis
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
            y_name, pred_name : columns of the ground truth and the prediction (default "y_true", "y_pred")
            missing_values : sentinel value(s) of these two columns meaning "no value" (e.g. -200)
            min_label_coverage : minimum share of rows with ground truth; below it the metric is recorded with an
                error (default 0.0: the metric is evaluated on the labelled rows, however few)
            min_labelled : minimum number of labelled rows (default 1)
        """
        sentinels = normalize_missing_values(kwargs.get("missing_values"))
        if sentinels:
            for column in (kwargs.get("y_name", "y_true"), kwargs.get("pred_name", "y_pred")):
                self._declare_missing(column, sentinels)
        for metric_name in metrics_list:
            if metric_name not in PerformanceMetricsFuncs._member_names_:
                raise error_msg(value=metric_name, message=f"unknown performance metric key {metric_name} given.")
            self._metrics_list.append(PerformanceMetric(metric_name=metric_name, **kwargs))

    def _declare_missing(self, column, sentinels: tuple):
        known = self._missing_values.setdefault(column, ())
        self._missing_values[column] = known + tuple(v for v in sentinels if v not in known)

    def add_drift_metrics(self, metrics_list: list, features_list: list = None, missing_values=None, nan_policy: str = "omit"):
        """Method to add drift metrics list to the analyzer

        Parameters
        ----------
        metrics_list : list
            List of drift metric and drift test names; "missing_rate" tests the share of missing values
        features_list : list
            List of features for drift metrics
        missing_values : scalar, list or dict, optional
            Sentinel values meaning "no value", turned into NaN: one value or a list for every feature, or a dict
            {feature: value or list} per feature (e.g. {"CO": -200})
        nan_policy : str, optional
            "omit" (default): missing values are set aside and their share recorded; "propagate" or "raise"
        """

        # TODO: better handling of numeric vs categorical variables
        if features_list is None:
            raise error_msg(
                value=None,
                message=f'{"features_list is required: the analyzer holds no data before run()."}',
            )

        def sentinels_of(feature):
            if isinstance(missing_values, dict):
                return normalize_missing_values(missing_values.get(feature))
            return normalize_missing_values(missing_values)

        for feature in features_list:
            if sentinels_of(feature):
                self._declare_missing(feature, sentinels_of(feature))

        for metric_name in metrics_list:
            if metric_name in DriftMetricsFuncs._member_names_:
                metric_class = DriftMetric
            elif metric_name in DriftTestMetricsFuncs._member_names_:
                metric_class = DriftTestMetric
            else:
                raise error_msg(value=metric_name, message=f"unknown drift metric key {metric_name} given.")
            for feature in features_list:
                self._metrics_list.append(
                    metric_class(
                        metric_name=metric_name,
                        feature_name=feature,
                        missing_values=sentinels_of(feature),
                        nan_policy=nan_policy,
                    )
                )

    def run(self, current: pd.DataFrame, reference: pd.DataFrame, options: dict = {}):
        """Method run() in analyzer from the list of metrics

        Parameters
        ----------
        current : DataFrame
            The input current (pandas DataFrame)
        reference : DataFrame
            The input reference (pandas DataFrame)
        options : dict,optional
            Keyword arguments per metric name ({"psi": {"threshold": 0.2}}), or per metric and feature
            ({"psi|income": {"threshold": 0.15}}, e.g. AACalibrator.options()), which takes precedence
        """

        # declared sentinels become NaN everywhere, summary statistics included
        current = self._mask_missing(current)
        reference = self._mask_missing(reference)

        ref_model_id_validation = _matches(reference.model_id, self._metadata["model_id"])
        ref_model_version_validation = _matches(reference.model_version, self._metadata["model_version"])
        df_reference = reference.loc[ref_model_id_validation & ref_model_version_validation]

        cur_model_id_validation = _matches(current.model_id, self._metadata["model_id"])
        cur_model_version_validation = _matches(current.model_version, self._metadata["model_version"])
        df_current = current.loc[cur_model_id_validation & cur_model_version_validation].copy()

        # format="mixed": timestamps from a store often mix precisions (with or without fractional seconds)
        df_current["pred_timestamp"] = pd.to_datetime(df_current["pred_timestamp"], format="mixed")

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
                message=self._metadata_error("reference", reference, ref_model_id_validation, ref_model_version_validation),
            )
        elif df_current.shape[0] == 0:
            raise error_msg(
                value=None,
                message=self._metadata_error("current", current, cur_model_id_validation, cur_model_version_validation),
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
                # options per feature ("metric|feature", e.g. calibrated thresholds) take precedence over per metric
                feature = getattr(metric, "_feature_name", None)
                kwargs = options.get(f"{metric._name}|{feature}", options.get(metric._name, {}))
                try:
                    # rows without ground truth are set aside by the metric itself (n_labelled, label_coverage)
                    if isinstance(metric, PerformanceMetric) and metric._y_name not in df_current.columns:
                        raise error_msg(value=None, message=f"{NO_GROUND_TRUTH} (no column {metric._y_name})")
                    self._results.append(metric.evaluate(current=df_current, reference=df_reference, **kwargs))
                except Exception as e:
                    logger.warning("Metric %s failed: %s", metric._name, e)
                    self._results.append(self._failed_result(metric, e))

    def _mask_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = [c for c in self._missing_values if c in data.columns]
        if not columns:
            return data
        data = data.copy()
        for column in columns:
            data[column] = mask_missing(data[column], self._missing_values[column])
        return data

    def _metadata_error(self, name: str, data: pd.DataFrame, id_match: pd.Series, version_match: pd.Series) -> str:
        """Say which filter emptied the dataset, and what the column holds"""
        versions = sorted({repr(v) for v in data.model_version.dropna().unique()[:5]})
        return (
            f"Wrong model metadata for {name} dataset: {len(data)} rows, {int(id_match.sum())} with "
            f"model_id={self._metadata['model_id']!r}, {int((id_match & version_match).sum())} also with "
            f"model_version={self._metadata['model_version']!r} (model_version values: {', '.join(versions)}; "
            f"dtype {data.model_version.dtype})"
        )

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
