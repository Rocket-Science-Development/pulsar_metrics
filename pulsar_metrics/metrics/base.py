#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Union, Optional

import pandas as pd
from pydantic import BaseModel, validator, Field

from ..utils import compare_to_threshold
from .enums import (
    MetricsType,
    DriftMetricsFuncs,
    DriftTestMetricsFuncs,
    PerformanceMetricsFuncs,
)


class MetricResults(BaseModel):

    """Data structure for the results of a metric"""

    metric_type: str
    metric_name: str
    feature_name: str = None
    metric_value: float = None
    drift_status: bool = None
    threshold: Union[float, int, list] = None
    conf_int: list = None
    #model_id: Optional[str]
    #model_version: Optional[str]
    #data_id: str = None
    #period_start: datetime = None
    #period_end: datetime = None
    #eval_timestamp: datetime = datetime.now()

    # @validator("eval_timestamp", always=True)
    # def timestamp_later_than_period_end(cls, v, values, **kwargs):
    #     if v < values["period_end"]:
    #         raise ValueError("Current timestamp earlier than period end")
    #     return v

    # TODO: validators for model id's, model's version, data_id, and metrics type
    @validator("metric_type", always=True)
    def metric_type_is_invalid(cls, v, **kwargs):
        if (v not in MetricsType._member_names_) and (v is not None):
            raise ValueError(f"Metric type should be None or one of {MetricsType._member_names_}")
        return v

    @validator("metric_name", always=True)
    def metric_name_is_invalid(cls, v, values, **kwargs):
        metric_names = (PerformanceMetricsFuncs._member_names_+
        DriftMetricsFuncs._member_names_+
        DriftTestMetricsFuncs._member_names_
        )
        if (v not in metric_names) and (values['metric_type'] != MetricsType.custom.value) :
            raise ValueError(f"Metric name {v} is invalid for {values['metric_type']} type")
        return v

class AbstractMetrics(ABC):

    """Base abstract class for metricsa"""

    def __init__(self, metric_name: str):

        """Parameters
        ----------
        - name: name of the metrics
        - data: dataset from which the metric is calculated
        """

        self._name = metric_name
        #self._data = data.copy(deep=True)  # Not sure I want to attach the data as an attribute ...

        # TODO: validation on the dataset ?

        # try:
        #     # TODO: better handling of date format
        #     data["pred_timestamp"] = pd.to_datetime(data["pred_timestamp"])
        #     self._model_id = str(data["model_id"].unique()[0])
        #     self._model_version = str(data["model_version"].unique()[0])
        #     self._period_start = data.pred_timestamp.min()
        #     self._period_end = data.pred_timestamp.max()
        #     self._result = None
        # except Exception as e:
        #     print(str(e))

    @property
    @abstractmethod
    def evaluate(self) -> MetricResults:
        raise NotImplementedError

    def _check_metrics_name(self):
        pass

    def get_result(self):
        return self._result

    # TODO: method to compare the metrics value to single value or interval thresholds


def CustomMetric(func):
    """Decorator for custom metrics"""

    def inner(metric_name: str) -> AbstractMetrics:
        class CustomClass(AbstractMetrics):
            def __init__(self, metric_name):
                super().__init__(metric_name)

            def evaluate(self, **kwargs):

                value = func(**kwargs)
                threshold = kwargs.get("threshold", None)
                upper_bound = kwargs.get("upper_bound", True)

                status = compare_to_threshold(value, threshold, upper_bound)

                self._result = MetricResults(
                    metric_name=self._name,
                    metric_type=MetricsType.custom.value,
                    #model_id=self._model_id,
                    #model_version=self._model_version,
                    metric_value=value,
                    conf_int=None,
                    drift_status=status,
                    threshold=threshold,
                    #period_start=self._period_start,
                    #period_end=self._period_end,
                )

                return self._result

        return CustomClass(metric_name=metric_name)

    return inner
