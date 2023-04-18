#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from abc import ABC, abstractmethod

# from datetime import datetime
from typing import Union

# import pandas as pd
from pydantic import BaseModel, validator

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..utils import compare_to_threshold
from .enums import (
    DriftMetricsFuncs,
    DriftTestMetricsFuncs,
    MetricsType,
    PerformanceMetricsFuncs,
)


class MetricResults(BaseModel):

    """Data structure for the results of a metric"""

    metric_type: str
    metric_name: str
    feature_name: str = None
    metric_value: Union[float, int, str] = None
    drift_status: bool = None
    threshold: Union[float, int, str, list] = None
    conf_int: list = None

    # TODO: validators for model id's, model's version, data_id, and metrics type
    @validator("metric_type", always=True)
    def metric_type_is_invalid(cls, v, **kwargs):
        if (v not in MetricsType._member_names_) and (v is not None):
            raise error_msg(
                value=None,
                message=f'{"ValueError: Metric type should be None or one of {MetricsType._member_names_}"}',
            )
        return v

    @validator("metric_name", always=True)
    def metric_name_is_invalid(cls, v, values, **kwargs):
        metric_names = (
            PerformanceMetricsFuncs._member_names_ + DriftMetricsFuncs._member_names_ + DriftTestMetricsFuncs._member_names_
        )
        if (v not in metric_names) and (values["metric_type"] not in [MetricsType.custom.value, MetricsType.statistics.value]):
            raise error_msg(
                value=None,
                message=f'{"ValueError:Metric name {v} is invalid"}',
            )
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

    @property
    @abstractmethod
    def evaluate(self) -> MetricResults:
        raise error_msg(
            value=None,
            message=f'{"NotImplementedError in evaluate() in AbstractMetrics class (base)"}',
        )

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
                    metric_value=value,
                    conf_int=None,
                    drift_status=status,
                    threshold=threshold,
                )

                return self._result

        return CustomClass(metric_name=metric_name)

    return inner
