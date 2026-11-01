#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from abc import ABC, abstractmethod

# from datetime import datetime
from typing import Optional, Union

# import pandas as pd
from pydantic import BaseModel, ValidationInfo, field_validator

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..utils import drift_from_threshold
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
    feature_name: Optional[str] = None
    metric_value: Optional[Union[float, int, str]] = None
    drift_status: Optional[bool] = None
    threshold: Optional[Union[float, int, str, list]] = None
    conf_int: Optional[list] = None
    error: Optional[str] = None
    # Group-level results (fairness): all optional, absent for the other metric types
    group: Optional[str] = None  # e.g. "sex=F" or, for intersections, "sex=F|region=North"
    reference_group: Optional[str] = None
    n: Optional[int] = None  # rows behind the value
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    method: Optional[str] = None  # how the value and the interval were obtained
    test_type: Optional[str] = None  # "fixed" or "sequential"
    e_value: Optional[float] = None  # sequential tests
    suppressed: Optional[bool] = None  # cell below the minimum size: value withheld

    # TODO: validators for model id's, model's version, data_id, and metrics type
    @field_validator("metric_type")
    @classmethod
    def metric_type_is_invalid(cls, v):
        if (v not in MetricsType._member_names_) and (v is not None):
            raise error_msg(
                value=None,
                message=f'{"ValueError: Metric type should be None or one of {MetricsType._member_names_}"}',
            )
        return v

    @field_validator("metric_name")
    @classmethod
    def metric_name_is_invalid(cls, v, info: ValidationInfo):
        metric_names = (
            PerformanceMetricsFuncs._member_names_ + DriftMetricsFuncs._member_names_ + DriftTestMetricsFuncs._member_names_
        )
        if (v not in metric_names) and (
            info.data.get("metric_type")
            not in [
                MetricsType.custom.value,
                MetricsType.statistics.value,
                MetricsType.fairness.value,
                MetricsType.explainability.value,
            ]
        ):
            raise error_msg(
                value=None,
                message=f'{"ValueError:Metric name {v} is invalid"}',
            )
        return v


class AbstractMetrics(ABC):
    """AbstractMetrics class for for metric"""

    def __init__(self, metric_name: str):
        """Constructor of the AbstractMetrics class

        Parameters
        ----------
        metric_name : str
            The input value for metric_name
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

                status = drift_from_threshold(value, threshold, upper_bound)

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
