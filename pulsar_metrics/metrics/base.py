#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

import pandas as pd
from pydantic import BaseModel, validator


class MetricsType(Enum):

    """Metrics type enumeration"""

    performance = "performance"
    drift = "drift"
    custom = "custom"


class MetricResults(BaseModel):

    """Data structure for the results of a metric"""

    metric_name: str = None
    type: str = None
    model_id: str = None
    model_version: str = None
    data_id: str = None
    feature: str = None
    value: float = None
    status: bool = None
    threshold: float = None
    period_start: datetime = None
    period_end: datetime = None
    eval_timestamp: datetime = datetime.now()
    conf_int: list = None

    @validator("eval_timestamp", always=True)
    def timestamp_later_than_period_end(cls, v, values, **kwargs):
        if v < values["period_end"]:
            raise ValueError("Current timestamp earlier than period end")
        return v

    # TODO: validators for model id's, model's version, data_id, and metrics type


class AbstractMetrics(ABC):

    """Base abstract class for metricsa"""

    def __init__(self, name: str, data: pd.DataFrame):

        """Parameters
        ----------
        - name: name of the metrics
        - data: dataset from which the metric is calculated
        """

        self._name = name
        self._data = data.copy(deep=True)  # Not sure I want to attach the data as an attribute ...

        # TODO: validation on the dataset ?

        try:
            # TODO: better handling of date format
            data["pred_timestamp"] = pd.to_datetime(data["pred_timestamp"])
            self._model_id = str(data["model_id"].unique()[0])
            self._model_version = str(data["model_version"].unique()[0])
            self._period_start = data.pred_timestamp.min()
            self._period_end = data.pred_timestamp.max()
            self._result = None
        except Exception as e:
            print(str(e))

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

    def inner(name: str, data: pd.DataFrame) -> AbstractMetrics:
        class CustomClass(AbstractMetrics):
            def __init__(self, name, data):
                super().__init__(name, data)

            def evaluate(self, **kwargs):

                value = func(**kwargs)
                threshold = kwargs.get("threshold", None)

                if isinstance(threshold, (int, float)):
                    status = value < threshold
                else:
                    status = None

                self._result = MetricResults(
                    metric_name=self._name,
                    type=MetricsType.custom.value,
                    model_id=self._model_id,
                    model_version=self._model_version,
                    value=value,
                    conf_int=None,
                    status=status,
                    threshold=threshold,
                    period_start=self._period_start,
                    period_end=self._period_end,
                )

                return self._result

        return CustomClass(name=name, data=data)

    return inner
