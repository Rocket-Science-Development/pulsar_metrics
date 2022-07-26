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

    name: str = None
    type: MetricsType = None
    model_id: str = None
    model_version: int = None
    data_id: str = None
    value: float = None
    status: bool = None
    threshold: float = None
    period_start: datetime = None
    period_end: datetime = None
    timestamp: datetime = datetime.now()
    conf_int: list = None

    @validator("timestamp", always=True)
    def timestamp_later_than_period_end(cls, v, values, **kwargs):
        if v < values["period_end"]:
            raise ValueError("Current timestamp earlier than period end")
        return v

    # TODO: validators for model id's, model's version, data_id, and metrics type


class AbstractMetrics(ABC):

    """Base abstract class for metricsa

    Parameters
    ----------
    - name: name of the metrics
    - data: dataset from which the metric is calculated
    """

    def __init__(self, name: str, data: pd.DataFrame):

        self._name = name
        self._data = data.copy(deep=True)  # Not sure I want to attach the data as an attribute ...

        # TODO: validation on the dataset ?

        try:
            self._model_id = str(data["model_id"].unique()[0])
            self._model_version = str(data["model_version"].unique()[0])
            self._period_start = data.date.min()
            self._period_end = data.date.max()
        except Exception as e:
            print(str(e))

    @property
    @abstractmethod
    def evaluate(self) -> MetricResults:
        raise NotImplementedError
