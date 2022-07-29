#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum


class AbstractAnalyzer(ABC):

    """Base abstract class for analyzers"""

    def __init__(self, name: str, data: pd.DataFrame):

        self._name = name
        self._data = data.copy(deep=True)  # Not sure I want to attach the data as an attribute ...

        # TODO: validation on the dataset ?

        try:
            self._model_id = str(data["model_id"].unique()[0])
            self._model_version = str(data["model_version"].unique()[0])
            self._period_start = data.date.min()
            self._period_end = data.date.max()
            self._result = None
        except Exception as e:
            print(str(e))

    @property
    @abstractmethod
    def run(self):
        raise NotImplementedError

    def add_performance_metrics(self, metricsNames: list):
        pass

    def add_drift_metrics(self, metricsNames: list):
        pass

    def add_drift_test_metrics(self, metricsNames: list):
        pass

    def get_result(self):
        return self._result

    def log_results(self):
        pass
