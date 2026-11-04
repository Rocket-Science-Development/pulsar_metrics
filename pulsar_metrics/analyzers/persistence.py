"""Persistence of drift verdicts across consecutive analysis windows.

A single window can flag a drift by chance: with a significance level of 5 %, a statistical test raises a false alarm
about once every twenty windows on stable data. DriftPersistence confirms a drift only when it is seen in `windows`
consecutive analyses, and clears it only after `windows` consecutive analyses without drift (hysteresis).
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Union

import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults

ResultsInput = Union[Iterable[MetricResults], pd.DataFrame]


@dataclass
class PersistentDrift:
    """Verdict of one metric after applying the persistence rule"""

    metric_type: str
    metric_name: str
    feature_name: Optional[str]
    drift_status: Optional[bool]  # raw verdict of the current window
    confirmed_drift: bool  # verdict after the persistence rule
    drift_streak: int  # consecutive windows with drift_status=True
    no_drift_streak: int  # consecutive windows with drift_status=False
    changed: bool  # confirmed_drift switched in this window


class DriftPersistence:
    """Confirm drift verdicts that persist over consecutive analysis windows

    Parameters
    ----------
    windows : int, optional
        Number of consecutive windows needed to confirm a drift, and to clear it (default 2). 1 keeps the raw verdicts.
    metric_types : tuple, optional
        Result types to track (default drift and performance; summary statistics have no verdict)

    Rules, per (metric_type, metric_name, feature_name):
    - drift_status=True increments the drift streak and resets the no-drift streak;
    - drift_status=False increments the no-drift streak and resets the drift streak;
    - drift_status=None (no threshold, or a metric in error) carries no information: both streaks are kept.
    The drift is confirmed when the drift streak reaches `windows`, and cleared when the no-drift streak reaches `windows`.
    """

    def __init__(self, windows: int = 2, metric_types: tuple = ("drift", "performance", "custom")):
        if not isinstance(windows, int) or windows < 1:
            raise error_msg(value=windows, message="windows must be a positive integer")
        self.windows = windows
        self.metric_types = metric_types
        self._state: dict = {}

    @staticmethod
    def _records(results: ResultsInput) -> list[dict]:
        if isinstance(results, pd.DataFrame):
            return results.to_dict(orient="records")
        return [r.model_dump() if isinstance(r, MetricResults) else dict(r) for r in results]

    def update(self, results: ResultsInput) -> list[PersistentDrift]:
        """Feed the results of one analysis window and return the verdicts after the persistence rule

        Parameters
        ----------
        results : list of MetricResults or pandas.DataFrame
            Output of Analyzer.get_result() or Analyzer.results_to_pandas() for one window

        Returns
        -------
        list of PersistentDrift
            One entry per tracked metric of the window
        """
        verdicts = []
        for record in self._records(results):
            if record.get("metric_type") not in self.metric_types:
                continue
            feature = record.get("feature_name")
            feature = None if pd.isna(feature) else feature
            key = (record["metric_type"], record["metric_name"], feature)
            state = self._state.setdefault(key, {"drift_streak": 0, "no_drift_streak": 0, "confirmed": False})

            status = record.get("drift_status")
            status = None if status is None or pd.isna(status) else bool(status)
            if status is True:
                state["drift_streak"] += 1
                state["no_drift_streak"] = 0
            elif status is False:
                state["no_drift_streak"] += 1
                state["drift_streak"] = 0

            previous = state["confirmed"]
            if state["drift_streak"] >= self.windows:
                state["confirmed"] = True
            elif state["no_drift_streak"] >= self.windows:
                state["confirmed"] = False
            state["changed"] = state["confirmed"] != previous

            verdicts.append(
                PersistentDrift(
                    metric_type=key[0],
                    metric_name=key[1],
                    feature_name=key[2],
                    drift_status=status,
                    confirmed_drift=state["confirmed"],
                    drift_streak=state["drift_streak"],
                    no_drift_streak=state["no_drift_streak"],
                    changed=state["confirmed"] != previous,
                )
            )
        return verdicts

    def update_to_pandas(self, results: ResultsInput) -> pd.DataFrame:
        """Same as update(), returned as a pandas DataFrame"""
        return pd.DataFrame([v.__dict__ for v in self.update(results)])

    def confirmed(self) -> list[tuple]:
        """Keys (metric_type, metric_name, feature_name) whose drift is currently confirmed"""
        return sorted((k for k, s in self._state.items() if s["confirmed"]), key=str)

    def reset(self):
        """Forget all history, e.g. after a new model version or a new reference"""
        self._state.clear()
