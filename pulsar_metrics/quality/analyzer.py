"""DataQualityAnalyzer: the data-quality checks run like the other analyzers (run, get_result, results_to_pandas,
pulsar_metrics.evidence.to_evidence)"""

from datetime import datetime
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

from ..analyzers.base import AbstractAnalyzer, _matches
from ..exceptions import CustomExceptionPulsarMetric as error_msg
from .checks import check
from .policy import DataQualityPolicy
from .profile import MAX_CATEGORIES, METADATA_COLUMNS, DataQualityProfile


class DataQualityAnalyzer(AbstractAnalyzer):
    """Checks windows against a data-quality reference profile

    The profile is either given (`profile=`, e.g. `DataQualityProfile.load("profile.yaml")` versioned with the
    model), learned with `fit(reference)`, or learned from the `reference` passed to the first `run()`.

    When the frames hold `model_id` / `model_version` columns and the analyzer has a model_id, rows are filtered on
    them as in Analyzer (compared as text). Unlike Analyzer, an empty window is not refused: its checks are recorded
    as not evaluable (`status="empty_window"`), the schema check still runs.

    Parameters
    ----------
    name, model_id, model_version, description : str
        As for Analyzer
    profile : DataQualityProfile, optional
    policy : DataQualityPolicy, dict or YAML path, optional
        Thresholds per check and column (default: DEFAULT_THRESHOLDS)
    columns, exclude, rules, duplicate_columns, max_categories :
        Passed to DataQualityProfile.fit when the analyzer learns its profile
    timestamp_column : str, optional
        Column giving the period of the window and of the reference (default "pred_timestamp")
    model_fingerprint : ModelFingerprint or dict, optional
        Digest of the analysed model (pulsar_metrics.fingerprint), carried into its evidence records
    """

    def __init__(
        self,
        name: str,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        description: str = None,
        profile: Optional[DataQualityProfile] = None,
        policy: Union[DataQualityPolicy, dict, str, None] = None,
        columns: Optional[Iterable[str]] = None,
        exclude: Iterable[str] = METADATA_COLUMNS,
        rules: Optional[Iterable] = None,
        duplicate_columns: Optional[Sequence[str]] = None,
        max_categories: int = MAX_CATEGORIES,
        timestamp_column: Optional[str] = "pred_timestamp",
        model_fingerprint=None,
    ):
        super().__init__(name, model_id, model_version, description, model_fingerprint=model_fingerprint)
        self.profile = profile
        self.policy = DataQualityPolicy.load(policy)
        self._fit_options = {
            "columns": columns,
            "exclude": exclude,
            "rules": rules,
            "duplicate_columns": duplicate_columns,
            "max_categories": max_categories,
            "timestamp_column": timestamp_column,
        }
        self._timestamp_column = timestamp_column

    def _select(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rows of the analyzer's model, when the frame carries model_id / model_version and the analyzer has them"""
        keep = pd.Series(True, index=data.index)
        for column, expected in (("model_id", self._model_id), ("model_version", self._model_version)):
            if expected is not None and column in data.columns:
                keep &= _matches(data[column], expected)
        return data if keep.all() else data.loc[keep]

    def fit(self, reference: pd.DataFrame, reference_id: Optional[str] = None) -> DataQualityProfile:
        """Learn the profile from a reference frame (rows of the analyzer's model) and keep it"""
        reference = self._select(reference)
        if reference.shape[0] == 0:
            raise error_msg(value=None, message="the reference holds no row of this model: no profile to learn")
        self.profile = DataQualityProfile.fit(reference, reference_id=reference_id, **self._fit_options)
        return self.profile

    def run(self, current: pd.DataFrame, reference: Optional[pd.DataFrame] = None, options: Optional[dict] = None):
        """Check the window against the profile

        Parameters
        ----------
        current : DataFrame
            Window to check
        reference : DataFrame, optional
            Reference to learn the profile from, when the analyzer has none yet (a held profile is kept: call fit()
            to learn a new one)
        options : dict, optional
            Policy for this run only (DataQualityPolicy, dict or YAML path), instead of the analyzer's policy
        """
        if self.profile is None:
            if reference is None:
                raise error_msg(value=None, message="no profile: pass profile=, call fit(reference) or run(current, reference)")
            self.fit(reference)
        policy = self.policy if options is None else DataQualityPolicy.load(options)
        window = self._select(current)
        period = {}
        if self._timestamp_column and self._timestamp_column in window.columns and window.shape[0]:
            stamps = pd.to_datetime(window[self._timestamp_column], format="mixed", errors="coerce")
            period = {"period_start": stamps.min(), "period_end": stamps.max()}
        self._metadata.update(
            {"period_start": None, "period_end": None, **period, "eval_timestamp": datetime.now(), "options": {}}
        )
        self._window_info = {
            "n_current": int(window.shape[0]),
            "n_reference": self.profile.n_rows,
            "reference_id": self.profile.reference_id,
            "reference_start": self.profile.reference_start,
            "reference_end": self.profile.reference_end,
        }
        self._results = check(self.profile, window, policy)
        return self._results
