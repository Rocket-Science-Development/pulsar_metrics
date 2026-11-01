import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.analyzers.persistence import DriftPersistence
from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.metrics.base import MetricResults


def window(status, metric_name="psi", feature_name="income", metric_type="drift"):
    value = None if status is None else 0.3
    return [
        MetricResults(
            metric_type=metric_type,
            metric_name=metric_name,
            feature_name=feature_name,
            metric_value=value,
            drift_status=status,
            threshold=0.2,
        )
    ]


def run_sequence(statuses, windows=2):
    persistence = DriftPersistence(windows=windows)
    return [persistence.update(window(s))[0] for s in statuses]


def test_single_window_false_alarm_is_not_confirmed():
    verdicts = run_sequence([False, False, True, False, False])
    assert [v.confirmed_drift for v in verdicts] == [False] * 5


def test_drift_is_confirmed_after_two_consecutive_windows():
    verdicts = run_sequence([False, True, True, True])
    assert [v.confirmed_drift for v in verdicts] == [False, False, True, True]
    assert [v.changed for v in verdicts] == [False, False, True, False]


def test_drift_is_cleared_after_two_consecutive_windows_without_drift():
    verdicts = run_sequence([True, True, False, True, False, False])
    assert [v.confirmed_drift for v in verdicts] == [False, True, True, True, True, False]
    assert verdicts[-1].changed


def test_missing_verdict_keeps_the_streaks():
    verdicts = run_sequence([True, None, True])
    assert [v.confirmed_drift for v in verdicts] == [False, False, True]
    assert verdicts[1].drift_status is None and verdicts[1].drift_streak == 1


@pytest.mark.parametrize("windows, expected", [(1, [True, False, True]), (3, [False, False, False])])
def test_number_of_windows_is_configurable(windows, expected):
    verdicts = run_sequence([True, False, True], windows=windows)
    assert [v.confirmed_drift for v in verdicts] == expected


@pytest.mark.parametrize("windows", [0, -1, 1.5])
def test_invalid_number_of_windows(windows):
    with pytest.raises(error_msg):
        DriftPersistence(windows=windows)


def test_metrics_are_tracked_independently_and_statistics_ignored():
    persistence = DriftPersistence()
    first = window(True, "psi", "income") + window(True, "psi", "age") + window(None, "mean", "age", "statistics")
    second = window(True, "psi", "income") + window(False, "psi", "age")
    assert len(persistence.update(first)) == 2
    result = {(v.metric_name, v.feature_name): v.confirmed_drift for v in persistence.update(second)}
    assert result == {("psi", "income"): True, ("psi", "age"): False}
    assert persistence.confirmed() == [("drift", "psi", "income")]
    persistence.reset()
    assert persistence.confirmed() == []


def _data(n, shift, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "income": rng.normal(shift, 1, n),
            "y_true": rng.integers(0, 2, n),
            "y_pred": rng.integers(0, 2, n),
            "model_id": "m1",
            "model_version": "1",
            "pred_timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        }
    )


def test_with_analyzer_results_over_successive_windows():
    reference = _data(1000, 0.0, 0)
    persistence = DriftPersistence(windows=2)
    confirmed = []
    # stable, isolated shift (false alarm), stable, then a lasting shift
    for k, shift in enumerate([0.0, 1.0, 0.0, 1.0, 1.0]):
        analyzer = Analyzer(name="window", model_id="m1", model_version="1")
        analyzer.add_drift_metrics(metrics_list=["ks_2samp"], features_list=["income"])
        analyzer.run(current=_data(300, shift, k + 1), reference=reference)
        table = persistence.update_to_pandas(analyzer.results_to_pandas())
        confirmed.append(bool(table.loc[table.metric_name == "ks_2samp", "confirmed_drift"].iloc[0]))
    assert confirmed == [False, False, False, False, True]
