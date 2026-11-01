import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg


def _make_data(n, shift, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(shift, 1, n),
            "y_true": rng.integers(0, 2, n),
            "y_pred": rng.integers(0, 2, n),
            "model_id": "m1",
            "model_version": "1",
            "pred_timestamp": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
        }
    )


# Package modules import without extra entries on sys.path
def test_modules_import_as_package():
    import pulsar_metrics.metrics.drift  # noqa: F401
    import pulsar_metrics.metrics.performance  # noqa: F401
    import pulsar_metrics.metrics.statistics  # noqa: F401


# add_drift_metrics() needs an explicit features list
def test_add_drift_metrics_requires_features_list():
    analyzer = Analyzer(name="test", model_id="m1", model_version="1")
    with pytest.raises(error_msg):
        analyzer.add_drift_metrics(metrics_list=["wasserstein"])


# results_to_pandas() returns the computed results
def test_results_to_pandas_returns_dataframe():
    analyzer = Analyzer(name="test", model_id="m1", model_version="1")
    analyzer.add_drift_metrics(metrics_list=["wasserstein"], features_list=["x1"])
    analyzer.run(current=_make_data(200, 0.5, 1), reference=_make_data(200, 0.0, 2), options={"wasserstein": {"threshold": 0.1}})
    results = analyzer.results_to_pandas()
    assert isinstance(results, pd.DataFrame)
    assert "wasserstein" in results["metric_name"].values
    assert "options" not in results.columns


# Metric enums keep their members (functools.partial is not a member by default from Python 3.13)
def test_metric_enums_are_populated():
    from pulsar_metrics.metrics.enums import DriftMetricsFuncs, DriftTestMetricsFuncs, PerformanceMetricsFuncs

    assert len(DriftMetricsFuncs) == 4
    assert len(DriftTestMetricsFuncs) == 7
    assert len(PerformanceMetricsFuncs) == 12
