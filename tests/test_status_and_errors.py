import logging

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.exceptions import MetricEvaluationError
from pulsar_metrics.metrics.drift import CustomDriftMetric, DriftMetric, DriftTestMetric
from pulsar_metrics.metrics.performance import PerformanceMetric
from pulsar_metrics.metrics.statistics import FeatureSummary
from pulsar_metrics.utils import drift_from_threshold

rng = np.random.default_rng(7)
REFERENCE = pd.DataFrame({"x": rng.normal(0, 1, 400)})
SHIFTED = pd.DataFrame({"x": rng.normal(2, 1, 400)})
CLASSIFICATION = pd.DataFrame({"y_true": [1, 1, 0, 0, 1, 0, 1, 0], "y_pred": [1, 0, 1, 0, 0, 1, 0, 1]})  # accuracy 0.25


def _make_data(n, seed, with_ground_truth=True):
    gen = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "x1": gen.normal(0, 1, n),
            "y_pred": gen.integers(0, 2, n),
            "model_id": "m1",
            "model_version": "1",
            "pred_timestamp": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
        }
    )
    if with_ground_truth:
        data["y_true"] = gen.integers(0, 2, n)
    return data


# drift_status: True always means drift / degradation
# ==============================


@pytest.mark.parametrize(
    "value, threshold, upper_bound, expected",
    [
        (0.8, 0.5, True, True),
        (0.2, 0.5, True, False),
        (0.2, 0.5, False, True),
        (0.8, 0.5, False, False),
        (0.8, [0.1, 0.5], True, True),
        (0.3, [0.1, 0.5], True, False),
        (0.8, None, True, None),
    ],
)
def test_drift_from_threshold(value, threshold, upper_bound, expected):
    assert drift_from_threshold(value, threshold, upper_bound) is expected


@pytest.mark.parametrize("current, expected", [(SHIFTED, True), (REFERENCE, False)])
def test_distance_metric_flags_drift_above_threshold(current, expected):
    metric = DriftMetric(metric_name="wasserstein", feature_name="x")
    result = metric.evaluate(current=current, reference=REFERENCE, threshold=0.5)
    assert result.drift_status is expected


@pytest.mark.parametrize("current, expected", [(SHIFTED, True), (REFERENCE, False)])
def test_statistical_test_flags_drift_below_alpha(current, expected):
    result = DriftTestMetric(metric_name="ks_2samp", feature_name="x").evaluate(current=current, reference=REFERENCE, alpha=0.05)
    assert result.drift_status is expected


def test_distance_metric_without_threshold_has_no_verdict():
    result = DriftMetric(metric_name="wasserstein", feature_name="x").evaluate(current=SHIFTED, reference=REFERENCE)
    assert result.metric_value > 0 and result.drift_status is None


@pytest.mark.parametrize(
    "metric_name, threshold, expected",
    [
        ("accuracy", 0.8, True),  # 0.25 below the minimum acceptable accuracy
        ("accuracy", 0.2, False),
        ("mae", 0.5, True),  # 0.75 above the maximum acceptable error
        ("mae", 0.9, False),
    ],
)
def test_performance_threshold_direction_follows_the_metric(metric_name, threshold, expected):
    result = PerformanceMetric(metric_name=metric_name).evaluate(current=CLASSIFICATION, threshold=threshold)
    assert result.drift_status is expected


def test_performance_upper_bound_can_be_overridden():
    result = PerformanceMetric(metric_name="accuracy").evaluate(current=CLASSIFICATION, threshold=0.2, upper_bound=True)
    assert result.drift_status is True


def test_custom_drift_metric_uses_the_same_convention():
    @CustomDriftMetric
    def mean_shift(current, reference, **kwargs):
        return abs(current.mean() - reference.mean())

    result = mean_shift(metric_name="mean_shift", feature_name="x").evaluate(current=SHIFTED, reference=REFERENCE, threshold=1.0)
    assert result.drift_status is True


# Explicit errors
# ==============================


def test_drift_metric_raises_on_missing_column():
    with pytest.raises(MetricEvaluationError, match="wasserstein"):
        DriftMetric(metric_name="wasserstein", feature_name="missing").evaluate(current=SHIFTED, reference=REFERENCE)


def test_drift_test_raises_on_missing_column():
    with pytest.raises(MetricEvaluationError, match="ks_2samp"):
        DriftTestMetric(metric_name="ks_2samp", feature_name="missing").evaluate(current=SHIFTED, reference=REFERENCE)


def test_performance_metric_raises_on_missing_column():
    with pytest.raises(MetricEvaluationError, match="accuracy"):
        PerformanceMetric(metric_name="accuracy", pred_name="missing").evaluate(current=CLASSIFICATION)


def test_feature_summary_raises_on_missing_feature():
    with pytest.raises(MetricEvaluationError):
        FeatureSummary(feature_name="missing").evaluate(SHIFTED, REFERENCE)


def test_feature_summary_reports_top_category():
    summary = FeatureSummary(feature_name="tool")
    summary.evaluate(pd.DataFrame({"tool": ["search", "search", "email"]}), pd.DataFrame({"tool": ["email"]}))
    top = [r for r in summary._result if r.metric_name == "top"]
    assert top and top[0].metric_value == "search" and top[0].threshold == "email"


@pytest.mark.parametrize("method, name", [("add_drift_metrics", "unknown"), ("add_performance_metrics", "unknown")])
def test_analyzer_rejects_unknown_metric_names(method, name):
    analyzer = Analyzer(name="test", model_id="m1", model_version="1")
    kwargs = {"features_list": ["x1"]} if method == "add_drift_metrics" else {}
    with pytest.raises(error_msg):
        getattr(analyzer, method)(metrics_list=[name], **kwargs)


def test_analyzer_records_failing_metrics_and_continues(caplog):
    analyzer = Analyzer(name="test", model_id="m1", model_version="1")
    analyzer.add_performance_metrics(metrics_list=["accuracy"])
    analyzer.add_drift_metrics(metrics_list=["wasserstein", "ks_2samp"], features_list=["x1", "missing"])
    with caplog.at_level(logging.WARNING, logger="pulsar_metrics.analyzers.base"):
        analyzer.run(
            current=_make_data(200, 1, with_ground_truth=False),
            reference=_make_data(200, 2),
            options={"wasserstein": {"threshold": 0.5}},
        )
    results = {(r.metric_name, r.feature_name): r for r in analyzer.get_result() if r.metric_type != "statistics"}

    assert "ground truth" in results[("accuracy", None)].error
    assert results[("wasserstein", "missing")].error and results[("wasserstein", "missing")].metric_value is None
    assert results[("wasserstein", "x1")].error is None and results[("wasserstein", "x1")].drift_status is False
    assert results[("ks_2samp", "x1")].error is None
    assert "Metric wasserstein failed" in caplog.text
    # Every result carries either a value or an error, never neither
    assert all((r.metric_value is not None) or r.error for r in analyzer.get_result())
