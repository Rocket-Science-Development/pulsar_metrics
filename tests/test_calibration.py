import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.calibration import AACalibrator
from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.metrics.drift import DriftMetric, DriftTestMetric
from pulsar_metrics.multiple_testing import benjamini_hochberg, fdr_correct


def ar1(n, phi, rng):
    """Autocorrelated series: successive values are not independent, as successive LLM requests"""
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal() * np.sqrt(1 - phi**2)
    return x


def frame(x):
    return pd.DataFrame({"score": x})


def fresh_pairs(n_pairs, window, phi, seed):
    """Independent A/A pairs, each from a new stretch of the same process"""
    rng = np.random.default_rng(seed)
    for _ in range(n_pairs):
        x = ar1(2 * window, phi, rng)
        yield x[:window], x[window:]


def test_iid_data_calibrated_alpha_is_close_to_nominal():
    rng = np.random.default_rng(0)
    cal = AACalibrator(window_size=100, n_draws=200, contiguous=False).fit(
        frame(rng.normal(size=2000)), [DriftTestMetric("ks_2samp", "score"), DriftMetric("wasserstein", "score")]
    )
    alpha = cal.options()["ks_2samp|score"]["alpha"]
    assert 0.02 < alpha < 0.12
    threshold = cal.options()["wasserstein|score"]["threshold"]
    false_alarms = np.mean([stats.wasserstein_distance(a, b) > threshold for a, b in fresh_pairs(200, 100, 0.0, 1)])
    assert false_alarms < 0.12


def test_calibration_controls_false_alarms_on_autocorrelated_traffic():
    rng = np.random.default_rng(1)
    reference = frame(ar1(6000, 0.9, rng))
    cal = AACalibrator(window_size=200, n_draws=200, contiguous=True).fit(reference, [DriftTestMetric("ks_2samp", "score")])
    alpha = cal.options()["ks_2samp|score"]["alpha"]

    pvalues = np.array([stats.ks_2samp(a, b).pvalue for a, b in fresh_pairs(150, 200, 0.9, 2)])
    naive_false_alarms = np.mean(pvalues < 0.05)
    calibrated_false_alarms = np.mean(pvalues < alpha)
    assert naive_false_alarms > 0.3  # the nominal 5 % is far off
    assert calibrated_false_alarms < 0.15
    assert cal.false_alarm_rate("ks_2samp", "score") > 0.3
    assert alpha < 0.05


def test_empirical_pvalue_ranks_observed_statistics():
    rng = np.random.default_rng(2)
    cal = AACalibrator(window_size=100, n_draws=99, contiguous=False).fit(
        frame(rng.normal(size=1000)), [DriftMetric("wasserstein", "score"), DriftTestMetric("ttest", "score")]
    )
    distance, test = cal.nulls[("wasserstein", "score")], cal.nulls[("ttest", "score")]
    assert distance.empirical_pvalue(10.0) == pytest.approx(1 / 100)  # far beyond every A/A draw
    assert distance.empirical_pvalue(0.0) == pytest.approx(1.0)
    assert test.empirical_pvalue(0.0) == pytest.approx(1 / 100)
    assert np.isnan(distance.empirical_pvalue(float("nan")))


def test_invalid_settings_and_inputs():
    with pytest.raises(error_msg):
        AACalibrator(window_size=1)
    with pytest.raises(error_msg):
        AACalibrator(window_size=10, quantile=0.4)
    small = frame(np.arange(30.0))
    with pytest.raises(error_msg, match="2 x window_size"):
        AACalibrator(window_size=20).fit(small, [DriftMetric("wasserstein", "score")])
    with pytest.raises(error_msg, match="no drift metric"):
        AACalibrator(window_size=5).fit(small, [])


def test_benjamini_hochberg_known_example():
    rejected, adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.005, 0.2, np.nan], alpha=0.05)
    assert list(rejected) == [True, True, True, True, False, False]
    np.testing.assert_allclose(adjusted[:5], [0.025, 0.05, 0.05, 0.025, 0.2])
    assert np.isnan(adjusted[5])


def test_benjamini_hochberg_rejects_less_than_raw_threshold_on_null_pvalues():
    rng = np.random.default_rng(3)
    pvalues = rng.uniform(size=40)
    rejected, _ = benjamini_hochberg(pvalues, alpha=0.05)
    assert rejected.sum() <= (pvalues < 0.05).sum()


def _window(n, shift, seed, start):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "model_id": "agent",
            "model_version": "1",
            "pred_timestamp": pd.date_range(start, periods=n, freq="min"),
            "steps": rng.poisson(4 + shift, n).astype(float),
            "latency": rng.normal(1.0, 0.2, n),
        }
    )


def test_analyzer_with_calibrated_options_and_fdr():
    reference = _window(3000, 0, 0, "2026-01-01")
    analyzer = Analyzer(name="agent", model_id="agent", model_version="1")
    analyzer.add_drift_metrics(metrics_list=["ks_2samp", "wasserstein"], features_list=["steps", "latency"])

    cal = AACalibrator(window_size=300, n_draws=100).fit(reference, analyzer._metrics_list)
    options = cal.options()
    assert set(options) == {"ks_2samp|steps", "ks_2samp|latency", "wasserstein|steps", "wasserstein|latency"}

    analyzer.run(current=_window(300, 2, 1, "2026-02-01"), reference=reference, options=options)
    results = analyzer.results_to_pandas()
    drift = results[results.metric_type == "drift"].copy()
    # the per-feature calibrated threshold was used
    row = drift[(drift.metric_name == "wasserstein") & (drift.feature_name == "steps")].iloc[0]
    assert row.threshold == pytest.approx(options["wasserstein|steps"]["threshold"])

    drift["p_empirical"] = cal.empirical_pvalues(drift)
    corrected = fdr_correct(drift, "p_empirical", alpha=0.05)
    verdict = corrected.groupby("feature_name")["drift_fdr"].any()
    assert bool(verdict["steps"]) is True  # shifted feature
    assert bool(verdict["latency"]) is False  # unchanged feature


def test_warns_when_the_reference_is_too_short_for_the_window():
    reference = frame(np.random.default_rng(4).normal(size=1000))
    with pytest.warns(UserWarning, match="unreliable"):
        AACalibrator(window_size=300, n_draws=50).fit(reference, [DriftTestMetric("ks_2samp", "score")])
