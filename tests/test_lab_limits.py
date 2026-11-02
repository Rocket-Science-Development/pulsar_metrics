"""Regression tests for the limits confirmed by the night lab (C1, C2, M1-M5, m1-m3, m16, m21).

Each test reproduces the situation measured by the lab and fails on the code before the fix.
"""

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.calibration import AACalibrator, CalibrationCache
from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.metrics.drift import DriftMetric, DriftTestMetric
from pulsar_metrics.metrics.performance import PerformanceMetric
from pulsar_metrics.multiple_testing import fdr_correct


def frame(**columns):
    return pd.DataFrame(columns)


@contextlib.contextmanager
def quiet():
    """Silence the calibration warnings of a test that does not check them"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def window(n, seed, shift=0.0, start="2026-01-01", n_features=1, version="1"):
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.normal(shift if i == 0 else 0.0, 1, n) for i in range(n_features)}
    data.update(
        {
            "model_id": "m1",
            "model_version": version,
            "pred_timestamp": pd.date_range(start, periods=n, freq="min").astype(str),
        }
    )
    return pd.DataFrame(data)


# ---------------------------------------------------------------- C1: p-value floor vs Benjamini-Hochberg
def test_c1_draws_are_raised_so_that_bh_can_reject_a_single_drift_among_40_tests():
    reference = window(4000, 0, n_features=40)
    features = [f"f{i}" for i in range(40)]
    specs = [DriftMetric("wasserstein", f) for f in features]
    with quiet():
        cal = AACalibrator(window_size=100).fit(reference, specs)
    # 1 / (n_draws + 1) must sit below 0.05 / 40 = 0.00125, with a margin of 2
    assert cal.n_draws_used >= 1599
    assert cal.pvalue_floor <= 0.05 / 40 / 2

    analyzer = Analyzer(name="c1", model_id="m1", model_version="1")
    analyzer.add_drift_metrics(["wasserstein"], features_list=features)
    analyzer.run(current=window(100, 9, shift=2.0, n_features=40), reference=reference)
    drift = analyzer.results_to_pandas().query("metric_type == 'drift'").copy()
    drift["p_empirical"] = cal.empirical_pvalues(drift)
    corrected = fdr_correct(drift, "p_empirical", alpha=0.05)
    flagged = set(corrected.loc[corrected.drift_fdr.astype(bool), "feature_name"])
    assert "f0" in flagged  # with 200 draws, f0 (floor 0.004975) could not pass 0.05/40


def test_c1_warns_when_the_automatic_increase_is_disabled():
    reference = window(2000, 1, n_features=30)
    specs = [DriftMetric("wasserstein", f"f{i}") for i in range(30)]
    with pytest.warns(UserWarning, match="cannot pass the correction"):
        cal = AACalibrator(window_size=100, auto_draws=False).fit(reference, specs)
    assert cal.n_draws_used == 200
    drift = pd.DataFrame({"metric_name": "wasserstein", "feature_name": [f"f{i}" for i in range(30)], "metric_value": 9.0})
    with pytest.warns(UserWarning, match="BH threshold"):
        cal.empirical_pvalues(drift)
    with pytest.warns(UserWarning, match="BH threshold"):
        fdr_correct(drift.assign(p=1 / 201), "p", pvalue_floor=cal.pvalue_floor)


# ---------------------------------------------------------------- C2: partial ground truth
def test_c2_performance_is_evaluated_on_labelled_rows_with_coverage():
    rng = np.random.default_rng(2)
    current = window(200, 2)
    current["y_pred"] = rng.integers(0, 2, 200)
    current["y_true"] = current["y_pred"].astype(float)
    current.loc[current.index[:36], "y_true"] = np.nan  # 18 % of labels missing, as in the lab
    current.loc[current.index[36:56], "y_true"] = 1 - current.loc[current.index[36:56], "y_pred"]

    analyzer = Analyzer(name="c2", model_id="m1", model_version="1")
    analyzer.add_performance_metrics(["accuracy"])
    analyzer.run(current=current, reference=window(200, 3), options={"accuracy": {"threshold": 0.95}})
    result = [r for r in analyzer.get_result() if r.metric_name == "accuracy"][0]
    assert result.error is None
    assert result.n_labelled == 164 and result.label_coverage == pytest.approx(0.82)
    assert result.metric_value == pytest.approx(144 / 164)
    assert result.drift_status is True


def test_c2_minimum_label_coverage_is_configurable():
    current = frame(y_true=[1.0, np.nan, np.nan, np.nan, 0.0], y_pred=[1, 1, 0, 0, 0])
    metric = PerformanceMetric("accuracy", min_label_coverage=0.5)
    result = metric.evaluate(current=current)
    assert "label coverage too low" in result.error and result.metric_value is None and result.drift_status is None
    assert result.n_labelled == 2 and result.label_coverage == pytest.approx(0.4)
    assert PerformanceMetric("accuracy").evaluate(current=current).metric_value == 1.0
    no_label = PerformanceMetric("accuracy").evaluate(current=frame(y_true=[np.nan, np.nan], y_pred=[1, 0]))
    assert "no ground truth" in no_label.error


def test_c2_label_sentinel_counts_as_unlabelled():
    current = frame(y_true=[1.0, -200.0, 0.0, -200.0], y_pred=[1, 1, 0, 0])
    result = PerformanceMetric("mse", missing_values=-200).evaluate(current=current)
    assert result.metric_value == 0.0 and result.n_labelled == 2


# ---------------------------------------------------------------- M1: sentinels and missing-rate drift
def test_m1_sentinel_is_not_analysed_as_a_measurement():
    rng = np.random.default_rng(4)
    reference = window(500, 5)
    current = window(500, 6)
    current.loc[current.index[: int(0.31 * 500)], "f0"] = -200.0  # 31 % of sentinels, as in week 32

    raw = Analyzer(name="raw", model_id="m1", model_version="1")
    raw.add_drift_metrics(["ks_2samp"], features_list=["f0"])
    raw.run(current=current, reference=reference)
    assert [r for r in raw.get_result() if r.metric_name == "ks_2samp"][0].drift_status is True

    declared = Analyzer(name="declared", model_id="m1", model_version="1")
    declared.add_drift_metrics(["ks_2samp", "missing_rate"], features_list=["f0"], missing_values={"f0": -200})
    declared.run(current=current, reference=reference)
    results = {r.metric_name: r for r in declared.get_result() if r.metric_type == "drift"}
    assert results["ks_2samp"].drift_status is False  # the measured values did not change
    assert results["ks_2samp"].missing_share_current == pytest.approx(0.31)
    assert results["missing_rate"].drift_status is True  # the missing rate did
    assert rng is not None
    mean = [r for r in declared.get_result() if r.metric_name == "mean" and r.feature_name == "f0"][0]
    assert abs(mean.metric_value) < 0.3  # summary statistics no longer average the sentinel in


def test_m1_missing_rate_without_missing_values_is_not_a_drift():
    result = DriftTestMetric("missing_rate", "x").evaluate(frame(x=[1.0, 2.0]), frame(x=[3.0, 4.0]))
    assert result.metric_value == 1.0 and result.drift_status is False


# ---------------------------------------------------------------- M2: NaN in drift tests
@pytest.mark.parametrize("name, cls", [("wasserstein", DriftMetric), ("ks_2samp", DriftTestMetric), ("psi", DriftMetric)])
def test_m2_nan_are_omitted_and_their_share_recorded(name, cls):
    rng = np.random.default_rng(7)
    reference = frame(x=rng.normal(0, 1, 400))
    reference.loc[:39, "x"] = np.nan
    current = frame(x=rng.normal(0, 1, 400))
    options = {"threshold": 0.5} if cls is DriftMetric else {}
    result = cls(name, "x").evaluate(current, reference, **options)
    assert result.error is None and result.metric_value is not None and not np.isnan(result.metric_value)
    assert result.drift_status is False
    assert result.missing_share_reference == pytest.approx(0.1) and result.missing_share_current == 0.0


@pytest.mark.parametrize(
    "name, cls, options", [("wasserstein", DriftMetric, {"threshold": 0.5}), ("ks_2samp", DriftTestMetric, {})]
)
def test_m2_nan_statistic_gives_no_verdict(name, cls, options):
    reference = frame(x=[0.0, np.nan, 1.0, 2.0])
    result = cls(name, "x", nan_policy="propagate").evaluate(frame(x=[0.0, 1.0, 2.0, 3.0]), reference, **options)
    assert result.drift_status is None and result.metric_value is None and "NaN" in result.error


def test_m2_nan_policy_raise():
    with pytest.raises(error_msg):
        DriftMetric("wasserstein", "x", nan_policy="raise").evaluate(frame(x=[0.0, 1.0]), frame(x=[np.nan, 1.0]))
    with pytest.raises(error_msg):
        DriftMetric("wasserstein", "x", nan_policy="drop")


# ---------------------------------------------------------------- M3: isolated calibration failures
def test_m3_one_failing_metric_does_not_stop_the_calibration():
    rng = np.random.default_rng(8)
    reference = frame(x=rng.normal(size=2000), text=["a"] * 2000)
    reference.loc[::7, "x"] = np.nan
    specs = [DriftMetric("psi", "x"), DriftTestMetric("ks_2samp", "x"), DriftMetric("wasserstein", "text")]
    with pytest.warns(UserWarning, match="calibration of wasserstein on text failed"):
        cal = AACalibrator(window_size=100).fit(reference, specs)
    assert {("psi", "x"), ("ks_2samp", "x")} <= set(cal.nulls)  # psi used to raise on the NaN and stop everything
    assert set(cal.errors) == {("wasserstein", "text")}
    assert "error" in cal.summary().columns


# ---------------------------------------------------------------- M4 and m2: constant reference
def test_m4_constant_reference_has_its_own_status():
    rng = np.random.default_rng(9)
    reference = frame(price=np.zeros(3000), load=rng.normal(size=3000))
    specs = [DriftTestMetric("ks_2samp", "price"), DriftMetric("wasserstein", "price"), DriftTestMetric("ks_2samp", "load")]
    with pytest.warns(UserWarning, match="constant in the reference"):
        cal = AACalibrator(window_size=100).fit(reference, specs)
    assert cal.nulls[("ks_2samp", "price")].status == "constant_reference"
    assert cal.nulls[("ks_2samp", "load")].status == "ok"
    assert cal.options()["wasserstein|price"]["reliable"] is False

    result = DriftMetric("wasserstein", "price").evaluate(frame(price=rng.normal(size=100)), reference, threshold=0.1)
    assert result.status == "constant_reference" and result.drift_status is True  # the change is real, and labelled
    assert DriftMetric("wasserstein", "load").evaluate(reference, reference, threshold=0.1).status is None


# ---------------------------------------------------------------- M5: unstable thresholds
def test_m5_short_reference_thresholds_come_with_an_interval_and_a_flag():
    rng = np.random.default_rng(10)
    with quiet():
        short = AACalibrator(window_size=100).fit(frame(x=rng.normal(size=1000)), [DriftTestMetric("ks_2samp", "x")])
        long = AACalibrator(window_size=100).fit(frame(x=rng.normal(size=4000)), [DriftTestMetric("ks_2samp", "x")])
    short_null, long_null = short.nulls[("ks_2samp", "x")], long.nulls[("ks_2samp", "x")]
    assert short_null.n_effective == 45  # 10 windows: about 45 disjoint pairs, as computed by the lab
    low, high = short_null.threshold_interval()
    assert low <= short_null.threshold <= high
    assert high - low > np.subtract(*long_null.threshold_interval()[::-1])
    assert short_null.reliable is False and long_null.reliable is True


def test_m5_shrinkage_towards_nominal_alpha_reduces_the_spread():
    plain, shrunk = [], []
    with quiet():
        for seed in range(12):
            reference = frame(x=np.random.default_rng(100 + seed).normal(size=1000))
            for draws, out in ((0, plain), (45, shrunk)):
                cal = AACalibrator(window_size=100, n_draws=100, shrink_draws=draws, seed=seed)
                out.append(cal.fit(reference, [DriftTestMetric("ks_2samp", "x")]).options()["ks_2samp|x"]["alpha"])
    assert np.std(shrunk) < 0.6 * np.std(plain)
    assert abs(np.mean(shrunk) - 0.05) < abs(np.mean(plain) - 0.05) + 0.01


# ---------------------------------------------------------------- m3: contiguous draws on a very short reference
def test_m3_contiguous_draws_need_four_windows_and_sample_without_rejection():
    reference = frame(x=np.random.default_rng(11).normal(size=400))
    with pytest.raises(error_msg, match="at least 4 windows"):
        AACalibrator(window_size=200, allow_short=True).fit(reference, [DriftTestMetric("ks_2samp", "x")])
    cal = AACalibrator(window_size=100, allow_short=True)
    draws = cal._draw_positions(400, 500, np.random.default_rng(0))
    assert all(abs(a.start - b.start) >= 100 and b.stop <= 400 for a, b in draws)  # disjoint, inside the reference


# ---------------------------------------------------------------- m16: reference read back from a CSV
def test_m16_reference_read_back_from_csv_is_accepted():
    reference = window(300, 12, version="1.0")
    buffer = io.StringIO()
    reference.to_csv(buffer, index=False)
    buffer.seek(0)
    reloaded = pd.read_csv(buffer)
    assert reloaded.model_version.dtype == float  # "1.0" became 1.0

    analyzer = Analyzer(name="csv", model_id="m1", model_version="1.0")
    analyzer.add_drift_metrics(["ks_2samp"], features_list=["f0"])
    analyzer.run(current=window(100, 13, version="1.0"), reference=reloaded)
    assert any(r.metric_name == "ks_2samp" and r.error is None for r in analyzer.get_result())

    wrong = Analyzer(name="csv", model_id="m1", model_version="2")
    wrong.add_drift_metrics(["ks_2samp"], features_list=["f0"])
    with pytest.raises(error_msg, match="300 with model_id='m1', 0 also with model_version='2'"):
        wrong.run(current=window(100, 13, version="2"), reference=reloaded)


# ---------------------------------------------------------------- m21: reusable calibration
def test_m21_null_distributions_are_cached_across_models_and_processes(tmp_path):
    reference = window(2000, 14, n_features=3)
    specs = [DriftTestMetric("ks_2samp", f"f{i}") for i in range(3)]
    cache = CalibrationCache(tmp_path / "nulls")
    with quiet():
        first = AACalibrator(window_size=100).fit(reference, specs, cache=cache)
        assert (cache.hits, cache.misses) == (0, 3)
        # another model with the same reference values for f0 and f1, but a different f2
        other = reference.copy()
        other["f2"] = other["f2"] + 1.0
        AACalibrator(window_size=100).fit(other, specs, cache=cache)
        assert (cache.hits, cache.misses) == (2, 4)
        # a new process: the cache is read from disk
        fresh = CalibrationCache(tmp_path / "nulls")
        again = AACalibrator(window_size=100).fit(reference, specs, cache=fresh)
    assert fresh.hits == 3
    assert again.options() == first.options()

    path = tmp_path / "calibration.json"
    first.save(path)
    loaded = AACalibrator.load(path)
    assert loaded.options() == first.options()
    np.testing.assert_array_equal(loaded.nulls[("ks_2samp", "f0")].values, first.nulls[("ks_2samp", "f0")].values)


def test_m21_cache_key_changes_with_the_draw_settings(tmp_path):
    reference = window(2000, 15)
    cache = CalibrationCache()
    with quiet():
        AACalibrator(window_size=100).fit(reference, [DriftTestMetric("ks_2samp", "f0")], cache=cache)
        AACalibrator(window_size=200, allow_short=True).fit(reference, [DriftTestMetric("ks_2samp", "f0")], cache=cache)
        AACalibrator(window_size=100, seed=1).fit(reference, [DriftTestMetric("ks_2samp", "f0")], cache=cache)
    assert cache.hits == 0 and len(cache) == 3
