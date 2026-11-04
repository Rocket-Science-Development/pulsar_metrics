"""Regression tests for the calibration limits confirmed by the P2 lab (TC-04, TS-04, TC-02).

Each test reproduces the situation measured by the lab and fails on the code before the fix.
"""

import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.calibration import (
    STATUS_SATURATED,
    AACalibrator,
    period_reference,
)
from pulsar_metrics.metrics.drift import DriftMetric, DriftTestMetric
from pulsar_metrics.metrics.performance import PerformanceMetric
from pulsar_metrics.metrics.utils import OTHER_LEVEL, chi2_test, get_population_counts


@contextlib.contextmanager
def quiet():
    """Silence the calibration warnings of a test that does not check them"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def tagged(data: pd.DataFrame, start="2026-01-01", freq="min") -> pd.DataFrame:
    data = data.copy()
    data["model_id"], data["model_version"] = "m1", "1"
    data["pred_timestamp"] = pd.date_range(start, periods=len(data), freq=freq)
    return data


def monthly_regimes(n_months=12, rows=1500, seed=0) -> pd.DataFrame:
    """Macro variable constant within each month, different every month (euribor3m, TC-04), plus a noise feature"""
    rng = np.random.default_rng(seed)
    rate = np.repeat(np.linspace(4.0, 5.0, n_months), rows)
    return pd.DataFrame({"rate": rate, "noise": rng.normal(size=n_months * rows)})


# ---------------------------------------------------------------- TC-04: saturated null
@pytest.mark.parametrize("compare_to", ["reference", "window"])
def test_tc04_null_at_the_bound_of_the_statistic_is_flagged_saturated(compare_to):
    reference = monthly_regimes()
    specs = [DriftTestMetric("ks_2samp", "rate"), DriftMetric("psi", "rate"), DriftMetric("wasserstein", "rate")]
    with pytest.warns(UserWarning, match="saturated"):
        cal = AACalibrator(window_size=1000, n_draws=200, compare_to=compare_to).fit(reference, specs)
    ks, psi, wasserstein = (cal.nulls[(m, "rate")] for m in ("ks_2samp", "psi", "wasserstein"))
    assert ks.share_at_bound > 0.05 and ks.status == STATUS_SATURATED and ks.reliable is False
    assert psi.share_at_bound > 0.05 and psi.status == STATUS_SATURATED
    assert wasserstein.status != STATUS_SATURATED  # unbounded: the advised alternative
    options = cal.options()
    assert options["ks_2samp|rate"]["calibration_status"] == STATUS_SATURATED

    # the status is carried to the results of Analyzer.run
    analyzer = Analyzer(name="m", model_id="m1", model_version="1")
    analyzer.add_drift_metrics(metrics_list=["ks_2samp", "psi"], features_list=["rate"])
    crisis = pd.DataFrame({"rate": np.full(1000, 0.93), "noise": 0.0})
    with quiet():
        analyzer.run(current=tagged(crisis, "2027-01-01"), reference=tagged(reference), options=options)
    rows = analyzer.results_to_pandas().query("metric_type == 'drift'")
    assert set(rows.status) == {STATUS_SATURATED}
    assert not rows.reliable.any()
    assert rows.reliability_reason.str.contains("saturated").all()


def test_tc04_p_values_tied_at_one_do_not_count_as_saturation():
    # refuted lab item (TC-02, section 4): ties at p = 1 are on the non-critical side
    reference = pd.DataFrame({"level": np.tile(list("abcde"), 600)})  # perfectly regular: chi2 p = 1
    with quiet():
        cal = AACalibrator(window_size=100, n_draws=100).fit(reference, [DriftTestMetric("chi2", "level")])
    null = cal.nulls[("chi2", "level")]
    assert null.share_at_bound == 0.0
    assert null.status != STATUS_SATURATED


def test_tc04_stationary_feature_is_not_saturated():
    reference = monthly_regimes()
    with quiet():
        cal = AACalibrator(window_size=1000, n_draws=100).fit(
            reference, [DriftTestMetric("ks_2samp", "noise"), DriftMetric("psi", "noise")]
        )
    assert all(null.status == "ok" for null in cal.nulls.values())


# ---------------------------------------------------------------- TC-04: window size and labelled rows
def test_tc04_window_far_from_the_calibrated_size_is_flagged_unreliable():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"x": rng.normal(size=6000)})
    with quiet():
        cal = AACalibrator(window_size=200, n_draws=100).fit(reference, [DriftTestMetric("ks_2samp", "x")])
    options = cal.options()
    assert options["ks_2samp|x"]["window_size"] == 200
    analyzer = Analyzer(name="m", model_id="m1", model_version="1")
    analyzer.add_drift_metrics(metrics_list=["ks_2samp"], features_list=["x"])

    def run(n):
        with quiet():
            analyzer.run(current=tagged(pd.DataFrame({"x": rng.normal(size=n)}), "2027-01-01"), reference=tagged(reference),
                         options=options)  # fmt: skip
        return analyzer.results_to_pandas().query("metric_type == 'drift'").iloc[0]

    small, same, large = run(30), run(250), run(1000)
    assert small.reliable is False or not small.reliable
    assert "30 rows" in small.reliability_reason and "200 rows" in small.reliability_reason
    assert bool(same.reliable) is True and same.n == 250
    assert not large.reliable
    with pytest.warns(UserWarning, match="more than 2x away from the calibrated window"):
        cal.empirical_pvalues(pd.DataFrame([small]))


def test_tc04_auc_needs_30_labelled_rows_and_both_classes():
    rng = np.random.default_rng(2)
    ten = pd.DataFrame({"y_true": [1] + [0] * 9, "y_pred": rng.uniform(size=10)})  # 2008-12: 10 rows, 1 positive
    result = PerformanceMetric("auc").evaluate(ten)
    assert result.metric_value is None and "minimum" in result.error and "30 rows" in result.error

    one_class = pd.DataFrame({"y_true": [0] * 40, "y_pred": rng.uniform(size=40)})
    result = PerformanceMetric("aucpr").evaluate(one_class)
    assert result.metric_value is None and "both classes" in result.error

    enough = pd.DataFrame({"y_true": [0, 1] * 20, "y_pred": rng.uniform(size=40)})
    assert PerformanceMetric("auc").evaluate(enough).metric_value is not None
    # the old behaviour stays available
    assert PerformanceMetric("auc", min_labelled=1).evaluate(ten).metric_value is not None


# ---------------------------------------------------------------- TS-04: calibration compares what run compares
def ar1(n, phi, rng):
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal() * np.sqrt(1 - phi**2)
    return x


def test_ts04_default_null_compares_a_window_with_the_rest_of_the_reference():
    rng = np.random.default_rng(3)
    reference = ar1(8000, 0.9, rng)
    w = 200
    specs = [DriftMetric("wasserstein", "x")]
    with quiet():
        by_reference = AACalibrator(window_size=w, n_draws=300).fit(pd.DataFrame({"x": reference}), specs)
        by_window = AACalibrator(window_size=w, n_draws=300, compare_to="window").fit(pd.DataFrame({"x": reference}), specs)
    assert by_reference.compare_to == "reference"
    t_ref = by_reference.options()["wasserstein|x"]["threshold"]
    t_win = by_window.options()["wasserstein|x"]["threshold"]
    # what Analyzer.run computes: fresh windows of the same process against the whole reference
    observed = np.array([stats.wasserstein_distance(ar1(w, 0.9, rng), reference) for _ in range(200)])
    assert np.mean(observed > t_ref) == pytest.approx(0.05, abs=0.05)
    assert np.mean(observed > t_win) < 0.02  # window vs window thresholds are too lenient for that comparison
    assert t_win > 1.2 * t_ref


def test_ts04_seasonal_reference_warns_between_window_variance():
    days = pd.date_range("2011-01-01", "2011-12-31 23:00", freq="h")
    rng = np.random.default_rng(4)
    season = 100 + 60 * np.sin(2 * np.pi * (days.day_of_year.to_numpy() - 100) / 365)
    reference = pd.DataFrame({"cnt": season + rng.normal(0, 20, len(days)), "pred_timestamp": days})
    with pytest.warns(UserWarning, match="period_reference"):
        cal = AACalibrator(window_size=168, n_draws=100).fit(reference, [DriftMetric("wasserstein", "cnt")])
    assert cal.nulls[("wasserstein", "cnt")].between_within_ratio > 10
    stationary = pd.DataFrame({"cnt": rng.normal(0, 20, len(days))})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal = AACalibrator(window_size=168, n_draws=100).fit(stationary, [DriftMetric("wasserstein", "cnt")])
    assert cal.nulls[("wasserstein", "cnt")].between_within_ratio < 3


def test_ts04_period_reference_selects_the_same_weeks_of_the_year():
    days = pd.date_range("2011-01-01", "2011-12-31 23:00", freq="h")
    reference = pd.DataFrame({"x": np.arange(len(days)), "pred_timestamp": days})
    window = pd.DataFrame({"pred_timestamp": pd.date_range("2012-06-04", periods=168, freq="h")})
    selected = period_reference(reference, window, weeks=2)
    span = pd.to_datetime(selected.pred_timestamp)
    # day of year: 2012 is a leap year, so the same day of year falls one calendar day later in 2011
    assert abs(span.min() - pd.Timestamp("2011-05-21")) <= pd.Timedelta(days=1)
    assert abs(span.max().normalize() - pd.Timestamp("2011-06-24")) <= pd.Timedelta(days=1)
    assert selected.index.is_monotonic_increasing
    # early January wraps to late December
    january = period_reference(reference, pd.Timestamp("2012-01-03"), weeks=1)
    months = set(pd.to_datetime(january.pred_timestamp).dt.month)
    assert months == {1, 12}


def test_ts04_period_reference_restores_detection_of_growth_on_seasonal_data():
    # hourly rentals: daily cycle, strong season, +40 rides/h in 2012 (smaller than the seasonal swing)
    days = pd.date_range("2011-01-01", "2012-12-31 23:00", freq="h")
    rng = np.random.default_rng(5)
    season = 100 + 80 * np.sin(2 * np.pi * (days.day_of_year.to_numpy() - 100) / 365)
    daily = 40 * np.sin(2 * np.pi * days.hour.to_numpy() / 24)
    growth = np.where(days.year == 2012, 40.0, 0.0)
    data = tagged(pd.DataFrame({"cnt": season + daily + growth + rng.normal(0, 20, len(days))}))
    data["pred_timestamp"] = days
    reference, production = data[days.year == 2011], data[days.year == 2012]
    weeks = [production.iloc[i : i + 168] for i in range(0, len(production) - 168, 168 * 2)]
    analyzer = Analyzer(name="m", model_id="m1", model_version="1")
    analyzer.add_drift_metrics(metrics_list=["wasserstein"], features_list=["cnt"])

    def detected(window, ref, compare_to="reference"):
        with quiet():
            cal = AACalibrator(window_size=168, n_draws=100, allow_short=True, compare_to=compare_to)
            cal.fit(ref, analyzer._metrics_list)
            analyzer.run(current=window, reference=ref, options=cal.options())
        return bool(analyzer.results_to_pandas().query("metric_type == 'drift'").drift_status.iloc[0])

    window_null = np.mean([detected(w, reference, "window") for w in weeks])  # behaviour before the fix
    static = np.mean([detected(w, reference) for w in weeks])
    by_period = np.mean([detected(w, period_reference(reference, w, weeks=5)) for w in weeks])
    assert window_null <= 0.1
    assert by_period >= 0.5
    assert by_period > static + 0.2


# ---------------------------------------------------------------- TC-02: level grouping
def airports(n, rng, levels=300, shift=0.0):
    """Zipf-like categorical feature with hundreds of sparse levels; `shift` moves rows onto the 5 largest"""
    weights = 1 / np.arange(1, levels + 1)
    weights /= weights.sum()
    values = rng.choice(levels, size=n, p=weights)
    moved = rng.uniform(size=n) < shift
    values[moved] = rng.integers(0, 5, moved.sum())
    return pd.Series([f"A{v:03d}" for v in values])


def test_tc02_grouping_is_chosen_on_the_reference_and_applied_alike():
    rng = np.random.default_rng(6)
    ref, new = airports(20000, rng), airports(1000, rng)
    counts = get_population_counts(new, ref, min_expected=5)
    shares = ref.value_counts(normalize=True)
    kept = [level for level in counts.index if level != OTHER_LEVEL]
    assert all(len(new) * shares[level] >= 5 for level in kept)
    assert OTHER_LEVEL in counts.index
    assert counts["new"].sum() == len(new) and counts["ref"].sum() == len(ref)  # no row lost
    assert len(get_population_counts(new, ref, max_levels=10)) == 10


def test_tc02_grouping_restores_chi2_power_and_reports_sparse_cells():
    rng = np.random.default_rng(7)
    reference = pd.DataFrame({"airport": airports(40000, rng)})
    raw = DriftTestMetric("chi2", "airport")
    grouped = DriftTestMetric("chi2", "airport", min_expected=5)
    power_raw, power_grouped = [], []
    for _ in range(40):
        current = pd.DataFrame({"airport": airports(2000, rng, shift=0.05)})
        with quiet():
            r = raw.evaluate(current, reference)
        g = grouped.evaluate(current, reference)
        power_raw.append(r.drift_status)
        power_grouped.append(g.drift_status)
    assert r.low_expected_share > 0.2  # reported on the result
    assert g.low_expected_share < r.low_expected_share
    assert np.mean(power_grouped) > np.mean(power_raw) + 0.2


def test_tc02_sparse_levels_warn_and_grouping_reaches_the_calibration():
    rng = np.random.default_rng(8)
    reference = pd.DataFrame({"airport": airports(20000, rng)})
    current = pd.DataFrame({"airport": airports(1000, rng)})
    with pytest.warns(UserWarning, match="min_expected=5"):
        DriftTestMetric("chi2", "airport").evaluate(current, reference)
    analyzer = Analyzer(name="m", model_id="m1", model_version="1")
    analyzer.add_drift_metrics(metrics_list=["chi2", "psi", "ks_2samp"], features_list=["airport"], min_expected=5)
    chi2 = analyzer._metrics_list[0]
    assert chi2._grouping == {"min_expected": 5}
    with quiet():
        cal = AACalibrator(window_size=1000, n_draws=50, fdr_alpha=None).fit(reference, analyzer._metrics_list[:2])
    a, b = slice(0, 1000), slice(1000, 20000)
    column = reference.airport
    expected = chi2_test(column.iloc[a], column.iloc[b], min_expected=5).pvalue
    assert chi2._value(column.iloc[a], column.iloc[b]) == pytest.approx(expected)
    assert cal.nulls[("chi2", "airport")].status == "ok"
