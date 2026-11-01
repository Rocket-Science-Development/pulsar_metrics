import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, entropy
from sklearn.metrics import mean_absolute_percentage_error

from pulsar_metrics.metrics.drift import DriftMetric, DriftTestMetric
from pulsar_metrics.metrics.performance import PerformanceMetric
from pulsar_metrics.metrics.utils import (
    chi2_test,
    get_population_counts,
    get_population_percentages,
    kl_divergence,
    population_stability_index,
)

TOOLS_REFERENCE = pd.Series(["search"] * 50 + ["calculator"] * 30 + ["calendar"] * 20)


# Counts
# ==============================


def test_population_counts_include_categories_missing_from_one_sample():
    new = pd.Series(["search", "search", "email"])
    counts = get_population_counts(new, TOOLS_REFERENCE)
    assert counts.loc["email", "ref"] == 0 and counts.loc["email", "new"] == 1
    assert counts.loc["calendar", "new"] == 0 and counts.loc["calendar", "ref"] == 20
    assert counts["ref"].sum() == len(TOOLS_REFERENCE) and counts["new"].sum() == len(new)


def test_population_counts_bin_numeric_series_on_shared_bins():
    rng = np.random.default_rng(0)
    reference, new = pd.Series(rng.normal(0, 1, 500)), pd.Series(rng.normal(1, 1, 300))
    counts = get_population_counts(new, reference)
    assert counts["ref"].sum() == 500 and counts["new"].sum() == 300


def test_binned_percentages_keep_labels():
    reference = pd.Series([90, 10], index=["a", "b"])
    new = pd.Series([10, 90], index=["a", "b"])
    percents = get_population_percentages(new, reference, binned=True)
    assert percents.loc["a", "ref"] == pytest.approx(0.9)
    assert percents.loc["a", "new"] == pytest.approx(0.1)


# KL divergence
# ==============================


def test_kl_is_zero_for_identical_distributions():
    assert kl_divergence(TOOLS_REFERENCE.copy(), TOOLS_REFERENCE) == pytest.approx(0.0, abs=1e-12)


def test_kl_matches_scipy_entropy():
    new = pd.Series(["search"] * 20 + ["calculator"] * 40 + ["calendar"] * 40)
    expected = entropy([0.2, 0.4, 0.4], [0.5, 0.3, 0.2])
    assert kl_divergence(new, TOOLS_REFERENCE, epsilon=0) == pytest.approx(expected, rel=1e-12)


def test_kl_is_finite_when_a_category_is_new():
    new = pd.Series(["search"] * 50 + ["email"] * 50)
    value = kl_divergence(new, TOOLS_REFERENCE)
    assert np.isfinite(value) and value > 1


def test_kl_returns_a_scalar_through_drift_metric():
    metric = DriftMetric(metric_name="kl", feature_name="tool")
    result = metric.evaluate(
        current=pd.DataFrame({"tool": ["search"] * 90 + ["calendar"] * 10}),
        reference=pd.DataFrame({"tool": TOOLS_REFERENCE}),
        threshold=0.1,
    )
    assert isinstance(result.metric_value, float)
    assert result.metric_value > 0.1


# Population stability index
# ==============================


def test_psi_matches_the_classic_formula_without_empty_bins():
    new = pd.Series(["search"] * 20 + ["calculator"] * 40 + ["calendar"] * 40)
    p, q = np.array([0.2, 0.4, 0.4]), np.array([0.5, 0.3, 0.2])
    order = ["search", "calculator", "calendar"]
    counts = get_population_counts(new, TOOLS_REFERENCE).loc[order]
    assert list(counts["new"]) == [20, 40, 40]
    assert population_stability_index(new, TOOLS_REFERENCE, epsilon=0) == pytest.approx(
        ((p - q) * np.log(p / q)).sum(), rel=1e-12
    )


def test_psi_is_finite_when_a_bin_is_empty_in_the_reference():
    new = pd.Series(["search"] * 50 + ["email"] * 50)
    value = population_stability_index(new, TOOLS_REFERENCE)
    assert np.isfinite(value) and value > 0.2


def test_psi_is_zero_for_identical_distributions():
    assert population_stability_index(TOOLS_REFERENCE.copy(), TOOLS_REFERENCE) == pytest.approx(0.0, abs=1e-12)


def test_psi_accepts_pre_binned_counts():
    reference = pd.Series([50, 30, 20], index=["a", "b", "c"])
    new = pd.Series([20, 40, 40], index=["a", "b", "c"])
    raw_new = pd.Series(["a"] * 20 + ["b"] * 40 + ["c"] * 40)
    raw_reference = pd.Series(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    assert population_stability_index(new, reference, binned=True) == pytest.approx(
        population_stability_index(raw_new, raw_reference), rel=1e-12
    )


# Chi-squared test
# ==============================


def test_chi2_matches_scipy_on_contingency_table():
    new = pd.Series(["search"] * 10 + ["calculator"] * 30 + ["calendar"] * 60)
    expected = chi2_contingency([[50, 30, 20], [10, 30, 60]])
    result = chi2_test(new, TOOLS_REFERENCE)
    order = list(get_population_counts(new, TOOLS_REFERENCE).index)
    assert sorted(order) == ["calculator", "calendar", "search"]
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-12)
    assert result.pvalue == pytest.approx(expected.pvalue, rel=1e-12)


def test_chi2_accepts_samples_of_different_sizes():
    new = pd.Series(["search"] * 5 + ["calculator"] * 3 + ["calendar"] * 2)
    assert chi2_test(new, TOOLS_REFERENCE).pvalue == pytest.approx(1.0)


@pytest.mark.parametrize(
    "current_tools, expected_drift",
    [
        (["search"] * 500 + ["calculator"] * 300 + ["calendar"] * 200, False),
        (["search"] * 100 + ["calculator"] * 100 + ["calendar"] * 300 + ["email"] * 500, True),
    ],
)
def test_chi2_detects_a_change_in_tool_distribution(current_tools, expected_drift):
    metric = DriftTestMetric(metric_name="chi2", feature_name="tool")
    result = metric.evaluate(
        current=pd.DataFrame({"tool": current_tools}),
        reference=pd.DataFrame({"tool": TOOLS_REFERENCE}),
        alpha=0.05,
    )
    assert result.drift_status is expected_drift


# MAPE
# ==============================


def test_mape_is_the_mean_absolute_percentage_error():
    data = pd.DataFrame({"y_true": [100.0, 200.0, 50.0], "y_pred": [110.0, 180.0, 50.0]})
    metric = PerformanceMetric(metric_name="mape")
    result = metric.evaluate(current=data, threshold=0.5)
    assert result.metric_value == pytest.approx(mean_absolute_percentage_error(data["y_true"], data["y_pred"]))
    assert result.metric_value == pytest.approx((0.1 + 0.1 + 0.0) / 3)
