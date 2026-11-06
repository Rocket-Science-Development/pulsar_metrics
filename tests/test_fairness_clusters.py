"""Several rows of one person in FairnessAnalyzer (lab FA-02): id_column warning, and score intervals of small groups
corrected for the clustering (effective counts). Synthetic patients and stays only."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy
from pulsar_metrics.fairness.planning import rate_interval

POLICY = dict(criteria=["demographic_parity_difference"], n_bootstrap=200, simultaneous=False)


def stays(rho, small=40, large=400, per_patient=4, seed=0):
    """Patients of two age bands, `per_patient` stays each; decisions share a patient propensity of intra-patient
    correlation `rho` (beta-binomial, mean 0.4)"""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for band, n in (("[0-10)", small), ("[10+)", large)):
        for _ in range(n):
            if rho == 0:
                p = 0.4
            else:
                total = 1 / rho - 1
                p = rng.beta(0.4 * total, 0.6 * total)
            for _ in range(per_patient):
                rows.append((f"p{pid}", band, int(rng.random() < p)))
            pid += 1
    return pd.DataFrame(rows, columns=["patient", "age", "y_pred"])


def small_group(data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = FairnessAnalyzer(["age"], FairnessPolicy(**POLICY), reference_groups={"age": "[10+)"}, **kwargs).run(data)
    return next(r for r in results if r.group == "age=[0-10)")


def test_repeated_ids_without_cluster_warn_and_are_noted_in_method():
    data = stays(0.5)
    with pytest.warns(UserWarning, match="patient appear on several rows"):
        result = FairnessAnalyzer(["age"], FairnessPolicy(**POLICY), id_column="patient").run(data)
    assert all("patient values repeat over 1760 rows, treated as independent rows (no cluster=)" in r.method for r in result)


def test_no_id_warning_with_cluster_or_unique_ids():
    data = stays(0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = FairnessAnalyzer(["age"], FairnessPolicy(**POLICY), id_column="patient", cluster="patient").run(data)
        FairnessAnalyzer(["age"], FairnessPolicy(**POLICY), id_column="patient").run(data.drop_duplicates("patient"))
    assert not any("appear on several rows" in str(w.message) for w in caught)
    assert not any("treated as independent rows" in r.method for r in result)
    with pytest.raises(error_msg, match="missing columns"):
        FairnessAnalyzer(["age"], FairnessPolicy(**POLICY), id_column="stay_id").run(data)


def test_without_id_column_nothing_is_guessed():
    data = stays(0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FairnessAnalyzer(["age"], FairnessPolicy(**POLICY)).run(data)
    assert not any("appear on several rows" in str(w.message) for w in caught)


def test_small_groups_keep_score_intervals_with_cluster():
    """FA-02: with cluster=, small groups used to switch to the cluster bootstrap, less prudent than Wilson"""
    result = small_group(stays(0.3), cluster="patient")
    assert "Wilson score interval (Newcombe)" in result.method and "effective counts of patient units" in result.method
    assert "bootstrap" not in result.method
    assert result.n == 40  # n counts patients


def test_one_row_per_cluster_gives_todays_interval_exactly():
    data = stays(0.3, per_patient=1)
    without = small_group(data)
    with_cluster = small_group(data, cluster="patient")
    assert "design effect 1.00" in with_cluster.method
    assert with_cluster.ci_low == pytest.approx(without.ci_low, abs=1e-12)
    assert with_cluster.ci_high == pytest.approx(without.ci_high, abs=1e-12)
    small, ref = data[data.age == "[0-10)"].y_pred, data[data.age == "[10+)"].y_pred
    expected = rate_interval(small.sum(), len(small), ref.sum(), len(ref), "difference", 0.95)
    assert (with_cluster.ci_low, with_cluster.ci_high) == pytest.approx(expected)


def test_intervals_widen_with_the_intra_cluster_correlation():
    widths = {}
    for rho in (0.0, 0.3, 0.8):
        r = small_group(stays(rho, seed=3), cluster="patient")
        widths[rho] = r.ci_high - r.ci_low
    rows_as_people = small_group(stays(0.8, seed=3))
    assert widths[0.0] < widths[0.3] < widths[0.8]
    assert widths[0.8] > 1.4 * (rows_as_people.ci_high - rows_as_people.ci_low)
    # no correlation: close to the interval of independent rows (the design effect is bounded below by 1)
    independent = small_group(stays(0.0, seed=3))
    assert widths[0.0] == pytest.approx(independent.ci_high - independent.ci_low, rel=0.25)


def test_small_group_verdict_is_more_prudent_with_cluster():
    """A small group of correlated stays: rows as people give an interval about sqrt(stays) too narrow"""
    data = stays(0.8, small=40, per_patient=6, seed=5)
    rows = small_group(data)
    patients = small_group(data, cluster="patient")
    assert (patients.ci_high - patients.ci_low) > 1.8 * (rows.ci_high - rows.ci_low)
    assert patients.ci_low <= rows.ci_low and patients.ci_high >= rows.ci_high


def test_large_groups_keep_the_cluster_bootstrap():
    data = stays(0.3, small=250, large=400)
    result = small_group(data, cluster="patient")
    assert "cluster bootstrap over 650 patient units" in result.method
