import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.fairness import BettingTest, FairnessMonitor


def run_pairs(rate_a, rate_b, n, tolerance=0.0, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    test = BettingTest(alpha, tolerance)
    for a, b in zip(rng.random(n) < rate_a, rng.random(n) < rate_b):
        if test.update(float(a), float(b)):
            break
    return test


def stream(rates: dict, n_batches: int, batch_size: int, seed=0, shift_at=None, shifted=None):
    """Batches of decisions; from batch `shift_at` on, the rates of `shifted` replace those of `rates`"""
    rng = np.random.default_rng(seed)
    groups = list(rates)
    for k in range(n_batches):
        current = shifted if shift_at is not None and k >= shift_at else rates
        group = rng.choice(groups, batch_size)
        rate = np.array([current[g] for g in group])
        yield pd.DataFrame({"group": group, "y_pred": (rng.random(batch_size) < rate).astype(int)})


def test_false_alarms_held_under_null():
    # equal rates, looked at after every pair: the false alarm probability stays at most alpha
    alarms = np.mean([run_pairs(0.4, 0.4, 2000, seed=s).rejected for s in range(200)])
    assert alarms <= 0.08


def test_detects_a_disparity():
    test = run_pairs(0.5, 0.35, 3000, seed=1)
    assert test.rejected and test.rejected_at < 1500
    assert test.e_value >= 20


def test_tolerance_absorbs_small_disparities():
    alarms = np.mean([run_pairs(0.4, 0.37, 2000, tolerance=0.05, seed=100 + s).rejected for s in range(100)])
    assert alarms <= 0.05


def test_rejection_is_final():
    test = run_pairs(0.6, 0.3, 3000, seed=2)
    at = test.rejected_at
    for _ in range(200):
        test.update(0.0, 1.0)
    assert test.rejected_at == at


def test_invalid_input():
    with pytest.raises(error_msg):
        BettingTest(alpha=1.5)
    with pytest.raises(error_msg):
        BettingTest().update(1.2, 0.0)


def test_monitor_flags_only_the_group_that_drifts():
    rates = {"A": 0.5, "B": 0.5, "C": 0.5}
    monitor = FairnessMonitor("group", "A", ["B", "C"], tolerance=0.05)
    for batch in stream(rates, 60, 200, seed=3, shift_at=20, shifted={"A": 0.5, "B": 0.5, "C": 0.3}):
        results = monitor.update(batch)
    by_group = {r.group: r for r in results}
    assert by_group["group=C"].drift_status is True
    assert by_group["group=B"].drift_status is False
    assert by_group["group=C"].test_type == "sequential" and by_group["group=C"].e_value >= 40
    assert by_group["group=C"].reference_group == "group=A"


def test_monitor_quiet_on_a_fair_stream():
    monitor = FairnessMonitor("group", "A", ["A", "B"])  # the reference group is dropped from the monitored groups
    assert monitor.groups == ["B"]
    for batch in stream({"A": 0.4, "B": 0.4}, 50, 100, seed=4):
        results = monitor.update(batch)
    assert results[0].drift_status is False and results[0].n > 1000


def test_monitor_equal_opportunity_uses_positives_only():
    rng = np.random.default_rng(5)
    n = 6000
    group = rng.choice(["A", "B"], n)
    y_true = rng.integers(0, 2, n)
    # same selection rate overall, but B is approved less often when the outcome is positive
    tpr = np.where(group == "A", 0.8, 0.55)
    fpr = np.where(group == "A", 0.2, 0.45)
    y_pred = (rng.random(n) < np.where(y_true == 1, tpr, fpr)).astype(int)
    data = pd.DataFrame({"group": group, "y_true": y_true, "y_pred": y_pred})
    parity = FairnessMonitor("group", "A", ["B"]).update(data)[0]
    opportunity = FairnessMonitor("group", "A", ["B"], criterion="equal_opportunity").update(data)[0]
    assert parity.drift_status is False
    assert opportunity.drift_status is True
    with pytest.raises(error_msg):
        FairnessMonitor("group", "A", ["B"], criterion="equal_opportunity").update(data.drop(columns="y_true"))


def test_monitor_bounds_unpaired_observations():
    monitor = FairnessMonitor("group", "A", ["B"], max_pending=100)
    monitor.update(pd.DataFrame({"group": ["A"] * 1000, "y_pred": [1] * 1000}))
    assert len(monitor._queues["B"][1]) == 100
