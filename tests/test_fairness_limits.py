"""Regression tests for the limits found by the night lab on pulsar_metrics.fairness (FA-01, FA-04, LB-01)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.fairness import (
    FairnessAnalyzer,
    FairnessMonitor,
    FairnessPolicy,
    counterfactual_analysis,
    equivalence_verdict,
)
from pulsar_metrics.fairness.planning import (
    flip_sets_needed,
    mover_ratio,
    paired_sets_needed,
    rate_interval,
    required_group_size,
    sequential_decisions_needed,
    wilson,
)


def two_groups(size, delta=0.0, reference=2000, rate=0.5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[rng.random(reference) < rate, rng.random(size) < rate + delta].astype(int)
    return pd.DataFrame({"language": ["fr"] * reference + ["other"] * size, "y_pred": y})


def other(results, metric="demographic_parity_difference"):
    return next(r for r in results if r.group == "language=other" and r.metric_name == metric)


# ----------------------------------------------------------------------------------------------- FA-01 M8: intersections
def five_attributes(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({a: rng.choice(list("ABCDEFGH"[:k]), n) for a, k in zip("abcde", (2, 8, 5, 6, 4))})
    frame["y_pred"] = (rng.random(n) < 0.4).astype(int)
    return frame


def test_intersections_stop_at_pairs_by_default():
    frame = five_attributes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = FairnessAnalyzer(list("abcde"), FairnessPolicy(n_bootstrap=50), intersections=True)
        default.run(frame)
        assert {r.feature_name.count("|") for r in default.get_result()} == {0, 1}  # 5 attributes + 10 pairs
        everything = FairnessAnalyzer(
            list("abcde"), FairnessPolicy(n_bootstrap=50), intersections=True, max_intersection_depth=None
        )
        everything.run(frame)
    # 2^5 - 1 attribute sets; the 5-way one has no cell of 30 rows and is skipped with a warning
    assert max(r.feature_name.count("|") for r in everything.get_result()) == 3
    with pytest.raises(error_msg):
        FairnessAnalyzer(["a"], max_intersection_depth=0)


def test_max_cells_and_coverage_report_the_rows_without_verdict():
    frame = five_attributes()
    capped = FairnessAnalyzer(list("abcde"), FairnessPolicy(n_bootstrap=50), intersections=True, max_cells=40)
    with pytest.warns(UserWarning, match="max_cells"):
        capped.run(frame)
    analyzer = FairnessAnalyzer(list("abcde"), FairnessPolicy(n_bootstrap=50), intersections=True, max_intersection_depth=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer.run(frame)
    messages = [str(w.message) for w in caught]
    assert any("suppressed cells" in m for m in messages) and any("combination skipped" in m for m in messages)
    coverage = {(r.feature_name, r.method.split(":")[0]): r for r in analyzer.coverage()}
    deep = coverage[("b|c|d|e", "demographic_parity_difference")]
    assert deep.metric_name == "no_verdict_share" and deep.metric_value > 0.9 and deep.threshold == 0.05
    assert "suppressed" in deep.method and coverage[("a", "demographic_parity_difference")].metric_value <= 1


def test_binomial_bootstrap_matches_the_row_bootstrap():
    """The exact binomial replicates replace the per-replicate Python loop: same spread of the disparity"""
    data = two_groups(800, delta=-0.1, seed=3)
    fast = FairnessAnalyzer(
        ["language"], FairnessPolicy(n_bootstrap=2000, interval="bootstrap"), reference_groups={"language": "fr"}
    )
    data["y_pred"] = data["y_pred"].astype(float) * 0.999999  # not 0/1: forces the row bootstrap
    slow = FairnessAnalyzer(
        ["language"], FairnessPolicy(n_bootstrap=2000, interval="bootstrap"), reference_groups={"language": "fr"}
    )
    f = other(fast.run(two_groups(800, delta=-0.1, seed=3)))
    s = other(slow.run(data))
    assert (f.ci_high - f.ci_low) == pytest.approx(s.ci_high - s.ci_low, rel=0.1)


# ----------------------------------------------------------------------------------------------- FA-01 m6, m7
def test_declared_groups_narrow_the_band_and_leave_the_others_exploratory():
    rng = np.random.default_rng(4)
    groups = list("ABCDEFGHIJ")
    frame = pd.DataFrame({"g": rng.choice(groups, 10000)})
    frame["y_pred"] = (rng.random(len(frame)) < 0.5).astype(int)
    policy = FairnessPolicy(criteria=["demographic_parity_difference"], interval="bootstrap", n_bootstrap=400)
    every = {r.group: r for r in FairnessAnalyzer(["g"], policy, reference_groups={"g": "A"}).run(frame)}
    declared = {
        r.group: r for r in FairnessAnalyzer(["g"], policy, reference_groups={"g": "A"}, declared_groups=["g=B"]).run(frame)
    }
    assert (declared["g=B"].ci_high - declared["g=B"].ci_low) < 0.8 * (every["g=B"].ci_high - every["g=B"].ci_low)
    assert declared["g=C"].drift_status is None and "exploratory" in declared["g=C"].method
    assert declared["g=B"].drift_status is not None or "rows of the group" in declared["g=B"].method
    with pytest.raises(error_msg):
        required_group_size(0.5, 0.5, power=1.0)


def test_inconclusive_groups_say_how_many_rows_would_settle_it():
    r = other(FairnessAnalyzer(["language"], reference_groups={"language": "fr"}).run(two_groups(60, delta=0.02, seed=5)))
    assert r.drift_status is None and "rows of the group for an 80 % chance" in r.method
    # lab FA-04: without bias, groups of 200 rows were certified 52 % of the time, groups of 500 rows 96 %
    assert 150 < required_group_size(0.5, 0.5, (-0.1, 0.1), reference_size=2000, power=0.5) < 230
    assert 250 < required_group_size(0.5, 0.5, (-0.1, 0.1), reference_size=2000) < 400
    assert required_group_size(0.6, 0.5, (-0.1, 0.1)) is None  # the disparity sits on the bound
    assert required_group_size(0.25, 0.5, (0.8, 1.25), kind="ratio") < 150


# ----------------------------------------------------------------------------------------------- FA-04 m8, m9
def test_score_intervals_keep_their_level_for_small_groups():
    """m8: the percentile bootstrap missed on one side 3 to 5 % of the time for groups of 30 to 100 rows (2.5 % nominal)"""
    rng = np.random.default_rng(1)
    for n, p in ((30, 0.2), (50, 0.5)):
        misses = np.zeros(2)
        reps = 4000
        for a, b in zip(rng.binomial(n, p, reps), rng.binomial(2000, 0.5, reps)):
            low, high = rate_interval(a, n, b, 2000, "difference")
            misses += (low > p - 0.5, high < p - 0.5)
        assert (misses / reps).max() < 0.036 and misses.sum() / reps < 0.06
    small = other(FairnessAnalyzer(["language"], reference_groups={"language": "fr"}).run(two_groups(50, seed=2)))
    assert "Wilson score interval (Newcombe)" in small.method
    ratio = other(
        FairnessAnalyzer(["language"], reference_groups={"language": "fr"}).run(two_groups(50, seed=2)), "disparate_impact_ratio"
    )
    assert "MOVER-R" in ratio.method and ratio.ci_low < ratio.metric_value < ratio.ci_high
    legacy = other(
        FairnessAnalyzer(["language"], FairnessPolicy(interval="bootstrap"), reference_groups={"language": "fr"}).run(
            two_groups(50, seed=2)
        )
    )
    assert "bootstrap" in legacy.method
    with pytest.raises(error_msg):
        FairnessPolicy(interval="exact")


def test_mover_ratio_and_wilson():
    low, high = wilson(0, 30)
    assert low == 0 and high == pytest.approx(3.84 / 33.84, abs=0.002)
    low, high = mover_ratio(0.4, wilson(400, 1000), 0.5, wilson(500, 1000))
    assert low < 0.8 < high < 0.9
    assert np.isnan(mover_ratio(0.4, (0.3, 0.5), 0.0, (0.0, 0.1))[0])


def test_equivalence_confidence_certifies_more_small_fair_groups():
    """m9: without bias, groups of 100 rows were certified within the zone in 5 % of runs"""
    default, tost = 0, 0
    for seed in range(40):
        data = two_groups(150, seed=seed)
        default += other(FairnessAnalyzer(["language"], reference_groups={"language": "fr"}).run(data)).drift_status is False
        policy = FairnessPolicy(equivalence_confidence=0.90)
        tost += other(FairnessAnalyzer(["language"], policy, reference_groups={"language": "fr"}).run(data)).drift_status is False
    assert tost > default + 4
    assert equivalence_verdict((-0.2, 0.2), (-0.08, 0.08), 0.0, (-0.1, 0.1)) is False
    assert equivalence_verdict((0.12, 0.3), (0.13, 0.28), 0.2, (-0.1, 0.1)) is True
    with pytest.raises(error_msg):
        FairnessPolicy(equivalence_confidence=1.2)


# ----------------------------------------------------------------------------------------------- FA-04 m10, LB-01 m17: probes
def probe_rows(n_sets, base=0.05, delta=0.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    u = rng.random(n_sets)
    for variant, p in (("ref", base), ("group", base + delta)):
        act = (u < p) != (rng.random(n_sets) < noise)
        rows.append(
            pd.DataFrame(
                {
                    "set_id": [f"cf-{i}" for i in range(n_sets)],
                    "variant": variant,
                    "control": "none",
                    "attribute": "name",
                    "outcome": np.where(act, "act", "other"),
                }
            )
        )
    u = rng.random(n_sets)
    for variant in ("pa", "pb"):
        act = (u < base) != (rng.random(n_sets) < noise)
        rows.append(
            pd.DataFrame(
                {
                    "set_id": [f"pl-{i}" for i in range(n_sets)],
                    "variant": variant,
                    "control": "placebo",
                    "attribute": "reference",
                    "outcome": np.where(act, "act", "other"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_small_probes_report_the_sets_needed_and_can_use_a_tost():
    """m10: at 30 sets without bias the action rate difference was 'within the zone' in only 10 % of runs"""
    default, tost = 0, 0
    for seed in range(20):
        rows = probe_rows(30, seed=seed)
        d = next(
            r for r in counterfactual_analysis(rows, "ref", action="act", n_resamples=300) if r.metric_name.endswith("difference")
        )
        default += d.drift_status is False
        if d.drift_status is None:
            assert "probe sets" in d.method
        t = next(
            r
            for r in counterfactual_analysis(rows, "ref", action="act", n_resamples=300, equivalence_confidence=0.90)
            if r.metric_name.endswith("difference")
        )
        tost += t.drift_status is False
    assert tost > default
    # no discordant set: the Wilson bound alone needs about 35 sets, 60 for an 80 % chance (lab: 79 % at 60 sets)
    assert 45 < paired_sets_needed(0.0, 0.0, (-0.1, 0.1)) < 80
    assert paired_sets_needed(0.0, 0.04, (-0.1, 0.1)) < paired_sets_needed(0.0, 0.2, (-0.1, 0.1))
    with pytest.raises(error_msg):
        paired_sets_needed(0.1, 0.05)


def test_excess_flip_rate_planner():
    """m17: excess flip rates of 0.035 over a floor of 0.02 stayed inconclusive at 200 sets (tolerance 0.05)"""
    needed = flip_sets_needed(0.055, 0.02, tolerance=0.05)
    assert needed is not None and needed > 200
    assert flip_sets_needed(0.02, 0.02, tolerance=0.05) < flip_sets_needed(0.04, 0.02, tolerance=0.05)
    rows = probe_rows(200, noise=0.03, seed=1)
    excess = next(
        r for r in counterfactual_analysis(rows, "ref", n_resamples=200) if r.metric_name == "counterfactual_excess_flip_rate"
    )
    assert excess.drift_status is False or "probe sets" in excess.method


# ----------------------------------------------------------------------------------------------- FA-01 M9, FA-04 m11: monitor
def stream(monitor, batches, gap_of_day, share=0.2, rate=0.5, seed=0, size=500):
    rng = np.random.default_rng(seed)
    first = None
    for day in range(batches):
        in_group = rng.random(size) < share
        p = np.where(in_group, rate + gap_of_day(day), rate)
        frame = pd.DataFrame({"g": np.where(in_group, "B", "A"), "y_pred": (rng.random(size) < p).astype(int)})
        result = monitor.update(frame)[0]
        if result.drift_status and first is None:
            first = day
    return first, result


def test_change_mode_ignores_a_stable_gap_and_catches_its_degradation():
    """M9: 23 of 23 stable gaps beyond the tolerance alarmed from the first year and stayed alarmed"""
    level = FairnessMonitor("g", "A", ["B"], tolerance=0.05)
    first, _ = stream(level, 40, lambda d: -0.13)
    assert first is not None  # level mode: the old, stable gap alarms
    history = two_groups(0)  # build a baseline frame with the same stable gap
    rng = np.random.default_rng(9)
    baseline_frame = pd.DataFrame({"g": rng.choice(["A", "B"], 8000, p=[0.8, 0.2])})
    baseline_frame["y_pred"] = (rng.random(8000) < np.where(baseline_frame["g"] == "B", 0.37, 0.5)).astype(int)
    change = FairnessMonitor("g", "A", ["B"], tolerance=0.05)
    gaps = change.fit_baseline(baseline_frame)
    assert gaps["B"] == pytest.approx(-0.13, abs=0.03) and change.baseline_n["B"][0] > 1000 and len(history) == 2000
    first, result = stream(change, 40, lambda d: -0.13, seed=1)
    assert first is None and "change from the baseline gap" in result.method
    first, result = stream(change, 60, lambda d: -0.13 if d < 10 else -0.30, seed=2)
    assert first is not None and first >= 10 and result.threshold[0] == pytest.approx(gaps["B"] - 0.05)


def test_acknowledge_rearms_the_test():
    monitor = FairnessMonitor("g", "A", ["B"], tolerance=0.05, baseline={"B": 0.0})
    first, _ = stream(monitor, 40, lambda d: -0.2)
    assert first is not None
    monitor.acknowledge("B", rebaseline=-0.2)
    assert monitor.alarms and monitor.alarms[0][0] == "B" and monitor.tests["B"].rejected is False
    first, _ = stream(monitor, 20, lambda d: -0.2, seed=3)
    assert first is None  # the accepted gap no longer alarms
    with pytest.raises(error_msg):
        monitor.acknowledge("Z")
    with pytest.raises(error_msg):
        FairnessMonitor("g", "A", ["B"]).acknowledge("B", rebaseline=0.1)
    with pytest.raises(error_msg):
        FairnessMonitor("g", "A", ["B"], baseline={"C": 0.1})


def test_pooled_reference_detects_small_groups_sooner():
    """m11: 1:1 pairs drop ~95 % of the reference decisions for a group of 5 %"""
    pooled_hits, paired_hits = 0, 0
    for seed in range(20):
        pooled = FairnessMonitor("g", "A", ["B"], tolerance=0.05)
        paired = FairnessMonitor("g", "A", ["B"], tolerance=0.05, reference_pooling=False)
        pooled_hits += stream(pooled, 40, lambda d: -0.1, share=0.05, rate=0.4, seed=seed)[0] is not None
        paired_hits += stream(paired, 40, lambda d: -0.1, share=0.05, rate=0.4, seed=seed)[0] is not None
    assert pooled_hits >= paired_hits + 3
    assert "reference pooled" in pooled.results()[0].method


def test_sequential_horizon_guidance():
    pooled = sequential_decisions_needed(0.1, 0.05, group_share=0.05)
    paired = sequential_decisions_needed(0.1, 0.05, group_share=0.05, reference_pooling=False)
    assert 10_000 < pooled < paired < 40_000
    assert sequential_decisions_needed(0.04, 0.05) is None
    with pytest.raises(error_msg):
        sequential_decisions_needed(0.1, group_share=1.0)
