import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy, verdict


def decisions(rates: dict, n: dict, seed=0, tpr=None, score_bias=None):
    """Synthetic decisions with known selection rates per group (and optionally labels and scores)"""
    rng = np.random.default_rng(seed)
    frames = []
    for group, rate in rates.items():
        size = n[group]
        y_true = rng.integers(0, 2, size)
        if tpr:
            # decision = 1 with probability tpr[group] when y_true = 1, 0.2 otherwise
            y_pred = np.where(y_true == 1, rng.random(size) < tpr[group], rng.random(size) < 0.2).astype(int)
        else:
            y_pred = (rng.random(size) < rate).astype(int)
        score = np.clip(y_true * 0.7 + 0.15 + (score_bias or {}).get(group, 0.0) + rng.normal(0, 0.05, size), 0, 1)
        frames.append(
            pd.DataFrame(
                {
                    "id": [f"{group}-{i}" for i in range(size)],
                    "group": group,
                    "y_pred": y_pred,
                    "y_true": y_true,
                    "y_pred_proba": score,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def result(analyzer, metric, group):
    return next(r for r in analyzer.get_result() if r.metric_name == metric and r.group == group)


def test_disparate_impact_is_flagged_when_the_interval_excludes_the_four_fifths_rule():
    data = decisions({"A": 0.5, "B": 0.3}, {"A": 2000, "B": 2000})
    analyzer = FairnessAnalyzer(["group"])
    analyzer.run(data)
    r = result(analyzer, "disparate_impact_ratio", "group=B")
    assert r.reference_group == "group=A" or r.reference_group == "group=B"
    assert r.metric_value == pytest.approx(0.6, abs=0.06)
    assert r.ci_high < 0.8 and r.drift_status is True
    assert r.metric_type == "fairness" and r.n == 2000 and r.threshold == [0.8, 1.25]
    d = result(analyzer, "demographic_parity_difference", "group=B")
    assert d.metric_value == pytest.approx(-0.2, abs=0.04) and d.drift_status is True


def test_no_disparity_is_confirmed_with_large_groups_and_inconclusive_with_small_ones():
    big = FairnessAnalyzer(["group"])
    big.run(decisions({"A": 0.4, "B": 0.4}, {"A": 5000, "B": 5000}, seed=1))
    assert result(big, "disparate_impact_ratio", "group=B").drift_status is False

    small = FairnessAnalyzer(["group"])
    small.run(decisions({"A": 0.4, "B": 0.4}, {"A": 60, "B": 40}, seed=1))
    r = result(small, "disparate_impact_ratio", "group=B")
    assert r.drift_status is None  # too few rows to conclude either way
    assert r.ci_low < 0.8 < r.ci_high or r.ci_low < 1.25 < r.ci_high


def test_small_groups_are_suppressed_without_value_or_count():
    analyzer = FairnessAnalyzer(["group"], FairnessPolicy(min_group_size=30))
    analyzer.run(decisions({"A": 0.5, "B": 0.5, "C": 0.1}, {"A": 500, "B": 500, "C": 12}))
    r = result(analyzer, "disparate_impact_ratio", "group=C")
    assert r.suppressed is True and r.metric_value is None and r.n is None and r.drift_status is None


def test_reference_group_defaults_to_the_largest_and_can_be_chosen():
    data = decisions({"A": 0.5, "B": 0.3}, {"A": 300, "B": 900})
    default = FairnessAnalyzer(["group"])
    default.run(data)
    assert {r.reference_group for r in default.get_result()} == {"group=B"}
    chosen = FairnessAnalyzer(["group"], reference_groups={"group": "A"})
    chosen.run(data)
    rates = data.groupby("group")["y_pred"].mean()
    assert result(chosen, "disparate_impact_ratio", "group=B").metric_value == pytest.approx(rates["B"] / rates["A"])
    with pytest.raises(error_msg, match="reference group"):
        FairnessAnalyzer(["group"], reference_groups={"group": "Z"}).run(data)


def test_label_based_criteria_must_be_requested_and_need_labels():
    data = decisions({"A": 0, "B": 0}, {"A": 3000, "B": 3000}, tpr={"A": 0.8, "B": 0.6}, seed=2)
    default = FairnessAnalyzer(["group"])
    default.run(data)
    assert {r.metric_name for r in default.get_result()} == {"disparate_impact_ratio", "demographic_parity_difference"}

    policy = FairnessPolicy(criteria=["equal_opportunity_difference", "equalized_odds_difference"])
    labelled = FairnessAnalyzer(["group"], policy, reference_groups={"group": "A"})
    labelled.run(data)
    eo = result(labelled, "equal_opportunity_difference", "group=B")
    assert eo.metric_value == pytest.approx(-0.2, abs=0.04) and eo.drift_status is True
    assert result(labelled, "equalized_odds_difference", "group=B").metric_value == pytest.approx(0.2, abs=0.04)

    unlabelled = FairnessAnalyzer(["group"], policy)
    unlabelled.run(data.drop(columns=["y_true"]))
    assert all(r.error and "labels" in r.error for r in unlabelled.get_result())


def test_calibration_difference_detects_a_miscalibrated_group():
    rng = np.random.default_rng(3)
    frames = []
    for group, shift in {"A": 0.0, "B": 0.2}.items():
        p = rng.uniform(0, 0.8, 3000)  # true probability; A's scores are calibrated, B's are 0.2 too high
        y_true = (rng.random(3000) < p).astype(int)
        frames.append(
            pd.DataFrame({"group": group, "y_true": y_true, "y_pred_proba": p + shift, "y_pred": (p + shift > 0.5).astype(int)})
        )
    analyzer = FairnessAnalyzer(["group"], FairnessPolicy(criteria=["calibration_difference"]), reference_groups={"group": "A"})
    analyzer.run(pd.concat(frames, ignore_index=True))
    r = result(analyzer, "calibration_difference", "group=B")
    assert r.metric_value == pytest.approx(0.2, abs=0.04) and r.drift_status is True


def test_intersections_and_join_on_pseudonymous_id():
    data = decisions({"A": 0.5, "B": 0.3}, {"A": 2000, "B": 2000}, seed=4)
    rng = np.random.default_rng(4)
    attributes = pd.DataFrame({"id": data["id"], "sex": data["group"], "region": rng.choice(["North", "South"], len(data))})
    predictions = data[["id", "y_pred", "y_true"]]  # protected attributes do not travel with predictions
    analyzer = FairnessAnalyzer(["sex", "region"], intersections=True)
    analyzer.run(predictions, attributes_data=attributes, on="id")
    features = {r.feature_name for r in analyzer.get_result()}
    assert features == {"sex", "region", "sex|region"}
    groups = {r.group for r in analyzer.get_result() if r.feature_name == "sex|region"}
    assert any(g.startswith("sex=B|region=") for g in groups)
    with pytest.raises(error_msg, match="`on`"):
        analyzer.run(predictions, attributes_data=attributes)


def test_simultaneous_band_is_wider_than_per_group_intervals():
    data = decisions({"A": 0.5, "B": 0.45, "C": 0.4, "D": 0.35}, {g: 800 for g in "ABCD"}, seed=5)
    per_group = FairnessAnalyzer(["group"], FairnessPolicy(simultaneous=False, n_bootstrap=400), reference_groups={"group": "A"})
    band = FairnessAnalyzer(["group"], FairnessPolicy(simultaneous=True, n_bootstrap=400), reference_groups={"group": "A"})
    per_group.run(data)
    band.run(data)
    for g in ("group=B", "group=C", "group=D"):
        p, s = result(per_group, "disparate_impact_ratio", g), result(band, "disparate_impact_ratio", g)
        assert (s.ci_high - s.ci_low) > (p.ci_high - p.ci_low)
    assert "simultaneous" in result(band, "disparate_impact_ratio", "group=B").method


def test_intervals_cover_the_true_disparity():
    covered = 0
    sims = 60
    for seed in range(sims):
        data = decisions({"A": 0.5, "B": 0.4}, {"A": 400, "B": 400}, seed=100 + seed)
        analyzer = FairnessAnalyzer(["group"], FairnessPolicy(n_bootstrap=200), reference_groups={"group": "A"}, seed=seed)
        analyzer.run(data)
        r = result(analyzer, "disparate_impact_ratio", "group=B")
        covered += r.ci_low <= 0.8 <= r.ci_high
    assert covered / sims >= 0.85


@pytest.mark.parametrize(
    "low, high, value, expected",
    [
        (0.6, 0.75, 0.7, True),
        (0.85, 1.1, 1.0, False),
        (0.7, 0.9, 0.8, None),
        (None, None, 0.5, True),
        (None, None, float("nan"), None),
    ],
)
def test_verdict(low, high, value, expected):
    assert verdict(low, high, value, (0.8, 1.25)) is expected


def test_policy_validation():
    with pytest.raises(error_msg, match="unknown fairness criteria"):
        FairnessPolicy(criteria=["accuracy"])
    with pytest.raises(error_msg):
        FairnessAnalyzer([])
