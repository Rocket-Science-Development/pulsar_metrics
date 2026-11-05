"""Complementary suppression: no suppressed group size can be computed by difference from what is published (lab FA-02)"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.explain import importance_by_group
from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy
from pulsar_metrics.fairness.disclosure import complementary_suppression

POLICY = dict(n_bootstrap=0, criteria=["demographic_parity_difference"])


def frame(sizes: dict, seed=0) -> pd.DataFrame:
    """One row per unit; `sizes` maps a tuple of attribute values (or a single value) to a count"""
    rng = np.random.default_rng(seed)
    rows = []
    for key, n in sizes.items():
        key = key if isinstance(key, tuple) else (key,)
        rows += [key] * n
    data = pd.DataFrame(rows, columns=["sex", "age"][: len(rows[0])])
    data["y_pred"] = (rng.random(len(data)) < 0.4).astype(int)
    return data


def published(analyzer):
    """{feature: {group label: n or None (suppressed)}}, and the coverage n per feature"""
    groups: dict = {}
    for r in analyzer.get_result():
        groups.setdefault(r.feature_name, {})[r.group] = None if r.suppressed else r.n
    coverage = {r.feature_name: r.n for r in analyzer.coverage()}
    return groups, coverage


def assert_not_recoverable(data, analyzer, attrs):
    """For the whole feature and every margin: suppressed cells are 0, or >= 2 holding >= min_group_size rows"""
    groups, coverage = published(analyzer)
    for depth in range(1, len(attrs) + 1):
        for subset in combinations(attrs, depth):
            feature = "|".join(subset)
            if feature not in groups:
                continue
            true = data.groupby(list(subset)).size()
            hidden = {}
            for key, n in true.items():
                key = key if isinstance(key, tuple) else (key,)
                label = "|".join(f"{a}={v}" for a, v in zip(subset, key))
                if label in groups[feature] and groups[feature][label] is None:
                    hidden[key] = n
            margins = [()] + [m for d in range(1, depth) for m in combinations(range(depth), d)]
            for margin in margins:
                by: dict = {}
                for key, n in hidden.items():
                    by.setdefault(tuple(key[i] for i in margin), []).append(n)
                for cells in by.values():
                    assert len(cells) >= 2 and sum(cells) >= 30, (feature, margin, len(cells))
            if hidden and coverage.get(feature) is not None:  # coverage n - published n = total suppressed, >= 30
                assert coverage[feature] - sum(n for n in groups[feature].values() if n is not None) >= 30


def test_a_single_small_group_takes_the_smallest_other_group_with_it():
    data = frame({"A": 600, "B": 300, "C": 200, "D": 12})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(**POLICY))
    analyzer.run(data)
    groups, coverage = published(analyzer)
    assert groups["sex"] == {"sex=B": 300, "sex=C": None, "sex=D": None}
    secondary = next(r for r in analyzer.get_result() if r.group == "sex=C")
    assert "complementary suppression" in secondary.method and secondary.metric_value is None
    assert coverage["sex"] == 512 and coverage["sex"] - 300 == 212  # only the sum of C and D is disclosed
    assert_not_recoverable(data, analyzer, ["sex"])


def test_two_small_groups_above_the_threshold_together_need_nothing_more():
    data = frame({"A": 600, "B": 300, "C": 18, "D": 16})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(**POLICY))
    analyzer.run(data)
    groups, _ = published(analyzer)
    assert groups["sex"] == {"sex=B": 300, "sex=C": None, "sex=D": None}
    assert all("fewer than 30" in r.method for r in analyzer.get_result() if r.suppressed)


def test_two_small_groups_below_the_threshold_together_take_a_third():
    data = frame({"A": 600, "B": 300, "C": 100, "D": 10, "E": 8})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(**POLICY))
    analyzer.run(data)
    assert published(analyzer)[0]["sex"]["sex=C"] is None
    assert_not_recoverable(data, analyzer, ["sex"])


def test_coverage_total_is_withheld_when_only_the_reference_is_left():
    data = frame({"A": 600, "B": 12})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(**POLICY))
    analyzer.run(data)
    (cov,) = analyzer.coverage()
    assert cov.n is None and cov.metric_value == pytest.approx(1.0)


def test_margins_of_an_intersection_are_protected():
    sizes = {
        ("M", "young"): 400,
        ("M", "mid"): 380,
        ("M", "old"): 300,
        ("F", "young"): 350,
        ("F", "mid"): 200,
        ("F", "old"): 9,  # alone in the F margin and the old margin
        ("X", "young"): 40,
        ("X", "mid"): 35,
        ("X", "old"): 31,
    }
    data = frame(sizes)
    analyzer = FairnessAnalyzer(["sex", "age"], FairnessPolicy(**POLICY), intersections=True)
    analyzer.run(data)
    groups, _ = published(analyzer)
    hidden = [g for g, n in groups["sex|age"].items() if n is None]
    assert "sex=F|age=old" in hidden and len(hidden) >= 3  # one more in the F margin, one more in the old margin
    assert_not_recoverable(data, analyzer, ["sex", "age"])


def test_can_be_turned_off_for_internal_reports():
    data = frame({"A": 600, "B": 300, "C": 200, "D": 12})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(complementary_suppression=False, **POLICY))
    analyzer.run(data)
    groups, coverage = published(analyzer)
    assert groups["sex"] == {"sex=B": 300, "sex=C": 200, "sex=D": None} and coverage["sex"] == 512


def test_helper_never_suppresses_the_reference_and_reports_unsafe():
    sizes = pd.Series({"A": 600, "B": 12})
    assert complementary_suppression(sizes, ["A"], ["B"], "A", 30) == ([], False)
    sizes = pd.Series({"A": 600, "B": 40, "C": 12})
    assert complementary_suppression(sizes, ["A", "B"], ["C"], "A", 30) == (["B"], True)
    assert complementary_suppression(sizes, ["A", "B", "C"], [], "A", 30) == ([], True)


def test_importance_by_group_applies_the_same_rule():
    rng = np.random.default_rng(0)
    attributions = pd.DataFrame(rng.normal(size=(1012, 3)), columns=["x", "y", "z"])
    groups = pd.Series(["A"] * 600 + ["B"] * 300 + ["C"] * 100 + ["D"] * 12, name="language")
    results = importance_by_group(attributions, groups, n_bootstrap=20)
    suppressed = {r.group for r in results if r.suppressed}
    assert suppressed == {"language=C", "language=D"}
    assert {r.group for r in results if not r.suppressed} == {"language=B"}
