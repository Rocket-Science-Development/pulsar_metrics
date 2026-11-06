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


# ----------------------------------------------------------------------------------------------- small_groups="merge"
def solvable(data, analyzer, attrs):
    """Hidden quantities a reader can compute exactly from what is published (linear system over the finest cells)

    Unknowns: the size of every cell of the original attributes crossed (the finest cells). Known, conservatively:
    the total of rows, every published group size (any feature, any depth, merged labels mapped back to their
    members) and every published coverage n (rows outside the reference group). A hidden quantity, the size of a
    member of a merge (alone or crossed with any value of another attribute) or of a suppressed cell, is computable
    iff its indicator row lies in the row space of the known equations. The success counts behind the published rates
    obey the same equations (one per published group, reference rate assumed known), so the same check covers them."""
    fine = sorted({tuple(str(v) for v in row) for row in data[attrs].itertuples(index=False)})
    to_published = {a: {} for a in attrs}
    for a, groups in analyzer.merged_groups.items():
        for label, members in groups.items():
            for m in members:
                to_published[a][m] = label

    def matches(cell, wanted):
        return all(to_published[a].get(cell[a], cell[a]) == v for a, v in wanted.items())

    def indicator(condition):
        return np.array([1.0 if condition(dict(zip(attrs, cell))) else 0.0 for cell in fine])

    def parse(label):
        return dict(part.split("=", 1) for part in label.split("|"))

    known = [np.ones(len(fine))]
    targets = {}
    for r in analyzer.get_result():
        if r.group is None:
            continue
        row = indicator(lambda c, w=parse(r.group): matches(c, w))
        if r.suppressed:
            targets[f"suppressed {r.group}"] = row
        elif r.n is not None:
            known.append(row)
    for r in analyzer.coverage():
        if r.n is not None:
            known.append(indicator(lambda c, w=parse(r.reference_group): not matches(c, w)))
    for a, groups in analyzer.merged_groups.items():
        for members in groups.values():
            for m in members:
                targets[f"member {a}={m}"] = indicator(lambda c, a=a, m=m: c[a] == m)
                for b in attrs:
                    for w in sorted(data[b].astype(str).unique()) if b != a else []:
                        targets[f"member {a}={m}|{b}={w}"] = indicator(lambda c, a=a, m=m, b=b, w=w: c[a] == m and c[b] == w)
    A = np.vstack(known)
    rank = np.linalg.matrix_rank(A)
    return [name for name, t in targets.items() if t.any() and np.linalg.matrix_rank(np.vstack([A, t])) == rank]


GENDER = {"Female": 620, "Male": 410, "Unknown/Invalid": 7}
MERGED = "Male+Unknown/Invalid"


def gender_frame(seed=0, with_age=False):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"sex": np.concatenate([[g] * n for g, n in GENDER.items()])})
    if with_age:
        data["age"] = rng.choice(["[0-40)", "[40-70)", "[70+)"], len(data), p=[0.3, 0.4, 0.3])
    data["y_pred"] = (rng.random(len(data)) < 0.4).astype(int)
    return data


def test_merge_publishes_a_result_where_suppression_publishes_none():
    data = gender_frame()
    suppress = FairnessAnalyzer(["sex"], FairnessPolicy(**POLICY))
    suppress.run(data)
    assert [r for r in suppress.get_result() if not r.suppressed] == []  # Male goes with Unknown/Invalid: nothing left
    merge = FairnessAnalyzer(["sex"], FairnessPolicy(small_groups="merge", **POLICY))
    merge.run(data)
    groups, coverage = published(merge)
    assert groups["sex"] == {f"sex={MERGED}": 417}
    assert merge.merged_groups == {"sex": {MERGED: ["Male", "Unknown/Invalid"]}}
    (result,) = merge.get_result()
    assert result.reference_group == "sex=Female" and result.metric_value is not None
    assert f"'{MERGED}' merges 'Male' + 'Unknown/Invalid'" in result.method
    expected = data.loc[data["sex"] != "Female", "y_pred"].mean() - data.loc[data["sex"] == "Female", "y_pred"].mean()
    assert result.metric_value == pytest.approx(expected)
    assert coverage["sex"] == 417
    assert solvable(data, merge, ["sex"]) == []  # 0 solvable equation
    assert solvable(data, suppress, ["sex"]) == []


def test_merge_is_the_same_in_every_intersection_and_discloses_nothing():
    data = gender_frame(seed=1, with_age=True)
    analyzer = FairnessAnalyzer(["sex", "age"], FairnessPolicy(small_groups="merge", **POLICY), intersections=True)
    analyzer.run(data)
    groups, _ = published(analyzer)
    sex_values = {part for feature in groups.values() for g in feature for part in g.split("|") if part.startswith("sex=")}
    assert sex_values <= {f"sex={MERGED}", "sex=Female"}  # the members of the merge are never published apart
    assert any(g.startswith(f"sex={MERGED}|age=") and n for g, n in groups["sex|age"].items())
    every = analyzer.get_result()
    assert all("merges 'Male' + 'Unknown/Invalid'" in r.method for r in every if r.feature_name.startswith("sex"))
    assert not any("merges" in r.method for r in every if r.feature_name == "age")
    assert solvable(data, analyzer, ["sex", "age"]) == []
    recoded = data.assign(sex=data["sex"].replace({"Male": MERGED, "Unknown/Invalid": MERGED}))
    assert_not_recoverable(recoded, analyzer, ["sex", "age"])


def test_the_verifier_finds_what_suppression_without_complement_discloses():
    data = frame({"A": 600, "B": 300, "C": 200, "D": 12})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(complementary_suppression=False, **POLICY))
    analyzer.run(data)
    assert solvable(data, analyzer, ["sex"]) == ["suppressed sex=D"]


def test_merge_never_absorbs_the_reference():
    data = frame({"A": 600, "B": 12})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(small_groups="merge", **POLICY))
    analyzer.run(data)
    assert analyzer.merged_groups == {}
    assert [r.suppressed for r in analyzer.get_result()] == [True]
    assert analyzer.coverage()[0].n is None


def test_merge_chains_small_groups_until_the_size_is_reached():
    # C and D together stay under 30: the merge takes the next smallest non-reference group (B), never A
    data = frame({"A": 500, "B": 480, "C": 10, "D": 8})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(small_groups="merge", **POLICY))
    analyzer.run(data)
    assert analyzer.merged_groups == {"sex": {"B+C+D": ["B", "C", "D"]}}
    (result,) = analyzer.get_result()
    assert result.group == "sex=B+C+D" and result.reference_group == "sex=A" and result.n == 498
    assert solvable(data, analyzer, ["sex"]) == []


def test_merge_keeps_the_reference_chosen_before_merging():
    data = frame({"A": 500, "B": 495, "C": 12})
    analyzer = FairnessAnalyzer(["sex"], FairnessPolicy(small_groups="merge", **POLICY))
    analyzer.run(data)
    (result,) = analyzer.get_result()
    assert result.reference_group == "sex=A" and result.group == "sex=B+C"  # B+C (507) is larger, A stays the reference


def test_small_groups_is_validated():
    with pytest.raises(Exception, match="small_groups"):
        FairnessPolicy(small_groups="round")
