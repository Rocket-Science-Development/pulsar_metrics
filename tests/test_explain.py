import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.explain import (
    AttributionDrift,
    describe,
    explain_decision,
    importance,
    importance_by_group,
    ndcg_at_k,
    top_k_jaccard,
    tree_attributions,
)

FEATURES = ["income", "hours", "age", "education", "region", "tenure"]


def attributions(n, scales, seed=0):
    """Synthetic attributions: feature j has mean |attribution| proportional to scales[j]"""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, (n, len(scales))) * np.asarray(scales)
    return pd.DataFrame(values, columns=FEATURES[: len(scales)])


REFERENCE_SCALES = [1.0, 0.7, 0.5, 0.3, 0.15, 0.05]


def by_name(results, name, feature=None):
    return [r for r in results if r.metric_name == name and (feature is None or r.feature_name == feature)]


def test_ranking_measures():
    ref = pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"])
    assert ndcg_at_k(ref, ref, 3) == pytest.approx(1.0)
    swapped = pd.Series([0.3, 0.5, 0.2], index=["a", "b", "c"])
    assert 0.8 < ndcg_at_k(ref, swapped, 3) < 1.0
    assert top_k_jaccard(ref, swapped, 2) == 1.0
    assert top_k_jaccard(ref, pd.Series([0.1, 0.3, 0.6], index=["a", "b", "c"]), 1) == 0.0
    assert importance(attributions(2000, REFERENCE_SCALES)).sum() == pytest.approx(1.0)


def test_no_drift_between_windows_of_the_same_model():
    drift = AttributionDrift(k=3).fit(attributions(3000, REFERENCE_SCALES, seed=1))
    results = drift.run(attributions(3000, REFERENCE_SCALES, seed=2))
    assert drift.stability["stable"]
    assert by_name(results, "attribution_ndcg_at_k")[0].drift_status is False
    assert by_name(results, "attribution_top_k_jaccard")[0].drift_status is False
    assert all(r.drift_status is False for r in by_name(results, "importance_share_difference"))


def test_drift_when_the_model_relies_on_another_feature():
    drift = AttributionDrift(k=3).fit(attributions(3000, REFERENCE_SCALES, seed=1))
    # "region" becomes the leading feature
    results = drift.run(attributions(3000, [1.0, 0.7, 0.5, 0.3, 1.6, 0.05], seed=3))
    assert by_name(results, "attribution_ndcg_at_k")[0].drift_status is True
    region = by_name(results, "importance_share_difference", "region")[0]
    assert region.drift_status is True and region.ci_low > 0.05
    assert by_name(results, "importance_share_difference", "tenure")[0].drift_status is False
    table = drift.results_to_pandas()
    assert set(table["metric_type"]) == {"explainability"}


def test_no_verdict_when_explanations_are_unstable():
    # nearly equal importances and few rows: the ranking itself is noise
    drift = AttributionDrift(k=3).fit(attributions(40, [1.0, 0.98, 0.96, 0.95, 0.94, 0.93], seed=4))
    results = drift.run(attributions(40, [1.0, 0.98, 0.96, 0.95, 0.94, 0.93], seed=5))
    assert not drift.stability["stable"]
    ndcg = by_name(results, "attribution_ndcg_at_k")[0]
    assert ndcg.drift_status is None and "not stable" in ndcg.method


def test_drift_input_checks():
    drift = AttributionDrift()
    with pytest.raises(error_msg):
        drift.run(attributions(100, REFERENCE_SCALES))
    drift.fit(attributions(100, REFERENCE_SCALES))
    with pytest.raises(error_msg):
        drift.run(attributions(100, REFERENCE_SCALES[:4]))
    bad = attributions(100, REFERENCE_SCALES)
    bad.iloc[0, 0] = np.nan
    with pytest.raises(error_msg):
        drift.run(bad)


def test_importance_by_group_flags_a_proxy_and_suppresses_small_groups():
    a = attributions(2000, REFERENCE_SCALES, seed=6)
    b = attributions(800, [1.0, 0.7, 0.5, 0.3, 1.2, 0.05], seed=7)  # region weighs much more for group B
    c = attributions(10, REFERENCE_SCALES, seed=8)
    data = pd.concat([a, b, c], ignore_index=True)
    groups = pd.Series(["A"] * 2000 + ["B"] * 800 + ["C"] * 10, name="language")
    results = importance_by_group(data, groups)
    region = [r for r in results if r.feature_name == "region" and r.group == "language=B"][0]
    assert region.drift_status is True and region.reference_group == "language=A" and region.n == 800
    tenure = [r for r in results if r.feature_name == "tenure" and r.group == "language=B"][0]
    assert tenure.drift_status is False
    small = [r for r in results if r.group == "language=C"]
    assert len(small) == 1 and small[0].suppressed and small[0].n is None and small[0].metric_value is None
    with pytest.raises(error_msg):
        importance_by_group(data, groups, reference_group="C")


def test_plain_language_decision():
    row = pd.Series({"income": 0.31, "hours": -0.12, "age": 0.02})
    values = pd.Series({"income": 52000.0, "hours": 20.0, "age": 41.5})
    labels = {"income": {"fr": "revenu", "en": "income"}, "hours": {"fr": "heures par semaine", "en": "hours per week"}}
    fr = explain_decision(row, values, lang="fr", top=2, labels=labels)
    assert "1. revenu (valeur : 52000) a pesé en faveur (+0,31)" in fr
    assert "2. heures par semaine (valeur : 20) a pesé en défaveur (−0,12)" in fr
    assert "age" not in fr and "pas une cause" in fr
    en = explain_decision(row, lang="en", top=1, labels=labels)
    assert "1. income weighed in favour (+0.31)" in en
    with pytest.raises(error_msg):
        explain_decision(row, lang="de")


def test_plain_language_drift_summary():
    drift = AttributionDrift(k=3).fit(attributions(3000, REFERENCE_SCALES, seed=1))
    results = drift.run(attributions(3000, [1.0, 0.7, 0.5, 0.3, 1.6, 0.05], seed=3))
    fr = describe(results, lang="fr", only_changes=True)
    assert any(s.startswith("Classement des facteurs (NDCG@3)") and s.endswith("Changement établi.") for s in fr)
    assert any("« region »" in s for s in fr)
    assert not any("« tenure »" in s for s in fr)
    en = describe(results, lang="en")
    assert len(en) == 2 + len(FEATURES)


def test_tree_attributions_are_additive():
    pytest.importorskip("shap")
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + 0.5 * X["b"] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)
    phi = tree_attributions(model, X)
    assert list(phi.columns) == list(X.columns) and len(phi) == 400
    # attributions add up to the log-odds, up to the constant expected value
    total = phi.sum(axis=1) - model.decision_function(X)
    assert total.std() < 1e-6
    assert importance(phi).idxmax() == "a"
