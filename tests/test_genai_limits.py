"""Regression tests for the limits found by the night lab on pulsar_metrics.genai (TX-02, IM-02, LM-01, LI-03, LS-01)."""

import time
import warnings
from functools import partial

import numpy as np
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.genai import (
    RECOMMENDED_CANARY,
    ComplianceMonitor,
    Rule,
    component_test,
    detection_power,
    embedding_drift,
    load_rules,
    mmd_test,
    model_equality,
    model_equality_power,
    novelty_test,
)


# ----------------------------------------------------------------------------------------------- embeddings
class Corpus:
    """Synthetic text embeddings: topics, a polarity axis, and texts in another language far from the rest"""

    def __init__(self, d=128, topics=20, polarity=2.5, language=6.0, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.d, self.polarity = d, polarity
        self.centers = rng.normal(0, 1, (topics, d)) * 0.35
        self.u = rng.normal(size=d)
        self.u /= np.linalg.norm(self.u)
        other = rng.normal(size=d)
        self.language = other / np.linalg.norm(other) * language

    def texts(self, n, positive=0.5, foreign=False):
        rng = self.rng
        sign = np.where(rng.random(n) < positive, 1.0, -1.0)
        x = self.centers[rng.integers(0, len(self.centers), n)] + rng.normal(0, 0.5, (n, self.d))
        x += np.outer(sign, self.u) * self.polarity
        return x + self.language if foreign else x


def _naive_mmd2(K, in_x):
    x, y = in_x, ~in_x
    n, m = x.sum(), y.sum()
    kxx = (K[np.ix_(x, x)].sum() - np.trace(K[np.ix_(x, x)])) / (n * (n - 1))
    kyy = (K[np.ix_(y, y)].sum() - np.trace(K[np.ix_(y, y)])) / (m * (m - 1))
    return float(kxx + kyy - 2 * K[np.ix_(x, y)].mean())


def test_vectorized_mmd_matches_the_direct_formula():
    from pulsar_metrics.genai.mmd import _mmd2_many

    rng = np.random.default_rng(3)
    z = rng.normal(size=(40, 5))
    K = np.exp(-((z[:, None, :] - z[None, :, :]) ** 2).sum(-1) / 4)
    labels = np.stack([rng.permutation(np.arange(40) < 15) for _ in range(7)], axis=1)
    fast = _mmd2_many(K, labels.astype(float), K.sum(axis=1), float(np.trace(K)))
    assert np.allclose(fast, [_naive_mmd2(K, labels[:, j]) for j in range(7)])


def test_mmd_cost_at_the_default_cap():
    """m4: 1000 x 1000 texts with 500 permutations took 11 to 18 s (one copy of the kernel matrix per permutation)"""
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(1000, 384)), rng.normal(size=(1000, 384))
    start = time.perf_counter()
    result = mmd_test(x, y)
    assert time.perf_counter() - start < 6 and result.n_reference == 1000 and 0 < result.p_value <= 1


def test_mmd_bandwidth_option():
    corpus = Corpus(seed=1)
    ref, cur = corpus.texts(100), corpus.texts(100)
    assert mmd_test(ref, cur, bandwidth=5.0, n_permutations=50).bandwidth == 5.0
    with pytest.raises(error_msg):
        mmd_test(ref, cur, bandwidth=0)


def test_novelty_test_catches_a_small_share_of_foreign_texts_that_mmd_misses():
    """M7: 5 % of texts in another language in a window of 300; MMD detected it in 40 % of the lab's windows"""
    corpus = Corpus(seed=2)
    ref = corpus.texts(500)
    mmd_hits, novelty_hits = 0, 0
    for _ in range(5):
        window = np.vstack([corpus.texts(285), corpus.texts(15, foreign=True)])
        mmd_hits += mmd_test(ref, window, n_permutations=200).p_value < 0.05
        result = novelty_test(ref, window, n_permutations=500)
        novelty_hits += result.p_value < 0.05
    assert novelty_hits >= 4 and mmd_hits <= 2
    # every foreign text (indices 285 to 299) is flagged, among about 5 % of ordinary texts
    assert set(range(285, 300)) <= set(result.novel.tolist()) and result.expected_share == pytest.approx(0.05, abs=0.02)


def test_novelty_and_component_tests_hold_their_level():
    corpus = Corpus(seed=4)
    novelty, hotelling, max_t = [], [], []
    for s in range(30):
        ref, cur = corpus.texts(200), corpus.texts(120)
        novelty.append(novelty_test(ref, cur, n_permutations=200, seed=s).p_value < 0.05)
        hotelling.append(component_test(ref, cur, n_permutations=200, seed=s).p_value < 0.05)
        max_t.append(component_test(ref, cur, statistic="max_t", n_permutations=200, seed=s).p_value < 0.05)
    assert sum(novelty) <= 4 and sum(hotelling) <= 4 and sum(max_t) <= 4  # 30 runs, 5 % nominal


def test_component_test_points_at_the_direction_that_moved():
    rng = np.random.default_rng(6)
    ref, cur = rng.normal(size=(300, 20)), rng.normal(size=(300, 20))
    cur[:, 0] += 0.5
    result = component_test(ref, cur, statistic="max_t", n_components=None, n_permutations=300)
    assert result.p_value < 0.01 and result.component == 0 and len(result.t_values) == 20
    assert component_test(ref, cur, n_components=None, n_permutations=300).p_value < 0.01
    with pytest.raises(error_msg):
        component_test(ref, cur, statistic="median")


def test_embedding_drift_wrappers():
    corpus = Corpus(seed=6)
    ref, cur = corpus.texts(200), np.vstack([corpus.texts(170), corpus.texts(30, foreign=True)])
    novelty = embedding_drift(ref, cur, test="novelty", n_permutations=200)
    assert novelty.metric_name == "embedding_novelty" and novelty.drift_status is True and novelty.metric_value > 0.1
    shift = embedding_drift(ref, cur, test="component", n_permutations=200)
    assert shift.metric_name == "embedding_mean_shift" and shift.p_value is not None
    assert embedding_drift(ref, cur, n_permutations=100).metric_name == "mmd"
    with pytest.raises(error_msg):
        embedding_drift(ref, cur, test="c2st")


def test_detection_power_gives_the_minimal_detectable_change():
    """M6 / m5: power by share of changed texts and window size, with its A/A false alarm rate"""
    corpus = Corpus(seed=7)
    ref, pool, positive = corpus.texts(300), corpus.texts(1000), corpus.texts(500, positive=1.0)
    grid = detection_power(
        ref, pool, positive, shares=(0.0, 0.6), sizes=(150,), test=partial(mmd_test, n_permutations=100), reps=8
    )
    by_share = {row["share"]: row for row in grid}
    assert by_share[0.0]["power"] <= 0.25 and by_share[0.6]["power"] >= 0.75
    assert by_share[0.6]["ci_low"] <= by_share[0.6]["power"] <= by_share[0.6]["ci_high"] and by_share[0.6]["reps"] == 8
    with pytest.raises(error_msg):
        detection_power(ref, pool[:10], positive, sizes=(150,), reps=1)


class _Welch:
    """Fast stand-in for a drift test: Welch t-test on the first coordinate"""

    def __call__(self, reference, current):
        from types import SimpleNamespace

        from scipy import stats

        return SimpleNamespace(p_value=float(stats.ttest_ind(reference[:, 0], current[:, 0], equal_var=False).pvalue))


def _offset_reference(n=30, d=2):
    """An i.i.d. N(0, 1) reference whose mean happens to sit away from 0 (as some references do, by chance)"""
    for seed in range(1000):
        reference = np.random.default_rng(seed).normal(size=(n, d))
        if abs(reference[:, 0].mean()) > 0.4:
            return reference
    raise AssertionError("no offset reference found")


def test_detection_power_aa_point_is_conditional_on_a_fixed_reference():
    """IM-01: on a fixed reference the A/A point is that reference's rate; resampled, it falls back near alpha"""
    reference = _offset_reference()
    pool = np.random.default_rng(1).normal(size=(3000, 2))
    changed = np.random.default_rng(2).normal(loc=1.0, size=(100, 2))
    kwargs = dict(shares=(0.0,), sizes=(30,), test=_Welch(), alpha=0.05, reps=400, seed=3)
    fixed = detection_power(reference, pool, changed, **kwargs)[0]
    resampled = detection_power(reference, pool, changed, resample_reference=True, **kwargs)[0]
    assert fixed["reference"] == "fixed" and resampled["reference"] == "resampled"
    assert fixed["power"] > 0.15  # conditional on an unlucky reference: far above alpha, with a narrow interval
    assert fixed["ci_low"] > 0.05
    assert 0.02 <= resampled["power"] <= 0.09  # independent repetitions: the false alarm rate of the test
    assert resampled["ci_low"] <= 0.05 <= resampled["ci_high"]


def test_detection_power_resampled_reference_is_disjoint_and_reproducible():
    reference = np.column_stack([np.arange(20.0), np.zeros(20)])
    pool = np.column_stack([np.arange(20.0, 120.0), np.zeros(100)])
    changed = np.column_stack([np.arange(1000.0, 1050.0), np.ones(50)])
    seen = []

    def recording_test(ref, cur):
        from types import SimpleNamespace

        seen.append((ref[:, 0].copy(), cur[:, 0].copy()))
        return SimpleNamespace(p_value=float(cur[:, 0].sum() % 1.0))

    grid = detection_power(
        reference, pool, changed, shares=(0.0, 0.2), sizes=(10,), test=recording_test, reps=15, seed=4, resample_reference=True
    )
    assert len(seen) == 30
    references = {tuple(sorted(ref)) for ref, _ in seen}
    assert len(references) > 1  # a new reference at each repetition
    for ref, cur in seen:
        assert len(ref) == 20 and len(cur) == 10
        assert not set(ref) & set(cur)  # the window never reuses a reference row
        assert set(ref) <= set(np.arange(120.0))  # the reference holds unchanged rows only
    seen.clear()
    again = detection_power(
        reference, pool, changed, shares=(0.0, 0.2), sizes=(10,), test=recording_test, reps=15, seed=4, resample_reference=True
    )
    assert again == grid


def test_detection_power_default_keeps_the_given_reference():
    reference = np.random.default_rng(0).normal(size=(20, 2))
    pool = np.random.default_rng(1).normal(size=(200, 2))
    changed = np.random.default_rng(2).normal(loc=1.0, size=(50, 2))
    seen = []

    def recording_test(ref, cur):
        from types import SimpleNamespace

        seen.append(ref)
        return SimpleNamespace(p_value=0.5)

    detection_power(reference, pool, changed, shares=(0.0,), sizes=(10,), test=recording_test, reps=5)
    assert all(ref is reference for ref in seen)


# ----------------------------------------------------------------------------------------------- model equality
WORDS = "le programme aide au loyer exige un revenu inférieur à quarante mille dollars par année pour les demandeurs".split()


def canary(prompts, per_prompt, style, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k, p in enumerate(prompts):
        base = np.random.default_rng(k).permutation(WORDS)
        out[p] = [" ".join(w if rng.random() > style else rng.choice(WORDS) for w in base) for _ in range(per_prompt)]
    return out


def test_model_equality_power_by_canary_size():
    """M23: a close model (style 0.30 vs 0.40, like a quantization) is missed at small canary sizes"""
    prompts = [f"q{i}" for i in range(48)]
    ref, cur = canary(prompts, 10, 0.30, 1), canary(prompts, 10, 0.40, 2)
    grid = model_equality_power(ref, cur, prompts=(6, 48), answers=(3, 10), reps=10, n_permutations=100)
    power = {(row["prompts"], row["answers_per_prompt"]): row["power"] for row in grid}
    assert power[(6, 3)] <= 0.3 and power[(48, 10)] >= 0.8
    with pytest.raises(error_msg):
        model_equality_power(ref, cur, prompts=(60,), answers=(5,), reps=1)


def test_small_canary_can_report_inconclusive():
    prompts = [f"q{i}" for i in range(12)]
    ref, cur = canary(prompts, 5, 0.30, 1), canary(prompts, 5, 0.30, 2)
    default = model_equality(ref, cur, n_permutations=100)
    assert default.drift_status is False and "usually missed" in default.method
    strict = model_equality(ref, cur, min_answers=RECOMMENDED_CANARY["answers"], n_permutations=100)
    assert strict.drift_status is None and "inconclusive" in strict.error
    assert RECOMMENDED_CANARY == {"prompts": 48, "answers_per_prompt": 10, "answers": 480}


# ----------------------------------------------------------------------------------------------- compliance
RULES = {
    "rules": [
        {"id": "auth-before-file", "template": "precedence", "events": ["authenticate", "create_file"], "severity": "critical"},
        {"id": "amount-after-eligibility", "template": "precedence", "events": ["check_eligibility", "compute_amount"]},
        {"id": "no-delete", "template": "absence", "events": ["delete_file"], "severity": "minor"},
    ]
}


def test_alerts_per_run_for_critical_rules():
    """C3: a run violating a critical rule must raise an alert by itself, with the rules version"""
    monitor = ComplianceMonitor(load_rules(RULES), version="rules/2")
    trace = ["create_file", "compute_amount", "delete_file"]
    alerts = monitor.alerts(trace)
    assert [a["rule_id"] for a in alerts] == ["auth-before-file"] and alerts[0]["rules_version"] == "rules/2"
    assert [a["rule_id"] for a in monitor.alerts(trace, min_severity="minor")] == [
        "auth-before-file",
        "amount-after-eligibility",
        "no-delete",
    ]
    assert monitor.alerts(["authenticate", "create_file"]) == []
    record = monitor.run_record(trace)
    assert record == {
        "violated_rules": "auth-before-file,amount-after-eligibility,no-delete",
        "critical_violations": 1,
        "max_severity": "critical",
        "rules_version": "rules/2",
    }
    assert monitor.run_record([])["max_severity"] is None
    with pytest.raises(error_msg):
        monitor.alerts(trace, min_severity="urgent")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Rule(id="x", formula="F a", severity="high")
    assert "unknown severity" in str(caught[0].message)


def _runs(n, violation_rate, rng):
    good, bad = ["check_eligibility", "compute_amount"], ["compute_amount"]
    return [bad if rng.random() < violation_rate else good for _ in range(n)]


def test_evaluate_against_a_reference_ignores_stable_violation_rates():
    """M26: a rule violated at a stable rate (29 % vs 29 %) alerted in 10 windows out of 10"""
    rng = np.random.default_rng(0)
    monitor = ComplianceMonitor(load_rules(RULES), version="rules/2")
    reference = _runs(1000, 0.29, rng)
    stable = [{r.feature_name: r for r in monitor.evaluate(_runs(120, 0.29, rng), reference=reference)} for _ in range(10)]
    absolute = [{r.feature_name: r for r in monitor.evaluate(_runs(120, 0.29, rng))} for _ in range(10)]
    assert sum(w["amount-after-eligibility"].drift_status for w in absolute) == 10
    assert sum(w["amount-after-eligibility"].drift_status for w in stable) <= 1
    worse = {r.feature_name: r for r in monitor.evaluate(_runs(120, 0.5, rng), reference=reference)}
    rule = worse["amount-after-eligibility"]
    assert rule.drift_status is True and rule.p_value < 0.01 and rule.threshold == pytest.approx(0.71, abs=0.04)
    assert "Fisher" in rule.method
    # critical rules keep the absolute verdict, unless absolute_from=None
    critical = ComplianceMonitor(load_rules(RULES)).evaluate(
        [["create_file"]] + [["authenticate"]] * 9, reference=[["create_file"]] * 5
    )
    assert {r.feature_name: r.drift_status for r in critical}["auth-before-file"] is True
    relative = ComplianceMonitor(load_rules(RULES)).evaluate(
        [["create_file"]] + [["authenticate"]] * 9, reference=[["create_file"]] * 5, absolute_from=None
    )
    assert {r.feature_name: r.drift_status for r in relative}["auth-before-file"] is False
    with pytest.raises(error_msg):
        monitor.evaluate([["a"]], reference=[])


def test_pca_drops_null_directions_when_there_are_fewer_texts_than_components():
    """Gram-branch PCA divided null eigenvectors by ~0: projections of 1e137 for 12 + 12 texts in 384 dimensions"""
    from pulsar_metrics.genai.mmd import _project

    for seed in (85, 98, 3):
        rng = np.random.default_rng(seed)
        x, y = rng.normal(size=(12, 384)), rng.normal(size=(12, 384))
        px, py = _project(x, y, 32)
        assert px.shape[1] <= 23 and np.abs(np.vstack([px, py])).max() < 1e3
        assert mmd_test(x, y).p_value == pytest.approx(mmd_test(x, y, n_components=None).p_value)
    assert _project(np.ones((6, 10)), np.ones((6, 10)), 4)[0].shape == (6, 10)  # no variance at all: unprojected
