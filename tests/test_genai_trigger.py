"""RAGDriftTrigger: the recommended drift trigger of a RAG agent (LR-03 replay of pulsar_demo, trigger "score")"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pulsar_metrics.analyzers.persistence import DriftPersistence
from pulsar_metrics.evidence import NOTE_CODES, REASON_CODES, SCHEMA_PATH, dumps, loads
from pulsar_metrics.exceptions import CustomExceptionPulsarMetric
from pulsar_metrics.genai import (
    RAG_TRIGGER_SIGNALS,
    RECOMMENDED_RAG_TRIGGER,
    RAGDriftTrigger,
    TriggerSignal,
)


def runs(rng, n, faithfulness_drop=0.0, empty=0.05, missing=0.0):
    """Synthetic agent_run records: faithfulness in [0, 1], empty retrievals, BM25 score of the top passage"""
    faithfulness = np.clip(rng.beta(5, 2, n) - faithfulness_drop, 0, 1)
    faithfulness[rng.random(n) < missing] = np.nan  # no answer scored
    return pd.DataFrame(
        {
            "det_faithfulness": faithfulness,
            "det_retrieval_empty": rng.random(n) < empty,
            "det_retrieval_top": rng.gamma(5, 2, n),
        }
    )


# ------------------------------------------------------------------------------------- replay logic, as in pulsar_demo
# lab/replay/declencheur_derive.py: window_pvalues (faithfulness, empty_retrieval, relevance_bm25), add_composites
# ("trigger_score": smallest p-value x 3) and DriftPersistence(windows=k) on the composite verdict p < alpha
def _clean(values):
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _mw_lower(win, ref):
    w, r = _clean(win), _clean(ref)
    if len(w) < 5 or len(r) < 5:
        return None
    return float(stats.mannwhitneyu(w, r, alternative="less").pvalue)


def _fisher_higher(win, ref):
    a, b = sum(bool(x) for x in win), sum(bool(x) for x in ref)
    return float(stats.fisher_exact([[a, len(win) - a], [b, len(ref) - b]], alternative="greater").pvalue)


def replay_pvalues(ref, win):
    empty = [r["empty"] or r["no_search"] for r in win], [r["empty"] or r["no_search"] for r in ref]
    return {
        "faithfulness": _mw_lower([r["faithfulness"] for r in win], [r["faithfulness"] for r in ref]),
        "empty_retrieval": _fisher_higher(*empty),
        "relevance_bm25": _mw_lower([r.get("top1_bm25") for r in win], [r.get("top1_bm25") for r in ref]),
    }


def replay_composite(pvalues):
    members = ("faithfulness", "empty_retrieval", "relevance_bm25")
    values = [pvalues.get(m) for m in members if pvalues.get(m) is not None]
    return min(1.0, len(members) * min(values)) if values else None


def lr03_runs(rng, n, level=0.0):
    """Runs in the shape of lab/results/LR-03/runs.jsonl: a corpus loss lowers faithfulness and the top score and
    empties more searches; some runs make no search (no score) and some answers are not scored"""
    out = []
    for _ in range(n):
        no_search = rng.random() < 0.03
        empty = (not no_search) and rng.random() < 0.04 + 0.3 * level
        faithfulness = None if rng.random() < 0.05 else float(np.clip(rng.beta(5, 2) - 0.4 * level * rng.random(), 0, 1))
        top = None if (no_search or empty) else float(rng.gamma(5, 2) * (1 - 0.5 * level * rng.random()))
        out.append({"faithfulness": faithfulness, "empty": empty, "no_search": no_search, "top1_bm25": top})
    return out


def as_records(rows):
    """LR-03 runs mapped to the agent_run fields of pulsar-data-collection"""
    return [
        {
            "det_faithfulness": r["faithfulness"],
            "det_retrieval_empty": bool(r["empty"] or r["no_search"]),
            "det_retrieval_top": r["top1_bm25"],
        }
        for r in rows
    ]


# ------------------------------------------------------------------------------------------------------------ defaults
def test_defaults_are_the_replay_configuration():
    trigger = RAGDriftTrigger()
    assert trigger.alpha == 0.05 and trigger.persistence == 2 and trigger.is_recommended
    assert [(s.name, s.column, s.test, s.alternative, s.min_values) for s in trigger.signals] == [
        ("faithfulness", "det_faithfulness", "manwu", "less", 5),
        ("empty_retrieval", "det_retrieval_empty", "rate_increase", None, 1),
        ("retrieval_top", "det_retrieval_top", "manwu", "less", 5),
    ]
    assert trigger.signals == RAG_TRIGGER_SIGNALS
    assert RECOMMENDED_RAG_TRIGGER["alpha"] == 0.05 and RECOMMENDED_RAG_TRIGGER["persistence"] == 2
    assert RECOMMENDED_RAG_TRIGGER["correction"] == "bonferroni"
    assert RECOMMENDED_RAG_TRIGGER["window_runs"] == 100 and RECOMMENDED_RAG_TRIGGER["reference_runs"] == 300
    assert RECOMMENDED_RAG_TRIGGER["measured"]["false_alarms_per_week"] == 0.08


def test_decisions_reproduce_the_replay_logic():
    """Same p-values, composite, raw and confirmed verdicts as declencheur_derive.py on a sequence of windows"""
    rng = np.random.default_rng(3)
    reference = lr03_runs(rng, 300)
    levels = [0.0, 0.0, 0.4, 0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.8, 0.8]
    windows = [lr03_runs(rng, 100, level) for level in levels]
    windows.append([{**r, "faithfulness": None} for r in lr03_runs(rng, 100)])  # faithfulness not computable
    trigger, tracker = RAGDriftTrigger(), DriftPersistence(windows=2)
    raw_seen, confirmed_seen = set(), set()
    for win in windows:
        expected = replay_pvalues(reference, win)
        composite = replay_composite(expected)
        flag = composite < 0.05
        record = {"metric_type": "drift", "metric_name": "trigger_score", "feature_name": None, "drift_status": flag}
        confirmed = tracker.update([record])[0].confirmed_drift

        decision = trigger.evaluate(as_records(win), as_records(reference))
        got = decision.p_values
        for ours, theirs in (
            ("faithfulness", "faithfulness"),
            ("empty_retrieval", "empty_retrieval"),
            ("retrieval_top", "relevance_bm25"),
        ):
            if expected[theirs] is None:
                assert got[ours] is None
            else:
                assert got[ours] == pytest.approx(expected[theirs], rel=1e-12, abs=1e-300)
        assert decision.p_value == pytest.approx(composite, rel=1e-12, abs=1e-300)
        assert decision.drift is flag
        assert decision.confirmed is confirmed
        raw_seen.add(flag)
        confirmed_seen.add(confirmed)
    assert raw_seen == {True, False} and confirmed_seen == {True, False}  # the sequence exercises both verdicts


# ---------------------------------------------------------------------------------------------------------------- A/A
def test_aa_windows_raise_no_confirmed_alert():
    rng = np.random.default_rng(11)
    reference = runs(rng, 300)
    trigger = RAGDriftTrigger()
    decisions = [trigger.evaluate(runs(rng, 100), reference) for _ in range(24)]  # one day at one window per hour
    assert not any(d.confirmed for d in decisions)
    assert sum(bool(d.drift) for d in decisions) <= 2


def test_aa_raw_false_alarms_stay_near_alpha():
    rng = np.random.default_rng(5)
    reference = runs(rng, 300)
    trigger = RAGDriftTrigger(persistence=1)
    alarms = sum(bool(trigger.evaluate(runs(rng, 100), reference).drift) for _ in range(300))
    assert alarms / 300 <= 0.06  # Bonferroni over 3 one-sided tests: at most alpha


# ----------------------------------------------------------------------------------------------------- injected drift
def test_faithfulness_drop_is_confirmed_at_the_second_window():
    rng = np.random.default_rng(2)
    reference = runs(rng, 300)
    trigger = RAGDriftTrigger()
    stable = [trigger.evaluate(runs(rng, 100), reference) for _ in range(3)]
    assert not any(d.confirmed for d in stable)
    first = trigger.evaluate(runs(rng, 100, faithfulness_drop=0.25), reference)
    second = trigger.evaluate(runs(rng, 100, faithfulness_drop=0.25), reference)
    assert first.drift is True and first.confirmed is False and first.changed is False
    assert first.signals[0].drift_status is True  # the faithfulness signal carries the alert
    assert second.drift is True and second.confirmed is True and second.changed is True

    pending = first.to_evidence("RAG-01", window={"id": "w4"})[-1]
    assert pending.metric.status == "alert" and "drift_not_confirmed" in pending.reliability.notes
    confirmed = second.to_evidence("RAG-01", window={"id": "w5"})[-1]
    assert confirmed.metric.name == "rag_drift_trigger" and confirmed.metric.family == "genai"
    assert confirmed.reliability.persistence.confirmed and confirmed.reliability.persistence.changed
    assert confirmed.reliability.persistence.windows == 2
    assert "drift_not_confirmed" not in confirmed.reliability.notes

    # back to normal: cleared after 2 windows without alert
    trigger.evaluate(runs(rng, 100), reference)
    cleared = trigger.evaluate(runs(rng, 100), reference)
    assert cleared.confirmed is False and cleared.changed is True


# -------------------------------------------------------------------------------------------- signals not evaluable
def test_a_missing_signal_column_is_not_evaluable_with_a_reason_code():
    rng = np.random.default_rng(4)
    reference = runs(rng, 300)
    current = runs(rng, 100, faithfulness_drop=0.25).drop(columns="det_retrieval_top")
    decision = RAGDriftTrigger().evaluate(current, reference)
    assert decision.not_evaluable == {"retrieval_top": "missing_signal"}
    assert decision.p_values["retrieval_top"] is None
    missing = decision.signals[2]
    assert missing.status == "missing_signal" and missing.drift_status is None and "absent from the window" in missing.error
    # the others still decide, with the factor of all three signals (as in the replay)
    assert decision.p_value == pytest.approx(min(1.0, 3 * decision.p_values["faithfulness"]))
    assert decision.drift is True

    records = decision.to_evidence("RAG-01", window={"id": "w1"})
    signal = records[2]
    assert signal.metric.status == "not_evaluable" and signal.reliability.reasons == ["missing_signal"]
    assert not signal.reliability.reliable
    combined = records[-1]
    assert combined.metric.status == "alert" and "partial_signals" in combined.reliability.notes
    for record in records:
        assert set(record.reliability.reasons) <= set(REASON_CODES)
        assert set(record.reliability.notes) <= set(NOTE_CODES)


def test_missing_in_the_reference_and_few_values():
    rng = np.random.default_rng(6)
    reference = runs(rng, 300).drop(columns="det_faithfulness")
    current = runs(rng, 100)
    current.loc[4:, "det_retrieval_top"] = np.nan  # 4 values left, below min_values=5
    decision = RAGDriftTrigger().evaluate(current, reference)
    assert decision.not_evaluable == {"faithfulness": "missing_signal", "retrieval_top": "few_values"}
    assert "absent from the reference" in decision.signals[0].error
    assert "4 non-missing values in the window" in decision.signals[2].error
    assert decision.p_value == pytest.approx(min(1.0, 3 * decision.p_values["empty_retrieval"]))
    reasons = [r.reliability.reasons for r in decision.to_evidence("RAG-01", window={"id": "w1"})[:3]]
    assert reasons == [["missing_signal"], [], ["few_values"]]


def test_no_evaluable_signal_gives_no_verdict_and_keeps_the_streaks():
    rng = np.random.default_rng(7)
    reference = runs(rng, 300)
    trigger = RAGDriftTrigger()
    assert trigger.evaluate(runs(rng, 100, faithfulness_drop=0.25), reference).drift is True
    empty = trigger.evaluate(pd.DataFrame({"other": range(100)}), reference)
    assert empty.p_value is None and empty.drift is None and empty.confirmed is False
    assert empty.persistence.drift_streak == 1  # no information: the streak is kept
    record = empty.to_evidence("RAG-01", window={"id": "w2"}, include_signals=False)
    assert len(record) == 1 and record[0].metric.status == "not_evaluable"
    assert record[0].reliability.reasons == ["no_signal"] and "partial_signals" not in record[0].reliability.notes
    # the next drifted window confirms
    assert trigger.evaluate(runs(rng, 100, faithfulness_drop=0.25), reference).confirmed is True


def test_a_failing_test_is_an_error_record_not_a_crash():
    rng = np.random.default_rng(8)
    reference, current = runs(rng, 300), runs(rng, 100)
    current["det_retrieval_empty"] = 0.5  # not an indicator: rate_increase refuses it
    decision = RAGDriftTrigger().evaluate(current, reference)
    assert decision.p_values["empty_retrieval"] is None and "0 or 1" in decision.not_evaluable["empty_retrieval"]
    record = decision.to_evidence("RAG-01", window={"id": "w1"})[1]
    assert record.metric.status == "error" and record.reliability.reasons == ["computation_error"]


# ------------------------------------------------------------------------------------------------------------ overrides
def test_overrides_of_alpha_persistence_signals_and_columns():
    rng = np.random.default_rng(9)
    reference = runs(rng, 300).rename(columns={"det_faithfulness": "faith"})
    current = runs(rng, 100, faithfulness_drop=0.25).rename(columns={"det_faithfulness": "faith"})

    trigger = RAGDriftTrigger(
        signals=["faithfulness", "empty_retrieval"], alpha=0.01, persistence=1, columns={"faithfulness": "faith"}
    )
    assert not trigger.is_recommended
    decision = trigger.evaluate(current, reference)
    assert decision.names == ("faithfulness", "empty_retrieval") and decision.settings["factor"] == 2
    assert decision.signals[0].feature_name == "faith" and decision.signals[0].threshold == pytest.approx(0.005)
    assert decision.p_value == pytest.approx(min(1.0, 2 * min(v for v in decision.p_values.values())))
    assert decision.result.threshold == 0.01
    assert decision.drift is True and decision.confirmed is True  # persistence 1: the raw verdict
    combined = decision.to_evidence("RAG-01", window={"id": "w1"})[-1]
    assert combined.details["trigger"]["preset"] == "custom" and combined.details["trigger"]["alpha"] == 0.01

    custom = TriggerSignal("abstention", "det_abstention", "rate_increase", min_values=1)
    frames = [f.assign(det_abstention=(rng.random(len(f)) < 0.2).astype(float)) for f in (reference, current)]
    trigger = RAGDriftTrigger(signals=["faithfulness", custom], columns={"faithfulness": "faith"}, persistence=3)
    decision = trigger.evaluate(frames[1], frames[0])
    assert decision.names == ("faithfulness", "abstention") and trigger.persistence == 3
    assert decision.drift is True and decision.confirmed is False


def test_invalid_settings_are_refused():
    with pytest.raises(CustomExceptionPulsarMetric):
        RAGDriftTrigger(alpha=0)
    with pytest.raises(CustomExceptionPulsarMetric):
        RAGDriftTrigger(persistence=0)
    with pytest.raises(CustomExceptionPulsarMetric):
        RAGDriftTrigger(signals=[])
    with pytest.raises(CustomExceptionPulsarMetric):
        RAGDriftTrigger(signals=["faithfulness", "faithfulness"])
    with pytest.raises(CustomExceptionPulsarMetric):
        RAGDriftTrigger(signals=["abstention"])
    with pytest.raises(CustomExceptionPulsarMetric):
        RAGDriftTrigger(columns={"abstention": "x"})
    with pytest.raises(CustomExceptionPulsarMetric):
        TriggerSignal("x", "x", "rate_increase", alternative="less")
    with pytest.raises(CustomExceptionPulsarMetric):
        TriggerSignal("x", "x", "not_a_test")
    with pytest.raises(CustomExceptionPulsarMetric):
        TriggerSignal("x", "x", min_values=0)


# ------------------------------------------------------------------------------------------------------------ evidence
def test_evidence_records_round_trip_and_follow_the_schema():
    rng = np.random.default_rng(10)
    reference = runs(rng, 300)
    trigger = RAGDriftTrigger()
    decision = trigger.evaluate(runs(rng, 12), reference)
    records = decision.to_evidence("RAG-01", window={"id": "2026-11-05T10"}, reference={"id": "ref-2026-11"}, model_id="rag")
    assert [r.metric.name for r in records] == ["manwu", "rate_increase", "manwu", "rag_drift_trigger"]
    assert [r.details.get("trigger_signal") for r in records[:3]] == ["faithfulness", "empty_retrieval", "retrieval_top"]
    assert records[1].details["reference_n"] == 300  # the rates of rate_increase are still read back
    combined = records[-1]
    assert combined.window.n_rows == 12 and combined.reference.n_rows == 300
    assert combined.metric.direction == "lower" and combined.metric.decision_basis == "p_value"
    assert combined.metric.p_value == pytest.approx(decision.p_value) and combined.metric.threshold == 0.05
    assert "small_window" in combined.reliability.notes  # 12 runs, below the 20 recommended
    assert [s["name"] for s in combined.details["trigger"]["signals"]] == ["faithfulness", "empty_retrieval", "retrieval_top"]
    assert loads(dumps(records)) == records


def test_evidence_records_validate_against_the_committed_schema():
    jsonschema = pytest.importorskip("jsonschema")
    rng = np.random.default_rng(13)
    reference = runs(rng, 300)
    current = runs(rng, 100).drop(columns="det_retrieval_top")
    records = RAGDriftTrigger().evaluate(current, reference).to_evidence("RAG-01", window={"id": "w1"})
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for record in records:
        jsonschema.validate(record.to_dict(), schema)


def test_reset_forgets_the_persistence():
    rng = np.random.default_rng(12)
    reference = runs(rng, 300)
    trigger = RAGDriftTrigger()
    trigger.evaluate(runs(rng, 100, faithfulness_drop=0.25), reference)
    trigger.reset()
    assert trigger.evaluate(runs(rng, 100, faithfulness_drop=0.25), reference).confirmed is False
