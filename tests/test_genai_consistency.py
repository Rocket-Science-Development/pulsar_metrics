"""Consistency of repeated answers: window metrics with bootstrap intervals, one-sided drift test, evidence export"""

import json

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.evidence import NOTE_CODES, REASON_CODES, SCHEMA_PATH, dumps, evidence_from_result, loads, to_evidence
from pulsar_metrics.genai import consistency, consistency_drift, question_agreement


def window(p_agree, n_questions=40, repeats=3, seed=0, prefix="q"):
    """Synthetic per-question frame: each answer is the majority answer with probability p_agree"""
    rng = np.random.default_rng(seed)
    answers = {}
    for i in range(n_questions):
        answers[f"{prefix}{i}"] = ["a" if rng.random() < p_agree else f"b{rng.integers(3)}" for _ in range(repeats)]
    return question_agreement(answers)


def test_question_agreement_counts_pairs_and_unreadable_answers():
    frame = question_agreement({"q1": ["yes", "yes", "yes"], "q2": ["yes", "no", "yes"], "q3": [None, None], "q4": ["x"]})
    rows = frame.set_index("question_id")
    assert rows.loc["q1", "agreement"] == 1.0 and rows.loc["q1", "full_agreement"]
    assert rows.loc["q2", "agreement"] == pytest.approx(1 / 3) and not rows.loc["q2", "full_agreement"]
    assert rows.loc["q3", "agreement"] == 0.0  # nothing read: agrees with nothing
    assert pd.isna(rows.loc["q4", "agreement"])  # a single answer has no pair


def test_window_metrics_with_bootstrap_intervals():
    agreement, full = consistency(window(0.9), model="llama3.2:3b", threshold=0.5)
    assert agreement.metric_name == "consistency_agreement" and full.metric_name == "consistency_full_agreement"
    assert agreement.feature_name == "llama3.2:3b" and agreement.n == 40
    assert agreement.ci_low < agreement.metric_value < agreement.ci_high
    assert 0 <= full.ci_low <= full.metric_value <= full.ci_high <= 1
    assert full.metric_value <= agreement.metric_value
    assert agreement.drift_status is False and full.drift_status is None  # no threshold given for the share
    assert agreement.reliable is None and agreement.status is None
    low = consistency(window(0.3), threshold=0.5)[0]
    assert low.drift_status is True


def test_few_repeats_and_few_questions_are_flagged():
    agreement, _ = consistency(window(0.9, repeats=2))
    assert agreement.status == "few_repeats" and agreement.reliable is False and "fewer than 3" in agreement.reliability_reason
    small = consistency(window(0.9, n_questions=5))[0]
    assert small.status == "few_questions" and small.reliable is False
    nothing = consistency({"q": ["only one"]})[0]
    assert nothing.metric_value is None and nothing.status == "few_repeats"


def test_drift_detects_a_drop_and_not_a_rise():
    reference, dropped = window(0.9, seed=1), window(0.5, seed=2)
    drop = consistency_drift(reference, dropped, model="m")
    assert drop.metric_name == "consistency_drift" and drop.metric_type == "drift"
    assert drop.metric_value < 0 and drop.ci_high < 0 and drop.p_value < 0.01 and drop.drift_status is True
    assert "paired on 40 shared questions" in drop.method
    rise = consistency_drift(dropped, reference)
    assert rise.drift_status is False and rise.p_value > 0.5  # one-sided: a gain is not an alert
    share = consistency_drift(reference, dropped, statistic="full_agreement")
    assert share.metric_name == "consistency_full_agreement_drift" and share.drift_status is True
    with pytest.raises(ValueError):
        consistency_drift(reference, dropped, statistic="median")


def test_drift_is_unpaired_when_the_questions_differ():
    result = consistency_drift(window(0.9, seed=1), window(0.5, seed=2, prefix="other"))
    assert "unpaired" in result.method and result.drift_status is True


def test_aa_false_alarm_rate_is_near_alpha():
    alarms = [
        consistency_drift(window(0.7, seed=2 * k), window(0.7, seed=2 * k + 1), n_permutations=500, n_boot=200).drift_status
        for k in range(100)
    ]
    assert np.mean(alarms) <= 0.12  # alpha 0.05, 100 A/A draws


def test_drift_with_too_few_questions_is_not_evaluable():
    result = consistency_drift(window(0.9, n_questions=1), window(0.9, n_questions=1))
    assert result.metric_value is None and result.drift_status is None and result.reliable is False
    record = evidence_from_result(result, "REG-0003", window={"id": "w1"})
    assert record.metric.status == "not_evaluable" and record.reliability.reasons == ["few_questions"]


def test_evidence_of_consistency_results():
    agreement, full = consistency(window(0.3), model="llama3.2:3b", threshold=0.5)
    record = to_evidence(agreement, "REG-0003", window={"id": "2026-W45"})[0]
    assert record.metric.family == "genai" and record.metric.status == "alert"
    assert (record.metric.direction, record.metric.decision_basis, record.metric.threshold) == ("lower", "value", 0.5)
    assert record.metric.ci_low == agreement.ci_low and record.reliability.reliable
    assert "nominal_threshold" in record.reliability.notes
    no_threshold = to_evidence(full, "REG-0003", window={"id": "2026-W45"})[0]
    assert no_threshold.metric.status == "no_verdict" and no_threshold.metric.direction == "none"

    drop = consistency_drift(window(0.9, seed=1), window(0.5, seed=2), model="llama3.2:3b")
    record = to_evidence([drop], "REG-0003", window={"id": "2026-W45"}, reference={"id": "2026-W40"})[0]
    assert record.metric.family == "genai" and record.metric.status == "alert"
    assert (record.metric.direction, record.metric.decision_basis) == ("lower", "p_value")
    assert record.metric.p_value == drop.p_value and record.metric.threshold == 0.05
    assert record.metric.value == pytest.approx(drop.metric_value) and record.metric.ci_high < 0

    few = consistency(window(0.9, repeats=2))[0]
    record = to_evidence(few, "REG-0003", window={"id": "w"})[0]
    assert record.reliability.reliable is False and record.reliability.reasons == ["few_repeats"]
    assert record.reliability.reason_text == few.reliability_reason
    for r in loads(dumps([record])):
        assert set(r.reliability.reasons) <= set(REASON_CODES) and set(r.reliability.notes) <= set(NOTE_CODES)


def test_consistency_records_validate_against_the_committed_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    results = consistency(window(0.8), threshold=0.5) + [consistency_drift(window(0.9, seed=1), window(0.5, seed=2))]
    for record in to_evidence(results, "REG-0003", window={"id": "w"}):
        jsonschema.validate(record.to_dict(), schema)


def test_a_missing_full_agreement_is_read_from_the_agreement():
    frame = window(0.8).astype({"full_agreement": object})
    expected = consistency(frame)[1].metric_value
    frame.loc[frame.index[:5], "full_agreement"] = None  # e.g. a frame assembled by hand
    full = consistency(frame)[1]
    assert full.metric_value == pytest.approx(expected) and not np.isnan(full.metric_value)
    frame.loc[frame.index[0], "full_agreement"] = "yes"
    with pytest.raises(ValueError, match="full_agreement"):
        consistency(frame)


def test_probe_frames_with_extra_columns_are_accepted():
    frame = window(0.8)
    frame["pair_scores"] = [[1.0, 0.0, 0.0]] * len(frame)
    frame.loc[0, "n_answers"] = 1  # set aside
    agreement = consistency(frame)[0]
    assert agreement.n == len(frame) - 1 and "1 set aside" in agreement.method
    with pytest.raises(ValueError):
        consistency(pd.DataFrame({"question_id": ["a"]}))
