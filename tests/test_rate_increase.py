"""rate_increase: one-sided exact test of the rate of a 0 / 1 indicator per run (abstentions, answers without support)"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.evidence import SCHEMA_PATH, dumps, loads, to_evidence
from pulsar_metrics.exceptions import MetricEvaluationError
from pulsar_metrics.metrics.drift import DriftTestMetric
from pulsar_metrics.metrics.utils import rate_increase_test


def frame(values, name="det_unsupported_answer"):
    return pd.DataFrame({name: values})


def test_p_value_is_the_one_sided_fisher_exact_test():
    result = rate_increase_test([1, 1, 1, 0, 0, 0, 0, 0], [1] * 4 + [0] * 76)
    expected = fisher_exact([[3, 5], [4, 76]], alternative="greater").pvalue
    assert result.pvalue == pytest.approx(expected)
    assert result.statistic == pytest.approx(3 / 8 - 4 / 80)
    assert (result.current_count, result.current_n, result.reference_count, result.reference_n) == (3, 8, 4, 80)


def test_only_a_rise_alerts():
    reference = frame([1.0] * 20 + [0.0] * 60)
    lower = DriftTestMetric("rate_increase", "det_unsupported_answer").evaluate(frame([0.0] * 8), reference)
    higher = DriftTestMetric("rate_increase", "det_unsupported_answer").evaluate(frame([1.0] * 6 + [0.0] * 2), reference)
    assert lower.drift_status is False and lower.metric_value == pytest.approx(1.0)
    assert higher.drift_status is True


def test_a_zero_base_rate_is_a_valid_reference():
    """The generic tests flag a reference without any positive as constant_reference (reliable=False); a rate starting
    from zero is the normal case of abstentions and unsupported answers"""
    reference, current = frame([0.0] * 80), frame([1.0, 1.0, 1.0, 0, 0, 0, 0, 0])
    generic = DriftTestMetric("manwu", "det_unsupported_answer").evaluate(current, reference, alternative="greater")
    assert generic.status == "constant_reference"
    result = DriftTestMetric("rate_increase", "det_unsupported_answer").evaluate(current, reference)
    assert result.status is None and result.reliable is None and result.drift_status is True
    assert result.metric_value == pytest.approx(fisher_exact([[3, 5], [0, 80]], alternative="greater").pvalue)
    # the asymptotic rank test on 0 / 1 values is far below the exact p-value on a small window
    assert generic.metric_value < result.metric_value / 1000
    nothing = DriftTestMetric("rate_increase", "det_unsupported_answer").evaluate(frame([0.0] * 8), reference)
    assert nothing.metric_value == 1.0 and nothing.drift_status is False


def test_false_alarms_stay_under_alpha_on_small_windows():
    rng = np.random.default_rng(0)
    alarms = 0
    for _ in range(500):
        reference = frame((rng.random(80) < 0.1).astype(float))
        current = frame((rng.random(8) < 0.1).astype(float))
        alarms += DriftTestMetric("rate_increase", "det_unsupported_answer").evaluate(current, reference).drift_status
    assert alarms / 500 <= 0.05


def test_booleans_and_missing_values():
    current = pd.DataFrame({"det_abstention": [True, True, None, False]})
    reference = pd.DataFrame({"det_abstention": [False] * 30 + [None]})
    result = DriftTestMetric("rate_increase", "det_abstention").evaluate(current, reference)
    assert "window 2/3" in result.method and "reference 0/30" in result.method
    assert result.missing_share_current == pytest.approx(0.25)


def test_values_other_than_zero_and_one_are_refused():
    with pytest.raises(MetricEvaluationError, match="0 or 1"):
        DriftTestMetric("rate_increase", "x").evaluate(frame([0.2, 0.9], "x"), frame([0.0, 1.0], "x"))


def test_analyzer_runs_it_and_the_evidence_carries_both_rates():
    def runs(values, version="1"):
        n = len(values)
        return pd.DataFrame(
            {
                "det_unsupported_answer": values,
                "model_id": "agent",
                "model_version": version,
                "pred_timestamp": pd.date_range("2026-11-01", periods=n, freq="min"),
                "period": "p",
                "y_pred": 0,
                "y_true": 0,
            }
        )

    analyzer = Analyzer(name="support", model_id="agent", model_version="1")
    analyzer.add_drift_metrics(["rate_increase"], features_list=["det_unsupported_answer"])
    analyzer.run(current=runs([1.0] * 4 + [0.0] * 4), reference=runs([0.0] * 76 + [1.0] * 4))
    result = [r for r in analyzer.get_result() if r.metric_name == "rate_increase"][0]
    assert result.drift_status is True

    record = to_evidence([result], "register-assistant", window={"id": "w1"})[0]
    assert record.metric.family == "drift" and record.metric.status == "alert"
    assert record.metric.direction == "lower" and record.metric.decision_basis == "p_value"
    assert record.metric.p_value == pytest.approx(result.metric_value)
    assert record.details["current_rate"] == 0.5 and record.details["current_n"] == 8
    assert record.details["reference_rate"] == 0.05 and record.details["reference_count"] == 4
    assert record.reliability.reliable is True
    assert loads(dumps([record]))[0].details == record.details

    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(json.loads(record.to_json()), json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
