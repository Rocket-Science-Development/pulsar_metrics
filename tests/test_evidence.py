"""Evidence records ("pulsar.evidence/1"): conversion of every result kind, identity, JSON round trip, schema contract"""

import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from pulsar_metrics import evidence as ev
from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.analyzers.persistence import DriftPersistence
from pulsar_metrics.calibration import AACalibrator
from pulsar_metrics.evidence import (
    NOTE_CODES,
    REASON_CODES,
    SCHEMA_PATH,
    SCHEMA_VERSION,
    Evidence,
    Window,
    evidence_from_analyzer,
    evidence_from_result,
    evidence_from_results,
    json_schema,
    to_evidence,
)
from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy, comparative_analysis
from pulsar_metrics.genai import ComplianceMonitor, embedding_drift, model_equality
from pulsar_metrics.metrics.base import MetricResults
from pulsar_metrics.metrics.drift import DriftMetric, DriftTestMetric

PRODUCED = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def window(n, shift, start, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1, n)
    return pd.DataFrame(
        {
            "model_id": "credit_model",
            "model_version": "1.0",
            "pred_timestamp": pd.date_range(start, periods=n, freq="min"),
            "income": x,
            "channel": rng.choice(["web", "branch"], size=n, p=[0.7, 0.3] if shift == 0 else [0.4, 0.6]),
            "y_true": (x + rng.normal(0, 1, n) > 0).astype(int),
            "y_pred": (x > 0).astype(int),
        }
    )


def run(current, reference, options=None, performance=("accuracy", "f1")):
    analyzer = Analyzer(name="weekly check", model_id="credit_model", model_version="1.0")
    analyzer.add_drift_metrics(metrics_list=["ks_2samp", "wasserstein"], features_list=["income"])
    analyzer.add_drift_metrics(metrics_list=["chi2"], features_list=["channel"])
    if performance:
        analyzer.add_performance_metrics(metrics_list=list(performance))
    analyzer.run(current=current, reference=reference, options=options or {})
    return analyzer


@pytest.fixture(scope="module")
def analyzer():
    return run(
        window(500, 0.8, "2026-02-01", 1),
        window(1000, 0.0, "2026-01-01", 0),
        options={"wasserstein": {"threshold": 0.2}, "accuracy": {"threshold": 0.7}},
    )


def by_name(records):
    return {(r.metric.name, r.metric.feature): r for r in records}


# ----------------------------------------------------------------------------------------------- Analyzer results
def test_analyzer_results_become_records_with_window_reference_and_model(analyzer):
    records = evidence_from_analyzer(analyzer, "REG-0001", reference_id="ref-2026-01", produced_at=PRODUCED)
    assert len(records) == 5  # summary statistics left out by default
    for r in records:
        assert r.schema_version == SCHEMA_VERSION
        assert (r.system_id, r.model_id, r.model_version) == ("REG-0001", "credit_model", "1.0")
        assert r.window.start == datetime(2026, 2, 1, tzinfo=timezone.utc)
        assert r.window.n_rows == 500
        assert (r.reference.id, r.reference.n_rows) == ("ref-2026-01", 1000)
        assert r.reference.start == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert r.provenance.producer == "pulsar-metrics" and r.provenance.run_name == "weekly check"
        assert r.provenance.produced_at == PRODUCED

    records = by_name(records)
    ks = records[("ks_2samp", "income")]
    assert ks.metric.family == "drift" and ks.metric.status == "alert"
    assert (ks.metric.direction, ks.metric.decision_basis, ks.metric.threshold) == ("lower", "p_value", 0.05)
    assert ks.metric.p_value == ks.metric.value and ks.metric.n == 500
    assert ks.reliability.reliable and "nominal_threshold" in ks.reliability.notes
    assert ks.reliability.calibration.method == "nominal"

    wasserstein = records[("wasserstein", "income")]
    assert (wasserstein.metric.direction, wasserstein.metric.decision_basis) == ("upper", "value")

    accuracy = records[("accuracy", "prediction")]
    assert accuracy.metric.family == "performance" and accuracy.metric.status == "ok"
    assert accuracy.metric.direction == "lower"  # a minimum: alert below it
    assert accuracy.reliability.quality.n_labelled == 500

    f1 = records[("f1", "prediction")]
    assert (f1.metric.status, f1.metric.direction) == ("no_verdict", "none")
    assert f1.reliability.calibration.method == "none" and "no_threshold" in f1.reliability.notes


def test_summary_statistics_are_data_quality_records_with_the_reference_value_in_details(analyzer):
    records = evidence_from_analyzer(analyzer, "REG-0001", include_statistics=True)
    stats = [r for r in records if r.metric.family == "data_quality"]
    assert stats and all(r.metric.threshold is None and r.metric.status == "no_verdict" for r in stats)
    assert any("reference_value" in r.details for r in stats)


def test_upper_bound_from_the_options_sets_the_direction():
    analyzer = run(
        window(300, 0.0, "2026-02-01", 2),
        window(600, 0.0, "2026-01-01", 3),
        options={"wasserstein": {"threshold": 0.5, "upper_bound": False}},
        performance=(),
    )
    record = by_name(evidence_from_analyzer(analyzer, "S"))[("wasserstein", "income")]
    assert record.metric.direction == "lower"


# ----------------------------------------------------------------------------------------------- reliability
def test_calibrated_threshold_fills_the_calibration_block():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"score": rng.normal(size=3000)})
    metrics = [DriftTestMetric("ks_2samp", "score"), DriftMetric("wasserstein", "score")]
    calibrator = AACalibrator(window_size=100, n_draws=99, contiguous=False).fit(reference, metrics)
    current = pd.DataFrame({"score": rng.normal(size=100)})
    result = metrics[1].evaluate(current, reference, **calibrator.options()["wasserstein|score"])
    record = evidence_from_result(result, "S", options=calibrator.options()["wasserstein|score"], calibrator=calibrator)
    calibration = record.reliability.calibration
    assert calibration.method == "aa_calibration" and calibration.null_status == "ok"
    assert calibration.window_size == 100 and calibration.n_draws == 99 and calibration.quantile == 0.95
    assert calibration.threshold_low <= record.metric.threshold <= calibration.threshold_high
    assert 0 < calibration.empirical_p_value <= 1 and calibration.p_value_floor == pytest.approx(0.01)
    assert record.reliability.reliable and "nominal_threshold" not in record.reliability.notes


def test_window_far_from_the_calibrated_size_is_unreliable_with_a_code():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"score": rng.normal(size=3000)})
    metric = DriftTestMetric("ks_2samp", "score")
    calibrator = AACalibrator(window_size=100, n_draws=99, contiguous=False).fit(reference, [metric])
    options = calibrator.options()["ks_2samp|score"]
    result = metric.evaluate(pd.DataFrame({"score": rng.normal(size=20)}), reference, **options)
    assert result.reliable is False
    record = evidence_from_result(result, "S", options=options)
    assert not record.reliability.reliable
    assert record.reliability.reasons == ["window_size_mismatch"]
    assert record.reliability.reason_text == result.reliability_reason


def test_short_and_saturated_calibrations_get_their_codes():
    short = MetricResults(metric_type="drift", metric_name="psi", feature_name="x", metric_value=0.1, threshold=0.2)
    short.drift_status, short.reliable, short.n = False, False, 100
    record = evidence_from_result(short, "S", options={"threshold": 0.2, "reliable": False, "calibration_status": "ok"})
    assert record.reliability.reasons == ["short_reference"]

    saturated = MetricResults(
        metric_type="drift", metric_name="psi", feature_name="x", metric_value=27.0, threshold=27.0, status="saturated"
    )
    saturated.reliable, saturated.drift_status = False, False
    record = evidence_from_result(saturated, "S", options={"calibration_status": "saturated", "reliable": False})
    assert record.reliability.reasons == ["saturated_null"] and not record.reliability.reliable


def test_constant_reference_and_failed_metrics_are_not_reliable():
    constant = MetricResults(
        metric_type="drift", metric_name="ks_2samp", feature_name="x", metric_value=0.5, threshold=0.05, drift_status=False
    )
    constant.status = "constant_reference"
    assert evidence_from_result(constant, "S").reliability.reasons == ["constant_reference"]

    failed = MetricResults(metric_type="drift", metric_name="psi", feature_name="x", error="boom")
    record = evidence_from_result(failed, "S")
    assert record.metric.status == "error" and record.metric.error == "boom"
    assert record.reliability.reasons == ["computation_error"]


def test_performance_without_labels_is_not_evaluable_and_few_labels_unreliable():
    current = window(200, 0.0, "2026-02-01", 4)
    current.loc[10:, "y_true"] = np.nan
    with pytest.warns(UserWarning, match="labelled rows"):
        analyzer = run(current, window(400, 0.0, "2026-01-01", 5), performance=("accuracy",))
    accuracy = by_name(evidence_from_analyzer(analyzer, "S"))[("accuracy", "prediction")]
    assert accuracy.metric.status == "no_verdict" and accuracy.metric.n == 10
    assert accuracy.reliability.reasons == ["few_labelled_rows"]
    assert "partial_label_coverage" in accuracy.reliability.notes
    assert accuracy.reliability.quality.label_coverage == pytest.approx(0.05)

    current["y_true"] = np.nan
    analyzer = run(current, window(400, 0.0, "2026-01-01", 5), performance=("accuracy",))
    accuracy = by_name(evidence_from_analyzer(analyzer, "S"))[("accuracy", "prediction")]
    assert accuracy.metric.status == "not_evaluable"
    assert accuracy.reliability.reasons == ["no_ground_truth"]


def test_persistence_counts_and_unconfirmed_alert(analyzer):
    persistence = DriftPersistence(windows=2)
    persistence.update(analyzer.get_result())
    record = by_name(evidence_from_analyzer(analyzer, "S", persistence=persistence))[("ks_2samp", "income")]
    assert record.reliability.persistence.model_dump() == {
        "windows": 2,
        "drift_streak": 1,
        "no_drift_streak": 0,
        "confirmed": False,
        "changed": False,
    }
    assert "drift_not_confirmed" in record.reliability.notes

    verdicts = persistence.update(analyzer.get_result())  # second window in drift: confirmed
    record = by_name(evidence_from_analyzer(analyzer, "S", persistence=persistence))[("ks_2samp", "income")]
    assert record.reliability.persistence.confirmed and record.reliability.persistence.changed
    assert "drift_not_confirmed" not in record.reliability.notes
    from_list = by_name(evidence_from_analyzer(analyzer, "S", persistence=verdicts))[("ks_2samp", "income")]
    assert from_list.reliability.persistence.drift_streak == 2 and from_list.reliability.persistence.windows is None


# ----------------------------------------------------------------------------------------------- other result kinds
def test_fairness_results_and_coverage():
    rng = np.random.default_rng(0)
    n = 2000
    group = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.39, 0.01])
    data = pd.DataFrame({"group": group, "y_pred": (rng.random(n) < np.where(group == "B", 0.3, 0.5)).astype(int)})
    fairness = FairnessAnalyzer(["group"], policy=FairnessPolicy(n_bootstrap=100))
    fairness.run(data)
    records = to_evidence(fairness, "REG-0003", model_id="loans", model_version="3")
    assert {r.metric.family for r in records} == {"fairness"}
    judged = [r for r in records if r.metric.group == "group=B" and r.metric.name == "disparate_impact_ratio"][0]
    assert judged.metric.direction == "outside" and judged.metric.decision_basis == "confidence_interval"
    assert isinstance(judged.metric.threshold, list) and judged.metric.ci_low is not None
    assert judged.metric.status == "alert"
    suppressed = [r for r in records if r.metric.group == "group=C"]
    assert suppressed and all(r.metric.status == "not_evaluable" for r in suppressed)
    assert all(r.reliability.reasons == ["suppressed_group"] and r.reliability.quality.suppressed for r in suppressed)
    coverage = [r for r in records if r.metric.name == "no_verdict_share"]
    assert {r.metric.subject_metric for r in coverage} == {"disparate_impact_ratio", "demographic_parity_difference"}
    assert len({r.evidence_id for r in records}) == len(records)  # group in the identity


def test_genai_results_are_their_own_family():
    rng = np.random.default_rng(0)
    drift = embedding_drift(rng.normal(0, 1, (60, 8)), rng.normal(0.6, 1, (60, 8)), n_permutations=99)
    record = evidence_from_result(drift, "REG-0002", window={"n_rows": 60}, reference={"id": "canary", "n_rows": 60})
    assert record.metric.family == "genai" and record.metric.status == "alert"
    assert (record.metric.decision_basis, record.metric.direction) == ("p_value", "lower")
    assert record.metric.p_value == drift.p_value and record.metric.value == drift.metric_value

    answers = {f"p{i}": [f"answer {i} {j % 2}" for j in range(4)] for i in range(6)}
    equality = model_equality(answers, answers, n_permutations=50)
    record = evidence_from_result(equality, "REG-0002")
    assert record.metric.family == "genai" and record.metric.status == "ok"
    assert "small_canary" in record.reliability.notes


def test_compliance_results_and_not_evaluable_rules():
    monitor = ComplianceMonitor(
        [
            {"id": "auth-first", "template": "precedence", "events": ["authenticate", "create_file"]},
            {"id": "confirmed", "formula": "G (create_file -> @user_confirmation)"},
        ],
        version="rules-v1",
    )
    traces = [["authenticate", "create_file"], ["create_file"], ["authenticate"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = monitor.evaluate(traces)
    records = by_name(evidence_from_results(results, "REG-0004"))
    first = records[("rule_compliance", "auth-first")]
    assert first.metric.family == "compliance" and first.metric.status == "alert"
    assert (first.metric.direction, first.metric.decision_basis, first.metric.n) == ("lower", "value", 3)
    assert first.metric.value == pytest.approx(2 / 3)
    skipped = records[("rule_compliance", "confirmed")]
    assert skipped.metric.status == "not_evaluable" and skipped.reliability.reasons == ["rule_not_evaluable"]


# ----------------------------------------------------------------------------------------------- identity, JSON, schema
def test_evidence_id_is_deterministic_and_ignores_value_and_production_time(analyzer):
    first = evidence_from_analyzer(analyzer, "REG-0001", reference_id="r")
    again = evidence_from_analyzer(analyzer, "REG-0001", reference_id="r", produced_at=PRODUCED)
    assert [r.evidence_id for r in first] == [r.evidence_id for r in again]
    assert len({r.evidence_id for r in first}) == len(first)
    assert all(len(r.evidence_id) == 64 for r in first)
    other_system = evidence_from_analyzer(analyzer, "REG-0009", reference_id="r")
    assert not {r.evidence_id for r in first} & {r.evidence_id for r in other_system}

    changed = first[0].model_copy(deep=True)
    changed.metric.value = 0.5
    assert ev.evidence_id(changed) == first[0].evidence_id


def test_json_round_trip_keeps_every_field(analyzer):
    persistence = DriftPersistence()
    persistence.update(analyzer.get_result())
    records = evidence_from_analyzer(analyzer, "REG-0001", reference_id="r", persistence=persistence, include_statistics=True)
    for record in records:
        text = record.to_json()
        back = Evidence.from_json(text)
        assert back == record
        assert back.to_json() == text
        assert Evidence.from_dict(json.loads(text)) == record
    assert ev.loads(ev.dumps(records)) == records


def test_json_is_strict_json_and_dates_are_utc_iso(analyzer):
    record = evidence_from_analyzer(analyzer, "REG-0001", produced_at=PRODUCED)[0]
    data = json.loads(record.to_json())
    assert data["window"]["start"] == "2026-02-01T00:00:00Z"
    assert data["provenance"]["produced_at"] == "2026-11-02T12:00:00Z"
    nan = MetricResults(metric_type="custom", metric_name="mine", metric_value=float("nan"))
    record = evidence_from_result(nan, "S", details={"rate": np.float64("inf"), "counts": np.array([1, 2])})
    text = record.to_json()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["details"]["counts"] == [1, 2]


def test_reader_ignores_unknown_fields_of_a_later_additive_version(analyzer):
    data = evidence_from_analyzer(analyzer, "REG-0001")[0].to_dict()
    data["new_top_level_field"] = 1
    data["metric"]["new_metric_field"] = "x"
    record = Evidence.from_dict(data)
    assert record.evidence_id == data["evidence_id"]


def test_other_major_versions_are_refused(analyzer):
    data = evidence_from_analyzer(analyzer, "REG-0001")[0].to_dict()
    data["schema_version"] = "pulsar.evidence/2"
    with pytest.raises(ValidationError):
        Evidence.from_dict(data)


def test_details_are_capped():
    result = MetricResults(metric_type="custom", metric_name="mine", metric_value=1.0)
    with pytest.raises(ValidationError, match="aggregates only"):
        evidence_from_result(result, "S", details={"rows": list(range(20000))})


def test_committed_json_schema_matches_the_model():
    """The contract cannot drift silently: regenerate with pulsar_metrics.evidence.write_json_schema()"""
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == json_schema(), "schemas/evidence-1.json is stale: run pulsar_metrics.evidence.write_json_schema()"
    assert committed["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert {"evidence_id", "system_id", "metric", "window", "reference", "reliability", "provenance"} <= set(
        committed["required"]
    )


def test_records_validate_against_the_committed_schema(analyzer):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for record in evidence_from_analyzer(analyzer, "REG-0001", include_statistics=True):
        jsonschema.validate(record.to_dict(), schema)


def test_reason_and_note_codes_are_documented(analyzer):
    records = evidence_from_analyzer(analyzer, "REG-0001", include_statistics=True)
    for record in records:
        assert set(record.reliability.reasons) <= set(REASON_CODES)
        assert set(record.reliability.notes) <= set(NOTE_CODES)


# ----------------------------------------------------------------------------------------------- review fixes
LATER = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def psi(value=0.1, threshold=0.2, drift_status=False, **kwargs):
    return MetricResults(
        metric_type="drift",
        metric_name="psi",
        feature_name="a",
        metric_value=value,
        threshold=threshold,
        drift_status=drift_status,
        **kwargs,
    )


def test_records_without_window_do_not_share_an_id_across_productions():
    """No window bounds nor id: two windows would overwrite each other at the receiver"""
    with pytest.warns(UserWarning, match="window"):
        first = evidence_from_result(psi(), "S", produced_at=PRODUCED)
    with pytest.warns(UserWarning, match="window"):
        later = evidence_from_result(psi(), "S", produced_at=LATER)
    assert first.evidence_id != later.evidence_id
    # with a window id (or bounds) the same finding sent again keeps its id
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = [evidence_from_result(psi(), "S", window={"id": "batch-42"}, produced_at=t) for t in (PRODUCED, LATER)]
        other = evidence_from_result(psi(), "S", window={"id": "batch-43"}, produced_at=PRODUCED)
    assert again[0].evidence_id == again[1].evidence_id != other.evidence_id
    assert Evidence.from_json(first.to_json()).evidence_id == first.evidence_id


def test_compliance_rule_set_version_is_part_of_the_identity():
    rules = [{"id": "auth-first", "template": "precedence", "events": ["authenticate", "create_file"]}]
    traces = [["authenticate", "create_file"], ["create_file"]]
    records = []
    for version in ("rules-v1", "rules-v2"):
        results = ComplianceMonitor(rules, version=version).evaluate(traces)
        records += evidence_from_results(results, "REG-0004", window={"id": "week-45"})
    assert records[0].details["rules_version"] == "rules-v1"
    assert records[0].evidence_id != records[1].evidence_id


@pytest.mark.parametrize("zone, exported", [((-np.inf, np.inf), [None, None]), ((0.8, np.inf), [0.8, None])])
def test_unbounded_fairness_zone_is_exported_with_null_bounds(zone, exported):
    result = MetricResults(
        metric_type="fairness",
        metric_name="error_rate_difference",
        feature_name="sex",
        group="sex=F",
        reference_group="sex=M",
        metric_value=0.3,
        threshold=list(zone),
        drift_status=False,
    )
    record = evidence_from_result(result, "S", window={"id": "w"})
    assert record.metric.threshold == exported and record.metric.direction == "outside"
    assert Evidence.from_json(record.to_json()) == record
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(record.to_dict(), json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def test_fairness_analyzer_with_the_default_unbounded_zone_exports():
    rng = np.random.default_rng(0)
    n = 600
    data = pd.DataFrame({"sex": rng.choice(["F", "M"], n), "y_true": rng.integers(0, 2, n), "y_pred": rng.integers(0, 2, n)})
    fairness = FairnessAnalyzer(["sex"], policy=FairnessPolicy(criteria=["error_rate_difference"], n_bootstrap=50))
    fairness.run(data)
    records = to_evidence(fairness, "S", window={"id": "w"})
    assert any(r.metric.name == "error_rate_difference" for r in records)


def test_nat_window_bounds_are_missing_values():
    assert Window(start=pd.NaT, end=pd.Timestamp("2026-01-02")).start is None
    record = evidence_from_result(psi(), "S", window={"start": pd.NaT, "end": pd.NaT, "id": "w"})
    assert record.window.start is None and json.loads(record.to_json())["window"]["start"] is None


def test_failed_run_does_not_relabel_the_previous_results():
    analyzer = run(window(300, 0.5, "2026-02-01", 6), window(600, 0.0, "2026-01-01", 7), performance=())
    first = evidence_from_analyzer(analyzer, "S", produced_at=PRODUCED)
    bad = window(300, 0.0, "2026-03-01", 8).assign(model_id="other_model")  # no row of the analyzer's model
    with pytest.raises(Exception):
        analyzer.run(current=bad, reference=window(600, 0.0, "2026-01-01", 7))
    again = evidence_from_analyzer(analyzer, "S", produced_at=PRODUCED)
    assert [r.window for r in again] == [r.window for r in first]
    assert [r.evidence_id for r in again] == [r.evidence_id for r in first]


def test_position_effect_is_exported_as_the_binomial_test_it_is():
    from tests.test_fairness_paired import comparative_rows

    results = comparative_analysis(comparative_rows(300, p_first=0.5, p_prefer_a=0.5))
    records = {r.metric.name: r for r in evidence_from_results(results, "S", window={"id": "w"})}
    position = records["position_effect"]
    assert (position.metric.decision_basis, position.metric.direction, position.metric.threshold) == ("p_value", "lower", 0.05)
    assert position.metric.p_value is not None and position.metric.p_value < 0.05 and position.metric.status == "alert"
    assert position.details["null_value"] == 0.5
    unusable = records["unusable_rate"]
    assert (unusable.metric.decision_basis, unusable.metric.direction) == ("value", "upper")  # point estimate


def test_direction_follows_the_exported_threshold():
    record = evidence_from_result(psi(threshold=float("nan"), drift_status=None), "S", window={"id": "w"})
    assert record.metric.threshold is None and record.metric.direction == "none" and record.metric.decision_basis == "none"


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_value_with_a_verdict_is_not_reliable(value):
    record = evidence_from_result(psi(value=value, drift_status=True), "S", window={"id": "w"})
    assert record.metric.value is None and record.metric.status == "alert"
    assert not record.reliability.reliable and record.reliability.reasons == ["non_finite_value"]
    assert record.details["value_non_finite"] == str(value)


def test_unstable_explainability_reference_has_a_reason_code():
    from pulsar_metrics.explain import AttributionDrift
    from tests.test_explain import attributions

    scales = [1.0, 0.98, 0.96, 0.95, 0.94, 0.93]
    drift = AttributionDrift(k=3).fit(attributions(40, scales, seed=4))
    results = drift.run(attributions(40, scales, seed=5))
    record = [r for r in evidence_from_results(results, "S", window={"id": "w"}) if r.metric.name == "attribution_ndcg_at_k"][0]
    assert record.metric.status == "no_verdict"
    assert not record.reliability.reliable and record.reliability.reasons == ["unstable_reference"]


def test_fairness_criteria_without_labels_are_not_evaluable():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"group": rng.choice(["A", "B"], 400), "y_pred": rng.integers(0, 2, 400)})
    fairness = FairnessAnalyzer(["group"], policy=FairnessPolicy(criteria=["equal_opportunity_difference"], n_bootstrap=50))
    fairness.run(data)
    record = [r for r in to_evidence(fairness, "S", window={"id": "w"}) if r.metric.name == "equal_opportunity_difference"][0]
    assert record.metric.status == "not_evaluable" and record.reliability.reasons == ["no_ground_truth"]


def test_performance_without_a_label_column_is_not_evaluable():
    current = window(200, 0.0, "2026-02-01", 9).drop(columns="y_true")
    reference = window(400, 0.0, "2026-01-01", 10).drop(columns="y_true")
    analyzer = run(current, reference, performance=("accuracy",))
    accuracy = [r for r in evidence_from_analyzer(analyzer, "S") if r.metric.name == "accuracy"][0]
    assert accuracy.metric.status == "not_evaluable" and accuracy.reliability.reasons == ["no_ground_truth"]


@pytest.mark.parametrize("share, status", [(0.6, "alert"), (0.01, "ok")])
def test_coverage_above_the_policy_maximum_alerts(share, status):
    result = MetricResults(
        metric_type="fairness",
        metric_name="no_verdict_share",
        feature_name="sex",
        metric_value=share,
        threshold=0.05,
        method="disparate_impact_ratio: share of the rows outside the reference group in suppressed or inconclusive groups",
        test_type="fixed",
    )
    record = evidence_from_result(result, "S", window={"id": "w"})
    assert (record.metric.status, record.metric.direction, record.metric.decision_basis) == (status, "upper", "value")


def test_calibration_block_reports_a_nominal_threshold_used_despite_a_calibrator():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"score": rng.normal(size=3000)})
    metric = DriftMetric("wasserstein", "score")
    calibrator = AACalibrator(window_size=100, n_draws=99, contiguous=False).fit(reference, [metric])
    result = metric.evaluate(pd.DataFrame({"score": rng.normal(size=100)}), reference, threshold=0.5)
    record = evidence_from_result(result, "S", options={"threshold": 0.5}, calibrator=calibrator, window={"id": "w"})
    assert record.reliability.calibration.method == "nominal"
    assert record.reliability.calibration.n_draws is None


def test_incoming_evidence_id_is_checked(analyzer):
    data = evidence_from_analyzer(analyzer, "REG-0001")[0].to_dict()
    tampered = {**data, "metric": {**data["metric"], "feature": "another"}}
    with pytest.raises(ValidationError, match="evidence_id"):
        Evidence.from_dict(tampered)
    with pytest.raises(ValidationError, match="evidence_id"):
        Evidence.from_json(json.dumps(tampered))
    assert Evidence.from_dict(data).evidence_id == data["evidence_id"]


def test_json_reader_and_writer_refuse_nan(analyzer):
    data = evidence_from_analyzer(analyzer, "REG-0001")[0].to_dict()
    data["metric"]["p_value"] = float("nan")
    with pytest.raises(ValueError):
        Evidence.from_dict(data)
    text = json.dumps([data])  # json.dumps writes NaN by default
    with pytest.raises(ValueError, match="NaN"):
        ev.loads(text)
    record = evidence_from_analyzer(analyzer, "REG-0001")[0]
    record.metric.p_value = float("inf")  # assignment is not validated
    with pytest.raises(ValueError):
        ev.dumps([record])
