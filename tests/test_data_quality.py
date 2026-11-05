"""Data-quality reference profile, policy, checks, DataQualityAnalyzer and their evidence records (family data_quality)"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.evidence import REASON_CODES, SCHEMA_PATH, Evidence, dumps, loads, to_evidence
from pulsar_metrics.exceptions import CustomExceptionPulsarMetric
from pulsar_metrics.quality import (
    DEFAULT_THRESHOLDS,
    NOT_EVALUABLE,
    DataQualityAnalyzer,
    DataQualityPolicy,
    DataQualityProfile,
    RowRule,
    check,
    check_categories,
    check_duplicates,
    check_nulls,
    check_ranges,
    check_rules,
    check_schema,
)

PRODUCED = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def frame(n=1000, seed=0, start="2026-01-01"):
    """Synthetic hourly weather-like frame: no person involved"""
    rng = np.random.default_rng(seed)
    temp = rng.normal(8, 9, n).round(1)
    return pd.DataFrame(
        {
            "model_id": "no2",
            "model_version": "1",
            "pred_timestamp": pd.date_range(start, periods=n, freq="h"),
            "temp": temp,
            "dew": (temp - rng.uniform(0, 6, n)).round(1),
            "rh": rng.integers(30, 101, n).astype(float),
            "station": rng.choice(["A", "B", "C"], n),
            "count": rng.integers(0, 10, n),
            "calm": rng.random(n) < 0.2,
        }
    )


@pytest.fixture(scope="module")
def reference():
    return frame(2000, 0)


@pytest.fixture(scope="module")
def profile(reference):
    return DataQualityProfile.fit(reference, rules=["dew <= temp", "rh <= 100"], reference_id="ref-2026-01")


def by_key(results):
    return {(r.metric_name, r.feature_name): r for r in results}


# ----------------------------------------------------------------------------------------------- profile
def test_profile_learns_types_nulls_ranges_domains_duplicates_and_rules(reference, profile):
    assert set(profile.columns) == {"temp", "dew", "rh", "station", "count", "calm"}  # metadata columns left out
    assert set(profile.ignored_columns) == {"model_id", "model_version", "pred_timestamp"}
    temp = profile.columns["temp"]
    assert (temp.kind, temp.null_share, temp.n_rows) == ("numeric", 0.0, 2000)
    assert temp.min == reference.temp.min() and temp.max == reference.temp.max()
    assert temp.quantiles["0.5"] == pytest.approx(reference.temp.median())
    assert profile.columns["station"].categories == ["A", "B", "C"] and profile.columns["station"].n_unique == 3
    assert profile.columns["calm"].kind == "boolean" and profile.columns["calm"].categories == ["False", "True"]
    assert profile.columns["count"].kind == "numeric"
    assert profile.duplicate_share == 0.0
    assert [(r.id, r.reference_share) for r in profile.rules] == [("dew <= temp", 0.0), ("rh <= 100", 0.0)]
    assert profile.reference_start == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert profile.n_rows == 2000 and profile.reference_id == "ref-2026-01"


def test_profile_keeps_the_kind_of_integer_and_boolean_columns_holding_missing_values():
    data = pd.DataFrame({"flag": [True, None, False, True], "n": [1, None, 3, 4], "label": ["x", "y", None, "x"]})
    profile = DataQualityProfile.fit(data)
    assert profile.columns["flag"].kind == "boolean"
    assert profile.columns["n"].kind == "numeric" and profile.columns["n"].null_share == 0.25
    assert profile.columns["label"].kind == "categorical"


def test_profile_keeps_no_domain_above_max_categories():
    data = pd.DataFrame({"id": [f"row-{i}" for i in range(60)], "level": ["a", "b"] * 30})
    profile = DataQualityProfile.fit(data, max_categories=50)
    assert profile.columns["id"].categories is None and profile.columns["id"].n_unique == 60
    assert profile.columns["level"].categories == ["a", "b"]


def test_profile_round_trips_through_json_and_yaml(profile, tmp_path):
    assert DataQualityProfile.from_json(profile.to_json()) == profile
    assert DataQualityProfile.from_yaml(profile.to_yaml()) == profile
    for name in ("profile.json", "profile.yaml"):
        path = profile.save(tmp_path / name)
        assert DataQualityProfile.load(path) == profile
    assert json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))["profile_version"] == "pulsar.dq-profile/1"


def test_profile_refuses_columns_and_rules_absent_from_the_reference(reference):
    with pytest.raises(CustomExceptionPulsarMetric):
        DataQualityProfile.fit(reference, columns=["temp", "pressure"])
    with pytest.raises(CustomExceptionPulsarMetric):
        DataQualityProfile.fit(reference, rules=["dew <= pressure"])


def test_row_rules_parse_columns_numbers_and_dicts():
    assert RowRule.parse("rh <= 100").right == 100.0
    assert RowRule.of({"id": "dew-below-temp", "expr": "dew<=temp"}).model_dump()["right"] == "temp"
    assert RowRule.of({"id": "r", "left": "a", "op": ">", "right": 0}).expression == "a > 0"
    with pytest.raises(CustomExceptionPulsarMetric):
        RowRule.parse("rh is 100")


# ----------------------------------------------------------------------------------------------- policy
def test_policy_defaults_are_the_reference_share_plus_a_margin():
    policy = DataQualityPolicy()
    assert policy.threshold("null_share", "rh", 0.02) == pytest.approx(0.07)
    assert policy.threshold("out_of_range_share", "temp") == DEFAULT_THRESHOLDS["out_of_range_share"]["margin"]
    assert policy.threshold("null_share", "rh", 0.99) == 1.0  # capped


def test_policy_per_check_and_per_column_thresholds_from_a_dict():
    policy = DataQualityPolicy.from_dict(
        {
            "null_share": {"max": 0.1},
            "out_of_range_share": 0.02,
            "duplicate_share": None,
            "columns": {"rh": {"null_share": {"margin": 0.3}}, "station": {"unseen_category_share": None, "schema": False}},
        }
    )
    assert policy.threshold("null_share", "temp", 0.05) == 0.1
    assert policy.threshold("null_share", "rh", 0.05) == pytest.approx(0.35)
    assert policy.threshold("out_of_range_share", "temp") == 0.02
    assert policy.threshold("duplicate_share", None) is None
    assert policy.threshold("unseen_category_share", "station") is None
    assert not policy.schema_checked("station") and policy.schema_checked("rh")
    assert DataQualityPolicy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()


def test_policy_from_a_yaml_file(tmp_path):
    path = tmp_path / "quality.yaml"
    path.write_text(
        "null_share: {margin: 0.1}\nrange_quantiles: [0.01, 0.99]\nallow_new_columns: true\n"
        "columns:\n  rh:\n    null_share: 0.4\n",
        encoding="utf-8",
    )
    policy = DataQualityPolicy.load(str(path))
    assert policy.threshold("null_share", "temp", 0.0) == 0.1
    assert policy.threshold("null_share", "rh", 0.0) == 0.4
    assert policy.range_quantiles == (0.01, 0.99) and policy.allow_new_columns


@pytest.mark.parametrize(
    "data",
    [
        {"nul_share": 0.1},  # typo
        {"null_share": 1.5},
        {"null_share": {"maximum": 0.1}},
        {"columns": {"rh": {"null_shar": 0.1}}},
        {"range_quantiles": [0.99, 0.01]},
    ],
)
def test_policy_refuses_unknown_keys_and_shares_out_of_bounds(data):
    with pytest.raises(CustomExceptionPulsarMetric):
        DataQualityPolicy.from_dict(data)


# ----------------------------------------------------------------------------------------------- checks: pass and alert
def test_a_clean_window_passes_every_check(profile):
    results = check(profile, frame(500, 1, "2026-03-01"))
    assert results and all(r.drift_status is False for r in results), [
        (r.metric_name, r.feature_name, r.metric_value, r.threshold) for r in results if r.drift_status is not False
    ]
    assert {r.metric_type for r in results} == {"data_quality"}


def test_schema_alerts_on_missing_new_and_retyped_columns(profile):
    current = frame(300, 2).drop(columns=["calm"]).assign(wind=1.0, count=lambda d: d["count"].astype(str))
    results = by_key(check_schema(profile, current))
    assert (results[("schema", "calm")].drift_status, results[("schema", "calm")].status) == (True, "missing_column")
    assert (results[("schema", "wind")].drift_status, results[("schema", "wind")].status) == (True, "new_column")
    assert results[("schema", "count")].status == "type_change" and "numeric" in results[("schema", "count")].method
    assert results[("schema", "temp")].drift_status is False and results[("schema", "temp")].threshold == 0
    allowed = by_key(check_schema(profile, current, {"allow_new_columns": True}))
    assert allowed[("schema", "wind")].drift_status is False
    ignored = by_key(check_schema(profile, current, {"columns": {"count": {"schema": False}}}))
    assert ("schema", "count") not in ignored


def test_schema_does_not_flag_an_integer_column_turned_float_by_missing_values(profile):
    current = frame(300, 3)
    current["count"] = current["count"].astype(float).mask(current.index % 10 == 0)
    assert by_key(check_schema(profile, current))[("schema", "count")].drift_status is False


def test_null_share_alerts_above_the_reference_share_plus_margin(profile):
    current = frame(400, 4)
    current.loc[current.index[:20], "rh"] = np.nan  # 5 %: at the margin, no alert
    assert by_key(check_nulls(profile, current))[("null_share", "rh")].drift_status is False
    current.loc[current.index[:60], "rh"] = np.nan  # 15 %
    result = by_key(check_nulls(profile, current))[("null_share", "rh")]
    assert (result.metric_value, result.threshold, result.drift_status) == (0.15, 0.05, True)
    relaxed = by_key(check_nulls(profile, current, {"columns": {"rh": {"null_share": 0.2}}}))[("null_share", "rh")]
    assert relaxed.drift_status is False


def test_out_of_range_share_catches_a_unit_change(profile):
    current = frame(400, 5)
    assert by_key(check_ranges(profile, current))[("out_of_range_share", "temp")].drift_status is False
    current["temp"] = current["temp"] * 9 / 5 + 32  # Celsius to Fahrenheit
    result = by_key(check_ranges(profile, current))[("out_of_range_share", "temp")]
    assert result.drift_status is True and result.metric_value > 0.3 and "above" in result.method


def test_range_from_quantiles_of_the_reference_moves_the_threshold_with_them(profile):
    results = by_key(check_ranges(profile, frame(2000, 6), {"range_quantiles": [0.01, 0.99]}))
    temp = results[("out_of_range_share", "temp")]
    assert temp.threshold == pytest.approx(0.03)  # 2 % outside by construction, plus the margin
    assert temp.metric_value == pytest.approx(0.02, abs=0.01) and temp.drift_status is False


def test_out_of_range_sets_missing_values_aside(profile):
    current = frame(400, 7)
    current.loc[current.index[:100], "temp"] = np.nan
    result = by_key(check_ranges(profile, current))[("out_of_range_share", "temp")]
    assert result.n == 300 and result.missing_share_current == 0.25


def test_unseen_category_share_catches_new_levels_without_copying_them(profile):
    current = frame(400, 8)
    assert by_key(check_categories(profile, current))[("unseen_category_share", "station")].drift_status is False
    current.loc[current.index[:40], "station"] = "Station-Z"
    result = by_key(check_categories(profile, current))[("unseen_category_share", "station")]
    assert (result.metric_value, result.drift_status) == (0.1, True)
    assert "1 level(s)" in result.method and "Station-Z" not in result.method  # counts only, never the values


def test_duplicate_share_on_all_columns_or_on_a_key(reference):
    profile = DataQualityProfile.fit(reference)
    current = frame(400, 9)
    assert check_duplicates(profile, current)[0].drift_status is False
    replayed = pd.concat([current, current.iloc[:40]], ignore_index=True)  # a batch sent twice
    result = check_duplicates(profile, replayed)[0]
    assert result.feature_name is None and result.metric_value == pytest.approx(40 / 440) and result.drift_status
    keyed = DataQualityProfile.fit(reference, duplicate_columns=["pred_timestamp"])
    assert keyed.duplicate_share == 0.0
    assert check_duplicates(keyed, replayed)[0].drift_status is True


def test_rule_violation_share(profile):
    current = frame(400, 10)
    assert all(r.drift_status is False for r in check_rules(profile, current))
    current.loc[current.index[:40], "dew"] = current["temp"][:40] + 2
    result = by_key(check_rules(profile, current))[("rule_violation_share", "dew <= temp")]
    assert (result.metric_value, result.drift_status, result.n) == (0.1, True, 400)


# ----------------------------------------------------------------------------------------------- checks: not evaluable
def test_an_empty_window_gives_not_evaluable_value_checks_and_a_schema_check(profile):
    results = check(profile, frame(10, 11).iloc[:0])
    schema = [r for r in results if r.metric_name == "schema"]
    assert schema and all(r.drift_status is False for r in schema)
    others = [r for r in results if r.metric_name != "schema"]
    assert others and all(r.status == "empty_window" and r.metric_value is None and r.drift_status is None for r in others)
    assert all(r.reliable is False and r.reliability_reason == NOT_EVALUABLE["empty_window"] for r in others)


def test_checks_that_cannot_run_say_why(profile):
    current = frame(300, 12).drop(columns=["rh"]).assign(wind=2.0, temp=lambda d: d["temp"].astype(str) + " C")
    current["station"] = np.nan
    results = by_key(check(profile, current))
    assert results[("null_share", "rh")].status == "missing_in_window"
    assert results[("out_of_range_share", "rh")].status == "missing_in_window"
    assert results[("rule_violation_share", "rh <= 100")].status == "missing_in_window"
    assert results[("null_share", "wind")].status == "not_in_reference"
    assert results[("out_of_range_share", "temp")].status == "type_mismatch"
    assert results[("unseen_category_share", "station")].status == "no_values_in_window"
    assert results[("null_share", "station")].drift_status is True  # the missing values are reported there
    assert results[("duplicate_share", None)].metric_value is not None  # on the columns still present


def test_reference_without_values_or_domain_is_not_evaluable():
    reference = pd.DataFrame({"x": [np.nan] * 5, "id": [f"r{i}" for i in range(5)]})
    profile = DataQualityProfile.fit(reference, max_categories=3)
    results = by_key(check(profile, pd.DataFrame({"x": [1.0, 2.0], "id": ["r1", "r9"]})))
    assert results[("out_of_range_share", "x")].status == "no_reference_values"
    assert results[("unseen_category_share", "id")].status == "no_reference_domain"


def test_disabled_checks_produce_no_result(profile):
    policy = {k: None for k in DEFAULT_THRESHOLDS} | {"schema": False}
    assert check(profile, frame(100, 13), policy) == []


# ----------------------------------------------------------------------------------------------- analyzer
def test_analyzer_learns_its_profile_from_the_first_reference_and_keeps_it(reference):
    analyzer = DataQualityAnalyzer("weekly quality", model_id="no2", model_version="1", rules=["dew <= temp"])
    analyzer.run(frame(300, 14, "2026-04-01"), reference)
    profile = analyzer.profile
    assert profile.n_rows == 2000 and all(r.drift_status is False for r in analyzer.get_result())
    analyzer.run(frame(300, 15, "2026-04-08"), frame(50, 99))  # a held profile is kept
    assert analyzer.profile is profile
    table = analyzer.results_to_pandas()
    assert {"metric_name", "feature_name", "metric_value", "threshold", "drift_status", "model_id"} <= set(table.columns)
    assert analyzer._metadata["period_start"] == pd.Timestamp("2026-04-08")


def test_analyzer_filters_on_the_model_and_records_an_empty_window(profile):
    analyzer = DataQualityAnalyzer("q", model_id="no2", model_version="1", profile=profile)
    other = frame(100, 16).assign(model_id="other")
    analyzer.run(other)
    assert {r.status for r in analyzer.get_result() if r.metric_name != "schema"} == {"empty_window"}
    with pytest.raises(CustomExceptionPulsarMetric):
        DataQualityAnalyzer("q").run(frame(10, 17))  # no profile, no reference


def test_analyzer_policy_and_per_run_options(profile):
    current = frame(300, 18)
    current.loc[current.index[:45], "rh"] = np.nan  # 15 %
    analyzer = DataQualityAnalyzer("q", profile=profile, policy={"columns": {"rh": {"null_share": 0.2}}})
    assert by_key(analyzer.run(current))[("null_share", "rh")].drift_status is False
    assert by_key(analyzer.run(current, options={}))[("null_share", "rh")].drift_status is True


# ----------------------------------------------------------------------------------------------- evidence
def test_evidence_records_of_the_checks_carry_family_threshold_direction_and_status(reference):
    analyzer = DataQualityAnalyzer("weekly quality", model_id="no2", model_version="1", rules=["dew <= temp"])
    analyzer.fit(reference, reference_id="ref-2026-01")
    current = frame(400, 19, "2026-05-01")
    current["temp"] = current["temp"] * 9 / 5 + 32
    analyzer.run(current)
    records = {(r.metric.name, r.metric.feature): r for r in to_evidence(analyzer, "SYS-0042", produced_at=PRODUCED)}
    assert len(records) == len(analyzer.get_result())
    assert {r.metric.family for r in records.values()} == {"data_quality"}

    temp = records[("out_of_range_share", "temp")]
    assert (temp.metric.status, temp.metric.direction, temp.metric.decision_basis) == ("alert", "upper", "value")
    assert temp.metric.threshold == 0.01 and temp.metric.value > 0.3 and temp.reliability.reliable
    assert temp.reliability.calibration.method == "nominal" and temp.reliability.notes == []
    rh = records[("null_share", "rh")]
    assert (rh.metric.status, rh.metric.threshold) == ("ok", 0.05)
    schema = records[("schema", "temp")]
    assert (schema.metric.status, schema.metric.value, schema.metric.threshold) == ("ok", 0, 0.0)

    assert temp.model_id == "no2" and temp.provenance.run_name == "weekly quality"
    assert temp.reference.id == "ref-2026-01" and temp.reference.n_rows == 2000
    assert temp.reference.start == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert temp.window.start == datetime(2026, 5, 1, tzinfo=timezone.utc) and temp.window.n_rows == 400
    assert loads(dumps(records.values()))[0] == next(iter(records.values()))


def test_evidence_of_checks_that_cannot_run_is_not_evaluable_with_a_reason_code(profile):
    analyzer = DataQualityAnalyzer("q", profile=profile)
    analyzer.run(frame(100, 20).drop(columns=["rh"]).assign(wind=1.0))
    records = {(r.metric.name, r.metric.feature): r for r in to_evidence(analyzer, "SYS", produced_at=PRODUCED)}
    missing = records[("null_share", "rh")]
    assert missing.metric.status == "not_evaluable" and missing.metric.value is None
    assert missing.reliability.reliable is False and missing.reliability.reasons == ["missing_in_window"]
    assert records[("null_share", "wind")].reliability.reasons == ["not_in_reference"]
    assert records[("schema", "rh")].metric.status == "alert" and records[("schema", "wind")].metric.status == "alert"
    assert records[("schema", "rh")].details["pulsar_status"] == "missing_column"
    for code in NOT_EVALUABLE:
        assert code in REASON_CODES

    analyzer.run(frame(10, 21).iloc[:0])
    with pytest.warns(UserWarning, match="without window start"):  # no row: no period
        empty = to_evidence(analyzer, "SYS")
    assert {r.reliability.reasons[0] for r in empty if r.metric.name != "schema"} == {"empty_window"}


def test_data_quality_records_validate_against_the_committed_schema(profile):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    analyzer = DataQualityAnalyzer("q", profile=profile)
    analyzer.run(frame(100, 22).drop(columns=["rh"]))
    for record in to_evidence(analyzer, "SYS", produced_at=PRODUCED):
        jsonschema.validate(record.to_dict(), schema)
        assert Evidence.from_json(record.to_json()) == record
