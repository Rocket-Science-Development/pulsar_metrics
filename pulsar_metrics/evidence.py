"""Evidence records: a stable, versioned and self-describing export of Pulsar results.

An evidence record is one finding about one monitored system: what was measured (metric, feature, value, threshold,
verdict), on which window and against which reference, how reliable the measurement is, and who produced it. It is
built from the existing result objects (`MetricResults`, the analyzers, `DriftPersistence`, `AACalibrator`) and does
not replace them: it is the format to hand results over to another system (a dashboard, an audit trail, a governance
service) without that system depending on pulsar_metrics.

The format is versioned by `schema_version` ("pulsar.evidence/1") and described by a JSON Schema shipped with the
package (`pulsar_metrics/schemas/evidence-1.json`, see `json_schema()`). Compatibility rules:

- additive changes keep "pulsar.evidence/1": a new optional field, a new value of an open vocabulary (family, reason
  and note codes), a new key in `details`. Consumers must ignore fields they do not know and treat unknown codes as
  unknown, not as errors;
- any other change is breaking and moves to "pulsar.evidence/2": a field removed, renamed or retyped, a field made
  required, a changed meaning of a value (e.g. of `status` or `direction`), a change of the fields hashed into
  `evidence_id`.

`details` holds aggregates only (counts, rates, statistics, settings): never raw rows, texts, identifiers of people or
any personal data. Its serialized size is capped (MAX_DETAILS_BYTES) as a guard against dumping data by mistake.

Records hold strict JSON only: NaN and infinities are refused by the models, by `loads()` and by `dumps()` (the
converters turn non-finite values into null, with a reason code for a value carrying a verdict).
"""

import hashlib
import json
import math
import re
import warnings
from datetime import datetime, timezone
from importlib import metadata
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .metrics import constant
from .metrics.base import MetricResults
from .metrics.drift import CONSTANT_REFERENCE, RATE_METRICS, SPARSE_SHARE, WINDOW_RATIO
from .metrics.enums import DriftTestMetricsFuncs
from .metrics.performance import MIN_LABELLED
from .quality.checks import METRIC_TYPE as DATA_QUALITY
from .quality.checks import NOT_EVALUABLE as DATA_QUALITY_NOT_EVALUABLE

SCHEMA_VERSION = "pulsar.evidence/1"
SCHEMA_ID = "https://github.com/Rocket-Science-Development/pulsar_metrics/schemas/evidence-1.json"
SCHEMA_PATH = Path(__file__).parent / "schemas" / "evidence-1.json"
PRODUCER = "pulsar-metrics"
MAX_DETAILS_BYTES = 64 * 1024  # details above this size are refused: aggregates only

Family = Literal["drift", "performance", "fairness", "genai", "compliance", "data_quality", "explainability", "custom"]
Status = Literal["ok", "alert", "no_verdict", "not_evaluable", "error"]
Direction = Literal["upper", "lower", "outside", "none"]
DecisionBasis = Literal["value", "p_value", "confidence_interval", "reference_test", "sequential_test", "none"]
CalibrationMethod = Literal["aa_calibration", "nominal", "none"]

# metric_type of MetricResults -> evidence family
_FAMILIES = {
    "drift": "drift",
    "performance": "performance",
    "fairness": "fairness",
    "agents": "compliance",
    "statistics": "data_quality",
    "data_quality": "data_quality",  # pulsar_metrics.quality checks
    "explainability": "explainability",
    "custom": "custom",
}
# drift results of pulsar_metrics.genai (embedding_drift, model_equality): value is a statistic, verdict on p_value
GENAI_METRICS = (
    "embedding_mean_shift",
    "embedding_novelty",
    "model_equality",
    "consistency_drift",
    "consistency_full_agreement_drift",
)
# window metrics of pulsar_metrics.genai.consistency (metric_type custom): value with an interval, minimum acceptable
CONSISTENCY_METRICS = ("consistency_agreement", "consistency_full_agreement")
CONSISTENCY_STATUSES = ("few_repeats", "few_questions")  # pulsar_metrics.genai.consistency: raw status -> reason code
SMALL_CANARY_ANSWERS = 480  # pulsar_metrics.genai.RECOMMENDED_CANARY["answers"]
POINT_ESTIMATE_DECISIONS = ("unusable_rate",)  # fairness results decided on the value, not on their interval
UNSTABLE_REFERENCE = "unstable_reference"  # pulsar_metrics.explain: the ranking is not stable on the reference
_NULL_STATUSES = {"saturated": "saturated_null", "degenerate_null": "degenerate_null", CONSTANT_REFERENCE: CONSTANT_REFERENCE}

# Codes of Reliability.reasons: each one makes `reliable` False
REASON_CODES = {
    "computation_error": "the metric could not be computed (see metric.error)",
    "no_ground_truth": "performance or fairness criterion that needs labels: no labelled row (or no label column)",
    "insufficient_labels": "performance: too few labelled rows, too low a label coverage, or a single class",
    "few_labelled_rows": f"performance: value computed on fewer than {MIN_LABELLED} labelled rows",
    "constant_reference": "the reference of the feature holds a single value: thresholds are degenerate",
    "saturated_null": "the calibrated null sits at the bound of the statistic: a drift beyond it cannot be seen",
    "degenerate_null": "the calibrated null takes at most 2 distinct values",
    "window_size_mismatch": f"the window is more than {WINDOW_RATIO}x smaller or larger than the calibrated window",
    "short_reference": "the calibration reference holds too few windows: the threshold varies a lot between references",
    "suppressed_group": "fairness: group below the minimum size, value withheld",
    "rule_not_evaluable": "compliance: no run carries what the rule reads (arguments, statuses, events)",
    "instrument_unusable": "fairness probe: the instrument cannot tell the variants apart",
    "flagged_unreliable": "Pulsar flagged the result reliable=False (see reason_text)",
    "non_finite_value": "the value is NaN or infinite (exported as null, see details.value_non_finite): the verdict "
    "cannot be checked against the threshold",
    "unstable_reference": "explainability: the attribution ranking is not stable on the reference window, no verdict",
    **DATA_QUALITY_NOT_EVALUABLE,  # pulsar_metrics.quality: empty_window, not_in_reference, missing_in_window, ...
    "few_repeats": "consistency: too few answers per question (median below min_repeats, default 3)",
    "few_questions": "consistency: too few questions in a window (below min_questions, default 10)",
}
# Codes of Reliability.notes: they lower the confidence without making `reliable` False
NOTE_CODES = {
    "nominal_threshold": "the threshold is a nominal value (e.g. alpha 0.05), not calibrated on A/A draws",
    "no_threshold": "no threshold was given: value only, no verdict",
    "sparse_cells": f"chi2 / psi / kl: more than {SPARSE_SHARE:.0%} of the cells have an expected count below 5",
    "missing_values_set_aside": "missing values (NaN or declared sentinels) were set aside before the computation",
    "partial_label_coverage": "performance: evaluated on the labelled rows only",
    "drift_not_confirmed": "alert in this window, not yet confirmed over consecutive windows (DriftPersistence)",
    "ecological_inference": "fairness: groups from a composition table, not from individual attributes",
    "small_canary": f"model equality: fewer than {SMALL_CANARY_ANSWERS} answers, a change is usually missed",
}


def _utc(value) -> Optional[datetime]:
    """Datetime (or ISO string, pandas Timestamp) in UTC; naive datetimes are taken as UTC; None, NaN and NaT give None"""
    # NaT is a datetime instance: the missing-value test comes first
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    elif isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat().replace("+00:00", "Z")


def _jsonable(value):
    """JSON-safe copy: numpy scalars and arrays to Python, NaN and infinities to None, datetimes to ISO 8601"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, datetime):
        return _iso(_utc(value))
    if value is None or isinstance(value, (str, int, float)):
        return value
    return str(value)


def _number(value) -> Optional[Union[int, float]]:
    value = _jsonable(value)
    if isinstance(value, bool) or not isinstance(value, Number):
        return None
    return value


class _Part(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_serialization_defaults_required=True, allow_inf_nan=False)


class Metric(_Part):
    """What was measured and the verdict"""

    family: Family = Field(description="Kind of finding: drift, performance, fairness, genai, compliance, data_quality, ...")
    name: str = Field(description="Metric name, e.g. ks_2samp, psi, accuracy, disparate_impact_ratio, mmd, rule_compliance")
    feature: Optional[str] = Field(None, description="Feature, model output or rule id the metric is about (None: whole frame)")
    group: Optional[str] = Field(None, description="Fairness: group label, e.g. 'sex=F' or 'sex=F|region=North'")
    reference_group: Optional[str] = Field(None, description="Fairness: group the group is compared with")
    subject_metric: Optional[str] = Field(
        None, description="Metric this one is about, e.g. the fairness criterion of a no_verdict_share coverage record"
    )
    value: Optional[Union[int, float, str]] = Field(None, description="Value of the metric (p-value for statistical tests)")
    threshold: Optional[Union[float, list[Optional[float]]]] = Field(
        None,
        description="Threshold or alpha; [low, high] for an acceptable zone, a null bound being unbounded (e.g. "
        "[0.8, null]). None: no threshold",
    )
    direction: Direction = Field(
        "none",
        description="upper: alert above the threshold; lower: alert below it; outside: alert outside [low, high]; "
        "none: no threshold",
    )
    decision_basis: DecisionBasis = Field(
        "none",
        description="What the threshold is compared with: value, p_value, confidence_interval (the interval must lie "
        "beyond the threshold), reference_test (test against the reference period), sequential_test (anytime-valid)",
    )
    p_value: Optional[float] = Field(None, description="p-value of the test, when there is one (BH-adjusted for compliance)")
    ci_low: Optional[float] = Field(None, description="Lower bound of the confidence interval of the value")
    ci_high: Optional[float] = Field(None, description="Upper bound of the confidence interval of the value")
    e_value: Optional[float] = Field(None, description="Sequential tests: e-value (evidence against no disparity)")
    n: Optional[int] = Field(None, description="Rows, units or runs behind the value")
    status: Status = Field(
        description="ok: no drift / degradation; alert: drift, degradation or violation detected; no_verdict: value "
        "without verdict (no threshold, inconclusive); not_evaluable: the data does not allow a value; error: failed"
    )
    error: Optional[str] = Field(None, description="Why the value or the verdict is missing")
    method: Optional[str] = Field(None, description="How the value and its interval were obtained (human-readable)")


class Window(_Part):
    """Current window the metric was computed on"""

    id: Optional[str] = Field(
        None,
        description="Identifier of the window chosen by the caller (batch, run or period id), hashed into evidence_id; "
        "needed when the window has no start and end",
    )
    start: Optional[datetime] = Field(None, description="First timestamp of the window (UTC, ISO 8601)")
    end: Optional[datetime] = Field(None, description="Last timestamp of the window (UTC, ISO 8601)")
    n_rows: Optional[int] = Field(None, description="Rows (predictions, texts, runs) in the window")

    @field_validator("start", "end", mode="before")
    @classmethod
    def _in_utc(cls, v):
        return _utc(v)


class Reference(_Part):
    """Reference the window is compared with"""

    id: Optional[str] = Field(None, description="Identifier of the reference dataset or period, chosen by the user")
    start: Optional[datetime] = Field(None, description="First timestamp of the reference (UTC, ISO 8601)")
    end: Optional[datetime] = Field(None, description="Last timestamp of the reference (UTC, ISO 8601)")
    n_rows: Optional[int] = Field(None, description="Rows in the reference")

    @field_validator("start", "end", mode="before")
    @classmethod
    def _in_utc(cls, v):
        return _utc(v)


class Calibration(_Part):
    """Where the threshold comes from"""

    method: CalibrationMethod = Field(
        "none", description="aa_calibration: quantile of A/A draws (AACalibrator); nominal: given value; none: no threshold"
    )
    null_status: Optional[str] = Field(
        None, description="Status of the calibrated null: ok, constant_reference, degenerate_null, saturated"
    )
    window_size: Optional[int] = Field(None, description="Window size the threshold was calibrated for")
    quantile: Optional[float] = Field(None, description="Quantile of the A/A distribution used as threshold")
    n_reference_windows: Optional[float] = Field(None, description="Reference rows / window_size")
    n_draws: Optional[int] = Field(None, description="A/A draws behind the null distribution")
    n_effective_draws: Optional[int] = Field(None, description="Draws carrying independent information")
    share_at_bound: Optional[float] = Field(None, description="Share of A/A draws at the bound of the statistic")
    threshold_low: Optional[float] = Field(None, description="Lower bound of the 90 % interval of the calibrated threshold")
    threshold_high: Optional[float] = Field(None, description="Upper bound of the 90 % interval of the calibrated threshold")
    empirical_p_value: Optional[float] = Field(None, description="Share of A/A draws at least as extreme as the value")
    p_value_floor: Optional[float] = Field(None, description="Smallest empirical p-value the null can give")


class Persistence(_Part):
    """Verdict over consecutive windows (DriftPersistence)"""

    windows: Optional[int] = Field(None, description="Consecutive windows needed to confirm or clear a drift")
    drift_streak: int = Field(description="Consecutive windows with an alert, this one included")
    no_drift_streak: int = Field(description="Consecutive windows without alert, this one included")
    confirmed: bool = Field(description="Drift confirmed after the persistence rule")
    changed: bool = Field(description="The confirmed verdict switched in this window")


class Quality(_Part):
    """Data-quality aggregates behind the value"""

    missing_share_current: Optional[float] = Field(None, description="Share of missing values set aside in the window")
    missing_share_reference: Optional[float] = Field(None, description="Share of missing values set aside in the reference")
    low_expected_share: Optional[float] = Field(None, description="chi2 / psi / kl: share of cells with expected count < 5")
    n_labelled: Optional[int] = Field(None, description="Performance: rows with ground truth and prediction")
    label_coverage: Optional[float] = Field(None, description="Performance: n_labelled / rows of the window")
    suppressed: Optional[bool] = Field(None, description="Fairness: value withheld (group below the minimum size)")


class Reliability(_Part):
    """Whether the measurement can be trusted, and why not"""

    reliable: bool = Field(
        description="False when any code is listed in reasons: the verdict is provisional or the value missing"
    )
    reasons: list[str] = Field(default_factory=list, description="Codes that make reliable False (see REASON_CODES)")
    notes: list[str] = Field(default_factory=list, description="Codes that lower the confidence only (see NOTE_CODES)")
    reason_text: Optional[str] = Field(None, description="Human-readable reliability_reason of Pulsar")
    calibration: Calibration = Field(default_factory=Calibration)
    persistence: Optional[Persistence] = None
    quality: Quality = Field(default_factory=Quality)


class Provenance(_Part):
    """Who produced the record, and when"""

    producer: str = Field(PRODUCER, description="Producing package")
    producer_version: str = Field(description="Version of the producing package")
    produced_at: datetime = Field(description="Time the record was produced (UTC, ISO 8601)")
    run_name: Optional[str] = Field(None, description="Name of the analysis that produced the result")

    @field_validator("produced_at", mode="before")
    @classmethod
    def _in_utc(cls, v):
        return _utc(v)


class Evidence(_Part):
    """One finding about one monitored system, in the "pulsar.evidence/1" format"""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_serialization_defaults_required=True,
        allow_inf_nan=False,
        protected_namespaces=(),  # model_id, model_version
        title="Pulsar evidence record",
    )

    schema_version: Literal["pulsar.evidence/1"] = Field(SCHEMA_VERSION, description="Format and major version")
    evidence_id: str = Field(
        "",
        description="sha256 of the identifying content (schema major version, system, model, metric, feature, group, "
        "subject metric, window, reference, compliance rules version; produced_at when the window has no start, end "
        "nor id): the same finding sent twice has the same id. Checked when a record is read",
    )
    system_id: str = Field(description="External identifier of the monitored system, supplied by the user")
    model_id: Optional[str] = Field(None, description="Model identifier")
    model_version: Optional[str] = Field(None, description="Model version")
    metric: Metric
    window: Window = Field(default_factory=Window)
    reference: Reference = Field(default_factory=Reference)
    reliability: Reliability
    provenance: Provenance
    details: dict[str, Any] = Field(
        default_factory=dict, description="Free aggregates (counts, rates, settings). Never raw rows or personal data"
    )

    @field_validator("model_id", "model_version", mode="before")
    @classmethod
    def _as_text(cls, v):
        return None if v is None else str(v)

    @field_validator("details", mode="before")
    @classmethod
    def _aggregates_only(cls, v):
        v = _jsonable(v or {})
        size = len(json.dumps(v))
        if size > MAX_DETAILS_BYTES:
            raise ValueError(f"details holds {size} bytes, more than {MAX_DETAILS_BYTES}: details are for aggregates only")
        return v

    @model_validator(mode="after")
    def _identify(self):
        expected = evidence_id(self)
        if self.evidence_id and self.evidence_id != expected:
            raise ValueError(
                f"evidence_id {self.evidence_id[:12]}... does not match the identifying content of the record "
                f"({expected[:12]}...): the record was altered, or its id computed otherwise"
            )
        self.evidence_id = expected
        return self

    # ------------------------------------------------------------------ JSON round trip
    def to_dict(self) -> dict:
        """JSON-compatible dict (datetimes as ISO 8601 strings)"""
        return self.model_dump(mode="json")

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        """Record of a dict; ValidationError when its evidence_id does not match its content, or it holds NaN"""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "Evidence":
        """Record of a JSON object; ValidationError when its evidence_id does not match its content, or it holds NaN"""
        return cls.model_validate_json(text)


def evidence_id(record: Evidence) -> str:
    """sha256 of the identifying content of a record

    Hashed: schema major version, system_id, model_id, model_version, metric family, name, feature, group,
    reference group and subject metric, window id, start and end, reference id, start and end, and for compliance
    records the rules version (details.rules_version). Not hashed: the value, the verdict, the reliability, the other
    details and the provenance: a recomputation of the same metric on the same window has the same id (receivers keep
    the record with the latest provenance.produced_at).

    A window with no start, end nor id cannot tell two windows apart: provenance.produced_at is then hashed too, so
    that a later window does not overwrite an earlier one at the receiver (the converters warn; pass window={"id": ...}
    to keep the same id when the same finding is sent again).
    """
    identity = {
        "schema": SCHEMA_VERSION,
        "system_id": record.system_id,
        "model_id": record.model_id,
        "model_version": record.model_version,
        "family": record.metric.family,
        "name": record.metric.name,
        "feature": record.metric.feature,
        "group": record.metric.group,
        "reference_group": record.metric.reference_group,
        "subject_metric": record.metric.subject_metric,
        "window": [record.window.id, _iso(record.window.start), _iso(record.window.end)],
        "reference": [record.reference.id, _iso(record.reference.start), _iso(record.reference.end)],
        "rules_version": record.details.get("rules_version") if record.metric.family == "compliance" else None,
    }
    if _no_window(record.window):
        identity["produced_at"] = _iso(record.provenance.produced_at)
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _no_window(window: Window) -> bool:
    return window.id is None and window.start is None and window.end is None


def json_schema() -> dict:
    """JSON Schema of the records, as committed in pulsar_metrics/schemas/evidence-1.json"""
    schema = Evidence.model_json_schema(mode="serialization")
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_ID, **schema}


def write_json_schema(path: Union[str, Path] = SCHEMA_PATH) -> Path:
    """Write json_schema() to `path` (default: the file shipped with the package)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _producer_version() -> str:
    """Version of the code that runs: the source tree's pyproject.toml when the package is imported from a checkout
    (an editable install keeps the metadata of the version it was installed at), else the installed metadata"""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8") if pyproject.is_file() else ""
    if f'name = "{PRODUCER}"' in text:  # this package's own checkout, not another project's pyproject
        for line in text.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "version" and value.strip():
                return value.strip().strip("\"'")
    try:
        return metadata.version(PRODUCER)
    except metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------------------------- converters
def _family(result: MetricResults) -> str:
    if result.metric_type == "drift" and (
        result.metric_name in GENAI_METRICS or (result.metric_name == "mmd" and result.p_value is not None)
    ):
        return "genai"  # pulsar_metrics.genai: embedding_drift, model_equality, consistency_drift
    if result.metric_name in CONSISTENCY_METRICS:
        return "genai"  # pulsar_metrics.genai.consistency
    return _FAMILIES.get(result.metric_type, "custom")


def _decision(result: MetricResults, family: str, threshold, upper_bound: Optional[bool]) -> tuple:
    """(direction, decision_basis) of a result, from the threshold as exported"""
    if threshold is None:
        return "none", "none"
    if result.test_type == "sequential":
        return "outside", "sequential_test"
    if result.metric_name in CONSISTENCY_METRICS:
        return "lower", "value"  # minimum acceptable consistency: alert below it
    if _position_test(result):
        return "lower", "p_value"  # two-sided binomial test against 0.5: alert when its p-value is below alpha
    if result.metric_name in DriftTestMetricsFuncs._member_names_ or family == "genai":
        return "lower", "p_value"
    if family == "compliance":
        return ("lower", "reference_test") if result.p_value is not None else ("lower", "value")
    interval = result.ci_low is not None or result.ci_high is not None
    interval = interval and result.metric_name not in POINT_ESTIMATE_DECISIONS
    basis = "confidence_interval" if family in ("fairness", "explainability") and interval else "value"
    if isinstance(threshold, list):
        return "outside", basis
    if family == "explainability":
        return "lower", basis  # rank stability: at least the threshold
    if upper_bound is None:
        upper_bound = not (family == "performance" and result.metric_name in constant.HIGHER_IS_BETTER_METRICS)
    return ("upper" if upper_bound else "lower"), basis


def _subject_metric(result: MetricResults) -> Optional[str]:
    """FairnessAnalyzer.coverage(): the criterion is the head of `method` ("disparate_impact_ratio: share of ...")"""
    if result.metric_name == "no_verdict_share" and result.method and ":" in result.method:
        return result.method.split(":", 1)[0].strip()
    return None


def _position_test(result: MetricResults) -> bool:
    """pulsar_metrics.fairness comparative_analysis: position_effect is decided by a binomial test (its p_value)"""
    return result.metric_name == "position_effect" and result.p_value is not None


def _alpha(method: Optional[str]) -> Optional[float]:
    match = re.search(r"alpha = ([0-9.eE+-]+)", method or "")
    return float(match.group(1)) if match else None


_RATES = re.compile(r"window (\d+)/(\d+) = [^ ]+ vs reference (\d+)/(\d+)")


def _rates(result: MetricResults) -> dict:
    """rate_increase (DriftTestMetric): the counts and rates of the window and of the reference, from `method`"""
    match = _RATES.search(result.method or "") if result.metric_name in RATE_METRICS else None
    if match is None:
        return {}
    count, n, ref_count, ref_n = (int(g) for g in match.groups())
    return {
        "current_count": count,
        "current_n": n,
        "current_rate": count / n,
        "reference_count": ref_count,
        "reference_n": ref_n,
        "reference_rate": ref_count / ref_n,
    }


def _rules_version(method: Optional[str]) -> Optional[str]:
    """ComplianceMonitor.evaluate: the rules version is in the method ("<formula> [<severity>; rules <version>]...")"""
    match = re.search(r"; rules ([^\]]+)\]", method or "")
    return match.group(1) if match else None


def _finite_number(value) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool) and math.isfinite(value)


def _threshold(threshold):
    """Threshold as exported: a non-finite scalar is no threshold, a non-finite bound of a zone is unbounded (None)"""
    if isinstance(threshold, list):
        return [_number(t) for t in threshold]
    return _number(threshold)


def _data_quality_not_evaluable(result: MetricResults) -> bool:
    return result.metric_type == DATA_QUALITY and result.status in DATA_QUALITY_NOT_EVALUABLE


def _status(result: MetricResults) -> str:
    if _data_quality_not_evaluable(result):
        return "not_evaluable"
    if result.suppressed or result.status == "not_evaluable":
        return "not_evaluable"
    if result.error is not None and result.metric_value is None and result.status in CONSISTENCY_STATUSES:
        return "not_evaluable"  # consistency: no question with enough answers
    if result.error is not None and result.metric_value is None:
        # results refused for their labels carry n_labelled (performance, fairness); failures of the computation do not
        return "not_evaluable" if result.n_labelled is not None else "error"
    if result.drift_status is True:
        return "alert"
    if result.drift_status is False:
        return "ok"
    if result.metric_name == "no_verdict_share" and _finite_number(result.metric_value) and _finite_number(result.threshold):
        # FairnessAnalyzer.coverage(): the threshold is the policy maximum of rows without verdict
        return "alert" if result.metric_value > result.threshold else "ok"
    return "no_verdict"


def _persistence(result: MetricResults, persistence) -> Optional[Persistence]:
    if persistence is None:
        return None
    key = (result.metric_type, result.metric_name, result.feature_name)
    state = getattr(persistence, "_state", None)
    if state is not None:  # DriftPersistence
        s = state.get(key)
        if s is None:
            return None
        return Persistence(
            windows=persistence.windows,
            drift_streak=s["drift_streak"],
            no_drift_streak=s["no_drift_streak"],
            confirmed=s["confirmed"],
            changed=s.get("changed", False),
        )
    for verdict in persistence:  # PersistentDrift list, as returned by DriftPersistence.update()
        if (verdict.metric_type, verdict.metric_name, verdict.feature_name) == key:
            return Persistence(
                drift_streak=verdict.drift_streak,
                no_drift_streak=verdict.no_drift_streak,
                confirmed=verdict.confirmed_drift,
                changed=verdict.changed,
            )
    return None


def _calibration(result: MetricResults, options: dict, calibrator, threshold) -> tuple:
    """(Calibration, short_reference): short_reference when the calibration is flagged unreliable for its reference
    length alone (null status ok)"""
    null = None
    if calibrator is not None:
        null = getattr(calibrator, "nulls", {}).get((result.metric_name, result.feature_name))
    if null is not None and not (_finite_number(threshold) and math.isclose(threshold, null.threshold, rel_tol=1e-9)):
        null = None  # the result was evaluated with another threshold than the calibrated one: report what was used
    if null is not None:
        low, high = null.threshold_interval()
        value = result.metric_value if isinstance(result.metric_value, Number) else None
        calibration = Calibration(
            method="aa_calibration",
            null_status=null.status,
            window_size=null.window_size or getattr(calibrator, "window_size", None),
            quantile=null.quantile,
            n_reference_windows=_number(null.n_windows),
            n_draws=len(null.values),
            n_effective_draws=_number(null.n_effective),
            share_at_bound=_number(null.share_at_bound),
            threshold_low=_number(low),
            threshold_high=_number(high),
            empirical_p_value=None if value is None else _number(null.empirical_pvalue(value)),
            p_value_floor=_number(null.pvalue_floor),
        )
        return calibration, null.status == "ok" and not null.reliable
    if "calibration_status" in options or "window_size" in options:
        calibration = Calibration(
            method="aa_calibration", null_status=options.get("calibration_status"), window_size=options.get("window_size")
        )
        return calibration, options.get("reliable") is False and calibration.null_status in (None, "ok")
    return Calibration(method="none" if threshold is None else "nominal"), False


def _reliability(
    result: MetricResults,
    family: str,
    status: str,
    calibration: Calibration,
    short_reference: bool,
    persistence,
    non_finite: bool = False,
) -> Reliability:
    reasons, notes = [], []

    def add(codes, code):
        if code not in codes:
            codes.append(code)

    if status == "error":
        add(reasons, "computation_error")
    if result.suppressed:
        add(reasons, "suppressed_group")
    if result.status == "not_evaluable":
        add(reasons, "rule_not_evaluable")
    if result.status == "instrument_unusable":
        add(reasons, "instrument_unusable")
    if result.status == UNSTABLE_REFERENCE:
        add(reasons, UNSTABLE_REFERENCE)
    if _data_quality_not_evaluable(result) or result.status in CONSISTENCY_STATUSES:
        add(reasons, result.status)
    if non_finite and status in ("ok", "alert", "no_verdict"):
        add(reasons, "non_finite_value")
    if family in ("performance", "fairness") and result.n_labelled is not None:
        if result.metric_value is None and result.error is not None:
            add(reasons, "no_ground_truth" if result.n_labelled == 0 else "insufficient_labels")
        elif result.n_labelled < MIN_LABELLED:
            add(reasons, "few_labelled_rows")
    for raw in (result.status, calibration.null_status):
        if raw in _NULL_STATUSES:
            add(reasons, _NULL_STATUSES[raw])
    window_size = calibration.window_size
    if window_size and result.n is not None and family == "drift":
        if result.n * WINDOW_RATIO < window_size or result.n > WINDOW_RATIO * window_size:
            add(reasons, "window_size_mismatch")
    if short_reference:
        add(reasons, "short_reference")
    if result.reliable is False and not reasons:
        add(reasons, "flagged_unreliable")

    if calibration.method == "nominal" and family in ("drift", "genai"):
        add(notes, "nominal_threshold")
    if calibration.method == "none" and status in ("ok", "alert", "no_verdict") and family != "data_quality":
        add(notes, "no_threshold")
    if result.low_expected_share is not None and result.low_expected_share > SPARSE_SHARE:
        add(notes, "sparse_cells")
    if (result.missing_share_current or 0) > 0 or (result.missing_share_reference or 0) > 0:
        add(notes, "missing_values_set_aside")
    if result.label_coverage is not None and result.label_coverage < 1 and result.metric_value is not None:
        add(notes, "partial_label_coverage")
    if result.status == "ecological":
        add(notes, "ecological_inference")
    if result.metric_name == "model_equality" and result.n is not None and result.n < SMALL_CANARY_ANSWERS:
        add(notes, "small_canary")
    if persistence is not None and status == "alert" and not persistence.confirmed:
        add(notes, "drift_not_confirmed")

    return Reliability(
        reliable=not reasons,
        reasons=reasons,
        notes=notes,
        reason_text=result.reliability_reason,
        calibration=calibration,
        persistence=persistence,
        quality=Quality(
            missing_share_current=_number(result.missing_share_current),
            missing_share_reference=_number(result.missing_share_reference),
            low_expected_share=_number(result.low_expected_share),
            n_labelled=result.n_labelled,
            label_coverage=_number(result.label_coverage),
            suppressed=result.suppressed,
        ),
    )


def evidence_from_result(
    result: MetricResults,
    system_id: str,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    window: Union[Window, dict, None] = None,
    reference: Union[Reference, dict, None] = None,
    options: Optional[dict] = None,
    calibrator=None,
    persistence=None,
    family: Optional[str] = None,
    details: Optional[dict] = None,
    run_name: Optional[str] = None,
    produced_at: Optional[datetime] = None,
) -> Evidence:
    """Evidence record of one MetricResults (any metric type: drift, performance, fairness, genai, agents, ...)

    Parameters
    ----------
    result : MetricResults
    system_id : str
        External identifier of the monitored system (e.g. its entry in an inventory of AI systems)
    model_id, model_version : str, optional
    window, reference : Window, Reference or dict, optional
        Current window (start, end, n_rows) and reference (id, start, end, n_rows)
    options : dict, optional
        Options the metric was evaluated with (the entry of Analyzer.run options for this metric: threshold or
        alpha, upper_bound, and the calibration fields of AACalibrator.options())
    calibrator : AACalibrator, optional
        Calibrator whose null distribution of (metric, feature) fills the calibration block
    persistence : DriftPersistence or list of PersistentDrift, optional
        Persistence state after this window was fed to it
    family : str, optional
        Overrides the family inferred from metric_type and metric_name
    details : dict, optional
        Extra aggregates, merged over the ones taken from the result (conf_int, raw status; for rate_increase the
        counts and rates of the window and of the reference). Never raw rows

    A record whose window has no start, end nor id gets an evidence_id that includes produced_at (UserWarning): pass
    window={"id": ...} (batch, run or period id) so that the same finding sent again keeps its id.
    """
    options = options or {}
    family = family or _family(result)
    threshold, extra = result.threshold, {}
    if (family == "data_quality" and result.metric_type != DATA_QUALITY) or not (
        threshold is None
        or (isinstance(threshold, Number) and not isinstance(threshold, bool))
        or (isinstance(threshold, list) and all(isinstance(t, Number) for t in threshold))
    ):
        # summary statistics: the "threshold" is the statistic of the reference, not a decision threshold (the checks
        # of pulsar_metrics.quality, metric_type "data_quality", carry a real one: alert above it)
        if threshold is not None:
            extra["reference_value"] = threshold
        threshold = None
    if _position_test(result):
        extra["null_value"] = threshold  # the share tested against; the threshold of the decision is alpha
        threshold = _alpha(result.method)
    threshold = _threshold(threshold)
    direction, basis = _decision(result, family, threshold, options.get("upper_bound"))
    status = _status(result)
    p_value = result.p_value
    if p_value is None and basis == "p_value" and family == "drift":
        p_value = result.metric_value  # DriftTestMetric: the value is the p-value
    calibration, short_reference = _calibration(result, options, calibrator, threshold)
    if result.conf_int is not None:
        extra["conf_int"] = result.conf_int
    if result.status is not None:
        extra["pulsar_status"] = result.status
    if result.test_type is not None:
        extra["test_type"] = result.test_type
    extra.update(_rates(result))
    if family == "compliance" and _rules_version(result.method) is not None:
        extra["rules_version"] = _rules_version(result.method)
    non_finite = isinstance(result.metric_value, float) and not math.isfinite(result.metric_value)
    if non_finite:
        extra["value_non_finite"] = str(result.metric_value)
    value = _jsonable(result.metric_value)
    metric = Metric(
        family=family,
        name=result.metric_name,
        feature=result.feature_name,
        group=result.group,
        reference_group=result.reference_group,
        subject_metric=_subject_metric(result),
        value=value,
        threshold=threshold,
        direction=direction,
        decision_basis=basis,
        p_value=_number(p_value),
        ci_low=_number(result.ci_low),
        ci_high=_number(result.ci_high),
        e_value=_number(result.e_value),
        n=_number(result.n if result.n is not None else result.n_labelled),
        status=status,
        error=result.error,
        method=result.method,
    )
    window = Window.model_validate(window or {})
    if _no_window(window):
        warnings.warn(
            f"evidence of {result.metric_name} without window start, end nor id: its evidence_id includes produced_at, "
            "so the same finding sent again gets a new id; pass window={'id': ...} (batch, run or period id)",
            UserWarning,
            stacklevel=2,
        )
    persistence = _persistence(result, persistence)
    return Evidence(
        system_id=system_id,
        model_id=model_id,
        model_version=model_version,
        metric=metric,
        window=window,
        reference=Reference.model_validate(reference or {}),
        reliability=_reliability(result, family, status, calibration, short_reference, persistence, non_finite),
        provenance=Provenance(
            producer_version=_producer_version(),
            produced_at=produced_at or datetime.now(timezone.utc),
            run_name=run_name,
        ),
        details={**extra, **(details or {})},
    )


def evidence_from_results(results: Iterable[MetricResults], system_id: str, **kwargs) -> list[Evidence]:
    """Evidence records of a list of MetricResults sharing one context (see evidence_from_result for the arguments)

    For FairnessAnalyzer.get_result() and coverage(), ComplianceMonitor.evaluate(), embedding_drift(),
    model_equality(), FairnessMonitor.results(), AttributionDrift results... `options`, when given, is the Analyzer
    run options: the entry of each metric is looked up as in Analyzer.run ("metric|feature", then "metric").
    """
    options = kwargs.pop("options", None) or {}
    out = []
    for result in results:
        entry = options.get(f"{result.metric_name}|{result.feature_name}", options.get(result.metric_name, {}))
        out.append(evidence_from_result(result, system_id, options=entry if isinstance(entry, dict) else {}, **kwargs))
    return out


def evidence_from_analyzer(
    analyzer,
    system_id: str,
    reference_id: Optional[str] = None,
    calibrator=None,
    persistence=None,
    include_statistics: bool = False,
    produced_at: Optional[datetime] = None,
) -> list[Evidence]:
    """Evidence records of the last Analyzer.run(): model, window and options are taken from the analyzer

    include_statistics : bool, optional
        Also export the summary statistics (family data_quality; default False: they carry no verdict)
    """
    results = analyzer.get_result()
    if results is None:
        return []
    meta = analyzer._metadata
    info = getattr(analyzer, "_window_info", {}) or {}
    window = Window(start=meta.get("period_start"), end=meta.get("period_end"), n_rows=info.get("n_current"))
    reference = Reference(
        id=reference_id or info.get("reference_id"),
        start=info.get("reference_start"),
        end=info.get("reference_end"),
        n_rows=info.get("n_reference"),
    )
    kept = [r for r in results if include_statistics or r.metric_type != "statistics"]
    return evidence_from_results(
        kept,
        system_id,
        model_id=meta.get("model_id"),
        model_version=meta.get("model_version"),
        window=window,
        reference=reference,
        options=meta.get("options") or {},
        calibrator=calibrator,
        persistence=persistence,
        run_name=meta.get("name"),
        produced_at=produced_at,
    )


def to_evidence(source, system_id: str, **kwargs) -> list[Evidence]:
    """Evidence records of any result source: an Analyzer or a DataQualityAnalyzer after run(), a FairnessAnalyzer
    after run() (results and coverage), one MetricResults or a list of them. kwargs go to the matching converter."""
    from .analyzers.base import AbstractAnalyzer
    from .fairness.analyzer import FairnessAnalyzer

    if isinstance(source, AbstractAnalyzer):
        return evidence_from_analyzer(source, system_id, **kwargs)
    if isinstance(source, FairnessAnalyzer):
        return evidence_from_results(list(source.get_result() or []) + source.coverage(), system_id, **kwargs)
    if isinstance(source, MetricResults):
        return [evidence_from_result(source, system_id, **kwargs)]
    return evidence_from_results(source, system_id, **kwargs)


def dumps(records: Iterable[Evidence]) -> str:
    """JSON array of records (strict JSON: ValueError on NaN or infinities)"""
    return json.dumps([r.to_dict() for r in records], ensure_ascii=False, allow_nan=False)


def _refuse_constant(name: str):
    raise ValueError(f"evidence JSON holds {name}: NaN and infinities are not valid JSON")


def loads(text: Union[str, bytes]) -> list[Evidence]:
    """Records of a JSON array (or of a single JSON object); ValueError on NaN or infinities, ValidationError on an
    evidence_id that does not match its record"""
    data = json.loads(text, parse_constant=_refuse_constant)
    return [Evidence.from_dict(d) for d in (data if isinstance(data, list) else [data])]
