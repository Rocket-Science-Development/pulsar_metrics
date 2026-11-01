"""Characterization tests: freeze the current output of every metric.

Every drift, drift test and performance metric is evaluated on a fixed synthetic dataset and
compared with tests/golden/metrics_reference.json. The goal is to detect any numerical change
when upgrading dependencies (Python, pandas, numpy, scipy, scikit-learn, pydantic).

Regenerate the reference file (only after an intended behaviour change):
    python -m tests.test_characterization
"""

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.metrics.drift import DriftMetric, DriftTestMetric
from pulsar_metrics.metrics.enums import DriftMetricsFuncs, DriftTestMetricsFuncs, PerformanceMetricsFuncs
from pulsar_metrics.metrics.performance import PerformanceMetric
from pulsar_metrics.metrics.statistics import FeatureSummary

GOLDEN_FILE = Path(__file__).parent / "golden" / "metrics_reference.json"
FEATURES = ["x1", "x2", "x3"]
PROBA_METRICS = ["auc", "aucpr", "brier", "log_loss"]
DRIFT_THRESHOLD = 0.5
PERFORMANCE_THRESHOLD = 0.5
ALPHA = 0.05
REL_TOL = 1e-9
ABS_TOL = 1e-12

# Metrics that currently fail on this dataset (known defects, tracked separately)
KNOWN_FAILURES: dict[str, str] = {}


def make_data(n: int, shift: float, start: str, rng: np.random.Generator) -> pd.DataFrame:
    x1 = rng.normal(0 + shift, 1, n)
    x2 = rng.exponential(1 + shift, n)
    x3 = rng.integers(0, 5, n).astype(float)
    proba = 1 / (1 + np.exp(-(x1 - 0.5 * x2)))
    y_true = (rng.uniform(size=n) < proba).astype(int)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y_true": y_true,
            "y_pred": (proba > 0.5).astype(int),
            "y_pred_proba": proba,
            "model_id": "m1",
            "model_version": "1",
            "pred_timestamp": pd.date_range(start, periods=n, freq="min").astype(str),
        }
    )


def make_datasets():
    rng = np.random.default_rng(42)
    reference = make_data(500, 0.0, "2026-01-01", rng)
    current = make_data(400, 0.4, "2026-02-01", rng)
    return reference, current


def data_fingerprint(reference: pd.DataFrame, current: pd.DataFrame) -> str:
    """Hash of the numeric inputs, rounded to 12 digits, to detect a change in the random stream"""
    digest = hashlib.sha256()
    for df in (reference, current):
        values = df[FEATURES + ["y_true", "y_pred", "y_pred_proba"]].to_numpy(dtype=float)
        digest.update(np.round(values, 12).tobytes())
    return digest.hexdigest()


def metric_key(metric_name: str, feature_name) -> str:
    return f"{metric_name}|{feature_name}"


def build_metrics():
    """Return (key, metric, evaluate kwargs) for every metric to characterize"""
    metrics = []
    for name in DriftMetricsFuncs._member_names_:
        for f in FEATURES:
            metrics.append((metric_key(name, f), DriftMetric(metric_name=name, feature_name=f), {"threshold": DRIFT_THRESHOLD}))
    for name in DriftTestMetricsFuncs._member_names_:
        for f in FEATURES:
            metrics.append((metric_key(name, f), DriftTestMetric(metric_name=name, feature_name=f), {"alpha": ALPHA}))
    for name in PerformanceMetricsFuncs._member_names_:
        pred_name = "y_pred_proba" if name in PROBA_METRICS else "y_pred"
        for bootstrap in (False, True):
            key = metric_key(name, "bootstrap" if bootstrap else None)
            kwargs = {"threshold": PERFORMANCE_THRESHOLD, "bootstrap": bootstrap}
            metrics.append((key, PerformanceMetric(metric_name=name, pred_name=pred_name), kwargs))
    return metrics


def evaluate(metric, current: pd.DataFrame, reference: pd.DataFrame, kwargs: dict):
    """Evaluate one metric; return its result or None (errors are printed by the library)"""
    with contextlib.redirect_stdout(io.StringIO()):
        metric.evaluate(current=current, reference=reference, **kwargs)
    result = getattr(metric, "_result", None)
    return None if result is None or result is True else result


def to_number(value):
    if value is None or isinstance(value, str):
        return value
    return float(value)


def serialize(result) -> dict:
    return {
        "metric_type": result.metric_type,
        "metric_value": to_number(result.metric_value),
        "drift_status": result.drift_status,
        "threshold": result.threshold if not isinstance(result.threshold, list) else [to_number(t) for t in result.threshold],
        "conf_int": None if result.conf_int is None else [to_number(c) for c in result.conf_int],
    }


def compute_all() -> dict:
    reference, current = make_datasets()
    results = {}
    for key, metric, kwargs in build_metrics():
        result = evaluate(metric, current, reference, kwargs)
        results[key] = None if result is None else serialize(result)
    with contextlib.redirect_stdout(io.StringIO()):
        for feature in FEATURES:
            summary = FeatureSummary(feature_name=feature)
            summary.evaluate(current, reference)
            for result in summary._result:
                results[metric_key(f"summary:{result.metric_name}", feature)] = serialize(result)
    return {"data_fingerprint": data_fingerprint(reference, current), "results": results}


def assert_close(actual, expected, label: str):
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), label
        for a, e in zip(actual, expected):
            assert_close(a, e, label)
    elif isinstance(expected, float):
        assert actual is not None and math.isclose(
            actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL
        ), f"{label}: {actual} != {expected}"
    else:
        assert actual == expected, f"{label}: {actual!r} != {expected!r}"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def computed():
    return compute_all()


def test_input_data_unchanged(golden, computed):
    assert computed["data_fingerprint"] == golden["data_fingerprint"], "the synthetic dataset changed (random stream)"


def test_same_metrics_evaluated(golden, computed):
    assert sorted(computed["results"]) == sorted(golden["results"])


def _keys():
    if not GOLDEN_FILE.exists():
        return []
    return sorted(json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["results"])


@pytest.mark.parametrize("key", _keys())
def test_metric_matches_reference(key, golden, computed):
    metric_name = key.split("|")[0]
    expected = golden["results"][key]
    actual = computed["results"][key]
    if metric_name in KNOWN_FAILURES:
        assert expected is None and actual is None, f"{key} now returns a result: update KNOWN_FAILURES and the reference"
        pytest.xfail(KNOWN_FAILURES[metric_name])
    assert actual is not None, f"{key} returned no result"
    for field, value in expected.items():
        assert_close(actual[field], value, f"{key}.{field}")


if __name__ == "__main__":
    GOLDEN_FILE.parent.mkdir(exist_ok=True)
    GOLDEN_FILE.write_text(json.dumps(compute_all(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written {GOLDEN_FILE}")
