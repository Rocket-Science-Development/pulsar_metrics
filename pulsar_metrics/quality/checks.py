"""Checks of a window against a data-quality profile: one MetricResults per check and column (or rule)

Every result has metric_type "data_quality", the share measured in `metric_value`, the threshold of the policy in
`threshold` and `drift_status=True` when the value is above it (alert). A check that cannot be evaluated has no value
and no verdict; its `status` gives the reason (NOT_EVALUABLE) and `reliable` is False.

Only counts and shares are reported: the values of the window (e.g. unseen levels, which may be personal data typed
in the wrong field) are never copied into the results.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from ..metrics.base import MetricResults
from .policy import DataQualityPolicy
from .profile import DataQualityProfile, as_levels, duplicate_share, observed_kind, text_columns

METRIC_TYPE = "data_quality"
# status of a result that cannot be evaluated -> why (also the reliability reason codes of pulsar_metrics.evidence)
NOT_EVALUABLE = {
    "empty_window": "data quality: the window holds no row",
    "not_in_reference": "data quality: the column is absent from the reference profile, nothing to compare it with",
    "missing_in_window": "data quality: a column the check reads is absent from the window (see the schema record)",
    "type_mismatch": "data quality: the column changed type (see the schema record), its range does not apply",
    "no_reference_values": "data quality: the reference holds no value of the column",
    "no_reference_domain": "data quality: the reference column has more levels than max_categories, no domain was kept",
    "no_values_in_window": "data quality: no value of the window to judge (all missing, see the null_share record)",
}


def _verdict(value: Optional[float], threshold: Optional[float]) -> Optional[bool]:
    """Alert (True) when the value is above the threshold; None without value or threshold"""
    if value is None or threshold is None:
        return None
    return bool(value > threshold)


def _result(name: str, feature: Optional[str], value, threshold, n: int, method: str, **fields) -> MetricResults:
    return MetricResults(
        metric_type=METRIC_TYPE,
        metric_name=name,
        feature_name=feature,
        metric_value=value,
        threshold=threshold,
        drift_status=_verdict(value, threshold),
        n=n,
        method=method,
        **fields,
    )


def _not_evaluable(name: str, feature: Optional[str], code: str, threshold=None, n: int = 0) -> MetricResults:
    return MetricResults(
        metric_type=METRIC_TYPE,
        metric_name=name,
        feature_name=feature,
        threshold=threshold,
        n=n,
        status=code,
        reliable=False,
        reliability_reason=NOT_EVALUABLE[code],
    )


def _new_columns(profile: DataQualityProfile, current: pd.DataFrame) -> list:
    ignored = set(profile.ignored_columns)
    return [str(c) for c in current.columns if str(c) not in profile.columns and str(c) not in ignored]


def check_schema(profile: DataQualityProfile, current: pd.DataFrame, policy=None) -> list:
    """`schema` per column: 1 when the column is missing from the window, new (absent from the reference, unless
    allow_new_columns) or of another kind (numeric, categorical, boolean, datetime), 0 otherwise; threshold 0

    The kind is read on the non-missing values, so that an integer or boolean column holding missing values keeps
    its kind; a column without any value in the window is not judged on its kind (its null_share record says it). A
    window without any row is not judged at all (`empty_window`): an empty query result has no columns either.
    """
    policy = DataQualityPolicy.load(policy)
    if not policy.schema:
        return []
    current = text_columns(current)
    if current.shape[0] == 0:
        return [_not_evaluable("schema", name, "empty_window", 0) for name in profile.columns if policy.schema_checked(name)]
    out = []
    for name, column in profile.columns.items():
        if not policy.schema_checked(name):
            continue
        if name not in current.columns:
            out.append(_result("schema", name, 1, 0, 0, "column missing from the window", status="missing_column"))
            continue
        values = current[name]
        kind = observed_kind(values)
        if values.notna().any() and kind != column.kind:
            method = f"type changed: {column.kind} ({column.dtype}) -> {kind} ({values.dtype})"
            out.append(_result("schema", name, 1, 0, int(values.shape[0]), method, status="type_change"))
        else:
            out.append(_result("schema", name, 0, 0, int(values.shape[0]), f"same type: {column.kind}"))
    for name in _new_columns(profile, current):
        if not policy.schema_checked(name):
            continue
        if policy.allow_new_columns:
            out.append(_result("schema", name, 0, 0, int(current.shape[0]), "new column (allowed)", status="new_column"))
        else:
            out.append(
                _result("schema", name, 1, 0, int(current.shape[0]), "new column, absent from the reference", status="new_column")
            )
    return out


def check_nulls(profile: DataQualityProfile, current: pd.DataFrame, policy=None) -> list:
    """`null_share` per column: share of missing values of the window (default threshold: reference share + 0.05)"""
    policy = DataQualityPolicy.load(policy)
    current = text_columns(current)
    out = []
    n = int(current.shape[0])
    for name, column in profile.columns.items():
        threshold = policy.threshold("null_share", name, column.null_share)
        if threshold is None:
            continue
        if n == 0:  # before the columns: an empty query result has none
            out.append(_not_evaluable("null_share", name, "empty_window", threshold))
        elif name not in current.columns:
            out.append(_not_evaluable("null_share", name, "missing_in_window", threshold))
        else:
            value = float(current[name].isna().mean())
            method = f"missing values: {int(current[name].isna().sum())} of {n} rows (reference {column.null_share:.4g})"
            out.append(_result("null_share", name, value, threshold, n, method))
    for name in _new_columns(profile, current):
        if policy.threshold("null_share", name) is not None:
            out.append(_not_evaluable("null_share", name, "not_in_reference", n=n))
    return out


def check_ranges(profile: DataQualityProfile, current: pd.DataFrame, policy=None) -> list:
    """`out_of_range_share` per numeric column: share of the non-missing values of the window outside the range of
    the reference (min-max, or the policy's range_quantiles); missing values are set aside, and an infinite value is
    in range when the reference held the same infinity"""
    policy = DataQualityPolicy.load(policy)
    current = text_columns(current)
    out = []
    for name, column in profile.columns.items():
        if column.kind != "numeric":
            continue
        quantiles = policy.quantiles_of(name)
        reference_share = 0.0 if quantiles is None else quantiles[0] + 1 - quantiles[1]
        threshold = policy.threshold("out_of_range_share", name, reference_share)
        if threshold is None:
            continue
        code = _unavailable(name, current, column.min is None)
        if code is None and observed_kind(current[name]) != "numeric" and current[name].notna().any():
            code = "type_mismatch"
        if code is not None:
            out.append(_not_evaluable("out_of_range_share", name, code, threshold, int(current.shape[0])))
            continue
        values = current[name].dropna().astype(float)
        low, high = column.bounds(quantiles)
        below = int(((values < low) & ~(column.neg_inf & (values == -np.inf))).sum())
        above = int(((values > high) & ~(column.pos_inf & (values == np.inf))).sum())
        span = "reference min-max" if quantiles is None else f"reference quantiles {quantiles[0]:g}-{quantiles[1]:g}"
        method = f"outside [{low:.6g}, {high:.6g}] ({span}): {below} below, {above} above, of {len(values)} values"
        out.append(
            _result(
                "out_of_range_share",
                name,
                (below + above) / len(values),
                threshold,
                len(values),
                method,
                missing_share_current=float(current[name].isna().mean()),
            )
        )
    return out


def check_categories(profile: DataQualityProfile, current: pd.DataFrame, policy=None) -> list:
    """`unseen_category_share` per categorical and boolean column: share of the non-missing values of the window
    whose level was not seen in the reference (compared as text); missing values are set aside"""
    policy = DataQualityPolicy.load(policy)
    current = text_columns(current)
    out = []
    for name, column in profile.columns.items():
        if column.kind not in ("categorical", "boolean"):
            continue
        threshold = policy.threshold("unseen_category_share", name, 0.0)
        if threshold is None:
            continue
        code = _unavailable(name, current, False)
        if code is None and column.categories is None:
            code = "no_reference_domain"
        if code is not None:
            out.append(_not_evaluable("unseen_category_share", name, code, threshold, int(current.shape[0])))
            continue
        levels = as_levels(current[name])
        unseen = ~levels.isin(set(column.categories))
        method = (
            f"{int(unseen.sum())} of {len(levels)} values in {levels[unseen].nunique()} level(s) unseen in the "
            f"reference ({len(column.categories)} levels)"
        )
        out.append(
            _result(
                "unseen_category_share",
                name,
                float(unseen.mean()),
                threshold,
                len(levels),
                method,
                missing_share_current=float(current[name].isna().mean()),
            )
        )
    return out


def check_duplicates(profile: DataQualityProfile, current: pd.DataFrame, policy=None) -> list:
    """`duplicate_share` of the frame (feature None): share of rows repeating an earlier row of the window, on the
    profile's duplicate_columns (a key) or every profiled column present in the window (default threshold: reference
    share + 0.01)"""
    policy = DataQualityPolicy.load(policy)
    current = text_columns(current)
    threshold = policy.threshold("duplicate_share", None, profile.duplicate_share)
    if threshold is None:
        return []
    n = int(current.shape[0])
    if n == 0:
        return [_not_evaluable("duplicate_share", None, "empty_window", threshold)]
    if profile.duplicate_columns:
        columns = list(profile.duplicate_columns)
        if any(c not in current.columns for c in columns):
            return [_not_evaluable("duplicate_share", None, "missing_in_window", threshold, n)]
        on = "key " + ", ".join(columns)
    else:  # profiled columns missing from the window are left out (their schema record alerts)
        columns = [c for c in profile.columns if c in current.columns]
        on = f"{len(columns)} of {len(profile.columns)} profiled columns"
        if not columns:
            return [_not_evaluable("duplicate_share", None, "missing_in_window", threshold, n)]
    value = duplicate_share(current[columns])
    method = f"{int(round(value * n))} of {n} rows repeat an earlier row ({on}; reference {profile.duplicate_share:.4g})"
    return [_result("duplicate_share", None, value, threshold, n, method)]


def check_rules(profile: DataQualityProfile, current: pd.DataFrame, policy=None) -> list:
    """`rule_violation_share` per row rule (feature = rule id): share of the judged rows (both sides present) that
    break the rule (default threshold: reference share + 0.01)"""
    policy = DataQualityPolicy.load(policy)
    current = text_columns(current)
    out = []
    for rule in profile.rules:
        threshold = policy.threshold("rule_violation_share", rule.id, rule.reference_share or 0.0)
        if threshold is None:
            continue
        n = int(current.shape[0])
        if n == 0:
            out.append(_not_evaluable("rule_violation_share", rule.id, "empty_window", threshold))
            continue
        if any(c not in current.columns for c in rule.columns):
            out.append(_not_evaluable("rule_violation_share", rule.id, "missing_in_window", threshold, n))
            continue
        violating, judged = rule.violations(current)
        if judged == 0:
            out.append(_not_evaluable("rule_violation_share", rule.id, "no_values_in_window", threshold, n))
            continue
        reference = rule.reference_share or 0.0  # a rule given without its reference share (not fitted)
        method = f"{rule.expression}: {violating} of {judged} judged rows break it (reference {reference:.4g})"
        out.append(_result("rule_violation_share", rule.id, violating / judged, threshold, judged, method))
    return out


def _unavailable(name: str, current: pd.DataFrame, no_reference_values: bool) -> Optional[str]:
    """Code of a value check that cannot run on the column, or None"""
    if current.shape[0] == 0:
        return "empty_window"
    if name not in current.columns:
        return "missing_in_window"
    if no_reference_values:
        return "no_reference_values"
    if not current[name].notna().any():
        return "no_values_in_window"
    return None


CHECK_FUNCTIONS = {
    "schema": check_schema,
    "null_share": check_nulls,
    "out_of_range_share": check_ranges,
    "unseen_category_share": check_categories,
    "duplicate_share": check_duplicates,
    "rule_violation_share": check_rules,
}


def check(
    profile: DataQualityProfile,
    current: pd.DataFrame,
    policy: Union[DataQualityPolicy, dict, str, None] = None,
) -> list:
    """Every data-quality check of a window against a reference profile

    Parameters
    ----------
    profile : DataQualityProfile
        Reference profile (DataQualityProfile.fit or load)
    current : DataFrame
        Window to check
    policy : DataQualityPolicy, dict or YAML path, optional
        Thresholds per check and column (default: DEFAULT_THRESHOLDS)

    Returns
    -------
    list of MetricResults
        schema, null_share, out_of_range_share, unseen_category_share, duplicate_share and rule_violation_share results
    """
    policy = DataQualityPolicy.load(policy)
    out = []
    for function in CHECK_FUNCTIONS.values():
        out += function(profile, current, policy)
    return out
