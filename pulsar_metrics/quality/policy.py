"""Thresholds of the data-quality checks, per check and per column, from a dict or a YAML file"""

from pathlib import Path
from typing import Optional, Sequence, Union

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from .profile import QUANTILES, quantile_key

# checks whose value is a share of rows: alert when the share of the window is above the threshold
SHARE_CHECKS = ("null_share", "out_of_range_share", "unseen_category_share", "duplicate_share", "rule_violation_share")
CHECKS = ("schema",) + SHARE_CHECKS
# Default thresholds: the share of the reference plus a margin (a reference with 2 % of missing values alerts above
# 7 %); out-of-range and unseen values are absent from the reference by construction (min-max range, levels seen)
DEFAULT_THRESHOLDS = {
    "null_share": {"margin": 0.05},
    "out_of_range_share": {"margin": 0.01},
    "unseen_category_share": {"margin": 0.01},
    "duplicate_share": {"margin": 0.01},
    "rule_violation_share": {"margin": 0.01},
}
_KEYS = set(CHECKS) | {"columns", "range_quantiles", "allow_new_columns"}
_COLUMN_KEYS = set(CHECKS) | {"range_quantiles"}


def _threshold(check: str, spec, where: str) -> Optional[dict]:
    """{"max": x} or {"margin": x} of a policy entry; None when the check is disabled"""
    if spec is None or spec is False:
        return None
    if spec is True:
        return dict(DEFAULT_THRESHOLDS[check])
    if isinstance(spec, (int, float)):
        spec = {"max": spec}
    if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in ("max", "margin"):
        raise error_msg(
            value=spec, message=f"{where}{check}: give a share, {{max: share}}, {{margin: share}}, or null to disable it"
        )
    ((key, value),) = spec.items()
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise error_msg(value=value, message=f"{where}{check}.{key} must be a share between 0 and 1")
    return {key: float(value)}


def _quantiles(spec, where: str) -> Optional[tuple]:
    if spec is None:
        return None
    if not isinstance(spec, (list, tuple)) or len(spec) != 2 or not 0 <= spec[0] < spec[1] <= 1:
        raise error_msg(value=spec, message=f"{where}range_quantiles must be [low, high] levels, 0 <= low < high <= 1")
    known = {quantile_key(q) for q in QUANTILES}
    unknown = [level for level in spec if quantile_key(float(level)) not in known]
    if unknown:  # the profile keeps these levels only: found here rather than when every check runs
        raise error_msg(value=unknown, message=f"{where}range_quantiles levels must be among {QUANTILES} (the profile's)")
    return float(spec[0]), float(spec[1])


class DataQualityPolicy:
    """Which data-quality checks to run and their thresholds

    Every share check alerts when the share of the window is **above** its threshold. A threshold is either absolute,
    `{"max": 0.1}` (or just `0.1`), or relative to the share of the reference, `{"margin": 0.05}`: reference share +
    margin, capped at 1. `null` (None) disables the check, `true` restores its default.

    Parameters
    ----------
    thresholds : dict, optional
        Threshold per check for every column, over DEFAULT_THRESHOLDS; "schema": False disables the schema check
    columns : dict, optional
        Overrides per column, or per rule id for rule_violation_share: {"rh": {"null_share": {"max": 0.3}}}; a column
        may also set its own "range_quantiles", or "schema": False to ignore its schema changes
    range_quantiles : [low, high], optional
        Acceptable range of numeric columns as quantile levels of the reference, e.g. [0.001, 0.999] (default None:
        the min and max of the reference). The reference share outside them (low + 1 - high) is the base of a margin
    allow_new_columns : bool, optional
        A column absent from the reference is a schema change (default False: alert)

    Example (YAML, read with DataQualityPolicy.load):

        null_share: {margin: 0.05}
        out_of_range_share: 0.02
        columns:
          rh: {null_share: {max: 0.3}}
          station: {unseen_category_share: null}
    """

    def __init__(
        self,
        thresholds: Optional[dict] = None,
        columns: Optional[dict] = None,
        range_quantiles: Optional[Sequence[float]] = None,
        allow_new_columns: bool = False,
    ):
        thresholds = thresholds or {}
        unknown = set(thresholds) - set(CHECKS)
        if unknown:
            raise error_msg(value=sorted(unknown), message=f"unknown data-quality checks {sorted(unknown)}; known: {CHECKS}")
        self.schema = bool(thresholds.get("schema", True))
        self.thresholds = {
            check: _threshold(check, thresholds[check], "") if check in thresholds else dict(DEFAULT_THRESHOLDS[check])
            for check in SHARE_CHECKS
        }
        self.range_quantiles = _quantiles(range_quantiles, "")
        self.allow_new_columns = bool(allow_new_columns)
        self.columns = {}
        for column, overrides in (columns or {}).items():
            if not isinstance(overrides, dict):
                raise error_msg(value=column, message=f"columns.{column} must be a mapping of checks to thresholds")
            unknown = set(overrides) - _COLUMN_KEYS
            if unknown:
                raise error_msg(value=sorted(unknown), message=f"columns.{column}: unknown keys {sorted(unknown)}")
            entry = {}
            for check, spec in overrides.items():
                where = f"columns.{column}."
                if check == "schema":
                    entry[check] = bool(spec)
                elif check == "range_quantiles":
                    entry[check] = _quantiles(spec, where)
                else:
                    entry[check] = _threshold(check, spec, where)
            self.columns[str(column)] = entry

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "DataQualityPolicy":
        data = dict(data or {})
        unknown = set(data) - _KEYS
        if unknown:
            raise error_msg(value=sorted(unknown), message=f"unknown policy keys {sorted(unknown)}; known: {sorted(_KEYS)}")
        return cls(
            thresholds={k: v for k, v in data.items() if k in CHECKS},
            columns=data.get("columns"),
            range_quantiles=data.get("range_quantiles"),
            allow_new_columns=data.get("allow_new_columns", False),
        )

    @classmethod
    def load(cls, source: Union["DataQualityPolicy", dict, str, Path, None]) -> "DataQualityPolicy":
        """Policy of a DataQualityPolicy, a dict, a YAML file path (needs PyYAML), or None (defaults)"""
        if isinstance(source, DataQualityPolicy):
            return source
        if isinstance(source, (str, Path)):
            from .profile import _yaml

            source = _yaml().safe_load(Path(source).read_text(encoding="utf-8")) or {}
        return cls.from_dict(source)

    def to_dict(self) -> dict:
        data = {"schema": self.schema, **self.thresholds, "allow_new_columns": self.allow_new_columns}
        if self.range_quantiles is not None:
            data["range_quantiles"] = list(self.range_quantiles)
        if self.columns:
            data["columns"] = {
                c: {k: list(v) if k == "range_quantiles" and v is not None else v for k, v in e.items()}
                for c, e in self.columns.items()
            }
        return data

    # ------------------------------------------------------------------ lookups
    def schema_checked(self, column: str) -> bool:
        return self.columns.get(column, {}).get("schema", self.schema)

    def quantiles_of(self, column: str) -> Optional[tuple]:
        return self.columns.get(column, {}).get("range_quantiles", self.range_quantiles)

    def threshold(self, check: str, feature: Optional[str], reference_share: float = 0.0) -> Optional[float]:
        """Threshold of a share check for a column (or rule id; None: the whole frame); None when disabled"""
        overrides = self.columns.get(feature, {}) if feature is not None else {}
        spec = overrides[check] if check in overrides else self.thresholds[check]
        if spec is None:
            return None
        if "max" in spec:
            return spec["max"]
        return min(1.0, (reference_share or 0.0) + spec["margin"])
