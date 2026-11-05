"""Data-quality reference profile: what the data looked like on a reference window, learned once and versioned with
the model.

Per column: its kind (numeric, categorical, boolean, datetime) and dtype, its share of missing values, its range and
quantiles (numeric), its domain of levels and cardinality (categorical, boolean); for the frame: the share of duplicate
rows; optionally row rules between columns ("dew <= temp"). The profile holds aggregates only, never raw rows. Levels
of a categorical column are stored only up to `max_categories` (identifiers and free text keep no domain).
"""

import json
import math
import operator
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..exceptions import CustomExceptionPulsarMetric as error_msg

PROFILE_VERSION = "pulsar.dq-profile/1"
# columns written by pulsar-data-collection next to the features: left out of the profile by default
METADATA_COLUMNS = ("model_id", "model_version", "pred_timestamp")
QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
MAX_CATEGORIES = 50

Kind = Literal["numeric", "categorical", "boolean", "datetime"]

_OPERATORS = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}
_RULE = re.compile(r"^\s*(?P<left>.+?)\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<right>.+?)\s*$")


def kind_of(column: pd.Series) -> str:
    """Kind of a column: numeric, boolean, datetime or categorical (anything else: text, categories, objects)"""
    if is_bool_dtype(column):
        return "boolean"
    if is_numeric_dtype(column):
        return "numeric"
    if is_datetime64_any_dtype(column):
        return "datetime"
    return "categorical"


def observed_kind(column: pd.Series) -> str:
    """Kind of the non-missing values of a column: a boolean or integer column holding missing values (object or
    float dtype) keeps its kind; a column without any value gets the kind of its dtype"""
    values = column.dropna()
    return kind_of(values.infer_objects() if not values.empty else column)


def quantile_key(level: float) -> str:
    """Key of a quantile level in ColumnProfile.quantiles, e.g. 0.01 -> "0.01" """
    return f"{level:g}"


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def as_levels(column: pd.Series) -> pd.Series:
    """Values of a categorical or boolean column as the text stored in the profile (missing values dropped)"""
    return column.dropna().astype(str)


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class ColumnProfile(_Model):
    """Reference profile of one column"""

    name: str
    kind: Kind
    dtype: str = Field(description="pandas dtype in the reference, e.g. float64, object")
    n_rows: int
    null_share: float = Field(description="Share of missing values (NaN, None, NaT) in the reference")
    n_unique: int = Field(description="Distinct non-missing values (cardinality)")
    min: Optional[Union[float, str]] = Field(None, description="Smallest value (numeric; ISO 8601 for datetimes)")
    max: Optional[Union[float, str]] = Field(None, description="Largest value (numeric; ISO 8601 for datetimes)")
    quantiles: dict[str, float] = Field(default_factory=dict, description="Numeric: quantiles of QUANTILES, by level")
    categories: Optional[list[str]] = Field(
        None, description="Categorical and boolean: levels seen in the reference; None above max_categories levels"
    )

    @classmethod
    def fit(cls, column: pd.Series, max_categories: int = MAX_CATEGORIES) -> "ColumnProfile":
        values = column.dropna()
        kind = observed_kind(column)
        n = int(column.shape[0])
        profile = {
            "name": str(column.name),
            "kind": kind,
            "dtype": str(column.dtype),
            "n_rows": n,
            "null_share": float(column.isna().mean()) if n else 0.0,
            "n_unique": int(values.nunique()),
        }
        if kind == "numeric" and not values.empty:
            numbers = values.astype(float)
            numbers = numbers[np.isfinite(numbers)]
            if not numbers.empty:
                profile["min"], profile["max"] = _finite(numbers.min()), _finite(numbers.max())
                profile["quantiles"] = {quantile_key(q): float(numbers.quantile(q)) for q in QUANTILES}
        elif kind == "datetime" and not values.empty:
            profile["min"] = pd.Timestamp(values.min()).isoformat()
            profile["max"] = pd.Timestamp(values.max()).isoformat()
        elif kind in ("categorical", "boolean"):
            levels = as_levels(column).unique()
            if len(levels) <= max_categories:
                profile["categories"] = sorted(levels.tolist())
        return cls(**profile)

    def bounds(self, quantiles: Optional[Sequence[float]] = None) -> tuple:
        """(low, high) of the acceptable range: min and max of the reference, or the given quantile levels of it"""
        if quantiles is None:
            return self.min, self.max
        missing = [q for q in quantiles if quantile_key(q) not in self.quantiles]
        if missing:
            raise error_msg(value=missing, message=f"quantile levels {missing} are not in the profile (see QUANTILES)")
        return self.quantiles[quantile_key(quantiles[0])], self.quantiles[quantile_key(quantiles[1])]


class RowRule(_Model):
    """Rule every row should satisfy: `left <op> right`, where right is a column or a number (e.g. "dew <= temp")

    Rows where a side is missing are not judged. `reference_share` is the share of violating rows in the reference,
    filled by DataQualityProfile.fit (the default threshold of the check sits just above it).
    """

    id: str
    left: str
    op: Literal["<=", "<", ">=", ">", "==", "!="]
    right: Union[str, float]
    reference_share: Optional[float] = None

    @classmethod
    def parse(cls, expression: str, id: Optional[str] = None) -> "RowRule":
        """Rule of an expression "column <op> column" or "column <op> number" """
        match = _RULE.match(expression)
        if not match:
            raise error_msg(value=expression, message="a row rule reads 'column <op> column|number', op in <, <=, >, >=, ==, !=")
        right = match["right"]
        try:
            right = float(right)
        except ValueError:
            pass
        return cls(id=id or expression.strip(), left=match["left"], op=match["op"], right=right)

    @classmethod
    def of(cls, rule) -> "RowRule":
        """Rule of a RowRule, an expression, or a dict ({"id", "expr"} or {"id", "left", "op", "right"})"""
        if isinstance(rule, RowRule):
            return rule
        if isinstance(rule, str):
            return cls.parse(rule)
        if isinstance(rule, dict) and "expr" in rule:
            parsed = cls.parse(rule["expr"], rule.get("id"))
            return parsed.model_copy(update={"reference_share": rule.get("reference_share")})
        return cls(**rule)

    @property
    def expression(self) -> str:
        right = self.right if isinstance(self.right, str) else f"{self.right:g}"
        return f"{self.left} {self.op} {right}"

    @property
    def columns(self) -> list:
        return [self.left] + ([self.right] if isinstance(self.right, str) else [])

    def violations(self, data: pd.DataFrame) -> tuple:
        """(violating rows, judged rows) of a frame holding the rule's columns"""
        left = pd.to_numeric(data[self.left], errors="coerce")
        if isinstance(self.right, str):
            right = pd.to_numeric(data[self.right], errors="coerce")
            judged = left.notna() & right.notna()
            right = right[judged]
        else:
            right, judged = self.right, left.notna()
        holds = _OPERATORS[self.op](left[judged], right)
        return int((~holds).sum()), int(judged.sum())


def duplicate_share(data: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> float:
    """Share of rows repeating an earlier row (on `subset`, default all columns); 0 for an empty frame"""
    if data.shape[0] == 0 or data.shape[1] == 0:
        return 0.0
    return float(data.duplicated(subset=list(subset) if subset else None).mean())


class DataQualityProfile(_Model):
    """Reference profile of a frame, to check later windows against (see pulsar_metrics.quality.check)

    Build it with `DataQualityProfile.fit(reference)`, save it with `save()` (JSON or YAML) next to the model, and
    read it back with `DataQualityProfile.load()`.
    """

    profile_version: Literal["pulsar.dq-profile/1"] = PROFILE_VERSION
    reference_id: Optional[str] = Field(None, description="Identifier of the reference dataset or period, chosen by the user")
    reference_start: Optional[datetime] = None
    reference_end: Optional[datetime] = None
    created_at: Optional[datetime] = None
    n_rows: int
    columns: dict[str, ColumnProfile]
    duplicate_columns: Optional[list[str]] = Field(
        None, description="Columns compared to find duplicate rows (None: every profiled column)"
    )
    duplicate_share: float = Field(description="Share of rows repeating an earlier row in the reference")
    ignored_columns: list[str] = Field(
        default_factory=list, description="Columns left out of the profile: not reported as new columns in a window"
    )
    rules: list[RowRule] = Field(default_factory=list)
    max_categories: int = MAX_CATEGORIES

    @field_validator("reference_start", "reference_end", "created_at", mode="before")
    @classmethod
    def _in_utc(cls, v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return None
        v = pd.Timestamp(v).to_pydatetime()
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    @classmethod
    def fit(
        cls,
        reference: pd.DataFrame,
        columns: Optional[Iterable[str]] = None,
        exclude: Iterable[str] = METADATA_COLUMNS,
        rules: Optional[Iterable] = None,
        duplicate_columns: Optional[Sequence[str]] = None,
        max_categories: int = MAX_CATEGORIES,
        reference_id: Optional[str] = None,
        timestamp_column: Optional[str] = "pred_timestamp",
    ) -> "DataQualityProfile":
        """Profile of a reference frame

        Parameters
        ----------
        reference : DataFrame
        columns : iterable of str, optional
            Columns to profile (default: every column but `exclude`)
        exclude : iterable of str, optional
            Columns left out when `columns` is not given (default: model_id, model_version, pred_timestamp)
        rules : iterable, optional
            Row rules: expressions ("dew <= temp", "rh <= 100"), dicts or RowRule
        duplicate_columns : sequence of str, optional
            Columns compared to find duplicate rows, e.g. a key, profiled or not (default: every profiled column)
        max_categories : int, optional
            Categorical columns with more levels keep no domain: their unseen-category check is not evaluable (default 50)
        reference_id : str, optional
            Identifier of the reference, carried to the evidence records
        timestamp_column : str, optional
            Column giving reference_start / reference_end, when present (default "pred_timestamp")
        """
        ignored = set()
        if columns is None:
            ignored = {str(c) for c in exclude or ()}
            columns = [c for c in reference.columns if str(c) not in ignored]
        columns = [str(c) for c in columns]
        ignored |= {str(c) for c in reference.columns if str(c) not in columns}
        absent = [c for c in columns if c not in reference.columns]
        if absent:
            raise error_msg(value=absent, message=f"columns {absent} are not in the reference")
        profiles = {c: ColumnProfile.fit(reference[c], max_categories) for c in columns}
        duplicates = list(duplicate_columns) if duplicate_columns else None
        absent = [c for c in duplicates or () if c not in reference.columns]
        if absent:
            raise error_msg(value=absent, message=f"duplicate_columns {absent} are not in the reference")
        fitted_rules = []
        for rule in rules or ():
            rule = RowRule.of(rule)
            absent = [c for c in rule.columns if c not in reference.columns]
            if absent:
                raise error_msg(value=rule.id, message=f"rule {rule.id} reads columns {absent} absent from the reference")
            violating, judged = rule.violations(reference)
            fitted_rules.append(rule.model_copy(update={"reference_share": violating / judged if judged else 0.0}))
        period = {}
        if timestamp_column and timestamp_column in reference.columns and reference.shape[0]:
            stamps = pd.to_datetime(reference[timestamp_column], format="mixed", errors="coerce")
            if stamps.notna().any():
                period = {"reference_start": stamps.min(), "reference_end": stamps.max()}
        return cls(
            reference_id=reference_id,
            created_at=datetime.now(timezone.utc),
            n_rows=int(reference.shape[0]),
            columns=profiles,
            duplicate_columns=duplicates,
            duplicate_share=duplicate_share(reference, duplicates or columns),
            rules=fitted_rules,
            ignored_columns=sorted(ignored),
            max_categories=max_categories,
            **period,
        )

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        """JSON-compatible dict (datetimes as ISO 8601 strings)"""
        return self.model_dump(mode="json")

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    def to_yaml(self) -> str:
        return _yaml().safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict) -> "DataQualityProfile":
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "DataQualityProfile":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, text: str) -> "DataQualityProfile":
        return cls.from_dict(_yaml().safe_load(text))

    def save(self, path: Union[str, Path]) -> Path:
        """Write the profile as YAML (.yaml, .yml) or JSON (any other suffix)"""
        path = Path(path)
        text = self.to_yaml() if path.suffix.lower() in (".yaml", ".yml") else self.to_json() + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataQualityProfile":
        """Read a profile written by save()"""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return cls.from_yaml(text) if path.suffix.lower() in (".yaml", ".yml") else cls.from_json(text)


def _yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise error_msg(value=None, message="reading or writing YAML needs PyYAML") from exc
    return yaml
