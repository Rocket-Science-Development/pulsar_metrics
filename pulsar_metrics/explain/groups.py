"""Does the model rely on different features for different groups? Importance shares per group against a reference group."""

from typing import Optional

import numpy as np
import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..fairness.criteria import verdict
from ..fairness.disclosure import complementary_suppression
from ..metrics.base import MetricResults
from .attributions import check_attributions


def _shares(magnitude: np.ndarray) -> np.ndarray:
    mean = magnitude.mean(axis=0)
    total = mean.sum()
    return mean / total if total > 0 else np.full(len(mean), 1 / len(mean))


def importance_by_group(
    attributions: pd.DataFrame,
    groups: pd.Series,
    reference_group: Optional[str] = None,
    min_group_size: int = 30,
    share_tolerance: float = 0.05,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
) -> list:
    """Share of the total importance of each feature in each group, minus that share in the reference group

    A feature that weighs much more for one group than for the others can act as a proxy of the protected attribute.
    The intervals come from a stratified bootstrap, simultaneous over groups and features (Bonferroni); groups below
    `min_group_size` are suppressed (no value, no count), with complementary suppression when a single group or fewer
    than `min_group_size` rows would be suppressed (see `pulsar_metrics.fairness.disclosure`). The default reference
    group is the largest.

    Parameters
    ----------
    attributions : DataFrame
        One row per prediction, one column per feature
    groups : Series
        Group of each row (aligned on the attributions' index), e.g. the protected attribute joined in memory
    """
    attributions = check_attributions(attributions)
    groups = groups.reindex(attributions.index)
    if groups.isna().any():
        raise error_msg(value=None, message="every attribution row needs a group")
    name = groups.name or "group"
    keys = groups.astype(str).to_numpy()
    sizes = pd.Series(keys).value_counts()
    kept = sizes[sizes >= min_group_size].index.tolist()
    if not kept:
        raise error_msg(value=name, message="every group is below min_group_size")
    reference = str(reference_group) if reference_group is not None else str(sizes[kept].idxmax())
    if reference not in kept:
        raise error_msg(value=reference, message="the reference group is missing or below min_group_size")
    primary = [g for g in sizes.index if g not in kept]
    secondary, _ = complementary_suppression(sizes, kept, primary, reference, min_group_size)
    kept = [g for g in kept if g not in secondary]

    magnitude = np.abs(attributions.to_numpy(dtype=float))
    columns = attributions.columns
    rows = {g: np.flatnonzero(keys == g) for g in kept}
    compared = [g for g in kept if g != reference]
    point = {g: _shares(magnitude[rows[g]]) for g in kept}

    rng = np.random.default_rng(seed)
    boot = {g: [] for g in compared}
    for _ in range(n_bootstrap):
        ref_b = _shares(magnitude[rng.choice(rows[reference], len(rows[reference]))])
        for g in compared:
            boot[g].append(_shares(magnitude[rng.choice(rows[g], len(rows[g]))]) - ref_b)

    tests = max(1, len(compared) * len(columns))
    tail = (1 - confidence) / (2 * tests)
    zone = (-share_tolerance, share_tolerance)
    out = []
    for g in compared:
        bounds = np.quantile(np.array(boot[g]), [tail, 1 - tail], axis=0)
        for j, feature in enumerate(columns):
            value = float(point[g][j] - point[reference][j])
            out.append(
                MetricResults(
                    metric_type="explainability",
                    metric_name="importance_share_difference_by_group",
                    feature_name=feature,
                    group=f"{name}={g}",
                    reference_group=f"{name}={reference}",
                    n=int(sizes[g]),
                    metric_value=value,
                    threshold=list(zone),
                    ci_low=float(bounds[0, j]),
                    ci_high=float(bounds[1, j]),
                    drift_status=verdict(float(bounds[0, j]), float(bounds[1, j]), value, zone),
                    method=f"share of the total importance in the group minus in the reference group; stratified bootstrap "
                    f"{n_bootstrap}, simultaneous over {tests} comparisons (Bonferroni)",
                    test_type="fixed",
                )
            )
    for g in sizes.index:
        if g not in kept:
            out.append(
                MetricResults(
                    metric_type="explainability",
                    metric_name="importance_share_difference_by_group",
                    group=f"{name}={g}",
                    reference_group=f"{name}={reference}",
                    suppressed=True,
                    method=(
                        "suppressed: complementary suppression, so that a smaller suppressed group cannot be computed "
                        "by difference"
                        if g in secondary
                        else f"suppressed: fewer than {min_group_size} rows"
                    ),
                )
            )
    return out
