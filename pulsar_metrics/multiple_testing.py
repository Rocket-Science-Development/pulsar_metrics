"""Multiple-testing correction: dozens of features are tested in every window."""

import numpy as np
import pandas as pd

from .exceptions import CustomExceptionPulsarMetric as error_msg


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg procedure controlling the false discovery rate at `alpha`

    Parameters
    ----------
    pvalues : array-like
        p-values; NaN values are ignored (never rejected, adjusted value NaN)
    alpha : float, optional
        Target false discovery rate (default 0.05)

    Returns
    -------
    tuple of numpy arrays
        (rejected, adjusted p-values), in the order of the input
    """
    if not 0 < alpha < 1:
        raise error_msg(value=alpha, message="alpha must be in (0, 1)")
    p = np.asarray(pvalues, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)
    adjusted = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return rejected, adjusted

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    # step-up: adjusted p-value = min over larger ranks, capped at 1
    adj_sorted = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    adjusted[valid] = adj
    rejected[valid] = adj <= alpha
    return rejected, adjusted


def fdr_correct(results: pd.DataFrame, pvalue_column: str, alpha: float = 0.05) -> pd.DataFrame:
    """Add `p_adjusted` and `drift_fdr` columns to a results table, using Benjamini-Hochberg on `pvalue_column`

    Typical use: `pvalue_column="p_empirical"` filled by AACalibrator.empirical_pvalues(), which gives one p-value per
    (metric, feature) whatever the kind of statistic.
    """
    out = results.copy()
    rejected, adjusted = benjamini_hochberg(out[pvalue_column].to_numpy(dtype=float), alpha)
    out["p_adjusted"] = adjusted
    out["drift_fdr"] = np.where(np.isnan(adjusted), None, rejected)
    return out
