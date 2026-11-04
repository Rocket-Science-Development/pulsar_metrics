#  Copyright Développement Science Fusée inc.
#  SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.special import rel_entr
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics.pairwise import pairwise_kernels

from ..exceptions import CustomExceptionPulsarMetric as error_msg

EPSILON = 1e-6  # additive smoothing of kl_divergence and population_stability_index
LOW_EXPECTED = 5  # a contingency cell with an expected count below it makes the chi2 approximation unreliable (Cochran)
OTHER_LEVEL = "__other__"  # level that gathers the levels merged by group_levels


def _as_float(series: pd.Series) -> pd.Series:
    """Numeric data are binned as float whatever their dtype (booleans are left as they are)

    pd.cut labels the bins of integer data differently from those of float data and merges neighbouring bins: on a
    0-4 integer feature, values 0 and 1 fell in one bin as int64 and in two as float64. Binning as float makes PSI, KL
    and chi2 independent of the dtype, e.g. of a column that becomes float in some windows only (masked sentinel), and
    keeps them equal to the A/A calibration, which runs on float arrays.
    """
    return series if is_bool_dtype(series) else series.astype(float)


def get_population_percentages(new: pd.Series, reference: pd.Series, binned: bool = False):
    """Return the population percentages of the two pandas series[new,reference]

    Parameters
    ----------
    new : pd.Series
        The input pandas Series of the new population
    reference : pd.Series
        The input pandas Series of the reference population
    binned : bool, optional
        if the population values have already been binned into identical bins.
        If the two pandas series have different lengths
        missing indices are imputed with zeros.

    Raises
    ------
        ValueError if input value have not correct

    Returns
    -------
    pd.Series
        returns population percentages of the two pandas series (new,reference)
    """

    if binned:
        percents = pd.concat([reference, new], axis=1, keys=["ref", "new"]).fillna(0)
        percents = percents / percents.sum()
    elif new.dtype != reference.dtype:
        raise error_msg(
            value=None,
            message=f'{"New and reference series should be numeric or object and should have the same type"}',
        )
    else:
        if is_numeric_dtype(new):
            new, reference = _as_float(new), _as_float(reference)
            bins = np.histogram_bin_edges(np.concatenate([reference, new]), bins="sturges")
            reference = pd.cut(reference, bins, include_lowest=True)
            new = pd.cut(new, bins, include_lowest=True)
        vector_all = pd.concat([reference, new], keys=["ref", "new"]).reset_index(0)
        percents = vector_all.groupby("level_0").value_counts(normalize=True).sort_index().unstack().T

    return percents


def get_population_counts(
    new: pd.Series, reference: pd.Series, min_expected: Optional[float] = None, max_levels: Optional[int] = None
) -> pd.DataFrame:
    """Return the counts of the two pandas series per category, or per shared bin for numeric series

    Parameters
    ----------
    new : pd.Series
        The input pandas Series of the new population
    reference : pd.Series
        The input pandas Series of the reference population
    min_expected, max_levels : optional
        Level grouping, see group_levels (default None: every category or bin is kept)

    Raises
    ------
        CustomExceptionPulsarMetric if the two series do not have the same type

    Returns
    -------
    pd.DataFrame
        one row per category or bin, columns "ref" and "new"; categories missing from a sample count zero
    """
    if new.dtype != reference.dtype:
        raise error_msg(
            value=None,
            message=f'{"New and reference series should be numeric or object and should have the same type"}',
        )
    if is_numeric_dtype(new):
        new, reference = _as_float(new), _as_float(reference)
        bins = np.histogram_bin_edges(np.concatenate([reference, new]), bins="sturges")
        reference = pd.cut(reference, bins, include_lowest=True)
        new = pd.cut(new, bins, include_lowest=True)
    counts = pd.concat([reference.value_counts(sort=False), new.value_counts(sort=False)], axis=1, keys=["ref", "new"])
    counts = counts.fillna(0)
    counts = counts.loc[counts.sum(axis=1) > 0]
    return group_levels(counts, min_expected=min_expected, max_levels=max_levels)


def group_levels(counts: pd.DataFrame, min_expected: Optional[float] = None, max_levels: Optional[int] = None) -> pd.DataFrame:
    """Merge the sparse levels of a count table into one "other" level, chosen on the reference proportions

    With hundreds of levels (airports, products...), most cells of the 2 x k table are nearly empty: chi2 spends its
    degrees of freedom on them and loses power, and psi / kl add noise (lab TC-02, 293 levels: power 0.05 raw vs 0.275
    grouped at 2000 rows). The levels to merge depend only on the reference proportions and on the size of the current
    sample, so the same levels are merged in the current window and in every A/A draw of a calibration. Levels absent
    from the reference (new categories) are merged too: they still weigh in "other", where a new category shows as a
    large current share against a small reference one.

    Parameters
    ----------
    counts : pd.DataFrame
        Table from get_population_counts (columns "ref" and "new")
    min_expected : float, optional
        Merge the levels whose expected count in the current sample, n_new x reference share, is below it (5 is the
        usual chi2 rule)
    max_levels : int, optional
        Keep at most this many levels, "other" included: the max_levels - 1 most frequent in the reference

    Returns
    -------
    pd.DataFrame
        the table with the merged levels summed into OTHER_LEVEL; unchanged when nothing is merged. The most frequent
        reference level is never merged, so that two levels at least remain.
    """
    if min_expected is None and max_levels is None:
        return counts
    if max_levels is not None and max_levels < 2:
        raise error_msg(value=max_levels, message="max_levels must be at least 2 (one level and 'other')")
    ref_total, new_total = counts["ref"].sum(), counts["new"].sum()
    if ref_total == 0 or len(counts) < 2:
        return counts
    shares = counts["ref"] / ref_total
    merged = pd.Series(False, index=counts.index)
    if min_expected is not None:
        merged |= new_total * shares < min_expected
    if max_levels is not None:
        merged |= shares.rank(method="first", ascending=False) > max_levels - 1
    merged.loc[shares.idxmax()] = False
    if not merged.any():
        return counts
    kept = counts.loc[~merged]
    kept.index = kept.index.astype(object)
    other = pd.DataFrame([counts.loc[merged].sum()], index=[OTHER_LEVEL])
    return pd.concat([kept, other])


def low_expected_share(counts: pd.DataFrame, min_expected: float = LOW_EXPECTED) -> Optional[float]:
    """Share of the cells of the 2 x k table whose expected count (row total x column total / n) is below min_expected

    Above 0.2, the chi2 approximation is commonly considered unreliable and psi / kl are dominated by sparse levels:
    group them (min_expected= or max_levels= of the drift metric).
    """
    total = counts.to_numpy().sum()
    if total == 0 or len(counts) == 0:
        return None
    expected = np.outer(counts.sum(axis=1).to_numpy(), counts.sum(axis=0).to_numpy()) / total
    return float((expected < min_expected).mean())


def kl_divergence(
    new: pd.Series,
    reference: pd.Series,
    epsilon: float = EPSILON,
    min_expected: Optional[float] = None,
    max_levels: Optional[int] = None,
) -> float:
    """Calculate the Kullback-Leibler divergence D_KL(new || reference) between two samples

    Both samples are binned on shared bins (numeric) or counted per category, then turned into
    probability distributions. `epsilon` is added to every probability before renormalisation so that
    a category absent from the reference sample gives a large but finite divergence.

    Parameters
    ----------
    new : pd.Series
        The input pandas Series of the new population
    reference : pd.Series
        The input pandas Series of the reference population
    epsilon : float, optional
        Additive smoothing applied to the probabilities
    min_expected, max_levels : optional
        Level grouping, see group_levels (default None: none)

    Returns
    -------
    float
        returns the KL divergence, zero when the two distributions are identical
    """
    counts = get_population_counts(new, reference, min_expected=min_expected, max_levels=max_levels)
    p = counts["new"] / counts["new"].sum() + epsilon
    q = counts["ref"] / counts["ref"].sum() + epsilon
    return float(rel_entr(p / p.sum(), q / q.sum()).sum())


def chi2_test(
    new: pd.Series, reference: pd.Series, min_expected: Optional[float] = None, max_levels: Optional[int] = None, **kwargs
):
    """Chi-squared test of homogeneity between the distributions of two samples

    Builds the 2 x k contingency table of counts (per category, or per shared bin for numeric series)
    and applies scipy.stats.chi2_contingency. The samples may have different sizes.

    Parameters
    ----------
    new : pd.Series
        The input pandas Series of the new population
    reference : pd.Series
        The input pandas Series of the reference population
    min_expected, max_levels : optional
        Level grouping, see group_levels (default None: none). With many sparse levels, min_expected=5 restores power.
    kwargs :
        keyworded variable length of arguments passed to scipy.stats.chi2_contingency

    Returns
    -------
    scipy.stats.contingency.Chi2ContingencyResult
        test result exposing statistic and pvalue
    """
    counts = get_population_counts(new, reference, min_expected=min_expected, max_levels=max_levels)
    return chi2_contingency(counts.T.to_numpy(), **kwargs)


def missing_rate_test(new, reference):
    """Fisher exact test of the share of missing values (NaN) between two samples

    Detects a change of the missing rate, e.g. a sensor that starts returning its "no value" sentinel more often. The
    sentinels should be turned into NaN first (`missing_values=` of DriftTestMetric or Analyzer.add_drift_metrics). For
    a DataFrame, a row counts as missing when any of its values is.

    Parameters
    ----------
    new : pd.Series or pd.DataFrame
        The input pandas Series of the new population
    reference : pd.Series or pd.DataFrame
        The input pandas Series of the reference population

    Returns
    -------
    scipy.stats result or SimpleNamespace
        exposing statistic (odds ratio, NaN when undefined) and pvalue; pvalue is 1.0 when no row, or every row, is
        missing in both samples
    """

    def counts(x):
        missing = x.isna().any(axis=1) if isinstance(x, pd.DataFrame) else x.isna()
        return int(missing.sum()), int((~missing).sum())

    new_missing, new_present = counts(new)
    ref_missing, ref_present = counts(reference)
    if new_missing + ref_missing == 0 or new_present + ref_present == 0:
        return SimpleNamespace(statistic=float("nan"), pvalue=1.0)
    return fisher_exact([[new_missing, new_present], [ref_missing, ref_present]])


def population_stability_index(
    new: pd.Series,
    reference: pd.Series,
    binned: bool = False,
    epsilon: float = EPSILON,
    min_expected: Optional[float] = None,
    max_levels: Optional[int] = None,
):
    """Calculate the Population Stability Index (PSI) between two samples

    PSI = sum((p_new - p_ref) * ln(p_new / p_ref)) over shared bins (numeric) or categories. `epsilon` is
    added to every proportion before renormalisation, as for kl_divergence, so that a bin empty in one
    sample gives a large but finite value instead of inf (or being silently skipped).

    Parameters
    ----------
    new : pd.Series
        The input pandas Series of the new population
    reference : pd.Series
        The input pandas Series of the reference population
    binned : bool, optional
        if the population values have already been binned into identical bins, i.e. the two series hold
        counts indexed by bin. Bins missing from one series count zero.
    epsilon : float, optional
        Additive smoothing applied to the proportions. It also caps the value: two samples without any shared bin give
        about 2 ln(1 / epsilon) = 27.6, however large the shift.
    min_expected, max_levels : optional
        Level grouping, see group_levels (default None: none)

    Returns
    -------
    float
        returns Population Stability Index (PSI) of two pandas series (new,reference)
    """
    if binned:
        counts = pd.concat([reference, new], axis=1, keys=["ref", "new"]).fillna(0)
        counts = group_levels(counts, min_expected=min_expected, max_levels=max_levels)
    else:
        counts = get_population_counts(new, reference, min_expected=min_expected, max_levels=max_levels)
    p = counts["new"] / counts["new"].sum() + epsilon
    q = counts["ref"] / counts["ref"].sum() + epsilon
    p, q = p / p.sum(), q / q.sum()
    return float(((p - q) * np.log(p / q)).sum())


def max_mean_discrepency(new: pd.DataFrame, reference: pd.DataFrame, kernel="linear", **kwargs):
    """Calculate the Maximum Mean Discrepency(MMD) between two samples[new,reference]

    Parameters
    ----------
    new : pd.DataFrame
        The input pandas Series of the new population
    reference : pd.DataFrame
        The input pandas Series of the reference population
    kernel : str, optional
        represent linear transformation
    kwargs :
        keyworded variable length of arguments to a function

    Returns
    -------
    pd.DataFrame
        returns Maximum Mean Discrepency(MMD) between two samples[new,reference]
    """

    if isinstance(new, pd.Series):
        new = new.to_frame()

    if isinstance(reference, pd.Series):
        reference = reference.to_frame()

    new = new.select_dtypes("number")
    reference = reference.select_dtypes("number")

    kxx = pairwise_kernels(new, new, metric=kernel, **kwargs)
    kyy = pairwise_kernels(reference, reference, metric=kernel, **kwargs)
    kxy = pairwise_kernels(new, reference, metric=kernel, **kwargs)

    return kxx.mean() + kyy.mean() - 2 * kxy.mean()
