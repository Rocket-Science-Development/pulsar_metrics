#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.special import rel_entr
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import pairwise_kernels

from ..exceptions import CustomExceptionPulsarMetric as error_msg


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
            bins = np.histogram_bin_edges(np.concatenate([reference, new]), bins="sturges")
            reference = pd.cut(reference, bins, include_lowest=True)
            new = pd.cut(new, bins, include_lowest=True)
        vector_all = pd.concat([reference, new], keys=["ref", "new"]).reset_index(0)
        percents = vector_all.groupby("level_0").value_counts(normalize=True).sort_index().unstack().T

    return percents


def get_population_counts(new: pd.Series, reference: pd.Series) -> pd.DataFrame:
    """Return the counts of the two pandas series per category, or per shared bin for numeric series

    Parameters
    ----------
    new : pd.Series
        The input pandas Series of the new population
    reference : pd.Series
        The input pandas Series of the reference population

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
        bins = np.histogram_bin_edges(np.concatenate([reference, new]), bins="sturges")
        reference = pd.cut(reference, bins, include_lowest=True)
        new = pd.cut(new, bins, include_lowest=True)
    counts = pd.concat([reference.value_counts(sort=False), new.value_counts(sort=False)], axis=1, keys=["ref", "new"])
    counts = counts.fillna(0)
    return counts.loc[counts.sum(axis=1) > 0]


def kl_divergence(new: pd.Series, reference: pd.Series, epsilon: float = 1e-6) -> float:
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

    Returns
    -------
    float
        returns the KL divergence, zero when the two distributions are identical
    """
    counts = get_population_counts(new, reference)
    p = counts["new"] / counts["new"].sum() + epsilon
    q = counts["ref"] / counts["ref"].sum() + epsilon
    return float(rel_entr(p / p.sum(), q / q.sum()).sum())


def chi2_test(new: pd.Series, reference: pd.Series, **kwargs):
    """Chi-squared test of homogeneity between the distributions of two samples

    Builds the 2 x k contingency table of counts (per category, or per shared bin for numeric series)
    and applies scipy.stats.chi2_contingency. The samples may have different sizes.

    Parameters
    ----------
    new : pd.Series
        The input pandas Series of the new population
    reference : pd.Series
        The input pandas Series of the reference population
    kwargs :
        keyworded variable length of arguments passed to scipy.stats.chi2_contingency

    Returns
    -------
    scipy.stats.contingency.Chi2ContingencyResult
        test result exposing statistic and pvalue
    """
    counts = get_population_counts(new, reference)
    return chi2_contingency(counts.T.to_numpy(), **kwargs)


def population_stability_index(new: pd.Series, reference: pd.Series, binned: bool = False):
    """Calculate the Population Stability Index (PSI) between two samples

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
        returns Population Stability Index (PSI) of two pandas series (new,reference)
    """

    percents = get_population_percentages(new, reference, binned)

    percent_diff = percents["new"] - percents["ref"]
    percent_ratio = percents["new"] / percents["ref"]

    return (percent_diff * np.log(percent_ratio)).sum()


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
