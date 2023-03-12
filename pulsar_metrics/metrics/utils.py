#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_kernels


def get_population_percentages(new: pd.Series, reference: pd.Series, binned: bool = False):
    """
    The function returns the population percentages of the two pandas series

    Parameters
        ----------
        - new: pandas Series of the new population
        - reference: pandas Series of the reference population
        - binned: if the population values have already been binned into identical bins.
        If the two pandas series have different lengths
        missing indices are imputed with zeros.
    """

    try:
        if is_numeric_dtype(new) & is_numeric_dtype(reference):
            if not binned:
                bins = np.histogram_bin_edges(np.concatenate([reference, new]), bins="sturges")
                reference = pd.cut(reference, bins, include_lowest=True)
                new = pd.cut(new, bins, include_lowest=True)
                vector_all = pd.concat([reference, new], keys=["ref", "new"]).reset_index(0)
                percents = vector_all.groupby("level_0").value_counts(normalize=True).sort_index().unstack().T
            else:
                percents = pd.concat([new, reference], axis=1, keys=["ref", "new"]).fillna(0)
                percents = percents / percents / sum()

            return percents
    except Exception as e:
        print(str(e))


def kl_divergence(new: pd.Series, reference: pd.Series, binned: bool = False):
    """
    Calculates the Kullback-Leibler divergence
    """

    percents = get_population_percentages(new, reference, binned)

    kl_div = entropy(percents["new"], percents["ref"])

    return kl_div


def psi(new: pd.Series, reference: pd.Series, binned: bool = False):
    """
    Calculates the Population Stability Index (PSI)
    """

    percents = get_population_percentages(new, reference, binned)

    psi = ((percents["new"] - percents["ref"]) * np.log(percents["new"] / percents["ref"])).sum()

    return psi


def mmd(new: pd.DataFrame, reference: pd.DataFrame, kernel="linear", **kwargs):
    """
    Calculates the Maximum Mean Discrepency between two samples
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
