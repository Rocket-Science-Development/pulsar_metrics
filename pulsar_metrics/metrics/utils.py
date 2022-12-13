#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

import numpy as np
import pandas as pd
from scipy.stats import entropy


def kl_divergence(new: pd.Series, reference: pd.Series):

    # Getting the same bins for both columns when they are numeric
    if (reference.dtype == float) & (new.dtype == float):
        bins = np.histogram_bin_edges(np.concatenate([reference, new]), bins="sturges")
        reference = pd.cut(reference, bins)
        new = pd.cut(new, bins)

    vector_all = pd.concat([reference, new], keys=["ref", "new"]).reset_index(0)

    percents = vector_all.groupby("level_0").value_counts(normalize=True).sort_index().unstack().T

    kl_div = entropy(percents["new"], percents["ref"])

    return kl_div
