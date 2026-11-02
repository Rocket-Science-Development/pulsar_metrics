#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from enum import Enum, member
from functools import partial

from scipy.stats import (
    cramervonmises_2samp,
    ks_2samp,
    levene,
    mannwhitneyu,
    ttest_ind,
    wasserstein_distance,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .utils import chi2_test, kl_divergence, max_mean_discrepency, population_stability_index


class MetricsType(Enum):
    """Metrics type enumeration"""

    performance = "performance"
    drift = "drift"
    custom = "custom"
    statistics = "statistics"
    fairness = "fairness"
    explainability = "explainability"
    agents = "agents"


class DriftMetricsFuncs(Enum):
    kl = member(partial(kl_divergence))
    psi = member(partial(population_stability_index))
    wasserstein = member(partial(wasserstein_distance))
    mmd = member(partial(max_mean_discrepency))


class DriftTestMetricsFuncs(Enum):
    ttest = member(partial(ttest_ind, equal_var=False))
    manwu = member(partial(mannwhitneyu))
    levene = member(partial(levene, center="mean"))
    bftest = member(partial(levene, center="median"))
    ks_2samp = member(partial(ks_2samp))
    CvM = member(partial(cramervonmises_2samp))
    chi2 = member(partial(chi2_test))


class PerformanceMetricsFuncs(Enum):
    """Set of performance metrics functions"""

    # Classification Metrics
    accuracy = member(partial(accuracy_score))
    precision = member(partial(precision_score))
    recall = member(partial(recall_score))
    f1 = member(partial(f1_score))
    log_loss = member(partial(log_loss))
    # Area under ROC Curve
    auc = member(partial(roc_auc_score))
    # Area under PR Curve
    aucpr = member(partial(average_precision_score))
    brier = member(partial(brier_score_loss))

    # Regression metrics
    mse = member(partial(mean_squared_error))
    mae = member(partial(mean_absolute_error))
    mape = member(partial(mean_absolute_percentage_error))
    r2 = member(partial(r2_score))
