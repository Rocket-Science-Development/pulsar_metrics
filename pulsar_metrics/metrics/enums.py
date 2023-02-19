#  Author:   Adel Benlagra  <abenlagra@rocketscience.one>

from enum import Enum
from functools import partial

from scipy.stats import (
    chisquare,
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
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .utils import kl_divergence, psi


class MetricsType(Enum):

    """Metrics type enumeration"""

    performance = "performance"
    drift = "drift"
    custom = "custom"
    statistics = "statistics"


class DriftMetricsFuncs(Enum):
    kl = partial(kl_divergence)
    psi = partial(psi)
    wasserstein = partial(wasserstein_distance)


class DriftTestMetricsFuncs(Enum):
    ttest = partial(ttest_ind, equal_var=False)
    manwu = partial(mannwhitneyu)
    levene = partial(levene, center="mean")
    bftest = partial(levene, center="median")
    ks_2samp = partial(ks_2samp)
    CvM = partial(cramervonmises_2samp)
    chi2 = partial(chisquare)


class PerformanceMetricsFuncs(Enum):

    """Set of performance metrics functions"""

    # Classification Metrics
    accuracy = partial(accuracy_score)
    precision = partial(precision_score)
    recall = partial(recall_score)
    f1 = partial(f1_score)
    log_loss = partial(log_loss)
    # Area under ROC Curve
    auc = partial(roc_auc_score)
    # Area under PR Curve
    aucpr = partial(average_precision_score)
    brier = partial(brier_score_loss)

    # Regression metrics
    mse = partial(mean_squared_error)
    mae = partial(mean_absolute_error)
    mape = partial(mean_absolute_error)
    r2 = partial(r2_score)
