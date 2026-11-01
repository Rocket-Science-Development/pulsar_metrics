"""Feature attributions (TreeSHAP) and the importance measures derived from them."""

import numpy as np
import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg


def tree_attributions(model, X: pd.DataFrame, positive_class: int = 1) -> pd.DataFrame:
    """SHAP values of a tree model (scikit-learn, XGBoost, LightGBM, CatBoost) for every row of X

    Needs the `explain` extra (`pip install pulsar-metrics[explain]`). For classifiers, the attributions of
    `positive_class` are returned. Any other attribution method can be used instead: the rest of the module only
    needs a DataFrame with one row per prediction and one column per feature.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise error_msg(value=None, message="tree_attributions needs shap: pip install pulsar-metrics[explain]") from exc
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = values[positive_class]
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, positive_class]
    return pd.DataFrame(values, index=X.index, columns=X.columns)


def check_attributions(attributions) -> pd.DataFrame:
    if not isinstance(attributions, pd.DataFrame) or attributions.empty:
        raise error_msg(value=None, message="attributions must be a non-empty DataFrame (rows x features)")
    if not np.isfinite(attributions.to_numpy(dtype=float)).all():
        raise error_msg(value=None, message="attributions contain missing or infinite values")
    return attributions


def importance(attributions: pd.DataFrame) -> pd.Series:
    """Share of the total attribution carried by each feature: mean |attribution|, normalised to sum to 1"""
    magnitude = np.abs(attributions.to_numpy(dtype=float)).mean(axis=0)
    total = magnitude.sum()
    shares = magnitude / total if total > 0 else np.full(len(magnitude), 1 / len(magnitude))
    return pd.Series(shares, index=attributions.columns)


def ndcg_at_k(reference: pd.Series, current: pd.Series, k: int) -> float:
    """How well the current ranking of features keeps the reference ranking, in [0, 1] (1 = same top k)

    The features are ranked by current importance; each one is worth its reference importance; the discounted gain
    is divided by the gain of the reference ranking itself.
    """
    k = min(k, len(reference))
    relevance = reference.reindex(current.index).fillna(0.0)
    order = current.sort_values(ascending=False, kind="stable").index[:k]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    gain = float((relevance[order].to_numpy() * discounts).sum())
    ideal = float((np.sort(reference.to_numpy())[::-1][:k] * discounts).sum())
    return gain / ideal if ideal > 0 else 1.0


def top_k_jaccard(reference: pd.Series, current: pd.Series, k: int) -> float:
    """Overlap of the k most important features: |intersection| / |union|"""
    a = set(reference.sort_values(ascending=False, kind="stable").index[:k])
    b = set(current.sort_values(ascending=False, kind="stable").index[:k])
    return len(a & b) / len(a | b)
