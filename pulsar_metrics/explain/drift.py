"""Drift of explanations: has the model changed what it relies on between a reference window and the current one?

Explanations are themselves noisy estimates, so every comparison carries a bootstrap interval, and the stability of the
explanations on the reference window is measured first: when the ranking of features is not stable at that sample
size, no verdict is given (the instrument, not the model, would be speaking).
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..fairness.criteria import verdict
from ..metrics.base import MetricResults
from .attributions import check_attributions, importance, ndcg_at_k, top_k_jaccard


def _shares(magnitude: np.ndarray, rows: np.ndarray, columns) -> pd.Series:
    mean = magnitude[rows].mean(axis=0)
    total = mean.sum()
    return pd.Series(mean / total if total > 0 else np.full(len(mean), 1 / len(mean)), index=columns)


def _at_least(threshold: float, low: float, high: float) -> Optional[bool]:
    """Verdict for a similarity that should stay above `threshold`: True = drift, False = stable, None = inconclusive"""
    if high < threshold:
        return True
    if low >= threshold:
        return False
    return None


class AttributionDrift:
    """Compare feature attributions of a current window with those of a reference window

    Parameters
    ----------
    k : int, optional
        Number of leading features compared (default 5)
    ndcg_threshold : float, optional
        Minimum NDCG@k of the current ranking against the reference one (default 0.9)
    jaccard_threshold : float, optional
        Minimum overlap of the top k features (default 0.6: with k = 5, one feature may leave the top k)
    share_tolerance : float, optional
        Acceptable change of a feature's share of the total importance (default 0.05)
    n_bootstrap : int, optional
    confidence : float, optional
        For the per-feature intervals, the confidence is Bonferroni-adjusted over the features
    seed : int, optional
    """

    def __init__(
        self,
        k: int = 5,
        ndcg_threshold: float = 0.9,
        jaccard_threshold: float = 0.6,
        share_tolerance: float = 0.05,
        n_bootstrap: int = 200,
        confidence: float = 0.95,
        seed: Optional[int] = 0,
    ):
        if k < 1 or n_bootstrap < 20 or not 0.5 < confidence < 1:
            raise error_msg(value=None, message="k >= 1, n_bootstrap >= 20 and 0.5 < confidence < 1 are required")
        self.k = k
        self.ndcg_threshold = ndcg_threshold
        self.jaccard_threshold = jaccard_threshold
        self.share_tolerance = share_tolerance
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.seed = seed
        self.reference = None
        self.reference_importance = None
        self.stability = None
        self._results = []

    def fit(self, reference: pd.DataFrame) -> "AttributionDrift":
        """Store the reference attributions and measure how stable their ranking is at this sample size"""
        reference = check_attributions(reference)
        self.reference = reference
        self.reference_importance = importance(reference)
        rng = np.random.default_rng(self.seed)
        magnitude = np.abs(reference.to_numpy(dtype=float))
        n = len(magnitude)
        scores = [
            ndcg_at_k(self.reference_importance, _shares(magnitude, rng.integers(0, n, n), reference.columns), self.k)
            for _ in range(self.n_bootstrap)
        ]
        low = float(np.quantile(scores, (1 - self.confidence) / 2))
        self.stability = {"ndcg_low": low, "stable": low >= self.ndcg_threshold, "n": n}
        return self

    def run(self, current: pd.DataFrame) -> list:
        if self.reference is None:
            raise error_msg(value=None, message="call fit(reference) before run(current)")
        current = check_attributions(current)
        missing = set(self.reference.columns) ^ set(current.columns)
        if missing:
            raise error_msg(value=sorted(missing), message="reference and current attributions must have the same features")
        current = current[self.reference.columns]
        columns = self.reference.columns
        ref_mag = np.abs(self.reference.to_numpy(dtype=float))
        cur_mag = np.abs(current.to_numpy(dtype=float))
        ref_imp = self.reference_importance
        cur_imp = importance(current)

        rng = np.random.default_rng(self.seed)
        n_ref, n_cur = len(ref_mag), len(cur_mag)
        ndcgs, jaccards, differences = [], [], []
        for _ in range(self.n_bootstrap):
            ref_b = _shares(ref_mag, rng.integers(0, n_ref, n_ref), columns)
            cur_b = _shares(cur_mag, rng.integers(0, n_cur, n_cur), columns)
            ndcgs.append(ndcg_at_k(ref_b, cur_b, self.k))
            jaccards.append(top_k_jaccard(ref_b, cur_b, self.k))
            differences.append((cur_b - ref_b).to_numpy())

        tail = (1 - self.confidence) / 2
        stable = self.stability["stable"]
        unstable_note = (
            f"; no verdict: the ranking is not stable on the reference window (NDCG@{self.k} lower bound "
            f"{self.stability['ndcg_low']:.3f} < {self.ndcg_threshold} with {self.stability['n']} rows)"
        )
        common = dict(metric_type="explainability", n=n_cur, test_type="fixed")

        out = []
        for name, value, samples, threshold, desc in (
            ("attribution_ndcg_at_k", ndcg_at_k(ref_imp, cur_imp, self.k), ndcgs, self.ndcg_threshold, "NDCG"),
            ("attribution_top_k_jaccard", top_k_jaccard(ref_imp, cur_imp, self.k), jaccards, self.jaccard_threshold, "Jaccard"),
        ):
            low, high = (float(q) for q in np.quantile(samples, [tail, 1 - tail]))
            method = f"{desc} of the top {self.k} features, bootstrap {self.n_bootstrap} on both windows" + (
                "" if stable else unstable_note
            )
            out.append(
                MetricResults(
                    metric_name=name,
                    metric_value=value,
                    threshold=threshold,
                    ci_low=low,
                    ci_high=high,
                    drift_status=_at_least(threshold, low, high) if stable else None,
                    method=method,
                    **common,
                )
            )

        per_feature_tail = (1 - self.confidence) / (2 * len(columns))
        bounds = np.quantile(np.array(differences), [per_feature_tail, 1 - per_feature_tail], axis=0)
        for j, feature in enumerate(columns):
            value = float(cur_imp[feature] - ref_imp[feature])
            low, high = float(bounds[0, j]), float(bounds[1, j])
            out.append(
                MetricResults(
                    metric_name="importance_share_difference",
                    feature_name=feature,
                    metric_value=value,
                    threshold=[-self.share_tolerance, self.share_tolerance],
                    ci_low=low,
                    ci_high=high,
                    drift_status=verdict(low, high, value, (-self.share_tolerance, self.share_tolerance)),
                    method=f"current minus reference share of the total importance; bootstrap {self.n_bootstrap}, "
                    f"simultaneous over {len(columns)} features (Bonferroni)",
                    **common,
                )
            )
        self._results = out
        return out

    def get_result(self) -> list:
        return self._results

    def results_to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self._results])
