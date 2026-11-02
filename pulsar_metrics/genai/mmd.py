"""Drift of text (requests, answers, retrieved passages) through their embeddings: MMD permutation test.

Feature tests (lengths, tools, tokens) miss a change of *content*: users asking about other topics, in another
language, with the same length. The maximum mean discrepancy between the embeddings of a reference window and of the
current window sees it. The kernel is a mixture of Gaussian kernels whose widths are multiples of the median distance
(0.5, 1, 2, 4), so that no width has to be tuned; the p-value comes from permutations, valid whatever the distribution
as long as the texts of a window are exchangeable.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults


@dataclass
class MMDResult:
    statistic: float  # unbiased MMD^2
    p_value: float
    n_reference: int
    n_current: int
    bandwidth: float  # median distance used as the unit of the kernel widths


def _project(reference: np.ndarray, current: np.ndarray, n_components: Optional[int]):
    """PCA fitted on both windows pooled, without their labels: the permutation test stays valid. (Fitted on the
    reference alone, the reference would be projected on its own noise directions and always look different.)"""
    if n_components is None or reference.shape[1] <= n_components:
        return reference, current
    pooled = np.vstack([reference, current])
    mean = pooled.mean(axis=0)
    _, _, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    basis = vt[:n_components].T
    return (reference - mean) @ basis, (current - mean) @ basis


def _mmd2(K: np.ndarray, in_x: np.ndarray) -> float:
    x, y = in_x, ~in_x
    n, m = x.sum(), y.sum()
    kxx = (K[np.ix_(x, x)].sum() - np.trace(K[np.ix_(x, x)])) / (n * (n - 1))
    kyy = (K[np.ix_(y, y)].sum() - np.trace(K[np.ix_(y, y)])) / (m * (m - 1))
    kxy = K[np.ix_(x, y)].mean()
    return float(kxx + kyy - 2 * kxy)


def mmd_test(
    reference,
    current,
    n_permutations: int = 500,
    widths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    n_components: Optional[int] = 32,
    max_samples: int = 1000,
    seed: Optional[int] = 0,
) -> MMDResult:
    """Permutation test of H0: the reference and current embeddings come from the same distribution

    Parameters
    ----------
    reference, current : array-like (n, d)
        Embeddings (one row per text) of the reference and current windows
    n_components : int, optional
        PCA dimensions, fitted on both windows pooled (default 32; None keeps the embeddings as they are)
    max_samples : int, optional
        Each window is subsampled to at most this many rows (the kernel matrix is quadratic)
    """
    rng = np.random.default_rng(seed)
    x, y = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise error_msg(value=None, message="reference and current must be 2-D arrays with the same number of columns")
    if len(x) < 5 or len(y) < 5:
        raise error_msg(value=None, message="at least 5 texts are needed in each window")
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    x, y = _project(x, y, n_components)
    z = np.vstack([x, y])
    sq = np.sum(z * z, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)
    median = float(np.sqrt(np.median(d2[np.triu_indices(len(z), 1)]))) or 1.0
    K = np.mean([np.exp(-d2 / (2 * (w * median) ** 2)) for w in widths], axis=0)
    labels = np.zeros(len(z), dtype=bool)
    labels[: len(x)] = True
    observed = _mmd2(K, labels)
    exceed = 0
    for _ in range(n_permutations):
        if _mmd2(K, rng.permutation(labels)) >= observed:
            exceed += 1
    return MMDResult(observed, (exceed + 1) / (n_permutations + 1), len(x), len(y), median)


def embedding_drift(reference, current, feature_name: str = "embedding", alpha: float = 0.05, **kwargs) -> MetricResults:
    """MMD test as a MetricResults (metric_type "drift", metric_name "mmd"); drift_status: p-value below alpha"""
    result = mmd_test(reference, current, **kwargs)
    return MetricResults(
        metric_type="drift",
        metric_name="mmd",
        feature_name=feature_name,
        metric_value=result.statistic,
        threshold=alpha,
        drift_status=result.p_value < alpha,
        n=result.n_current,
        method=f"MMD^2 (unbiased), Gaussian mixture kernel, widths x median distance {result.bandwidth:.3g}; "
        f"permutation p-value {result.p_value:.4f}; reference {result.n_reference} texts",
        test_type="fixed",
        p_value=result.p_value,
    )
