"""Drift of text (requests, answers, retrieved passages) through their embeddings: MMD permutation test.

Feature tests (lengths, tools, tokens) miss a change of *content*: users asking about other topics, in another
language, with the same length. The maximum mean discrepancy between the embeddings of a reference window and of the
current window sees it. The kernel is a mixture of Gaussian kernels whose widths are multiples of the median distance
(0.5, 1, 2, 4), so that no width has to be tuned; the p-value comes from permutations, valid whatever the distribution
as long as the texts of a window are exchangeable.

The MMD is an omnibus test: it spreads its power over every direction of the embedding space. Two kinds of change are
typically missed at a few hundred texts per window (night lab, TX-02):

- a change of **mix** along one direction (share of positive reviews from 50 % to 65 %): `component_test` (mean shift
  of the PCA components, whitened Hotelling statistic or per-component max-t) concentrates on it;
- a **small share of texts unlike the reference** (5 % of texts in another language): `novelty_test` scores each text
  by its distance to the nearest reference texts and tests the share of novel texts.

`detection_power` estimates, by simulation on your own embeddings, the share of windows of a given size in which a
test detects a given change: the minimal detectable change per window size.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults

_CHUNK = 64  # permutations evaluated per matrix product


@dataclass
class MMDResult:
    statistic: float  # unbiased MMD^2
    p_value: float
    n_reference: int
    n_current: int
    bandwidth: float  # median distance used as the unit of the kernel widths


def _check(reference, current, min_rows: int = 5):
    x, y = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise error_msg(value=None, message="reference and current must be 2-D arrays with the same number of columns")
    if len(x) < min_rows or len(y) < min_rows:
        raise error_msg(value=None, message=f"at least {min_rows} texts are needed in each window")
    return x, y


def _subsample(x, y, max_samples, rng):
    if max_samples is not None and len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if max_samples is not None and len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    return x, y


def _project(reference: np.ndarray, current: np.ndarray, n_components: Optional[int]):
    """PCA fitted on both windows pooled, without their labels: the permutation test stays valid. (Fitted on the
    reference alone, the reference would be projected on its own noise directions and always look different.)"""
    if n_components is None or reference.shape[1] <= n_components:
        return reference, current
    pooled = np.vstack([reference, current])
    mean = pooled.mean(axis=0)
    centred = pooled - mean
    # eigendecomposition of the smaller of the covariance (d x d) and Gram (N x N) matrices: same subspace as the SVD,
    # several times faster (the projection is defined up to the sign of each axis, which no test here depends on)
    if centred.shape[1] <= centred.shape[0]:
        values, vectors = np.linalg.eigh(centred.T @ centred)
        basis = vectors[:, ::-1][:, :n_components]
    else:
        values, vectors = np.linalg.eigh(centred @ centred.T)
        top = vectors[:, ::-1][:, :n_components]
        basis = centred.T @ top / np.sqrt(np.maximum(values[::-1][:n_components], 1e-300))
    return (reference - mean) @ basis, (current - mean) @ basis


def _mmd2(K: np.ndarray, in_x: np.ndarray) -> float:
    """Unbiased MMD^2 for one labelling (kept for reference; the test uses the vectorized `_mmd2_many`)"""
    return float(_mmd2_many(K, in_x[:, None].astype(float), K.sum(axis=1), float(np.trace(K)))[0])


def _mmd2_many(K: np.ndarray, W: np.ndarray, row_sums: np.ndarray, trace: float) -> np.ndarray:
    """Unbiased MMD^2 for each column of the 0/1 label matrix W (N x P, 1 = reference), with one matrix product.

    With w a label column: sum_xx = w'Kw, sum_xy = w'K1 - w'Kw, sum_yy = 1'K1 - sum_xx - 2 sum_xy; the diagonal is
    removed from the within-window sums. No copy of K per permutation (the former np.ix_ indexing copied it)."""
    diag = np.diag(K)
    n = W[:, 0].sum()
    m = len(W) - n
    KW = K @ W
    sxx = np.einsum("ij,ij->j", W, KW)
    sxy = row_sums @ W - sxx
    syy = row_sums.sum() - sxx - 2 * sxy
    dx = diag @ W
    kxx = (sxx - dx) / (n * (n - 1))
    kyy = (syy - (trace - dx)) / (m * (m - 1))
    return kxx + kyy - 2 * sxy / (n * m)


def _permutation_labels(labels: np.ndarray, n_permutations: int, rng):
    """Yield blocks of permuted label columns (same random stream as one rng.permutation per permutation)"""
    done = 0
    while done < n_permutations:
        size = min(_CHUNK, n_permutations - done)
        yield np.stack([rng.permutation(labels) for _ in range(size)], axis=1)
        done += size


def mmd_test(
    reference,
    current,
    n_permutations: int = 500,
    widths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    n_components: Optional[int] = 32,
    max_samples: int = 1000,
    seed: Optional[int] = 0,
    bandwidth: Optional[float] = None,
) -> MMDResult:
    """Permutation test of H0: the reference and current embeddings come from the same distribution

    Parameters
    ----------
    reference, current : array-like (n, d)
        Embeddings (one row per text) of the reference and current windows
    n_permutations : int, optional
        Permutations of the reference / current labels (default 500)
    widths : sequence of float, optional
        Widths of the Gaussian kernels, in units of `bandwidth` (default 0.5, 1, 2, 4). Smaller widths look at local
        structure (a small cluster of new texts), larger ones at global shifts.
    n_components : int, optional
        PCA dimensions, fitted on both windows pooled (default 32; None keeps the embeddings as they are). The lab
        measured the power on a polarity mix shift for none / 8 / 32 / 128 components: 0.05 / 0.50 / 0.55 / 0.20.
    max_samples : int, optional
        Each window is subsampled to at most this many rows (the kernel matrix is quadratic)
    bandwidth : float, optional
        Unit of the kernel widths (default: the median pairwise distance of the pooled windows)
    """
    rng = np.random.default_rng(seed)
    x, y = _check(reference, current)
    x, y = _subsample(x, y, max_samples, rng)
    x, y = _project(x, y, n_components)
    z = np.vstack([x, y])
    sq = np.sum(z * z, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)
    if bandwidth is None:
        median = float(np.sqrt(np.median(d2[np.triu_indices(len(z), 1)]))) or 1.0
    else:
        if bandwidth <= 0:
            raise error_msg(value=bandwidth, message="bandwidth must be positive")
        median = float(bandwidth)
    K = np.mean([np.exp(-d2 / (2 * (w * median) ** 2)) for w in widths], axis=0)
    labels = np.zeros(len(z), dtype=bool)
    labels[: len(x)] = True
    row_sums, trace = K.sum(axis=1), float(np.trace(K))
    observed = float(_mmd2_many(K, labels[:, None].astype(float), row_sums, trace)[0])
    exceed = 0
    for block in _permutation_labels(labels, n_permutations, rng):
        exceed += int(np.sum(_mmd2_many(K, block.astype(float), row_sums, trace) >= observed))
    return MMDResult(observed, (exceed + 1) / (n_permutations + 1), len(x), len(y), median)


# ----------------------------------------------------------------------------------------------- mean shift of components
@dataclass
class ComponentTestResult:
    statistic: float  # Hotelling: whitened squared distance of the means; max_t: largest |t| over the components
    p_value: float
    n_reference: int
    n_current: int
    statistic_name: str
    component: Optional[int] = None  # max_t: component with the largest |t| (0 = first principal component)
    t_values: list = field(default_factory=list)  # max_t: t value of every component


def component_test(
    reference,
    current,
    n_components: Optional[int] = 32,
    statistic: str = "hotelling",
    shrinkage: float = 1.0,
    n_permutations: int = 1000,
    max_samples: Optional[int] = 5000,
    seed: Optional[int] = 0,
) -> ComponentTestResult:
    """Permutation test of a shift of the mean embedding, on the PCA components fitted on both windows pooled

    A change of the mix of texts (more positive reviews, more questions of one kind) moves the mean embedding along a
    few directions; the Gaussian-kernel MMD spreads its power over all of them, this test does not.

    Parameters
    ----------
    statistic : str, optional
        "hotelling" (default): squared distance of the window means, whitened by the pooled covariance with ridge
        `shrinkage` (Hotelling T^2 with a ridge; the statistic of the best linear classifier between the windows).
        "max_t": largest |Welch t| over the components (per-dimension test; the permutation distribution of the
        maximum keeps the level over all components, and `component` tells which one moved).
    shrinkage : float, optional
        Ridge added to the covariance, as a share of its mean eigenvalue (default 1.0). Small values whiten fully
        (sensitive to shifts along low-variance directions, also to their noise); large values approach the plain
        distance of the means (linear-kernel MMD), which favours the high-variance directions.

    The covariance is computed on the pooled windows, without labels, so a single matrix serves every permutation and
    the p-value is exact.
    """
    if statistic not in ("hotelling", "max_t"):
        raise error_msg(value=statistic, message="statistic must be 'hotelling' or 'max_t'")
    if shrinkage < 0:
        raise error_msg(value=shrinkage, message="shrinkage must be >= 0")
    rng = np.random.default_rng(seed)
    x, y = _check(reference, current)
    x, y = _subsample(x, y, max_samples, rng)
    x, y = _project(x, y, n_components)
    z = np.vstack([x, y])
    z = z - z.mean(axis=0)
    n, m = len(x), len(y)
    labels = np.zeros(len(z), dtype=bool)
    labels[:n] = True

    if statistic == "hotelling":
        cov = np.cov(z, rowvar=False).reshape(z.shape[1], z.shape[1])
        ridge = shrinkage * max(float(np.trace(cov)) / len(cov), 1e-12)
        whiten = np.linalg.inv(cov + ridge * np.eye(len(cov)))
        scale = n * m / (n + m)

        def stat(W):  # W: N x P labels; the pooled mean is 0, so mean_x - mean_y = sum_x (1/n + 1/m)
            diff = (z.T @ W) * (1 / n + 1 / m)
            return scale * np.einsum("ip,ij,jp->p", diff, whiten, diff)

    else:
        sq = z * z

        def welch(W):
            sx, sxx = z.T @ W, sq.T @ W
            sy, syy = z.sum(axis=0)[:, None] - sx, sq.sum(axis=0)[:, None] - sxx
            mx, my = sx / n, sy / m
            vx = np.maximum(sxx / n - mx * mx, 0) * n / (n - 1)
            vy = np.maximum(syy / m - my * my, 0) * m / (m - 1)
            se = np.sqrt(vx / n + vy / m)
            return np.where(se > 0, (mx - my) / np.where(se > 0, se, 1), 0.0)

        def stat(W):
            return np.max(np.abs(welch(W)), axis=0)

    observed_labels = labels[:, None].astype(float)
    observed = float(stat(observed_labels)[0])
    exceed = 0
    for block in _permutation_labels(labels, n_permutations, rng):
        exceed += int(np.sum(stat(block.astype(float)) >= observed))
    result = ComponentTestResult(observed, (exceed + 1) / (n_permutations + 1), n, m, statistic)
    if statistic == "max_t":
        t = welch(observed_labels)[:, 0]
        result.component = int(np.argmax(np.abs(t)))
        result.t_values = [float(v) for v in t]
    return result


# ----------------------------------------------------------------------------------------------- novelty
@dataclass
class NoveltyTestResult:
    novel_share: float  # share of current texts above the novelty threshold
    expected_share: float  # share expected without change (1 - quantile)
    p_value: float
    threshold: float  # novelty score above which a text is novel
    n_reference: int  # reference texts used as neighbours
    n_calibration: int  # reference texts used to set the threshold
    n_current: int
    scores: np.ndarray  # novelty score of every current text (mean distance to its k nearest reference texts)
    novel: np.ndarray  # indices of the current texts flagged as novel


def _knn_score(points: np.ndarray, anchors: np.ndarray, k: int) -> np.ndarray:
    sq_p, sq_a = np.sum(points * points, axis=1), np.sum(anchors * anchors, axis=1)
    out = np.empty(len(points))
    for start in range(0, len(points), 512):
        block = points[start : start + 512]
        d2 = np.maximum(sq_p[start : start + 512, None] + sq_a[None, :] - 2 * block @ anchors.T, 0)
        nearest = np.partition(d2, k - 1, axis=1)[:, :k]
        out[start : start + 512] = np.sqrt(nearest).mean(axis=1)
    return out


def novelty_test(
    reference,
    current,
    k: int = 10,
    quantile: float = 0.95,
    calibration_share: float = 0.5,
    n_components: Optional[int] = 32,
    n_permutations: int = 2000,
    max_samples: Optional[int] = 5000,
    seed: Optional[int] = 0,
) -> NoveltyTestResult:
    """Test of H0: no more current texts are far from the reference than reference texts themselves

    Each text gets a novelty score, its mean distance to its `k` nearest neighbours in one part of the reference.
    The other part of the reference (`calibration_share`) sets the threshold, its `quantile`; the test compares the
    share of current texts above it with the share of calibration texts above it (permutation p-value, exact since
    the scores do not depend on the labels). Built for a small share of texts unlike the reference: another language,
    another product, spam, prompt-injection campaigns. The indices of the novel texts are returned for review.
    """
    if not 0 < quantile < 1 or not 0 < calibration_share < 1:
        raise error_msg(value=None, message="0 < quantile < 1 and 0 < calibration_share < 1 are required")
    rng = np.random.default_rng(seed)
    x, y = _check(reference, current, min_rows=max(5, 2 * k))
    x, _ = _subsample(x, x[:1], max_samples, rng)
    if max_samples is not None and len(y) > max_samples:
        raise error_msg(value=len(y), message=f"current window above max_samples ({max_samples}): split it")
    x, y = _project(x, y, n_components)
    order = rng.permutation(len(x))
    n_cal = int(round(calibration_share * len(x)))
    if n_cal < 5 or len(x) - n_cal < k:
        raise error_msg(value=len(x), message="reference too small for the calibration split and k neighbours")
    anchors, calibration = x[order[n_cal:]], x[order[:n_cal]]
    cal_scores, cur_scores = _knn_score(calibration, anchors, k), _knn_score(y, anchors, k)
    threshold = float(np.quantile(cal_scores, quantile))
    pooled = np.concatenate([cal_scores, cur_scores])
    labels = np.zeros(len(pooled), dtype=bool)
    labels[n_cal:] = True  # current

    def share_above(lab):  # statistic: novel share of the current minus that of the calibration (threshold recomputed)
        cal = pooled[~lab]
        t = np.quantile(cal, quantile)
        return np.mean(pooled[lab] > t) - np.mean(cal > t)

    observed = share_above(labels)
    exceed = int(sum(share_above(rng.permutation(labels)) >= observed for _ in range(n_permutations)))
    novel = np.flatnonzero(cur_scores > threshold)
    return NoveltyTestResult(
        novel_share=float(len(novel) / len(y)),
        expected_share=float(np.mean(cal_scores > threshold)),
        p_value=(exceed + 1) / (n_permutations + 1),
        threshold=threshold,
        n_reference=len(anchors),
        n_calibration=n_cal,
        n_current=len(y),
        scores=cur_scores,
        novel=novel,
    )


# ----------------------------------------------------------------------------------------------- power
def detection_power(
    reference,
    pool,
    changed,
    shares: Sequence[float] = (0.0, 0.05, 0.1, 0.3),
    sizes: Sequence[int] = (100, 300),
    test: Optional[Callable] = None,
    alpha: float = 0.05,
    reps: int = 20,
    seed: Optional[int] = 0,
) -> list:
    """Share of windows in which `test` detects a change, by share of changed texts and window size (simulation)

    Parameters
    ----------
    reference : array-like (n, d)
        Reference embeddings, as used in production
    pool : array-like
        Embeddings of texts like the reference but not in it (held-out reference period): unchanged texts
    changed : array-like
        Embeddings of the changed texts (e.g. positive reviews only, texts in another language, images with the
        corruption): a current window of size n holds round(share * n) of them, the rest drawn from `pool`
    test : callable, optional
        (reference, current) -> result with a `p_value` (default: `mmd_test` with its defaults); use
        functools.partial to set options, e.g. partial(component_test, statistic="max_t")
    shares, sizes : sequences
        Grid of the simulation; share 0 measures the false alarm rate (A/A) of the same setting

    Returns one dict per (share, size): detection rate, Wilson 95 % interval, reps. The smallest share whose rate
    reaches 0.8 at a size is the minimal detectable change at that window size, for that encoder and that test.
    """
    test = test or mmd_test
    rng = np.random.default_rng(seed)
    pool, changed = np.asarray(pool, dtype=float), np.asarray(changed, dtype=float)
    out = []
    for size in sizes:
        for share in shares:
            n_changed = int(round(share * size))
            if n_changed > len(changed) or size - n_changed > len(pool):
                raise error_msg(value=size, message="pool or changed too small for this window size")
            hits = 0
            for _ in range(reps):
                window = np.vstack(
                    [
                        pool[rng.choice(len(pool), size - n_changed, replace=False)],
                        changed[rng.choice(len(changed), n_changed, replace=False)],
                    ]
                )
                hits += int(test(reference, window).p_value < alpha)
            ci = stats.binomtest(hits, reps).proportion_ci(method="wilson")
            out.append(
                {
                    "share": share,
                    "size": size,
                    "power": hits / reps,
                    "ci_low": float(ci.low),
                    "ci_high": float(ci.high),
                    "reps": reps,
                }
            )
    return out


# ----------------------------------------------------------------------------------------------- MetricResults
def embedding_drift(
    reference, current, feature_name: str = "embedding", alpha: float = 0.05, test: str = "mmd", **kwargs
) -> MetricResults:
    """Drift test on embeddings as a MetricResults; drift_status: p-value below alpha

    test : "mmd" (default; metric_type "drift", metric_name "mmd"), "component" (`component_test`, metric_name
    "embedding_mean_shift") or "novelty" (`novelty_test`, metric_name "embedding_novelty", metric_value = share of
    novel texts). The last two are reported with metric_type "custom" until their names join the drift metric list.
    """
    if test == "component":
        r = component_test(reference, current, **kwargs)
        where = f"; largest shift on component {r.component}" if r.component is not None else ""
        return MetricResults(
            metric_type="custom",
            metric_name="embedding_mean_shift",
            feature_name=feature_name,
            metric_value=r.statistic,
            threshold=alpha,
            drift_status=r.p_value < alpha,
            n=r.n_current,
            method=f"shift of the mean PCA components ({r.statistic_name}), permutation p-value {r.p_value:.4f}{where}; "
            f"reference {r.n_reference} texts",
            test_type="fixed",
            p_value=r.p_value,
        )
    if test == "novelty":
        r = novelty_test(reference, current, **kwargs)
        return MetricResults(
            metric_type="custom",
            metric_name="embedding_novelty",
            feature_name=feature_name,
            metric_value=r.novel_share,
            threshold=alpha,
            drift_status=r.p_value < alpha,
            n=r.n_current,
            method=f"share of texts farther from their nearest reference texts than {r.expected_share:.1%} of reference "
            f"texts: {r.novel_share:.1%} ({len(r.novel)} texts); permutation p-value {r.p_value:.4f}; "
            f"{r.n_reference} neighbours, {r.n_calibration} calibration texts",
            test_type="fixed",
            p_value=r.p_value,
        )
    if test != "mmd":
        raise error_msg(value=test, message="test must be 'mmd', 'component' or 'novelty'")
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
