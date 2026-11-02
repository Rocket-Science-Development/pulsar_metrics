"""Score intervals for rates and sample-size planning: how many rows, probe sets or decisions a fairness verdict needs.

Small groups get no verdict (interval straddling a bound) or no value (suppressed cell). That is honest, but it hides
the question users ask next: how much data would settle it? The functions here answer it for the verdicts of
`FairnessAnalyzer`, `counterfactual_analysis` and `FairnessMonitor`, from the rates observed (or expected).

Intervals of rate differences and ratios use Wilson score intervals combined by Newcombe's hybrid method (difference)
and the MOVER-R method of Donner and Zou (ratio): unlike the percentile bootstrap, they keep their level for groups of
a few dozen rows (night lab FA-04: the percentile bootstrap missed on one side 3 to 5 % of the time instead of 2.5 %).
"""

import math
from functools import lru_cache
from statistics import NormalDist
from typing import Optional, Tuple

import numpy as np

from ..exceptions import CustomExceptionPulsarMetric as error_msg

_MAX_SIZE = 10**8


@lru_cache(maxsize=256)
def _z(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise error_msg(value=confidence, message="confidence must be in (0, 1)")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson(successes: float, n: float, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval of a proportion (closed form; NaN when n = 0)"""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = _z(confidence)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denominator
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def newcombe_difference(p1: float, ci1: tuple, p2: float, ci2: tuple) -> Tuple[float, float]:
    """Interval of p1 - p2 for two independent proportions from their Wilson intervals (Newcombe hybrid score)"""
    d = p1 - p2
    return (d - math.hypot(p1 - ci1[0], ci2[1] - p2), d + math.hypot(ci1[1] - p1, p2 - ci2[0]))


def mover_ratio(p1: float, ci1: tuple, p2: float, ci2: tuple) -> Tuple[float, float]:
    """Interval of p1 / p2 from the intervals of p1 and p2 (MOVER-R, Donner and Zou 2012); NaN when p2 = 0"""
    l1, u1 = ci1
    l2, u2 = ci2
    if p2 <= 0:
        return (float("nan"), float("nan"))
    product = p1 * p2
    low_den = u2 * (2 * p2 - u2)
    high_den = l2 * (2 * p2 - l2)
    if low_den > 0:
        low = (product - math.sqrt(max(product**2 - l1 * u2 * (2 * p1 - l1) * (2 * p2 - u2), 0.0))) / low_den
    else:
        low = 0.0
    if high_den > 0:
        high = (product + math.sqrt(max(product**2 - u1 * l2 * (2 * p1 - u1) * (2 * p2 - l2), 0.0))) / high_den
    else:
        high = float("inf")
    return (max(low, 0.0), high)


def rate_interval(
    successes_group: float,
    n_group: float,
    successes_reference: float,
    n_reference: float,
    kind: str = "difference",
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Score interval of the difference (group - reference) or ratio (group / reference) of two rates"""
    if kind not in ("difference", "ratio"):
        raise error_msg(value=kind, message="kind must be 'difference' or 'ratio'")
    if n_group <= 0 or n_reference <= 0:
        return (float("nan"), float("nan"))
    p1, p2 = successes_group / n_group, successes_reference / n_reference
    ci1, ci2 = wilson(successes_group, n_group, confidence), wilson(successes_reference, n_reference, confidence)
    return newcombe_difference(p1, ci1, p2, ci2) if kind == "difference" else mover_ratio(p1, ci1, p2, ci2)


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _verdict_probability(value: float, low: float, high: float, zone: tuple, z: float) -> float:
    """Probability that the verdict is conclusive when the true disparity is `value` and the interval around an
    estimate spans [estimate - (value - low), estimate + (high - value)]: normal approximation, se = half-width / z"""
    if not (np.isfinite(low) and np.isfinite(high)):
        return 0.0
    below, above = value - low, high - value
    se = max((below + above) / (2 * z), 1e-12)
    lo, hi = zone
    within = max(_phi((hi - above - value) / se) - _phi((lo + below - value) / se), 0.0)
    outside = _phi((lo - above - value) / se) + 1 - _phi((hi + below - value) / se)
    return within + outside


def _smallest(conclusive, start: int) -> Optional[int]:
    """Smallest n >= start with conclusive(n) True, assuming monotonicity; None if none below _MAX_SIZE"""
    if conclusive(start):
        return start
    low, high = start, start * 2
    while not conclusive(high):
        low, high = high, high * 2
        if high > _MAX_SIZE:
            return None
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if conclusive(middle) else (middle, high)
    return high


def required_group_size(
    rate_group: float,
    rate_reference: float,
    zone: tuple = (-0.1, 0.1),
    kind: str = "difference",
    reference_size: Optional[int] = None,
    confidence: float = 0.95,
    n_groups: int = 1,
    power: float = 0.8,
) -> Optional[int]:
    """Rows of the group needed for a conclusive verdict (inside or outside `zone`) with probability `power`

    Parameters
    ----------
    rate_group, rate_reference : float
        True (expected, or observed as a proxy) selection, true positive or false positive rates
    zone : tuple
        Acceptable zone of the difference or of the ratio (4/5 rule: (0.8, 1.25), kind="ratio")
    reference_size : int, optional
        Rows of the reference group, held fixed (default: as many as the group)
    confidence : float, optional
        Level of the interval (that of `equivalence_confidence` for a "within the zone" verdict)
    n_groups : int, optional
        Groups judged together: the level is split over them (Bonferroni), as in a simultaneous analysis
    power : float, optional
        Probability of a conclusive verdict (default 0.8); the estimate fluctuates around the true disparity, so a
        size that would conclude *at* the true rates concludes only about half the time

    Returns None when no size suffices: the disparity lies on (or too close to) a bound of the zone, or the fixed
    reference group is too small. Lab check (FA-04, zone +-0.1, rate 0.5, reference of 2,000 rows, no bias): groups of
    30 / 50 / 100 / 200 / 500 rows were certified 0 / 0 / 5 / 52 / 96 % of the time; this function gives about 190
    rows for power 0.5 and 300 for power 0.8.
    """
    if not 0 < power < 1:
        raise error_msg(value=power, message="power must be in (0, 1)")
    level = 1 - (1 - confidence) / max(n_groups, 1)
    z = _z(level)
    value = rate_group - rate_reference if kind == "difference" else rate_group / rate_reference if rate_reference else None
    if value is None:
        return None

    def conclusive(n):
        m = reference_size if reference_size is not None else n
        low, high = rate_interval(rate_group * n, n, rate_reference * m, m, kind, level)
        return _verdict_probability(value, low, high, zone, z) >= power

    return _smallest(conclusive, 2)


def paired_sets_needed(
    difference: float,
    discordant_rate: float,
    zone: tuple = (-0.1, 0.1),
    confidence: float = 0.95,
    power: float = 0.8,
) -> Optional[int]:
    """Probe sets needed for a conclusive verdict on a paired action rate difference, with probability `power`

    difference : expected (or observed) difference of the action rate, group minus reference
    discordant_rate : share of sets whose outcome differs between the two variants (b + c) / n, at least |difference|

    Normal approximation of the paired difference: variance (discordant_rate - difference^2) / n. With few discordant
    sets the interval is bounded by the share of discordant sets itself: with none observed among n sets, its Wilson
    upper bound z^2 / (n + z^2) must fall inside the zone (0.11 at 30 sets: never conclusive for a zone of +-0.1).
    """
    if discordant_rate < abs(difference) or not 0 <= discordant_rate <= 1:
        raise error_msg(value=discordant_rate, message="discordant_rate must lie in [|difference|, 1]")
    z = _z(confidence)
    width = min(abs(zone[0]), abs(zone[1]))
    variance = max(discordant_rate - difference**2, 0.0)

    def conclusive(n):
        half = max(z * math.sqrt(variance / n), wilson(0, n, confidence)[1] if discordant_rate == 0 else 0.0)
        return _verdict_probability(difference, difference - half, difference + half, zone, z) >= power

    if min(abs(difference - zone[0]), abs(difference - zone[1])) == 0 or width == 0:
        return None
    return _smallest(conclusive, 2)


def flip_sets_needed(
    flip_rate: float,
    placebo_rate: float,
    tolerance: float = 0.05,
    placebo_sets: Optional[int] = None,
    confidence: float = 0.95,
    power: float = 0.8,
) -> Optional[int]:
    """Probe sets needed for a conclusive verdict on the excess flip rate (flip rate - placebo flip rate, Newcombe),
    with probability `power`

    placebo_sets : int, optional
        Placebo comparisons, held fixed (default: as many as the probe sets)

    The verdict is "established" when the lower bound exceeds `tolerance`, "absent" when the upper bound is below it.
    In the lab (LB-01, 200 sets) excess flip rates of 0.035 to 0.038 with placebo floors near 0.02 stayed
    inconclusive: their upper bounds (0.077 to 0.078) exceeded the tolerance of 0.05.
    """
    value = flip_rate - placebo_rate
    z = _z(confidence)

    def conclusive(n):
        m = placebo_sets if placebo_sets is not None else n
        ci1, ci2 = wilson(flip_rate * n, n, confidence), wilson(placebo_rate * m, m, confidence)
        low, high = newcombe_difference(flip_rate, ci1, placebo_rate, ci2)
        return _verdict_probability(value, low, high, (-np.inf, tolerance), z) >= power

    if value == tolerance:
        return None
    return _smallest(conclusive, 2)


def sequential_decisions_needed(
    gap: float,
    tolerance: float = 0.05,
    group_share: float = 0.5,
    alpha: float = 0.05,
    n_groups: int = 1,
    rate: float = 0.5,
    reference_pooling: bool = True,
) -> Optional[int]:
    """Rough number of decisions (all groups) before `FairnessMonitor` establishes a gap beyond the tolerance

    Growth-rate approximation of the betting test: the log-wealth grows by about (|gap| - tolerance)^2 / (2 v) per
    pair, v the variance of a pair's difference (2 r (1 - r) with 1:1 pairs, r (1 - r)(1 + share / (1 - share)) with
    the reference pooled), and the null is rejected when it reaches log(2 n_groups / alpha). The Online Newton Step
    bets need a burn-in, so actual delays are typically 1 to 2 times this figure; use it to size the horizon, then
    check with the bias-injection bench (benchmarks/fairness_drift.py). Returns None when |gap| <= tolerance.

    Lab FA-04: a drop of 0.1 on a group of 5 % of the stream (tolerance 0.05) was established within 20,000
    decisions in 22 % of runs (1:1 pairs); this approximation gives about 29,500 decisions with 1:1 pairs and
    15,500 with the reference pooled.
    """
    excess = abs(gap) - tolerance
    if excess <= 0:
        return None
    if not 0 < group_share < 1:
        raise error_msg(value=group_share, message="group_share must be in (0, 1)")
    p = rate * (1 - rate)
    if reference_pooling:
        variance = p * (1 + min(group_share / (1 - group_share), 1.0))
    else:
        variance = 2 * p
    pairs = math.log(2 * n_groups / alpha) * 2 * variance / excess**2
    return int(math.ceil(pairs / group_share))
