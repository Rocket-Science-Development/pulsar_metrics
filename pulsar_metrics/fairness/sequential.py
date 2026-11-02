"""Anytime-valid fairness monitoring: sequential tests by betting (Chugg, Cortes-Gomez, Hamilton et al., NeurIPS 2023).

Re-running a fixed-sample test at every window inflates false alarms: with enough looks, a test at 5 % fires on a fair
system almost surely. A test by betting stays valid whenever it is looked at. Paired observations (one from the
monitored group, one from the reference group) are used to bet on their difference; the wealth is an e-process and,
by Ville's inequality, the probability that it ever exceeds 1/alpha on a fair system is at most alpha.

Two questions can be asked of a stream: is the gap beyond the tolerance (level, the default), or has it moved beyond
the tolerance from the gap of a reference period (change, `baseline` / `fit_baseline`)? A gap that was already there
when monitoring started alarms at once in level mode and stays alarmed, which masks any later degradation; the
change mode watches the degradation itself.
"""

import math
from collections import deque
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults

_ONS_STEP = 2 / (2 - math.log(3))


class BettingTest:
    """Two-sample test of H0: |mean(A) - mean(B) - center| <= tolerance, for observations in [0, 1], valid at any
    stopping time

    Two one-sided bets (A above B + tolerance, A below B - tolerance), each with Online Newton Step bet sizes; their
    average is an e-value. The null is rejected, once and for all, when it reaches 1 / alpha.

    Parameters
    ----------
    alpha : float, optional
        Probability of ever raising a false alarm on a system within the tolerance (default 0.05)
    tolerance : float, optional
        Disparity considered acceptable, in the units of the observations (default 0.0: strict equality)
    center : float, optional
        Expected difference under the null (default 0.0); a baseline gap, to test its change rather than its level
    """

    def __init__(self, alpha: float = 0.05, tolerance: float = 0.0, center: float = 0.0):
        if not 0 < alpha < 1 or not 0 <= tolerance < 1 or not -1 < center < 1:
            raise error_msg(value=None, message="0 < alpha < 1, 0 <= tolerance < 1 and -1 < center < 1 are required")
        self.alpha = alpha
        self.tolerance = tolerance
        self.center = center
        self._max_bet = 1 / (2 * (1 + tolerance + abs(center)))  # keeps the wealth positive for any pair in [0, 1]
        self._log_wealth = {1: 0.0, -1: 0.0}
        self._bet = {1: 0.0, -1: 0.0}
        self._curvature = {1: 1.0, -1: 1.0}
        self.n = 0
        self._sum_difference = 0.0
        self.rejected_at: Optional[int] = None
        self._e_value_at_rejection: Optional[float] = None

    @property
    def e_value(self) -> float:
        return 0.5 * (math.exp(self._log_wealth[1]) + math.exp(self._log_wealth[-1]))

    @property
    def evidence(self) -> float:
        """e-value when the null was rejected (the wealth may fall afterwards), otherwise the current e-value"""
        return self._e_value_at_rejection if self.rejected else self.e_value

    @property
    def rejected(self) -> bool:
        return self.rejected_at is not None

    @property
    def mean_difference(self) -> float:
        return self._sum_difference / self.n if self.n else float("nan")

    def update(self, a: float, b: float) -> bool:
        """Feed one pair (observation of A, observation of B); return True if the null is rejected"""
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise error_msg(value=(a, b), message="observations must lie in [0, 1]")
        difference = a - b
        self.n += 1
        self._sum_difference += difference
        for side in (1, -1):
            payoff = side * (difference - self.center) - self.tolerance
            bet = self._bet[side]
            self._log_wealth[side] += math.log1p(bet * payoff)
            gradient = payoff / (1 + bet * payoff)
            self._curvature[side] += gradient * gradient
            self._bet[side] = min(max(bet + _ONS_STEP * gradient / self._curvature[side], 0.0), self._max_bet)
        if self.rejected_at is None and self.e_value >= 1 / self.alpha:
            self.rejected_at = self.n
            self._e_value_at_rejection = self.e_value
        return self.rejected


CRITERIA = {
    "demographic_parity": "selection rates (decisions of every row)",
    "equal_opportunity": "true positive rates (decisions of rows with a positive outcome)",
}


class FairnessMonitor:
    """Continuous, anytime-valid monitoring of the disparity between groups and a reference group

    Feed it batches of decisions as they arrive (`update`); each monitored group is compared with the reference group by
    a BettingTest. The overall false alarm probability is held at `alpha` by splitting it over the monitored groups
    (Bonferroni), so the groups must be declared upfront.

    Parameters
    ----------
    attribute : str
        Protected attribute column
    reference_group : str
    groups : sequence of str
        Groups to compare with the reference group
    criterion : str, optional
        "demographic_parity" (default) or "equal_opportunity" (needs `y_true`)
    alpha : float, optional
        Probability of ever raising a false alarm while every group stays within the tolerance (default 0.05)
    tolerance : float, optional
        Acceptable absolute difference of rates (default 0.05)
    max_pending : int, optional
        Upper bound of the unpaired observations kept per group and side (default 10,000)
    baseline : dict, optional
        Gap (group rate minus reference rate) of each group over a reference period. Each test then asks whether the
        gap has moved more than `tolerance` away from its baseline (change mode) instead of whether it exceeds the
        tolerance (level mode, default). `fit_baseline` computes it from a reference frame.
    reference_pooling : bool, optional
        Pair each group decision with the mean of a disjoint share of the batch's reference decisions instead of a
        single one (default True; False is the 1:1 pairing of 0.10 and before). Every reference decision is used,
        so the variance of a pair falls from about 2 r (1 - r) to r (1 - r) (1 + share of the group): faster
        detection for small groups, still valid (the chunks depend on counts only, never on the values). Lab-like
        bench (group of 5 %, drop of 0.1, tolerance 0.05, 20,000 decisions): detected in 52 % of runs instead of 16 %.

    Pairs join observations of the same batch, or of the previous one: unpaired observations older than the current
    batch are dropped. Pairing a group's decisions with old reference decisions would turn a change shared by every
    group (a seasonal rise of approvals) into a false disparity. Dropping depends on counts only, never on the values,
    so validity is kept.
    """

    def __init__(
        self,
        attribute: str,
        reference_group: str,
        groups: Sequence[str],
        criterion: str = "demographic_parity",
        alpha: float = 0.05,
        tolerance: float = 0.05,
        y_pred: str = "y_pred",
        y_true: str = "y_true",
        max_pending: int = 10_000,
        baseline: Optional[Dict[str, float]] = None,
        reference_pooling: bool = True,
    ):
        if criterion not in CRITERIA:
            raise error_msg(value=criterion, message=f"criterion must be one of {sorted(CRITERIA)}")
        groups = [str(g) for g in groups if str(g) != str(reference_group)]
        if not groups:
            raise error_msg(value=None, message="at least one group other than the reference group is required")
        self.attribute = attribute
        self.reference_group = str(reference_group)
        self.groups = groups
        self.criterion = criterion
        self.alpha = alpha
        self.tolerance = tolerance
        self.columns = {"y_pred": y_pred, "y_true": y_true}
        self.reference_pooling = reference_pooling
        self.baseline: Optional[Dict[str, float]] = None
        self.baseline_n: Dict[str, tuple] = {}
        self.alarms: list = []  # acknowledged alarms: (group, pair at which it was established, e-value)
        self.tests = {g: BettingTest(alpha / len(groups), tolerance) for g in groups}
        if baseline is not None:
            self._set_baseline({str(k): float(v) for k, v in baseline.items()})
        # unpaired (group observations, reference observations) per group
        self._queues = {g: (deque(maxlen=max_pending), deque(maxlen=max_pending)) for g in groups}

    def _set_baseline(self, baseline: Dict[str, float]):
        missing = [g for g in self.groups if g not in baseline]
        if missing:
            raise error_msg(value=missing, message="baseline needs a gap for every monitored group")
        if any(test.n for test in self.tests.values()):
            raise error_msg(value=None, message="set the baseline before the first update")
        self.baseline = {g: baseline[g] for g in self.groups}
        self.tests = {g: BettingTest(self.alpha / len(self.groups), self.tolerance, self.baseline[g]) for g in self.groups}

    def fit_baseline(self, frame: pd.DataFrame) -> Dict[str, float]:
        """Set the baseline gaps from the decisions of a reference period (switches to change mode) and return them

        The baseline is a point estimate: the test is exact for H0 |gap - baseline| <= tolerance with the baseline
        taken as given, so choose a tolerance above its uncertainty (`baseline_n` keeps the rows behind each gap)."""
        keys = frame[self.attribute].astype(str)
        reference = self._observations(frame[keys == self.reference_group])
        if not len(reference):
            raise error_msg(value=self.reference_group, message="no decision of the reference group in the frame")
        gaps = {}
        for g in self.groups:
            observations = self._observations(frame[keys == g])
            if not len(observations):
                raise error_msg(value=g, message="no decision of this group in the baseline frame")
            gaps[g] = float(observations.mean() - reference.mean())
            self.baseline_n[g] = (len(observations), len(reference))
        self._set_baseline(gaps)
        return dict(gaps)

    def acknowledge(self, group: str, rebaseline: Optional[float] = None):
        """Re-arm the test of a group after its alarm was handled; the alarm is kept in `alarms`

        A new test starts from scratch on the next decisions, with a fresh alpha budget for that group: the guarantee
        holds for each armed period, not for the whole history. `rebaseline` (change mode only) sets a new baseline
        gap for the group, e.g. the gap accepted after review."""
        group = str(group)
        if group not in self.tests:
            raise error_msg(value=group, message="unknown monitored group")
        test = self.tests[group]
        if test.rejected:
            self.alarms.append((group, test.rejected_at, test.evidence))
        center = test.center
        if rebaseline is not None:
            if self.baseline is None:
                raise error_msg(value=None, message="rebaseline needs change mode (baseline or fit_baseline)")
            self.baseline[group] = center = float(rebaseline)
        self.tests[group] = BettingTest(self.alpha / len(self.groups), self.tolerance, center)
        for queue in self._queues[group]:
            queue.clear()

    def _observations(self, frame: pd.DataFrame):
        if self.criterion == "equal_opportunity":
            if self.columns["y_true"] not in frame:
                raise error_msg(value=None, message="equal_opportunity needs the y_true column")
            frame = frame[frame[self.columns["y_true"]] == 1]
        return frame[self.columns["y_pred"]].astype(float)

    def update(self, batch: pd.DataFrame) -> list:
        """Feed a batch of decisions (any size, any group mix) and return the current verdict per group"""
        keys = batch[self.attribute].astype(str)
        reference = self._observations(batch[keys == self.reference_group]).tolist()
        for g in self.groups:
            group_queue, reference_queue = self._queues[g]
            observations = self._observations(batch[keys == g]).tolist()
            group_queue.extend(observations)
            test = self.tests[g]
            if self.reference_pooling:
                pairs = min(len(group_queue), len(reference))
                if pairs:
                    for chunk in np.array_split(np.asarray(reference, dtype=float), pairs):
                        test.update(group_queue.popleft(), float(chunk.mean()))
                while len(group_queue) > len(observations):
                    group_queue.popleft()
                continue
            reference_queue.extend(reference)
            while group_queue and reference_queue:
                test.update(group_queue.popleft(), reference_queue.popleft())
            # keep only leftovers of this batch: the next pairs stay contemporaneous
            while len(group_queue) > len(observations):
                group_queue.popleft()
            while len(reference_queue) > len(reference):
                reference_queue.popleft()
        return self.results()

    def results(self) -> list:
        out = []
        for g, test in self.tests.items():
            mode = f"change from the baseline gap {test.center:+.3f}" if self.baseline is not None else "level of the gap"
            pairing = "reference pooled per batch" if self.reference_pooling else "1:1 pairs"
            out.append(
                MetricResults(
                    metric_type="fairness",
                    metric_name=f"sequential_{self.criterion}",
                    feature_name=self.attribute,
                    group=f"{self.attribute}={g}",
                    reference_group=f"{self.attribute}={self.reference_group}",
                    n=test.n,
                    metric_value=test.mean_difference if test.n else None,
                    threshold=[test.center - self.tolerance, test.center + self.tolerance],
                    drift_status=test.rejected,
                    e_value=test.evidence,
                    method=(
                        f"test by betting (ONS bets), two-sided, {mode}, tolerance {self.tolerance}, {pairing}, "
                        f"alpha {self.alpha} split over {len(self.groups)} groups; "
                        + (f"established at pair {test.rejected_at}" if test.rejected else "not established")
                    ),
                    test_type="sequential",
                )
            )
        return out
