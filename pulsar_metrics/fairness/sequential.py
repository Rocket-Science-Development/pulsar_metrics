"""Anytime-valid fairness monitoring: sequential tests by betting (Chugg, Cortes-Gomez, Hamilton et al., NeurIPS 2023).

Re-running a fixed-sample test at every window inflates false alarms: with enough looks, a test at 5 % fires on a fair
system almost surely. A test by betting stays valid whenever it is looked at. Paired observations (one from the
monitored group, one from the reference group) are used to bet on their difference; the wealth is an e-process and,
by Ville's inequality, the probability that it ever exceeds 1/alpha on a fair system is at most alpha.
"""

import math
from collections import deque
from typing import Optional, Sequence

import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults

_ONS_STEP = 2 / (2 - math.log(3))


class BettingTest:
    """Two-sample test of H0: |mean(A) - mean(B)| <= tolerance, for observations in [0, 1], valid at any stopping time

    Two one-sided bets (A above B + tolerance, A below B - tolerance), each with Online Newton Step bet sizes; their
    average is an e-value. The null is rejected, once and for all, when it reaches 1 / alpha.

    Parameters
    ----------
    alpha : float, optional
        Probability of ever raising a false alarm on a system within the tolerance (default 0.05)
    tolerance : float, optional
        Disparity considered acceptable, in the units of the observations (default 0.0: strict equality)
    """

    def __init__(self, alpha: float = 0.05, tolerance: float = 0.0):
        if not 0 < alpha < 1 or not 0 <= tolerance < 1:
            raise error_msg(value=None, message="0 < alpha < 1 and 0 <= tolerance < 1 are required")
        self.alpha = alpha
        self.tolerance = tolerance
        self._max_bet = 1 / (2 * (1 + tolerance))  # keeps the wealth positive for any pair in [0, 1]
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
            payoff = side * difference - self.tolerance
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
        self.tests = {g: BettingTest(alpha / len(groups), tolerance) for g in groups}
        # unpaired (group observations, reference observations) per group
        self._queues = {g: (deque(maxlen=max_pending), deque(maxlen=max_pending)) for g in groups}

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
            reference_queue.extend(reference)
            test = self.tests[g]
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
            out.append(
                MetricResults(
                    metric_type="fairness",
                    metric_name=f"sequential_{self.criterion}",
                    feature_name=self.attribute,
                    group=f"{self.attribute}={g}",
                    reference_group=f"{self.attribute}={self.reference_group}",
                    n=test.n,
                    metric_value=test.mean_difference if test.n else None,
                    threshold=[-self.tolerance, self.tolerance],
                    drift_status=test.rejected,
                    e_value=test.evidence,
                    method=(
                        f"test by betting (ONS bets), two-sided, tolerance {self.tolerance}, "
                        f"alpha {self.alpha} split over {len(self.groups)} groups; established at pair {test.rejected_at}"
                        if test.rejected
                        else f"test by betting (ONS bets), two-sided, tolerance {self.tolerance}, "
                        f"alpha {self.alpha} split over {len(self.groups)} groups; not established"
                    ),
                    test_type="sequential",
                )
            )
        return out
