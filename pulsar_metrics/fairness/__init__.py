"""Group fairness of decisions: disparities between groups with confidence intervals and small-cell suppression."""

from .analyzer import FairnessAnalyzer, FairnessPolicy
from .criteria import CRITERIA, DEFAULT_CRITERIA, EQUALIZED_ODDS, verdict
from .sequential import BettingTest, FairnessMonitor

__all__ = [
    "BettingTest",
    "CRITERIA",
    "DEFAULT_CRITERIA",
    "EQUALIZED_ODDS",
    "FairnessAnalyzer",
    "FairnessMonitor",
    "FairnessPolicy",
    "verdict",
]
