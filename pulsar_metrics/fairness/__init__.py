"""Group fairness of decisions: disparities between groups with confidence intervals and small-cell suppression."""

from .analyzer import FairnessAnalyzer, FairnessPolicy
from .criteria import CRITERIA, DEFAULT_CRITERIA, EQUALIZED_ODDS, verdict

__all__ = ["CRITERIA", "DEFAULT_CRITERIA", "EQUALIZED_ODDS", "FairnessAnalyzer", "FairnessPolicy", "verdict"]
