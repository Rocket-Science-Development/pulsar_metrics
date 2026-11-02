"""Group fairness of decisions: disparities between groups with confidence intervals and small-cell suppression."""

from .analyzer import FairnessAnalyzer, FairnessPolicy
from .criteria import CRITERIA, DEFAULT_CRITERIA, EQUALIZED_ODDS, equivalence_verdict, verdict
from .paired import comparative_analysis, counterfactual_analysis
from .planning import (
    flip_sets_needed,
    paired_sets_needed,
    required_group_size,
    sequential_decisions_needed,
)
from .sequential import BettingTest, FairnessMonitor

__all__ = [
    "BettingTest",
    "CRITERIA",
    "DEFAULT_CRITERIA",
    "EQUALIZED_ODDS",
    "FairnessAnalyzer",
    "FairnessMonitor",
    "FairnessPolicy",
    "comparative_analysis",
    "equivalence_verdict",
    "flip_sets_needed",
    "paired_sets_needed",
    "required_group_size",
    "sequential_decisions_needed",
    "counterfactual_analysis",
    "verdict",
]
