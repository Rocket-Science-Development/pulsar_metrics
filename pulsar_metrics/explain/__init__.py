"""Explanations of a model and their drift: attributions, ranking stability, importance by group, plain language."""

from .attributions import importance, ndcg_at_k, top_k_jaccard, tree_attributions
from .drift import AttributionDrift
from .groups import importance_by_group
from .language import describe, explain_decision

__all__ = [
    "AttributionDrift",
    "describe",
    "explain_decision",
    "importance",
    "importance_by_group",
    "ndcg_at_k",
    "top_k_jaccard",
    "tree_attributions",
]
