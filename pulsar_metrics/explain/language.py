"""Plain-language explanations, in French and English, of a decision and of explanation-drift results.

The wording describes what the model relied on, never a cause in the world, and states the uncertainty when there is
one. Feature labels can be given per language: {"WKHP": {"fr": "heures travaillées par semaine", "en": "hours worked
per week"}}.
"""

from typing import Optional

import pandas as pd

from ..exceptions import CustomExceptionPulsarMetric as error_msg

LANGUAGES = ("fr", "en")

_TEXT = {
    "fr": {
        "decision_intro": "Facteurs qui ont le plus pesé dans cette décision :",
        "up": "a pesé en faveur",
        "down": "a pesé en défaveur",
        "value": "valeur",
        "decision_note": "Ces facteurs décrivent le fonctionnement du modèle, pas une cause ni un jugement sur la personne.",
        "ndcg": "Classement des facteurs (NDCG@{k}) : {v} [{lo} ; {hi}], seuil {t}.",
        "jaccard": "Recouvrement des {k} premiers facteurs : {v} [{lo} ; {hi}], seuil {t}.",
        "share": "Poids de « {f} » : {d} points de pourcentage [{lo} ; {hi}].",
        "share_group": "Pour {g}, le poids de « {f} » diffère de {d} points de pourcentage par rapport à {r} [{lo} ; {hi}].",
        "suppressed": "{g} : effectif trop petit, résultat non publié.",
        True: "Changement établi.",
        False: "Pas de changement au-delà de la tolérance.",
        None: "Non concluant : l'intervalle chevauche le seuil.",
    },
    "en": {
        "decision_intro": "Factors that weighed most in this decision:",
        "up": "weighed in favour",
        "down": "weighed against",
        "value": "value",
        "decision_note": "These factors describe how the model works, not a cause nor a judgement about the person.",
        "ndcg": "Ranking of factors (NDCG@{k}): {v} [{lo}; {hi}], threshold {t}.",
        "jaccard": "Overlap of the top {k} factors: {v} [{lo}; {hi}], threshold {t}.",
        "share": "Weight of “{f}”: {d} percentage points [{lo}; {hi}].",
        "share_group": "For {g}, the weight of “{f}” differs by {d} percentage points from {r} [{lo}; {hi}].",
        "suppressed": "{g}: group too small, result withheld.",
        True: "Change established.",
        False: "No change beyond the tolerance.",
        None: "Inconclusive: the interval straddles the threshold.",
    },
}


def _check(lang: str) -> dict:
    if lang not in LANGUAGES:
        raise error_msg(value=lang, message=f"lang must be one of {LANGUAGES}")
    return _TEXT[lang]


def _number(x: float, lang: str, digits: int = 2, signed: bool = False) -> str:
    text = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return text.replace(".", ",").replace("-", "−") if lang == "fr" else text


def _label(feature: str, labels: Optional[dict], lang: str) -> str:
    label = (labels or {}).get(feature, feature)
    return label.get(lang, feature) if isinstance(label, dict) else str(label)


def explain_decision(
    attributions: pd.Series,
    values: Optional[pd.Series] = None,
    lang: str = "fr",
    top: int = 3,
    labels: Optional[dict] = None,
) -> str:
    """The `top` features with the largest attribution for one decision, in plain language

    `attributions` holds the attribution of each feature for this decision (one row of the attribution DataFrame);
    `values`, optional, holds the feature values to quote.
    """
    text = _check(lang)
    order = attributions.abs().sort_values(ascending=False, kind="stable").index[:top]
    lines = [text["decision_intro"]]
    for rank, feature in enumerate(order, start=1):
        contribution = float(attributions[feature])
        direction = text["up"] if contribution >= 0 else text["down"]
        quoted = ""
        if values is not None and feature in values.index:
            value = values[feature]
            if isinstance(value, float) and value.is_integer():
                value = int(value)  # codes and counts stored as floats
            shown = _number(value, lang) if isinstance(value, float) else str(value)
            quoted = f" ({text['value']} : {shown})" if lang == "fr" else f" ({text['value']}: {shown})"
        lines.append(f"{rank}. {_label(feature, labels, lang)}{quoted} {direction} ({_number(contribution, lang, signed=True)})")
    lines.append(text["decision_note"])
    return "\n".join(lines)


def describe(results: list, lang: str = "fr", labels: Optional[dict] = None, only_changes: bool = False) -> list:
    """One sentence per explainability result (AttributionDrift, importance_by_group), with interval and verdict

    `only_changes` keeps the established changes and the inconclusive results, and drops the stable ones.
    """
    text = _check(lang)
    sentences = []
    for r in results:
        if r.metric_type != "explainability" or (only_changes and r.drift_status is False):
            continue
        if r.suppressed:
            sentences.append(text["suppressed"].format(g=r.group))
            continue
        lo, hi = r.ci_low, r.ci_high
        if r.metric_name in ("attribution_ndcg_at_k", "attribution_top_k_jaccard"):
            k = r.method.split("top ")[1].split(" ")[0] if r.method and "top " in r.method else "k"
            key = "ndcg" if r.metric_name == "attribution_ndcg_at_k" else "jaccard"
            body = text[key].format(
                k=k, v=_number(r.metric_value, lang), lo=_number(lo, lang), hi=_number(hi, lang), t=_number(r.threshold, lang)
            )
        else:
            pp = dict(
                f=_label(r.feature_name, labels, lang),
                d=_number(100 * r.metric_value, lang, 1, signed=True),
                lo=_number(100 * lo, lang, 1, signed=True),
                hi=_number(100 * hi, lang, 1, signed=True),
            )
            if r.metric_name == "importance_share_difference_by_group":
                body = text["share_group"].format(g=r.group, r=r.reference_group, **pp)
            else:
                body = text["share"].format(**pp)
        sentences.append(f"{body} {text[r.drift_status]}")
    return sentences
