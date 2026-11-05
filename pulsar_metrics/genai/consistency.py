"""Consistency of the answers of an LLM application: does it answer the same question the same way?

The input is one row per question asked several times (pulsar_data_collection.genai.ConsistencyProbe, `questions`
frame): its mean pairwise agreement in [0, 1] (`agreement`), whether every pair agrees (`full_agreement`) and the
number of answers (`n_answers`). A mapping question -> normalized answers is accepted too (`question_agreement`).

- `consistency`: two window metrics, the mean agreement over the questions and the share of questions whose answers
  all agree, each with a bootstrap interval over the questions (the questions are the units: the answers to one
  question are not independent);
- `consistency_drift`: one-sided test that consistency **dropped** against a reference window. Paired on the
  questions both windows share (sign-flip permutation of the per-question differences), else unpaired (permutation of
  the questions between the windows). The value is the difference current - reference, with a bootstrap interval.

Results are MetricResults; `pulsar_metrics.evidence.to_evidence` exports them in the genai family. Few answers per
question (`min_repeats`, default 3) or few questions (`min_questions`, default 10) make the result unreliable
(`status` "few_repeats" or "few_questions", `reliable=False`): with 2 answers per question the agreement of a
question is 0 or 1, and a handful of questions gives an interval too wide to act on.
"""

import numbers
from itertools import combinations
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..metrics.base import MetricResults

AGREEMENT = "consistency_agreement"
FULL_AGREEMENT = "consistency_full_agreement"
AGREEMENT_DRIFT = "consistency_drift"
FULL_AGREEMENT_DRIFT = "consistency_full_agreement_drift"
WINDOW_METRICS = (AGREEMENT, FULL_AGREEMENT)
DRIFT_METRICS = (AGREEMENT_DRIFT, FULL_AGREEMENT_DRIFT)
FEW_REPEATS = "few_repeats"
FEW_QUESTIONS = "few_questions"
STATISTICS = {"agreement": "agreement", "full_agreement": "full_agreement"}


def question_agreement(answers: Mapping[str, Sequence]) -> pd.DataFrame:
    """Per-question agreement of normalized answers: {question_id: [answer, ...]} -> frame

    Two answers agree when they are equal; a None answer (nothing could be read) agrees with nothing.
    """
    rows = []
    for question_id, values in answers.items():
        values = list(values)
        pairs = [float(a is not None and a == b) for a, b in combinations(values, 2)]
        rows.append(
            {
                "question_id": question_id,
                "n_answers": len(values),
                "agreement": float(np.mean(pairs)) if pairs else None,
                "full_agreement": bool(min(pairs) == 1.0) if pairs else None,
            }
        )
    return pd.DataFrame(rows, columns=["question_id", "n_answers", "agreement", "full_agreement"])


def _frame(questions: Union[pd.DataFrame, Mapping]) -> pd.DataFrame:
    """Usable questions (at least 2 answers and an agreement), with the columns question_id, n_answers, agreement and
    full_agreement (float 0 / 1; a missing one is read from the agreement, 1.0 or not; anything but a boolean or 0 / 1
    is refused); `attrs["excluded"]` counts the questions set aside"""
    frame = questions.copy() if isinstance(questions, pd.DataFrame) else question_agreement(questions)
    missing = {"agreement", "n_answers"} - set(frame.columns)
    if missing:
        raise ValueError(f"the questions frame needs the columns {sorted(missing)}")
    if "question_id" not in frame:
        frame["question_id"] = [str(i) for i in range(len(frame))]
    frame["agreement"] = pd.to_numeric(frame["agreement"], errors="coerce")
    frame["n_answers"] = pd.to_numeric(frame["n_answers"], errors="coerce").fillna(0)
    if "full_agreement" not in frame:
        frame["full_agreement"] = frame["agreement"] >= 1.0
    usable = frame[(frame["n_answers"] >= 2) & frame["agreement"].notna()].copy()
    given = usable["full_agreement"].astype(object)
    full = given.where(given.notna(), usable["agreement"] >= 1.0)  # a missing flag (None, NaN): from the agreement
    flags = pd.to_numeric(full.map(lambda v: float(v) if isinstance(v, (bool, np.bool_, numbers.Real)) else np.nan))
    if not flags.isin([0.0, 1.0]).all():
        raise ValueError("full_agreement must be a boolean (or 0 / 1) per question")
    usable["full_agreement"] = flags.astype(float)
    if usable["question_id"].duplicated().any():
        raise ValueError("question_id must be unique within a window")
    usable.attrs["excluded"] = int(len(frame) - len(usable))
    return usable


def _bootstrap_ci(values: np.ndarray, n_boot: int, confidence: float, rng) -> tuple:
    if len(values) < 2:
        return None, None
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    tail = (1 - confidence) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def _reliability(frames: Sequence[pd.DataFrame], min_repeats: int, min_questions: int) -> tuple:
    """(status code or None, reason text or None) from the answers per question and the number of questions"""
    texts, code = [], None
    for frame in frames:
        median = float(frame["n_answers"].median()) if len(frame) else 0.0
        if median < min_repeats and code is None:
            code = FEW_REPEATS
        if median < min_repeats:
            texts.append(f"median {median:g} answers per question, fewer than {min_repeats}")
    for frame in frames:
        if len(frame) < min_questions:
            code = code or FEW_QUESTIONS
            texts.append(f"{len(frame)} questions, fewer than {min_questions}")
    return code, ("; ".join(dict.fromkeys(texts)) or None)


def consistency(
    questions: Union[pd.DataFrame, Mapping],
    model: str = "endpoint",
    threshold: Optional[float] = None,
    full_agreement_threshold: Optional[float] = None,
    min_repeats: int = 3,
    min_questions: int = 10,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
) -> list:
    """Mean agreement and share of questions in full agreement over a window, with bootstrap intervals

    Parameters
    ----------
    questions : DataFrame or mapping
        One row per question (`agreement`, `n_answers`, optionally `full_agreement` and `question_id`), e.g.
        ConsistencyProbe(...).run(...).questions; or {question_id: [normalized answers]}
    model : str, optional
        Name of the system (feature_name of the results)
    threshold, full_agreement_threshold : float, optional
        Minimum acceptable value of each metric: alert (drift_status True) when the value is below it. Default None:
        value and interval only, no verdict
    min_repeats, min_questions : int, optional
        Below a median of `min_repeats` answers per question or `min_questions` questions the result is flagged
        unreliable
    n_boot, confidence, seed : bootstrap over the questions

    Returns [consistency_agreement, consistency_full_agreement] (metric_type "custom", exported in the genai family).
    """
    frame = _frame(questions)
    rng = np.random.default_rng(seed)
    code, reason = _reliability([frame], min_repeats, min_questions)
    out = []
    for name, column, floor in (
        (AGREEMENT, "agreement", threshold),
        (FULL_AGREEMENT, "full_agreement", full_agreement_threshold),
    ):
        values = frame[column].to_numpy(dtype=float)
        what = "mean pairwise agreement" if column == "agreement" else "share of questions whose answers all agree"
        if not len(values):
            out.append(
                MetricResults(
                    metric_type="custom",
                    metric_name=name,
                    feature_name=model,
                    threshold=floor,
                    n=0,
                    status=FEW_REPEATS,
                    reliable=False,
                    reliability_reason="no question with at least two answers",
                    error="no question with at least two answers",
                    method=what,
                )
            )
            continue
        value = float(values.mean())
        low, high = _bootstrap_ci(values, n_boot, confidence, rng)
        out.append(
            MetricResults(
                metric_type="custom",
                metric_name=name,
                feature_name=model,
                metric_value=value,
                threshold=floor,
                drift_status=None if floor is None else bool(value < floor),
                n=int(len(values)),
                ci_low=low,
                ci_high=high,
                conf_int=None if low is None else [low, high],
                status=code,
                reliable=False if code else None,
                reliability_reason=reason,
                method=f"{what} over {len(values)} questions (median {frame['n_answers'].median():g} answers each, "
                f"{frame.attrs['excluded']} set aside with fewer than 2); percentile bootstrap over the questions, "
                f"{confidence:.0%} interval, {n_boot} draws",
                test_type="fixed",
            )
        )
    return out


def consistency_drift(
    reference: Union[pd.DataFrame, Mapping],
    current: Union[pd.DataFrame, Mapping],
    model: str = "endpoint",
    statistic: str = "agreement",
    alpha: float = 0.05,
    paired: Union[str, bool] = "auto",
    min_repeats: int = 3,
    min_questions: int = 10,
    n_permutations: int = 2000,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
) -> MetricResults:
    """One-sided test that consistency dropped from the reference to the current window

    Parameters
    ----------
    reference, current : DataFrame or mapping
        Per-question frames (see `consistency`)
    statistic : str, optional
        "agreement" (mean pairwise agreement, default; metric consistency_drift) or "full_agreement" (share of
        questions in full agreement; metric consistency_full_agreement_drift)
    alpha : float, optional
        Alert (drift_status True) when the p-value is below it
    paired : "auto", True or False, optional
        Paired on the shared question ids (sign-flip permutation of the differences) or unpaired (permutation of the
        questions between the windows). "auto": paired when the windows share at least 2 questions and every current
        question is in the reference
    min_repeats, min_questions : see `consistency`
    n_permutations, n_boot, confidence, seed : permutation test and bootstrap interval of the difference

    The value is mean(current) - mean(reference) (negative: consistency dropped), p_value the one-sided permutation
    p-value (H1: the current window is less consistent).
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {sorted(STATISTICS)}")
    name = AGREEMENT_DRIFT if statistic == "agreement" else FULL_AGREEMENT_DRIFT
    ref, cur = _frame(reference), _frame(current)
    shared = sorted(set(ref["question_id"]) & set(cur["question_id"]))
    if paired == "auto":
        paired = len(shared) >= 2 and set(cur["question_id"]) <= set(ref["question_id"])
    rng = np.random.default_rng(seed)
    if paired:
        ref, cur = ref.set_index("question_id").loc[shared], cur.set_index("question_id").loc[shared]
    code, reason = _reliability([ref, cur], min_repeats, min_questions)
    if len(ref) < 2 or len(cur) < 2:
        return MetricResults(
            metric_type="drift",
            metric_name=name,
            feature_name=model,
            threshold=alpha,
            n=int(len(cur)),
            status=code or FEW_QUESTIONS,
            reliable=False,
            reliability_reason=reason or "fewer than 2 usable questions",
            error="fewer than 2 usable questions in a window" + (" (shared, paired test)" if paired else ""),
            test_type="fixed",
        )
    x, y = ref[STATISTICS[statistic]].to_numpy(dtype=float), cur[STATISTICS[statistic]].to_numpy(dtype=float)
    observed = float(y.mean() - x.mean())
    if paired:
        d = y - x
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(d)))
        null = (signs * d).mean(axis=1)
        boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
        design = f"paired on {len(d)} shared questions, sign-flip permutation"
    else:
        pooled, n_ref = np.concatenate([x, y]), len(x)
        null = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(pooled)
            null[i] = perm[n_ref:].mean() - perm[:n_ref].mean()
        boot = y[rng.integers(0, len(y), size=(n_boot, len(y)))].mean(axis=1) - x[
            rng.integers(0, len(x), size=(n_boot, len(x)))
        ].mean(axis=1)
        design = f"unpaired, {len(x)} reference and {len(y)} current questions, permutation of the questions"
    # one-sided: a difference at most as large as the observed one (with a tolerance for ties of discrete values)
    p_value = float((1 + np.sum(null <= observed + 1e-12)) / (1 + n_permutations))
    tail = (1 - confidence) / 2
    low, high = float(np.quantile(boot, tail)), float(np.quantile(boot, 1 - tail))
    what = "mean pairwise agreement" if statistic == "agreement" else "share of questions in full agreement"
    return MetricResults(
        metric_type="drift",
        metric_name=name,
        feature_name=model,
        metric_value=observed,
        threshold=alpha,
        drift_status=bool(p_value < alpha),
        n=int(len(y)),
        ci_low=low,
        ci_high=high,
        conf_int=[low, high],
        p_value=p_value,
        status=code,
        reliable=False if code else None,
        reliability_reason=reason,
        method=f"drop of the {what}: current {y.mean():.4f} - reference {x.mean():.4f}; {design}, one-sided "
        f"p-value over {n_permutations} draws; percentile bootstrap {confidence:.0%} interval of the difference",
        test_type="fixed",
    )
