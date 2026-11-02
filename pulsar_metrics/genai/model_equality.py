"""Is the API still serving the same model? Model equality test on the answers to a fixed canary set.

A provider can swap the model behind an endpoint (a quantized version, another model, a new system prompt) without
notice. The canary set is a fixed list of prompts answered several times, at a temperature above zero, at reference
time and later. Following Gao et al. (ICLR 2025, "Model equality testing: which model is this API serving?"), the
answers are compared with an MMD whose kernel only compares answers to the same prompt, by the share of positions
where their tokens agree (Hamming kernel); the p-value comes from permuting the reference / current labels within
each prompt.

The power depends on how close the two models are. Another model or a new system prompt is detected with the default
canary set; a change of quantization is not: in the night lab (LM-01), llama3.2 3B q4_K_M against q8_0 was detected in
20 % of runs at 12 prompts x 5 answers (60 answers), 40 % at 24 x 5, 60 % at 48 x 5 and 100 % at 48 x 10 (480 answers;
5 runs per cell, so the last figure has a 95 % lower bound near 0.5). `model_equality_power` measures it on a pilot
of the change you want to catch (e.g. two quantizations served locally) before choosing the canary size.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults

# Canary size needed to detect a change of quantization (night lab LM-01): 48 prompts x 10 answers
RECOMMENDED_CANARY = {"prompts": 48, "answers_per_prompt": 10, "answers": 480}


def _tokens(text: str) -> list:
    return text.lower().split()


def _hamming_gram(samples: List[list], length: int) -> np.ndarray:
    """k(x, y) = share of the first `length` positions where x and y hold the same token (padding counts as a token)"""
    padded = np.array([[hash(t) for t in s[:length]] + [0] * (length - len(s[:length])) for s in samples])
    return (padded[:, None, :] == padded[None, :, :]).mean(axis=2)


def _prompt_mmd2(K: np.ndarray, in_x: np.ndarray) -> float:
    x, y = in_x, ~in_x
    n, m = x.sum(), y.sum()
    if n < 2 or m < 2:
        return 0.0
    kxx = (K[np.ix_(x, x)].sum() - n) / (n * (n - 1))  # the diagonal of a Hamming gram is 1
    kyy = (K[np.ix_(y, y)].sum() - m) / (m * (m - 1))
    return float(kxx + kyy - 2 * K[np.ix_(x, y)].mean())


@dataclass
class ModelEqualityResult:
    statistic: float
    p_value: float
    prompts: int
    answers_reference: int
    answers_current: int


def model_equality_test(
    reference: Dict[str, List[str]],
    current: Dict[str, List[str]],
    tokenize: Callable[[str], list] = _tokens,
    length: int = 30,
    n_permutations: int = 1000,
    seed: Optional[int] = 0,
) -> ModelEqualityResult:
    """Permutation test of H0: the same model answered the canary prompts in both periods

    Parameters
    ----------
    reference, current : dict
        Prompt -> answers (several answers per prompt, sampled at a temperature above zero)
    tokenize : callable, optional
        Text -> tokens (default: lower-cased words); the model's own tokenizer is a stronger choice when available
    length : int, optional
        Number of leading tokens compared (default 30)
    """
    prompts = [p for p in reference if p in current and len(reference[p]) >= 2 and len(current[p]) >= 2]
    if not prompts:
        raise error_msg(value=None, message="the canary sets share no prompt with at least two answers on each side")
    rng = np.random.default_rng(seed)
    grams, labels = [], []
    for p in prompts:
        samples = [tokenize(a) for a in reference[p]] + [tokenize(a) for a in current[p]]
        grams.append(_hamming_gram(samples, length))
        labels.append(np.array([True] * len(reference[p]) + [False] * len(current[p])))
    observed = float(np.mean([_prompt_mmd2(K, lab) for K, lab in zip(grams, labels)]))
    exceed = 0
    for _ in range(n_permutations):
        stat = np.mean([_prompt_mmd2(K, rng.permutation(lab)) for K, lab in zip(grams, labels)])
        if stat >= observed:
            exceed += 1
    return ModelEqualityResult(
        observed,
        (exceed + 1) / (n_permutations + 1),
        len(prompts),
        sum(len(reference[p]) for p in prompts),
        sum(len(current[p]) for p in prompts),
    )


def model_equality_power(
    reference: Dict[str, List[str]],
    current: Dict[str, List[str]],
    prompts: Sequence[int] = (12, 24, 48),
    answers: Sequence[int] = (5, 10),
    alpha: float = 0.05,
    reps: int = 50,
    seed: Optional[int] = 0,
    **kwargs,
) -> list:
    """Detection rate of `model_equality_test` by canary size, on a pilot where the change is known to exist

    reference and current are a pilot canary run of the two models to tell apart (e.g. the q8_0 and q4_K_M
    quantizations of the same model, served locally), with at least max(prompts) prompts and max(answers) answers per
    prompt on each side. Each rep draws that many prompts and answers per prompt without replacement and runs the
    test (kwargs: tokenize, length, n_permutations). Returns one dict per (prompts, answers): detection rate with its
    Wilson 95 % interval. Answers drawn from one generated batch are correlated: generate the pilot in several batches.
    Run it also on two pilot runs of the *same* model to check the false alarm rate.
    """
    shared = [p for p in reference if p in current]
    rng = np.random.default_rng(seed)
    kwargs.setdefault("n_permutations", 200)
    out = []
    for n_prompts in prompts:
        for n_answers in answers:
            usable = [p for p in shared if len(reference[p]) >= n_answers and len(current[p]) >= n_answers]
            if len(usable) < n_prompts:
                raise error_msg(
                    value=n_prompts, message=f"the pilot has {len(usable)} prompts with {n_answers} answers on each side"
                )
            hits = 0
            for _ in range(reps):
                chosen = rng.choice(len(usable), n_prompts, replace=False)
                ref, cur = {}, {}
                for i in chosen:
                    p = usable[i]
                    ref[p] = [reference[p][j] for j in rng.choice(len(reference[p]), n_answers, replace=False)]
                    cur[p] = [current[p][j] for j in rng.choice(len(current[p]), n_answers, replace=False)]
                result = model_equality_test(ref, cur, seed=int(rng.integers(1 << 31)), **kwargs)
                hits += int(result.p_value < alpha)
            ci = stats.binomtest(hits, reps).proportion_ci(method="wilson")
            out.append(
                {
                    "prompts": n_prompts,
                    "answers_per_prompt": n_answers,
                    "answers": n_prompts * n_answers,
                    "power": hits / reps,
                    "ci_low": float(ci.low),
                    "ci_high": float(ci.high),
                    "reps": reps,
                }
            )
    return out


def model_equality(
    reference, current, model: str = "endpoint", alpha: float = 0.05, min_answers: Optional[int] = None, **kwargs
) -> MetricResults:
    """Model equality test as a MetricResults (metric_type "drift", metric_name "model_equality")

    min_answers : int, optional
        Current answers below which a non-significant result is reported as inconclusive (drift_status None, with
        the reason in `error`) instead of "no change" (False). Default None keeps False; 480 (RECOMMENDED_CANARY)
        is the size at which the lab detected a change of quantization. A significant result is always True.
    """
    result = model_equality_test(reference, current, **kwargs)
    significant = result.p_value < alpha
    status, error, note = significant, None, ""
    if not significant and result.answers_current < RECOMMENDED_CANARY["answers"]:
        note = (
            f"; below {RECOMMENDED_CANARY['prompts']} prompts x {RECOMMENDED_CANARY['answers_per_prompt']} answers a "
            "change of quantization is usually missed (see model_equality_power)"
        )
    if not significant and min_answers is not None and result.answers_current < min_answers:
        status = None
        error = f"inconclusive: {result.answers_current} current answers, at least {min_answers} needed to rule out a change"
    return MetricResults(
        metric_type="drift",
        metric_name="model_equality",
        feature_name=model,
        metric_value=result.statistic,
        threshold=alpha,
        drift_status=status,
        error=error,
        n=result.answers_current,
        method=f"MMD with a Hamming kernel within prompts (Gao et al., ICLR 2025); {result.prompts} canary prompts, "
        f"{result.answers_reference} reference and {result.answers_current} current answers; permutation p-value "
        f"{result.p_value:.4f}{note}",
        test_type="fixed",
        p_value=result.p_value,
    )
