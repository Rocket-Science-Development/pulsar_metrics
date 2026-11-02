"""Is the API still serving the same model? Model equality test on the answers to a fixed canary set.

A provider can swap the model behind an endpoint (a quantized version, another model, a new system prompt) without
notice. The canary set is a fixed list of prompts answered several times, at a temperature above zero, at reference
time and later. Following Gao et al. (ICLR 2025, "Model equality testing: which model is this API serving?"), the
answers are compared with an MMD whose kernel only compares answers to the same prompt, by the share of positions
where their tokens agree (Hamming kernel); the p-value comes from permuting the reference / current labels within
each prompt.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults


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


def model_equality(reference, current, model: str = "endpoint", alpha: float = 0.05, **kwargs) -> MetricResults:
    """Model equality test as a MetricResults (metric_type "drift", metric_name "model_equality")"""
    result = model_equality_test(reference, current, **kwargs)
    return MetricResults(
        metric_type="drift",
        metric_name="model_equality",
        feature_name=model,
        metric_value=result.statistic,
        threshold=alpha,
        drift_status=result.p_value < alpha,
        n=result.answers_current,
        method=f"MMD with a Hamming kernel within prompts (Gao et al., ICLR 2025); {result.prompts} canary prompts, "
        f"{result.answers_reference} reference and {result.answers_current} current answers; permutation p-value "
        f"{result.p_value:.4f}",
        test_type="fixed",
        p_value=result.p_value,
    )
