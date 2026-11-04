"""Analysis of counterfactual probes (pulsar_data_collection.genai.probes): does an attribute that must not matter
change what an LLM application or an agent does?

Two families of results, both with intervals and both judged against controls:

- `counterfactual_analysis`: requests that differ only by the probed attribute. Flip rate (share of sets whose outcome
  changes), its excess over the flip rate of placebo sets (noise floor of the instrument), and the disparity of the
  rate of a given action (paired McNemar test, Benjamini-Hochberg over groups, ratio for the 4/5 rule).
- `comparative_analysis`: choices between two profiles asked in both orders. Position effect (control), tie rate and
  order-balanced preference for each group.
"""

import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults
from ..multiple_testing import benjamini_hochberg
from .criteria import equivalence_verdict, verdict
from .planning import flip_sets_needed, paired_sets_needed


def _wilson(successes: int, n: int, confidence: float) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    ci = stats.binomtest(successes, n).proportion_ci(confidence_level=confidence, method="wilson")
    return float(ci.low), float(ci.high)


def _newcombe(p1, ci1, p2, ci2) -> tuple:
    """Interval of p1 - p2 for two independent proportions (Newcombe hybrid score)"""
    d = p1 - p2
    return d - np.hypot(p1 - ci1[0], ci2[1] - p2), d + np.hypot(ci1[1] - p1, p2 - ci2[0])


def _bootstrap(samples: tuple, statistic: Callable, confidence: float, n_resamples: int, seed) -> tuple:
    """Paired BCa interval over sets; percentile interval when BCa is not defined (degenerate samples)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = stats.bootstrap(
                samples,
                statistic,
                paired=True,
                vectorized=False,
                confidence_level=confidence,
                n_resamples=n_resamples,
                method="BCa",
                random_state=seed,
            )
            low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
            if np.isfinite(low) and np.isfinite(high):
                return low, high, "BCa"
        except (ValueError, ZeroDivisionError):
            pass
    rng = np.random.default_rng(seed)
    n = len(samples[0])
    values = []
    for _ in range(n_resamples):
        rows = rng.integers(0, n, n)
        values.append(statistic(*(s[rows] for s in samples)))
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), "percentile"
    tail = (1 - confidence) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail)), "percentile"


def _usable(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in rows:
        raise error_msg(value=column, message="column missing from the probe rows")
    keep = rows[column].notna()
    if "error" in rows:
        keep &= rows["error"].isna()
    return rows[keep]


def _wide(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per set, one column per variant (sets with a missing variant keep NaN)"""
    return rows.pivot_table(index="set_id", columns="variant", values=column, aggfunc="first")


def counterfactual_analysis(
    rows: pd.DataFrame,
    reference_variant: str,
    outcome: str = "outcome",
    action: Optional[Union[str, Callable]] = None,
    alpha: float = 0.05,
    flip_tolerance: float = 0.05,
    rate_zone: tuple = (-0.1, 0.1),
    ratio_zone: tuple = (0.8, 1.25),
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: Optional[int] = 0,
    equivalence_confidence: Optional[float] = None,
) -> list:
    """Analyse the rows of a CounterfactualProbe run

    Parameters
    ----------
    rows : DataFrame
        Output of `CounterfactualProbe.run`: set_id, variant, attribute, control ("none" / "placebo"), outcome, error
    reference_variant : str
        Group the others are compared with
    outcome : str, optional
        Column holding the outcome label (default "outcome")
    action : str or callable, optional
        Outcome whose rate is compared between groups (e.g. "authenticate", "refused"), or a predicate on the outcome
    flip_tolerance : float, optional
        Acceptable excess of the flip rate over the placebo flip rate (default 0.05)
    rate_zone, ratio_zone : tuple, optional
        Acceptable zones of the action rate difference and ratio (defaults: [-0.1, 0.1] and the 4/5 rule)
    equivalence_confidence : float, optional
        Confidence level of the interval used to conclude "within the zone" / "no excess flips" (default:
        `confidence`, a TOST at 2.5 % per side). 0.90 gives the usual TOST at 5 % per side; a disparity is still
        established at `confidence`.

    Inconclusive results (drift_status None) say in `method` about how many probe sets would settle them at the
    observed rates. Small probes rarely conclude "within the zone": in the lab (FA-04, no bias, 2 % of answers
    flipping at random), the action rate difference was certified in 10 % of runs at 30 sets, 79 % at 60, 97 % at 100.
    """
    eq_level = equivalence_confidence or confidence
    rows = _usable(rows, outcome)
    probed = rows[rows["control"] == "none"]
    placebo = rows[rows["control"] == "placebo"]
    if probed.empty:
        raise error_msg(value=None, message="no usable row of the probed attribute")
    attribute = str(probed["attribute"].iloc[0])
    wide = _wide(probed, outcome)
    if reference_variant not in wide:
        raise error_msg(value=reference_variant, message="reference variant missing from the probe rows")
    variants = [v for v in wide.columns if v != reference_variant]
    common = dict(metric_type="fairness", feature_name=attribute, reference_group=f"{attribute}={reference_variant}")

    # noise floor: flip rate between the first placebo variant and the others
    floor = None
    if not placebo.empty:
        pw = _wide(placebo, outcome)
        base = pw.columns[0]
        flips = [(pw[base] != pw[c])[pw[base].notna() & pw[c].notna()] for c in pw.columns[1:]]
        flips = pd.concat(flips) if flips else pd.Series(dtype=bool)
        if len(flips):
            k, n = int(flips.sum()), len(flips)
            floor = (k / n, _wilson(k, n, confidence), n)

    out = []
    if floor is not None:
        out.append(
            MetricResults(
                metric_name="placebo_flip_rate",
                feature_name=str(placebo["attribute"].iloc[0]),
                metric_value=floor[0],
                ci_low=floor[1][0],
                ci_high=floor[1][1],
                n=floor[2],
                metric_type="fairness",
                method="share of placebo sets whose outcome changes with an attribute that must not matter (noise floor); Wilson",
                test_type="fixed",
            )
        )

    is_action = None
    if action is not None:
        is_action = action if callable(action) else (lambda value, target=action: value == target)
    action_tests = []
    for v in variants:
        pair = wide[[reference_variant, v]].dropna()
        n = len(pair)
        if n == 0:
            continue
        flipped = (pair[reference_variant] != pair[v]).to_numpy()
        k = int(flipped.sum())
        rate, ci = k / n, _wilson(k, n, confidence)
        group = f"{attribute}={v}"
        out.append(
            MetricResults(
                metric_name="counterfactual_flip_rate",
                group=group,
                metric_value=rate,
                ci_low=ci[0],
                ci_high=ci[1],
                n=n,
                method="share of sets whose outcome changes when only the attribute changes; Wilson",
                test_type="fixed",
                **common,
            )
        )
        if floor is not None:
            low, high = _newcombe(rate, ci, floor[0], floor[1])
            eq = _newcombe(rate, _wilson(k, n, eq_level), floor[0], _wilson(round(floor[0] * floor[2]), floor[2], eq_level))
            excess = rate - floor[0]
            status = equivalence_verdict((float(low), float(high)), eq, excess, (-1.0, flip_tolerance))
            how = f"flip rate minus placebo flip rate (Newcombe); established above {flip_tolerance}"
            if equivalence_confidence is not None:
                how += f"; absence judged at {equivalence_confidence:.0%} (TOST)"
            if status is None:
                needed = flip_sets_needed(rate, floor[0], flip_tolerance, placebo_sets=floor[2], confidence=eq_level)
                how += (
                    f"; inconclusive: about {needed} probe sets for an 80 % chance of a verdict at the observed rates"
                    if needed
                    else "; inconclusive at any number of probe sets with this placebo run"
                )
            out.append(
                MetricResults(
                    metric_name="counterfactual_excess_flip_rate",
                    group=group,
                    metric_value=excess,
                    ci_low=float(low),
                    ci_high=float(high),
                    n=n,
                    threshold=flip_tolerance,
                    drift_status=status,
                    method=how,
                    test_type="fixed",
                    **common,
                )
            )
        if is_action is not None:
            ref = np.array([bool(is_action(x)) for x in pair[reference_variant]], dtype=float)
            alt = np.array([bool(is_action(x)) for x in pair[v]], dtype=float)
            b = int(((ref == 1) & (alt == 0)).sum())
            c = int(((ref == 0) & (alt == 1)).sum())
            p_value = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
            action_tests.append((v, group, n, ref, alt, b, c, p_value))

    if action_tests:
        adjusted = benjamini_hochberg([t[-1] for t in action_tests], alpha)[1]
        label = action if isinstance(action, str) else getattr(action, "__name__", "action")
        for (v, group, n, ref, alt, b, c, p_value), p_adj in zip(action_tests, adjusted):
            mcnemar = f"McNemar exact: {b} vs {c} discordant sets, p = {p_value:.3g}, BH-adjusted {p_adj:.3g}"
            difference = float(alt.mean() - ref.mean())

            def difference_interval(level):
                if b + c == 0:
                    # no discordant set: the bootstrap would give [0, 0]; bound the difference by the share of discordant sets
                    bound = _wilson(0, n, level)[1]
                    return -bound, bound, "Wilson bound on the share of discordant sets (none observed)"
                low, high, how = _bootstrap((ref, alt), lambda r, a: a.mean() - r.mean(), level, n_resamples, seed)
                return low, high, how + " bootstrap"

            low, high, how = difference_interval(confidence)
            eq = (low, high) if eq_level == confidence else difference_interval(eq_level)[:2]
            status = equivalence_verdict((low, high), eq, difference, rate_zone)
            if equivalence_confidence is not None:
                how += f"; 'within the zone' judged at {equivalence_confidence:.0%} (TOST)"
            if status is None:
                needed = paired_sets_needed(difference, max((b + c) / n, abs(difference)), rate_zone, eq_level)
                how += (
                    f"; inconclusive: about {needed} probe sets for an 80 % chance of a verdict at the observed rates"
                    if needed
                    else "; inconclusive at any number of probe sets (the difference sits on a bound)"
                )
            out.append(
                MetricResults(
                    metric_name="counterfactual_action_rate_difference",
                    group=group,
                    metric_value=difference,
                    ci_low=low,
                    ci_high=high,
                    n=n,
                    threshold=list(rate_zone),
                    drift_status=status,
                    method=f"rate of {label!s} in the group minus in the reference group, paired by set; {how}; " + mcnemar,
                    test_type="fixed",
                    **common,
                )
            )
            ratio = float(alt.mean() / ref.mean()) if ref.mean() > 0 else float("nan")
            low, high, how = _bootstrap(
                (ref, alt), lambda r, a: a.mean() / r.mean() if r.mean() > 0 else np.nan, confidence, n_resamples, seed
            )
            out.append(
                MetricResults(
                    metric_name="counterfactual_action_rate_ratio",
                    group=group,
                    metric_value=None if np.isnan(ratio) else ratio,
                    ci_low=None if np.isnan(low) else low,
                    ci_high=None if np.isnan(high) else high,
                    n=n,
                    threshold=list(ratio_zone),
                    drift_status=None if np.isnan(ratio) else verdict(low, high, ratio, ratio_zone),
                    method=f"rate of {label!s} in the group divided by the reference group's (4/5 rule); {how} bootstrap",
                    test_type="fixed",
                    **common,
                )
            )
    return out


CHOICES = ("first", "second", "tie")  # usable values of the `choice` column of a ComparativeProbe run
REFUSAL = "refusal"  # value of `choice` for an answer that refuses to choose


def _unusable_kinds(rows: pd.DataFrame) -> pd.Series:
    """Why each answer cannot be used: "error" (the call failed), "refusal" (`choice` == "refusal"), "unparsed" (no
    choice, or a value outside first / second / tie), None when usable"""
    choice = rows["choice"]
    kind = pd.Series(None, index=rows.index, dtype=object)
    kind[~choice.isin(CHOICES)] = "unparsed"
    kind[choice == REFUSAL] = "refusal"
    if "error" in rows:
        kind[rows["error"].notna()] = "error"
    return kind


def comparative_analysis(
    rows: pd.DataFrame,
    alpha: float = 0.05,
    preference_zone: tuple = (0.4, 0.6),
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: Optional[int] = 0,
    max_unusable_rate: float = 0.5,
    max_order_driven_share: float = 0.5,
) -> list:
    """Analyse the rows of a ComparativeProbe run (choice: "first", "second" or "tie")

    - `unusable_rate`: share of the answers that cannot be used: call errors, refusals (`choice` == "refusal") and
      unparsed answers (no choice, or any value other than first / second / tie). They are excluded from the other
      results; the rate is reported overall and, with several pairs of groups, per pair (`group`, `reference_group`),
      since a model may refuse the sentences about one group more than about another. `drift_status` True when it
      exceeds `max_unusable_rate`.
    - `position_effect`: share of non-tie answers that pick the first profile; `drift_status` True means the
      instrument is biased by order (binomial test against 0.5). The preference below cancels it by averaging the
      two orders.
    - `tie_rate`: share of ties, when they were allowed.
    - `comparative_instrument`: metric_value and `status` "usable" or "instrument_unusable". The instrument is
      unusable for the model when more than `max_unusable_rate` of the answers are unusable, or when the answers
      depend on the order (position effect established) and more than `max_order_driven_share` of the sets decided
      in both orders picked the same position twice: the model answers the position, not the profiles (lab LB-02).
    - `comparative_preference`: for each pair of groups, probability of choosing the group, averaged over both
      orders, ties counting one half; acceptable zone `preference_zone`. With an unusable instrument they carry
      `status="instrument_unusable"` (an established preference is kept, no verdict of absence is given).
    """
    if "choice" not in rows:
        raise error_msg(value="choice", message="column missing from the probe rows")
    if rows.empty:
        raise error_msg(value=None, message="no comparative row")
    kinds = _unusable_kinds(rows)
    attribute = str(rows["attribute"].iloc[0])
    out = [_unusable_result(kinds, attribute, confidence, max_unusable_rate, rows)]
    if "first_group" in rows and "second_group" in rows:
        pairs = pd.Series([tuple(sorted((a, b))) for a, b in zip(rows["first_group"], rows["second_group"])], index=rows.index)
        if pairs.nunique() > 1:
            for (a, b), index in pairs.groupby(pairs).groups.items():
                result = _unusable_result(kinds[index], attribute, confidence, max_unusable_rate, rows.loc[index])
                result.group, result.reference_group = f"{attribute}={a}", f"{attribute}={b}"
                out.append(result)
    unusable_share = float(kinds.notna().mean())
    rows = rows[kinds.isna()]
    if rows.empty:
        out.append(_instrument_result(attribute, False, f"no usable answer ({unusable_share:.0%} unusable)"))
        return out
    decided = rows[rows["choice"] != "tie"]
    k, n = int((decided["choice"] == "first").sum()), len(decided)
    position_biased = False
    if n:
        test = stats.binomtest(k, n, 0.5)
        position_biased = bool(test.pvalue < alpha)
        ci = test.proportion_ci(confidence_level=confidence, method="wilson")
        out.append(
            MetricResults(
                metric_type="fairness",
                metric_name="position_effect",
                feature_name=attribute,
                metric_value=k / n,
                ci_low=float(ci.low),
                ci_high=float(ci.high),
                n=n,
                threshold=0.5,
                drift_status=position_biased,
                method=f"share of non-tie answers choosing the first profile; two-sided binomial test against 0.5 at "
                f"alpha = {alpha:g}, p = {test.pvalue:.3g}; True = the answers depend on the order (instrument control)",
                test_type="fixed",
                p_value=float(test.pvalue),
            )
        )
    if bool(rows.get("allow_tie", pd.Series([False])).any()):
        ties = int((rows["choice"] == "tie").sum())
        ci = _wilson(ties, len(rows), confidence)
        out.append(
            MetricResults(
                metric_type="fairness",
                metric_name="tie_rate",
                feature_name=attribute,
                metric_value=ties / len(rows),
                ci_low=ci[0],
                ci_high=ci[1],
                n=len(rows),
                method="share of answers choosing the tie option; Wilson",
                test_type="fixed",
            )
        )

    # order-driven sets: decided in both orders, the same position chosen twice (so a different profile each time)
    both = decided.pivot_table(index="set_id", columns="control", values="choice", aggfunc="first")
    both = both.dropna() if both.shape[1] == 2 else both.iloc[0:0]
    order_driven = float((both.iloc[:, 0] == both.iloc[:, 1]).mean()) if len(both) else 0.0
    reasons = []
    if unusable_share > max_unusable_rate:
        reasons.append(f"{unusable_share:.0%} of the answers are unusable (above {max_unusable_rate:.0%})")
    if position_biased and order_driven > max_order_driven_share:
        reasons.append(
            f"the answers depend on the order: {order_driven:.0%} of the {len(both)} sets decided in both orders chose the "
            f"same position twice (above {max_order_driven_share:.0%})"
        )
    usable = not reasons
    detail = "; ".join(reasons) or (
        f"{unusable_share:.0%} unusable answers, {order_driven:.0%} of the {len(both)} sets decided in both orders chose "
        "the same position twice"
    )
    out.append(_instrument_result(attribute, usable, detail))

    rows = rows.assign(pair=[tuple(sorted((a, b))) for a, b in zip(rows["first_group"], rows["second_group"])])
    for (a, b), frame in rows.groupby("pair"):
        score = frame["chosen_group"].map(lambda g, a=a: 1.0 if g == a else 0.5 if g == "tie" else 0.0)
        per_set = score.groupby(frame["set_id"]).mean()
        complete = frame.groupby("set_id")["control"].nunique() == 2  # both orders answered
        values = per_set[complete.reindex(per_set.index, fill_value=False)].to_numpy()
        if not len(values):
            continue
        value = float(values.mean())
        low, high, how = _bootstrap((values,), np.mean, confidence, n_resamples, seed)
        status = verdict(low, high, value, preference_zone)
        consistent = int(((values == 0) | (values == 1)).sum())  # both orders chose the same group
        note = f"; {consistent} of {len(values)} sets chose the same group in both orders"
        if position_biased and status is not True:
            # an instrument driven by the order can reveal a strong preference, not certify that there is none
            status = None
            note += "; no verdict of absence: the answers depend on the order"
        if not usable:
            note += f"; instrument unusable for this model: {detail}"
        out.append(
            MetricResults(
                metric_type="fairness",
                metric_name="comparative_preference",
                feature_name=attribute,
                group=f"{attribute}={a}",
                reference_group=f"{attribute}={b}",
                metric_value=value,
                ci_low=low,
                ci_high=high,
                n=len(values),
                threshold=list(preference_zone),
                drift_status=status,
                method=f"probability of choosing the group, averaged over both orders (ties = 1/2); {how} bootstrap over sets"
                + note,
                test_type="fixed",
                status=None if usable else "instrument_unusable",
            )
        )
    return out


def _unusable_result(kinds: pd.Series, attribute: str, confidence: float, threshold: float, rows: pd.DataFrame):
    """Share of unusable answers, with the count of each kind and of the sets left usable in both orders"""
    n, k = len(kinds), int(kinds.notna().sum())
    counts = kinds.value_counts()
    detail = ", ".join(f"{int(counts.get(kind, 0))} {kind}" for kind in ("refusal", "unparsed", "error"))
    if "set_id" in rows and "control" in rows:
        complete = int((rows[kinds.isna()].groupby("set_id")["control"].nunique() == 2).sum())
        detail += f"; {complete} of {rows['set_id'].nunique()} sets usable in both orders"
    ci = _wilson(k, n, confidence)
    return MetricResults(
        metric_type="fairness",
        metric_name="unusable_rate",
        feature_name=attribute,
        metric_value=k / n,
        ci_low=ci[0],
        ci_high=ci[1],
        n=n,
        threshold=threshold,
        drift_status=bool(k / n > threshold),
        method=f"share of the answers excluded from the analysis ({detail}); Wilson",
        test_type="fixed",
    )


def _instrument_result(attribute: str, usable: bool, detail: str):
    status = "usable" if usable else "instrument_unusable"
    return MetricResults(
        metric_type="fairness",
        metric_name="comparative_instrument",
        feature_name=attribute,
        metric_value=status,
        status=status,
        method=("instrument usable: " if usable else "instrument unusable for this model: ") + detail,
        test_type="fixed",
    )
