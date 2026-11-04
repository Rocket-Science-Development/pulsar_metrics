import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.fairness import comparative_analysis, counterfactual_analysis


def probe_rows(n_sets, p_auth, noise=0.05, n_placebo=None, seed=0):
    """Rows like CounterfactualProbe.run: outcome = first tool; p_auth[group] = probability of asking to authenticate"""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sets):
        base = rng.random()  # shared by the set: identical request except the name
        for group, p in p_auth.items():
            jitter = rng.random() < noise
            outcome = "authenticate" if (base < p) != jitter else "check_eligibility"
            rows.append({"set_id": f"cf-{s}", "variant": group, "attribute": "name", "control": "none", "outcome": outcome})
    for s in range(n_sets if n_placebo is None else n_placebo):
        base = rng.random()
        for day in ("tuesday", "thursday"):
            jitter = rng.random() < noise
            outcome = "authenticate" if (base < p_auth["fr"]) != jitter else "check_eligibility"
            rows.append({"set_id": f"pl-{s}", "variant": day, "attribute": "day", "control": "placebo", "outcome": outcome})
    frame = pd.DataFrame(rows)
    frame["error"] = None
    return frame


def get(results, name, group=None):
    return [r for r in results if r.metric_name == name and (group is None or r.group == group)]


def test_no_effect_stays_at_the_noise_floor():
    rows = probe_rows(400, {"fr": 0.2, "en": 0.2, "ar": 0.2})
    results = counterfactual_analysis(rows, "fr", action="authenticate")
    floor = get(results, "placebo_flip_rate")[0]
    assert 0.03 < floor.metric_value < 0.2
    for group in ("name=en", "name=ar"):
        assert get(results, "counterfactual_excess_flip_rate", group)[0].drift_status is False
        assert get(results, "counterfactual_action_rate_difference", group)[0].drift_status is False


def test_induced_bias_is_detected_on_the_affected_group_only():
    rows = probe_rows(400, {"fr": 0.2, "en": 0.2, "ar": 0.6})
    results = counterfactual_analysis(rows, "fr", action="authenticate")
    ar = get(results, "counterfactual_action_rate_difference", "name=ar")[0]
    assert ar.drift_status is True and ar.ci_low > 0.1 and "McNemar" in ar.method
    assert get(results, "counterfactual_excess_flip_rate", "name=ar")[0].drift_status is True
    assert get(results, "counterfactual_action_rate_ratio", "name=ar")[0].drift_status is True
    assert get(results, "counterfactual_action_rate_difference", "name=en")[0].drift_status is False
    assert get(results, "counterfactual_flip_rate", "name=ar")[0].reference_group == "name=fr"


def test_failed_requests_are_excluded_and_inputs_checked():
    rows = probe_rows(50, {"fr": 0.2, "en": 0.2})
    rows.loc[0, "error"] = "TimeoutError"
    results = counterfactual_analysis(rows, "fr")
    assert get(results, "counterfactual_flip_rate", "name=en")[0].n == 49
    assert not get(results, "counterfactual_action_rate_difference")  # no action given
    with pytest.raises(error_msg):
        counterfactual_analysis(rows, "de")


def test_without_placebo_no_excess_is_reported():
    rows = probe_rows(100, {"fr": 0.2, "en": 0.2}, n_placebo=0)
    results = counterfactual_analysis(rows, "fr", action=lambda tool: tool == "authenticate")
    assert not get(results, "counterfactual_excess_flip_rate") and not get(results, "placebo_flip_rate")
    assert get(results, "counterfactual_action_rate_difference", "name=en")[0].method.startswith("rate of")


def comparative_rows(n_sets, p_first, p_prefer_a, tie=0.0, seed=0):
    """Both orders per set; the model prefers the first slot with p_first and group a with p_prefer_a"""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sets):
        for control, (first, second) in (("none", ("a", "b")), ("order_swapped", ("b", "a"))):
            if rng.random() < tie:
                choice = "tie"
            elif rng.random() < p_first:
                choice = "first"
            else:
                choice = "first" if (rng.random() < p_prefer_a) == (first == "a") else "second"
            chosen = first if choice == "first" else second if choice == "second" else "tie"
            rows.append(
                {
                    "set_id": f"cmp-{s}",
                    "variant": f"{first}|{second}",
                    "attribute": "name",
                    "control": control,
                    "choice": choice,
                    "first_group": first,
                    "second_group": second,
                    "chosen_group": chosen,
                    "allow_tie": True,
                    "error": None,
                }
            )
    return pd.DataFrame(rows)


def test_position_effect_is_flagged_and_cancelled():
    # strong position bias, no preference between groups
    results = comparative_analysis(comparative_rows(300, p_first=0.5, p_prefer_a=0.5))
    position = get(results, "position_effect")[0]
    assert position.drift_status is True and position.metric_value > 0.65
    preference = get(results, "comparative_preference")[0]
    # balanced estimate is still close to 1/2, but a biased instrument cannot certify the absence of preference
    assert preference.drift_status is None and abs(preference.metric_value - 0.5) < 0.05
    assert "depend on the order" in preference.method


def test_order_only_answers_give_no_verdict():
    # the model always answers the second slot (seen with llama3.2:3b): no information on the groups
    results = comparative_analysis(
        comparative_rows(60, p_first=0.0, p_prefer_a=0.5)
        .assign(choice="second")
        .pipe(lambda f: f.assign(chosen_group=f["second_group"]))
    )
    assert get(results, "position_effect")[0].drift_status is True
    preference = get(results, "comparative_preference")[0]
    assert preference.drift_status is None and "0 of 60 sets" in preference.method


def test_comparative_preference_detected():
    results = comparative_analysis(comparative_rows(300, p_first=0.0, p_prefer_a=0.8, tie=0.1))
    preference = get(results, "comparative_preference")[0]
    assert preference.drift_status is True and preference.group == "name=a" and preference.ci_low > 0.6
    assert get(results, "position_effect")[0].drift_status is False
    assert 0.05 < get(results, "tie_rate")[0].metric_value < 0.15


def test_no_discordant_set_gives_a_bounded_interval_not_zero_width():
    rows = probe_rows(60, {"fr": 0.0, "en": 0.0}, noise=0.0, n_placebo=0)
    difference = get(counterfactual_analysis(rows, "fr", action="authenticate"), "counterfactual_action_rate_difference")[0]
    assert difference.metric_value == 0 and difference.ci_low < -0.04 and difference.ci_high > 0.04
    assert difference.drift_status is False  # within the default zone [-0.1, 0.1]


def test_unusable_answers_are_reported_not_dropped_silently():
    # 20 % unparsed answers (choice None), 10 % refusals: excluded from the analysis, counted in unusable_rate
    rows = comparative_rows(300, p_first=0.0, p_prefer_a=0.5, seed=3)
    rng = np.random.default_rng(3)
    draw = rng.random(len(rows))
    rows.loc[draw < 0.2, "choice"] = None
    rows.loc[(draw >= 0.2) & (draw < 0.3), "choice"] = "refusal"
    rows.loc[rows["choice"].isin([None, "refusal"]) | rows["choice"].isna(), "chosen_group"] = None
    results = comparative_analysis(rows)
    unusable = get(results, "unusable_rate")[0]
    excluded = int(rows["choice"].isna().sum() + (rows["choice"] == "refusal").sum())
    assert unusable.metric_value == pytest.approx(excluded / len(rows)) and unusable.n == len(rows)
    assert f"{int((rows['choice'] == 'refusal').sum())} refusal" in unusable.method
    assert "sets usable in both orders" in unusable.method
    assert unusable.drift_status is False  # under max_unusable_rate (0.5)
    # refusals are not decisions: the position effect only counts first / second
    assert get(results, "position_effect")[0].n == int(rows["choice"].isin(["first", "second"]).sum())
    assert get(results, "comparative_instrument")[0].status == "usable"


def test_order_driven_or_unusable_instrument_gets_an_explicit_status():
    order_only = comparative_rows(60, p_first=0.0, p_prefer_a=0.5).assign(choice="second")
    order_only["chosen_group"] = order_only["second_group"]
    results = comparative_analysis(order_only)
    instrument = get(results, "comparative_instrument")[0]
    assert instrument.metric_value == "instrument_unusable" and "same position twice" in instrument.method
    preference = get(results, "comparative_preference")[0]
    assert preference.status == "instrument_unusable" and preference.drift_status is None
    # mostly refused: unusable as well, and no exception when nothing is usable
    refused = comparative_rows(60, p_first=0.0, p_prefer_a=0.8).assign(choice="refusal")
    results = comparative_analysis(refused)
    assert get(results, "unusable_rate")[0].metric_value == 1 and get(results, "unusable_rate")[0].drift_status is True
    assert get(results, "comparative_instrument")[0].status == "instrument_unusable"
    assert not get(results, "comparative_preference")
