"""Compliance rules over rich traces: atom predicates, argument binding, conversation events, not-evaluable verdicts

Lab LA-02 (tau-bench retail): the confirmation policy failed on 36 of 36 runs with a write on the tool-name trace,
against 23 real violations once user confirmations are in the trace; 10 of 15 retail policies need arguments,
outputs or events. The traces below are synthetic replicas (no tau-bench data)."""

import math
import random
import warnings

import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.genai import ComplianceMonitor, Rule, load_rules, trace_from
from pulsar_metrics.genai.compliance import NOT_EVALUABLE, NotEvaluableWarning, RuleNotEvaluable, holds, parse


def tool(name, status="ok", **args):
    return {"name": name, "kind": "tool", "args": args, "status": status}


def event(name):
    return {"name": name, "kind": "event"}


def names(trace):
    return [e["name"] for e in trace if e["kind"] == "tool"]


WRITES = ["cancel_pending_order", "modify_pending_order_address", "return_delivered_order_items"]

# ----------------------------------------------------------------------------------------------- backward compatibility
OLD_RULES = {
    "rules": [
        {"id": "auth-before-file", "template": "precedence", "events": ["authenticate", "create_file"], "severity": "critical"},
        {"id": "in-scope", "template": "allowed_tools", "tools": ["search_programs", "authenticate", "create_file"]},
        {"id": "no-repeat-file", "formula": "G (create_file -> !X create_file)"},
        {"id": "auth-then-file", "template": "chain_response", "events": ["authenticate", "create_file"]},
    ]
}


@pytest.mark.parametrize(
    "rich",
    [
        [tool("search_programs"), event("user_msg"), tool("create_file"), tool("authenticate")],
        [tool("authenticate"), event("user_msg"), tool("create_file"), event("user_msg"), tool("create_file")],
        [tool("authenticate"), event("agent_msg"), tool("create_file")],
        [tool("delete_file"), event("user_msg")],
        [event("user_msg")],
    ],
)
def test_existing_rules_give_the_same_verdict_on_rich_and_name_traces(rich):
    """Rules over tool names judge a rich trace as its tool sequence: the events they do not name are invisible
    (an event between two calls must not satisfy "!X create_file" or break "X create_file")"""
    monitor = ComplianceMonitor(load_rules(OLD_RULES), version="rules/1")
    assert ComplianceMonitor.accepts_rich_trace is True
    tools = names(rich)
    assert monitor.violations(rich) == monitor.violations(tools)
    assert monitor.run_record(rich) == monitor.run_record(trace_from(">".join(tools) or "none"))
    assert monitor.alerts(rich) == monitor.alerts(tools)


def test_a_plain_atom_matches_a_rich_element_by_name():
    assert holds(parse("F user_confirmation"), [tool("a"), event("user_confirmation")]) is True
    assert holds(parse("a U b"), [tool("a"), "a", tool("b")]) is True
    assert Rule(id="x", formula="F b").check([{"name": "b"}]) is True


# ----------------------------------------------------------------------------------------------- atom predicates
RICH = [
    tool("find_user_id_by_email", email="ana@example.test"),
    event("user_confirmation"),
    tool("get_order_details", order_id="#W1"),
    tool("cancel_pending_order", status="error", order_id="#W1", reason="no longer needed"),
    tool("get_order_details", order_id=42),
]


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("F cancel_pending_order[order_id=#W1]", True),
        ("F cancel_pending_order[order_id=#W2]", False),
        ("F cancel_pending_order[order_id]", True),
        ("F cancel_pending_order[payment_method_id]", False),
        ("F cancel_pending_order[reason='no longer needed']", True),
        ('F cancel_pending_order[order_id="#W1", reason!=other]', True),
        ("F cancel_pending_order[status=error]", True),
        ("F get_order_details[status=error]", False),
        ("F get_order_details[order_id=42]", True),  # values compared as text
        ("F *[status=error]", True),
        ("G (*[status!=error] | cancel_pending_order)", True),
        ("F @user_confirmation", True),
        ("F @get_order_details", False),  # a tool is not an event
        ("F *[order_id=#W1] & F @*", True),
        ("(!cancel_pending_order[order_id=#W1] U get_order_details[order_id=#W1])", True),
    ],
)
def test_atom_predicates(formula, expected):
    assert Rule(id="r", formula=formula).verdict(RICH) is expected


def test_atom_predicate_on_an_argument_named_status():
    trace = [{"name": "get_order_details", "kind": "tool", "args": {"status": "pending"}, "status": "ok"}]
    assert Rule(id="r", formula="F get_order_details[args.status=pending]").verdict(trace) is True
    assert Rule(id="r", formula="F get_order_details[status=pending]").verdict(trace) is False


@pytest.mark.parametrize("bad", ["a[]", "a[status]", "a[=x]", "a[x=1", "@", "a[x y]"])
def test_atom_predicate_syntax_errors(bad):
    with pytest.raises(error_msg):
        parse(bad)


def test_missing_data_makes_the_verdict_unknown_not_false():
    """Kleene semantics: an element without args (or status) leaves the atom unknown; known elements still decide"""
    no_args = [{"name": "cancel_pending_order", "kind": "tool"}]
    rule = Rule(id="r", formula="F cancel_pending_order[order_id=#W1]")
    assert rule.verdict(no_args) is None
    with pytest.raises(RuleNotEvaluable):
        rule.check(no_args)
    assert rule.verdict(no_args + [tool("cancel_pending_order", order_id="#W1")]) is True
    assert Rule(id="r", formula="!F x[status=error]").verdict([{"name": "x", "kind": "tool", "args": {}}]) is None
    # a str element of a rich trace has a name only
    assert Rule(id="r", formula="F @user_confirmation").verdict([tool("a"), "user_confirmation"]) is None


# ----------------------------------------------------------------------------------------------- argument binding
P5 = {"id": "P5", "template": "precedence", "events": ["get_order_details", WRITES], "bind": ["order_id"]}


def test_precedence_bound_on_an_argument():
    """LA-02: P5 (order looked up before it is changed) is argument-blind: 18 writes accepted on another order"""
    blind, bound = Rule(**{k: v for k, v in P5.items() if k != "bind"}), Rule(**P5)
    other_order = [tool("get_order_details", order_id="#W1"), tool("cancel_pending_order", order_id="#W2")]
    assert blind.check(other_order) is True and bound.check(other_order) is False
    same = [
        tool("get_order_details", order_id="#W1"),
        tool("get_order_details", order_id="#W2"),
        event("user_confirmation"),
        tool("cancel_pending_order", order_id="#W2"),
        tool("return_delivered_order_items", order_id="#W1", item_ids=["1"]),
    ]
    assert bound.check(same) is True
    assert bound.check([tool("get_order_details", order_id="#W1")]) is True
    assert bound.check([tool("search")]) is True and bound.check([]) is True
    # a write without the bound argument can be neither tied to nor excluded from an order
    assert bound.verdict(same + [tool("cancel_pending_order")]) is None
    assert bound.verdict(trace_from("get_order_details>cancel_pending_order")) is None


def test_response_bound_on_two_arguments():
    rule = Rule(id="r", template="response", events=["open_ticket", "close_ticket"], bind=["user_id", "ticket"])
    ok = [tool("open_ticket", user_id=1, ticket="a"), tool("open_ticket", user_id=2, ticket="a")]
    ok += [tool("close_ticket", user_id=2, ticket="a"), tool("close_ticket", user_id=1, ticket="a")]
    assert rule.check(ok) is True
    assert rule.check(ok[:3]) is False


def test_bound_rule_from_a_yaml_file(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n"
        "  - id: status-checked\n"
        "    template: precedence\n"
        "    events: [get_order_details, [cancel_pending_order, modify_pending_order_address]]\n"
        "    bind: [order_id]\n"
        "    severity: major\n"
        "  - id: confirm-before-write\n"
        "    template: alternate_precedence\n"
        "    events: ['@user_confirmation', [cancel_pending_order, modify_pending_order_address]]\n"
        "    severity: critical\n",
        encoding="utf-8",
    )
    monitor = ComplianceMonitor(load_rules(str(path)), version="retail/2")
    trace = [tool("get_order_details", order_id="#W1"), tool("cancel_pending_order", order_id="#W2")]
    assert monitor.violations(trace) == ["status-checked", "confirm-before-write"]
    assert [a["rule_id"] for a in monitor.alerts(trace)] == ["confirm-before-write"]
    with pytest.raises(error_msg):
        Rule(id="x", template="allowed_tools", tools=["a"], bind=["order_id"])


# ----------------------------------------------------------------------------------------------- events
def _retail_runs(seed=0):
    """40 synthetic retail runs as in LA-02: 4 without write, 13 confirming every write, 23 violating"""
    rng = random.Random(seed)
    runs = []

    def start():
        return [tool("find_user_id_by_email", email="user@example.test"), event("user_msg"), tool("get_user_details")]

    for _ in range(4):
        runs.append(start() + [tool("get_order_details", order_id="#W0"), event("agent_msg")])
    for _ in range(13):
        trace = start()
        for k in range(rng.randint(1, 2)):
            trace += [tool("get_order_details", order_id=f"#W{k}"), event("agent_msg"), event("user_confirmation")]
            trace += [event("user_msg"), tool(rng.choice(WRITES), order_id=f"#W{k}")]
        runs.append(trace)
    for i in range(23):
        trace = start() + [tool("get_order_details", order_id="#W1")]
        if i % 2:  # confirmation for the first write only
            trace += [event("user_confirmation"), tool(rng.choice(WRITES), order_id="#W1"), event("user_msg")]
        trace += [tool(rng.choice(WRITES), order_id="#W1")]
        runs.append(trace)
    return runs


def test_confirmation_before_every_write_needs_the_events():
    """LA-02: P2 fails on every run with a write on the tool-name trace (36 / 36); with the confirmation events, only
    the 23 real violations remain"""
    runs = _retail_runs()
    stored = [trace_from(">".join(names(t))) for t in runs]
    with_write = sum(any(n in WRITES for n in names(t)) for t in runs)
    assert len(runs) == 40 and with_write == 36

    today = Rule(id="P2", template="alternate_precedence", events=["user_confirmation", WRITES], severity="critical")
    assert sum(not today.check(t) for t in stored) == 36  # structural false alarms
    assert sum(not today.check(t) for t in runs) == 23  # same rule, rich trace: the event matches by name

    rule = Rule(id="P2", template="alternate_precedence", events=["@user_confirmation", WRITES], severity="critical")
    assert rule.needs_rich_trace and not today.needs_rich_trace
    assert sum(not rule.check(t) for t in runs) == 23
    assert all(rule.verdict(t) is None for t in stored)  # not evaluable, never a silent violation

    monitor = ComplianceMonitor([rule], version="retail/2")
    result = monitor.evaluate(runs)[0]
    assert result.n == 40 and result.metric_value == pytest.approx(17 / 40) and result.drift_status is True
    stored_result = monitor.evaluate(stored)[0]
    assert stored_result.status == NOT_EVALUABLE and stored_result.n == 0
    assert stored_result.metric_value is None and stored_result.drift_status is None
    assert "40 runs not evaluable" in stored_result.method


# ----------------------------------------------------------------------------------------------- name-only traces
def test_name_only_traces_report_rules_needing_more_as_not_evaluable():
    monitor = ComplianceMonitor(
        load_rules(
            OLD_RULES["rules"][:1]
            + [
                {
                    "id": "P2",
                    "template": "alternate_precedence",
                    "severity": "critical",
                    "events": ["@user_confirmation", "create_file"],
                }
            ]
        ),
        version="rules/3",
    )
    stored = trace_from("create_file>authenticate")
    assert monitor.not_evaluable(stored) == ["P2"]
    with pytest.warns(NotEvaluableWarning, match="P2"):
        assert monitor.violations(stored) == ["auth-before-file"]
    with pytest.warns(NotEvaluableWarning):
        assert [a["rule_id"] for a in monitor.alerts(stored)] == ["auth-before-file"]
    assert monitor.run_record(stored) == {
        "violated_rules": "auth-before-file",
        "critical_violations": 1,
        "max_severity": "critical",
        "rules_version": "rules/3",
        "not_evaluable_rules": "P2",
    }
    assert monitor.score(stored) == 0.0
    assert monitor.not_evaluable(trace_from("none")) == ["P2"]  # an empty tool_sequence says nothing about events
    assert monitor.not_evaluable([]) == []  # an empty rich trace: nothing happened
    rich = [tool("authenticate"), event("user_confirmation"), tool("create_file")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert monitor.violations(rich) == [] and "not_evaluable_rules" not in monitor.run_record(rich)
    assert math.isnan(ComplianceMonitor([monitor.rules[1]]).score(stored))


def test_evaluate_counts_only_the_runs_a_rule_can_be_judged_on():
    rule = {"id": "P2", "template": "alternate_precedence", "events": ["@user_confirmation", "write"], "severity": "minor"}
    monitor = ComplianceMonitor(load_rules([rule]))
    good = [event("user_confirmation"), tool("write")]
    bad = [tool("write")]
    window = [good] * 8 + [bad] * 2 + [trace_from("write")] * 5
    result = monitor.evaluate(window)[0]
    assert result.n == 10 and result.metric_value == pytest.approx(0.8) and "5 runs not evaluable" in result.method
    relative = monitor.evaluate(window, reference=[good] * 7 + [bad] * 3 + [trace_from("write")])[0]
    assert relative.p_value is not None and relative.drift_status is False and relative.threshold == pytest.approx(0.7)
    # no evaluable reference run: absolute verdict
    fallback = monitor.evaluate(window, reference=[trace_from("write")])[0]
    assert fallback.p_value is None and fallback.drift_status is True


def test_a_bound_rule_still_matches_an_event_named_by_a_plain_atom():
    """Review: under bind, a plain atom no longer matched events, giving false violations"""
    trace = [{"name": "user_confirmation", "kind": "event"}, {"name": "cancel", "kind": "tool", "args": {"order_id": 1}}]
    plain = Rule(id="p", template="precedence", events=["user_confirmation", "cancel"])
    bound = Rule(id="b", template="precedence", events=["user_confirmation", "cancel"], bind=["order_id"])
    assert plain.verdict(trace) is True and bound.verdict(trace) is True
    assert bound.verdict([trace[1]]) is False  # no confirmation at all
