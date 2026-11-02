import numpy as np
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.genai import (
    ComplianceMonitor,
    Rule,
    embedding_drift,
    load_rules,
    mmd_test,
    model_equality,
    model_equality_test,
    trace_from,
)
from pulsar_metrics.genai.compliance import holds, parse

# ----------------------------------------------------------------------------------------------- MMD


def embeddings(n, shift=0.0, seed=0, d=64):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, d))
    x[:, :3] += shift  # a change of topic moves a few directions
    return x


def test_mmd_holds_its_level_under_the_null():
    p_values = [
        mmd_test(embeddings(80, seed=s), embeddings(80, seed=100 + s), n_permutations=200, seed=s).p_value for s in range(40)
    ]
    assert np.mean(np.array(p_values) < 0.05) <= 0.12  # 40 runs: 5 % nominal, sampling noise allowed


def test_mmd_detects_a_shift_of_content():
    result = mmd_test(embeddings(150, seed=1), embeddings(150, shift=0.8, seed=2), n_permutations=300)
    assert result.p_value < 0.01 and result.statistic > 0
    metric = embedding_drift(
        embeddings(150, seed=1), embeddings(150, shift=0.8, seed=2), feature_name="requests", n_permutations=200
    )
    assert metric.metric_name == "mmd" and metric.drift_status is True and metric.p_value < 0.05


def test_mmd_input_checks():
    with pytest.raises(error_msg):
        mmd_test(embeddings(10), embeddings(10, d=32))
    with pytest.raises(error_msg):
        mmd_test(embeddings(3), embeddings(10))


# ----------------------------------------------------------------------------------------------- model equality

WORDS = "le programme aide au loyer exige un revenu inférieur à quarante mille dollars par année pour les demandeurs".split()


def answers(prompts, per_prompt, style, seed):
    """Synthetic answers: each 'model' has its own way of phrasing (word choice probabilities)"""
    rng = np.random.default_rng(seed)
    out = {}
    for k, p in enumerate(prompts):
        base = np.random.default_rng(k).permutation(WORDS)  # what the prompt calls for
        out[p] = []
        for _ in range(per_prompt):
            words = [w if rng.random() > style else rng.choice(WORDS) for w in base]
            out[p].append(" ".join(words))
    return out


PROMPTS = [f"question {i}" for i in range(15)]


def test_same_model_is_not_flagged():
    flags = [
        model_equality_test(answers(PROMPTS, 6, 0.3, s), answers(PROMPTS, 6, 0.3, 50 + s), n_permutations=200, seed=s).p_value
        < 0.05
        for s in range(20)
    ]
    assert sum(flags) <= 3


def test_another_model_is_detected():
    result = model_equality(answers(PROMPTS, 6, 0.1, 1), answers(PROMPTS, 6, 0.5, 2), model="llama3.2:3b", n_permutations=300)
    assert result.drift_status is True and result.metric_name == "model_equality" and result.p_value < 0.01


def test_model_equality_needs_shared_prompts():
    with pytest.raises(error_msg):
        model_equality_test({"a": ["x", "y"]}, {"b": ["x", "y"]})


# ----------------------------------------------------------------------------------------------- LTLf compliance


@pytest.mark.parametrize(
    "formula, trace, expected",
    [
        ("F b", ["a", "b"], True),
        ("F b", ["a"], False),
        ("G a", [], True),
        ("F a", [], False),
        ("X b", ["a", "b"], True),
        ("X b", ["a"], False),  # strong next at the last position
        ("a U b", ["a", "a", "b"], True),
        ("a U b", ["a", "c", "b"], False),
        ("G (a -> F b)", ["a", "c", "b", "a"], False),
        ("G (a -> F b)", ["a", "c", "b"], True),
        ("!(a & b) | c", ["c"], True),
    ],
)
def test_ltlf_semantics(formula, trace, expected):
    assert holds(parse(formula), trace) is expected


def test_ltlf_parse_errors():
    for bad in ("a &", "(a | b", "U b", "a ? b"):
        with pytest.raises(error_msg):
            parse(bad)


RULES = {
    "rules": [
        {
            "id": "auth-before-file",
            "template": "precedence",
            "events": ["authenticate", "create_file"],
            "description": "a file is opened only after authentication",
            "severity": "critical",
        },
        {
            "id": "in-scope",
            "template": "allowed_tools",
            "tools": ["search_programs", "check_eligibility", "compute_amount", "authenticate", "create_file"],
        },
        {"id": "no-repeat-file", "formula": "G (create_file -> !X create_file)"},
    ]
}


def test_declare_templates_and_violations():
    monitor = ComplianceMonitor(load_rules(RULES), version="rules-v1")
    assert monitor.violations(["search_programs", "authenticate", "create_file"]) == []
    assert monitor.violations(["create_file", "authenticate"]) == ["auth-before-file"]
    assert monitor.violations(["delete_file"]) == ["in-scope"]
    assert monitor.violations(["authenticate", "create_file", "create_file"]) == ["no-repeat-file"]
    assert monitor.violations([]) == []
    assert monitor.score(["create_file", "delete_file"]) == pytest.approx(1 / 3)


def test_compliance_rates_by_rule():
    monitor = ComplianceMonitor(load_rules(RULES), version="rules-v1")
    traces = [["authenticate", "create_file"]] * 18 + [["create_file"]] * 2
    results = {r.feature_name: r for r in monitor.evaluate(traces)}
    auth = results["auth-before-file"]
    assert auth.metric_value == pytest.approx(0.9) and auth.drift_status is True and auth.n == 20
    assert auth.ci_low < 0.9 < auth.ci_high and "2 violating runs" in auth.method and "rules-v1" in auth.method
    assert results["in-scope"].drift_status is False


def test_rule_definition_errors_and_traces():
    with pytest.raises(error_msg):
        Rule(id="x", template="precedence", events=["a"])
    with pytest.raises(error_msg):
        Rule(id="x", template="nonexistent", events=["a"])
    with pytest.raises(error_msg):
        ComplianceMonitor([Rule(id="x", formula="F a"), Rule(id="x", formula="F b")])
    assert trace_from("search_programs>check_eligibility") == ["search_programs", "check_eligibility"]
    assert trace_from("none") == []
