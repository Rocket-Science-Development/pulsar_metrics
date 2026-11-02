"""Compliance of agent trajectories with behavioural rules, written in linear temporal logic on finite traces (LTLf).

A trajectory is the sequence of events of one run (the tools called, in order, e.g. "search_programs",
"authenticate", "create_file"). A rule is either a Declare template ("precedence: create_file only after
authenticate", "absence: delete_file", "allowed tools") or an LTLf formula over event names:

    !a   a & b   a | b   a -> b   X a (next, strong)   F a (eventually)   G a (always)   a U b (until)

Rules are versioned data (a YAML file), so that the rules applied to a period are known exactly. Each run gets the
list of the rules it violates (`violations`), and the per-run alert of its severe violations (`alerts`, from each
rule's `severity`: info, minor, major, critical); each rule gets a compliance rate over the runs, with a Wilson
interval, judged either in absolute terms or against a reference period (`evaluate(..., reference=...)`).
"""

import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from scipy import stats

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults
from ..multiple_testing import benjamini_hochberg

SEVERITIES = ("info", "minor", "major", "critical")  # increasing


def severity_rank(severity: str) -> int:
    """Position of a severity in SEVERITIES (unknown severities rank as "major")"""
    severity = str(severity).lower()
    return SEVERITIES.index(severity) if severity in SEVERITIES else SEVERITIES.index("major")


# ----------------------------------------------------------------------------------------------- LTLf formulas
_TOKEN = re.compile(r"\s*(->|[()!&|]|\w[\w.:-]*)")
_UNARY = {"!", "X", "F", "G"}


def _tokenize(text: str) -> list:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            raise error_msg(value=text[pos:], message="unexpected character in the LTLf formula")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


class _Parser:
    """formula := or ('->' formula)? ; or := and ('|' and)* ; and := until ('&' until)* ; until := unary ('U' unary)?"""

    def __init__(self, text: str):
        self.tokens, self.i = _tokenize(text), 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise error_msg(value=token, message=f"LTLf formula: expected {expected or 'a term'}")
        self.i += 1
        return token

    def parse(self):
        node = self.implication()
        if self.peek() is not None:
            raise error_msg(value=self.peek(), message="LTLf formula: unexpected token")
        return node

    def implication(self):
        left = self.disjunction()
        if self.peek() == "->":
            self.take()
            return ("->", left, self.implication())
        return left

    def disjunction(self):
        node = self.conjunction()
        while self.peek() == "|":
            self.take()
            node = ("|", node, self.conjunction())
        return node

    def conjunction(self):
        node = self.until()
        while self.peek() == "&":
            self.take()
            node = ("&", node, self.until())
        return node

    def until(self):
        node = self.unary()
        if self.peek() == "U":
            self.take()
            return ("U", node, self.unary())
        return node

    def unary(self):
        token = self.take()
        if token in _UNARY:
            return (token, self.unary())
        if token == "(":
            node = self.implication()
            self.take(")")
            return node
        if token in ("true", "false"):
            return (token,)
        if token in ("->", "|", "&", ")", "U"):
            raise error_msg(value=token, message="LTLf formula: operator without operand")
        return ("atom", token)


def parse(formula: str):
    return _Parser(formula).parse()


def holds(node, trace: Sequence[str], i: int = 0, memo: Optional[dict] = None) -> bool:
    """Finite-trace semantics at position i (X is the strong next: false at the last position)"""
    memo = {} if memo is None else memo
    key = (id(node), i)
    if key in memo:
        return memo[key]
    n, op = len(trace), node[0]
    if op == "atom":
        value = i < n and trace[i] == node[1]
    elif op == "true":
        value = True
    elif op == "false":
        value = False
    elif op == "!":
        value = not holds(node[1], trace, i, memo)
    elif op == "&":
        value = holds(node[1], trace, i, memo) and holds(node[2], trace, i, memo)
    elif op == "|":
        value = holds(node[1], trace, i, memo) or holds(node[2], trace, i, memo)
    elif op == "->":
        value = (not holds(node[1], trace, i, memo)) or holds(node[2], trace, i, memo)
    elif op == "X":
        value = i + 1 < n and holds(node[1], trace, i + 1, memo)
    elif op == "F":
        value = any(holds(node[1], trace, j, memo) for j in range(i, n))
    elif op == "G":
        value = all(holds(node[1], trace, j, memo) for j in range(i, n))
    elif op == "U":
        value = False
        for j in range(i, n):
            if holds(node[2], trace, j, memo):
                value = True
                break
            if not holds(node[1], trace, j, memo):
                break
    else:  # pragma: no cover - the parser builds only the nodes above
        raise error_msg(value=op, message="unknown LTLf operator")
    memo[key] = value
    return value


# ----------------------------------------------------------------------------------------------- rules
TEMPLATES = {
    "existence": lambda a: f"F {a}",
    "absence": lambda a: f"!F {a}",
    "init": lambda a: a,
    "precedence": lambda a, b: f"(!{b} U {a}) | G !{b}",  # b only after a
    "response": lambda a, b: f"G ({a} -> F {b})",  # every a is followed by b
    "chain_response": lambda a, b: f"G ({a} -> X {b})",  # every a is immediately followed by b
    "not_succession": lambda a, b: f"G ({a} -> !F {b})",  # no b after a
}


@dataclass
class Rule:
    """A behavioural rule: an LTLf formula, or a Declare template with its events, or the list of allowed tools"""

    id: str
    description: str = ""
    formula: Optional[str] = None
    template: Optional[str] = None
    events: Sequence[str] = field(default_factory=tuple)
    tools: Sequence[str] = field(default_factory=tuple)  # template "allowed_tools"
    severity: str = "major"

    def __post_init__(self):
        if str(self.severity).lower() not in SEVERITIES:
            warnings.warn(f"rule {self.id}: unknown severity {self.severity!r} (known: {SEVERITIES}); ranked as major")
        if self.template == "allowed_tools":
            if not self.tools:
                raise error_msg(value=self.id, message="allowed_tools needs a list of tools")
            self._allowed = set(self.tools)
            self.formula = "G (" + " | ".join(self.tools) + ")"
            self._node = None
            return
        if self.template is not None:
            if self.template not in TEMPLATES:
                raise error_msg(value=self.template, message=f"unknown template; known: {sorted(TEMPLATES)} + allowed_tools")
            try:
                self.formula = TEMPLATES[self.template](*self.events)
            except TypeError as exc:
                raise error_msg(value=self.id, message=f"wrong number of events for {self.template}") from exc
        if not self.formula:
            raise error_msg(value=self.id, message="a rule needs a formula or a template")
        self._node = parse(self.formula)

    def check(self, trace: Sequence[str]) -> bool:
        if self.template == "allowed_tools":
            return all(event in self._allowed for event in trace)
        return holds(self._node, list(trace))


def load_rules(source) -> list:
    """Rules from a list of dicts, a dict with a "rules" key, or a YAML file path (needs PyYAML)"""
    if isinstance(source, str):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise error_msg(value=None, message="reading a YAML file needs PyYAML") from exc
        with open(source, encoding="utf-8") as f:
            source = yaml.safe_load(f)
    if isinstance(source, dict):
        source = source.get("rules", [])
    return [Rule(**{k: v for k, v in r.items() if k in Rule.__dataclass_fields__}) for r in source]


class ComplianceMonitor:
    """Check agent trajectories against rules; per run: the violated rules and the alert of the severe ones; per rule:
    the compliance rate, absolute or against a reference period"""

    def __init__(self, rules: Iterable, version: str = "unversioned"):
        self.rules = [r if isinstance(r, Rule) else Rule(**r) for r in rules]
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise error_msg(value=None, message="rule ids must be unique")
        self.version = version

    def violations(self, trace: Sequence[str]) -> list:
        return [r.id for r in self.rules if not r.check(trace)]

    def alerts(self, trace: Sequence[str], min_severity: str = "critical") -> list:
        """Per-run alert: the violated rules of severity `min_severity` or above (default: critical), most severe first

        One run violating a critical rule (e.g. a file created before authentication) is an incident in itself; it
        must not wait for a window's compliance rate. Each alert is a dict: rule_id, severity, description, formula,
        rules_version. An empty list means no alert."""
        if str(min_severity).lower() not in SEVERITIES:
            raise error_msg(value=min_severity, message=f"min_severity must be one of {SEVERITIES}")
        floor = severity_rank(min_severity)
        violated = [r for r in self.rules if severity_rank(r.severity) >= floor and not r.check(trace)]
        violated.sort(key=lambda r: -severity_rank(r.severity))
        return [
            {
                "rule_id": r.id,
                "severity": str(r.severity).lower(),
                "description": r.description,
                "formula": r.formula,
                "rules_version": self.version,
            }
            for r in violated
        ]

    def run_record(self, trace: Sequence[str]) -> dict:
        """Fields to store with the run (e.g. in the agent_run record): violated_rules (comma-separated ids),
        critical_violations (count), max_severity (None when no violation), rules_version"""
        violated = [r for r in self.rules if not r.check(trace)]
        worst = max(violated, key=lambda r: severity_rank(r.severity), default=None)
        return {
            "violated_rules": ",".join(r.id for r in violated),
            "critical_violations": sum(severity_rank(r.severity) == SEVERITIES.index("critical") for r in violated),
            "max_severity": None if worst is None else str(worst.severity).lower(),
            "rules_version": self.version,
        }

    def score(self, trace: Sequence[str]) -> float:
        """Share of the rules a run complies with"""
        return 1 - len(self.violations(trace)) / len(self.rules) if self.rules else 1.0

    def evaluate(
        self,
        traces: Sequence[Sequence[str]],
        min_compliance: float = 1.0,
        confidence: float = 0.95,
        reference: Optional[Sequence[Sequence[str]]] = None,
        alpha: float = 0.05,
        absolute_from: Optional[str] = "critical",
    ) -> list:
        """One MetricResults per rule (metric_type "agents"), compliance rate with a Wilson interval

        Without `reference`: drift_status True when some runs violate the rule (compliance below `min_compliance`,
        default: every run must comply).

        With `reference` (the trajectories of a reference period): drift_status True when the rule is violated
        *more often than in the reference* (one-sided Fisher exact test, Benjamini-Hochberg at `alpha` over the rules
        tested this way), so that a rule violated at a stable rate no longer alerts in every window. Rules of severity
        `absolute_from` or above (default: critical) keep the absolute verdict; None tests every rule against the
        reference. `p_value` holds the BH-adjusted p-value of the rules tested against the reference, `threshold`
        their reference compliance rate.
        """
        traces = [list(t) for t in traces]
        if not traces:
            raise error_msg(value=None, message="no trajectory to evaluate")
        if reference is not None:
            reference = [list(t) for t in reference]
            if not reference:
                raise error_msg(value=None, message="the reference holds no trajectory")
        floor = None if absolute_from is None else severity_rank(absolute_from)
        n = len(traces)
        rows = []
        for rule in self.rules:
            ok = sum(rule.check(t) for t in traces)
            relative = reference is not None and (floor is None or severity_rank(rule.severity) < floor)
            p_value, ref_ok = None, None
            if relative:
                ref_ok = sum(rule.check(t) for t in reference)
                table = [[n - ok, ok], [len(reference) - ref_ok, ref_ok]]  # violations first: "greater" = more now
                p_value = float(stats.fisher_exact(table, alternative="greater").pvalue)
            rows.append((rule, ok, relative, p_value, ref_ok))
        tested = [r[3] for r in rows if r[2]]
        rejected, adjusted = benjamini_hochberg(tested, alpha) if tested else ([], [])
        verdicts = iter(zip(rejected, adjusted))

        out = []
        for rule, ok, relative, p_value, ref_ok in rows:
            ci = stats.binomtest(ok, n).proportion_ci(confidence_level=confidence, method="wilson")
            rate = ok / n
            base = f"{rule.formula} [{rule.severity}; rules {self.version}]; {n - ok} violating runs; Wilson"
            if relative:
                drift, p_adj = next(verdicts)
                ref_rate = ref_ok / len(reference)
                status, threshold, adjusted_p = bool(drift), ref_rate, float(p_adj)
                method = (
                    f"{base}; against the reference compliance {ref_rate:.3f} ({len(reference)} runs): "
                    f"Fisher one-sided p = {p_value:.3g}, BH-adjusted {p_adj:.3g} over {len(tested)} rules"
                )
            else:
                status, threshold, adjusted_p = rate < min_compliance, min_compliance, None
                method = base + ("; absolute verdict (severity)" if reference is not None else "")
            out.append(
                MetricResults(
                    metric_type="agents",
                    metric_name="rule_compliance",
                    feature_name=rule.id,
                    metric_value=rate,
                    ci_low=float(ci.low),
                    ci_high=float(ci.high),
                    n=n,
                    threshold=threshold,
                    drift_status=status,
                    method=method,
                    test_type="fixed",
                    p_value=adjusted_p,
                )
            )
        return out


def trace_from(tool_sequence: str) -> list:
    """Trajectory from the `tool_sequence` field of the agent_run records ("search_programs>check_eligibility")"""
    if not tool_sequence or tool_sequence == "none":
        return []
    return tool_sequence.split(">")
