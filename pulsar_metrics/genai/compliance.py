"""Compliance of agent trajectories with behavioural rules, written in linear temporal logic on finite traces (LTLf).

A trajectory is the sequence of events of one run (the tools called, in order, e.g. "search_programs",
"authenticate", "create_file"). A rule is either a Declare template ("precedence: create_file only after
authenticate", "absence: delete_file", "allowed tools") or an LTLf formula over event names:

    !a   a & b   a | b   a -> b   X a (next, strong)   F a (eventually)   G a (always)   a U b (until)

A trace element is either a name (str, as rebuilt by `trace_from` from the `tool_sequence` field) or, in a rich trace,
a mapping {"name": str, "kind": "tool" | "event", "args": dict (tools), "status": "ok" | "error" | "timeout" |
"soft_failure" (tools)}; conversation events (e.g. "user_confirmation") sit in time order among the tool calls. An
atom matches an element by its name; atom predicates read the rest of a rich element:

    tool[order_id=W123]   tool[order_id]   tool[reason!=other]   tool[status=error]   @user_confirmation   *[status=error]

Rules are versioned data (a YAML file), so that the rules applied to a period are known exactly. Each run gets the
list of the rules it violates (`violations`), and the per-run alert of its severe violations (`alerts`, from each
rule's `severity`: info, minor, major, critical); each rule gets a compliance rate over the runs, with a Wilson
interval, judged either in absolute terms or against a reference period (`evaluate(..., reference=...)`). A rule that
needs arguments, statuses or events cannot be judged on a name-only trace: it is reported as not evaluable
(`not_evaluable`), never as a silent violation or pass.
"""

import json
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from scipy import stats

from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..metrics.base import MetricResults
from ..multiple_testing import benjamini_hochberg

SEVERITIES = ("info", "minor", "major", "critical")  # increasing
NOT_EVALUABLE = "not_evaluable"  # MetricResults.status of a rule no run of the window could be judged on


def severity_rank(severity: str) -> int:
    """Position of a severity in SEVERITIES (unknown severities rank as "major")"""
    severity = str(severity).lower()
    return SEVERITIES.index(severity) if severity in SEVERITIES else SEVERITIES.index("major")


class RuleNotEvaluable(error_msg):
    """Raised by `Rule.check` when the trace lacks what the rule reads (arguments, statuses, events)"""


class NotEvaluableWarning(UserWarning):
    """Some rules could not be judged on a trace (name-only trace, or elements without the arguments they read)"""


class NameTrace(list):
    """A name-only trace (list of tool names, as `trace_from` rebuilds it): rules needing more are not evaluable"""


# ----------------------------------------------------------------------------------------------- trace elements
_MISSING = object()
_NO_VALUE = object()  # binding value that no element carries


def _name(element) -> str:
    return element.get("name") if isinstance(element, Mapping) else str(element)


def _is_event(element) -> bool:
    return isinstance(element, Mapping) and element.get("kind") == "event"


def _text(value) -> str:
    """Values are compared as text: "W123" == "W123", 5 == "5", True == "true", None == "null" """
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)


def _name_only(trace) -> bool:
    """A NameTrace, or a non-empty trace without any mapping (an empty plain list is a run where nothing happened)"""
    if isinstance(trace, NameTrace):
        return True
    return len(trace) > 0 and not any(isinstance(e, Mapping) for e in trace)


def _and3(a, b):
    """Kleene conjunction: None is "unknown" (the trace lacks what the atom reads)"""
    if a is False or b is False:
        return False
    return None if a is None or b is None else True


def _or3(a, b):
    if a is True or b is True:
        return True
    return None if a is None or b is None else False


def _not3(a):
    return None if a is None else not a


@dataclass(frozen=True)
class Atom:
    """An atom with predicates: name ("*" for any tool), event (the "@" prefix), conditions (where, key, op, value)

    where is "args", "status" or "kind"; op is "has" (argument present), "=" or "!=", or "bind" (argument equal to a
    binding value, see Rule.bind)."""

    name: str
    event: bool = False
    conditions: tuple = ()

    def match(self, element):
        """True, False, or None when the element lacks the data a condition reads"""
        if self.event:
            if self.name != "*" and _name(element) != self.name:
                return False
            kind = element.get("kind", _MISSING) if isinstance(element, Mapping) else _MISSING
            if kind is _MISSING:
                return None
            if kind != "event":
                return False
        elif _is_event(element) or (self.name != "*" and _name(element) != self.name):
            return False
        value = True
        for condition in self.conditions:
            value = _and3(value, _condition(element, *condition))
            if value is False:
                return False
        return value


def _condition(element, where, key, op, expected):
    if where == "args":
        args = element.get("args", _MISSING) if isinstance(element, Mapping) else _MISSING
        if not isinstance(args, Mapping):
            return None
        if op == "has":
            return key in args
        if op == "bind":
            return None if key not in args else _text(args[key]) == expected
        actual = args.get(key, _MISSING)
    else:  # "status" or "kind": fields of the element itself
        if not isinstance(element, Mapping) or key not in element:
            return None
        actual = element[key]
    equal = actual is not _MISSING and _text(actual) == expected
    return equal if op == "=" else not equal


# ----------------------------------------------------------------------------------------------- LTLf formulas
_TOKEN = re.compile(r"""\s*(->|[()!&|]|@?(?:\w[\w.:-]*|\*)(?:\[(?:[^\]"']|"[^"]*"|'[^']*')*\])?)""")
_ATOM = re.compile(r"(@?)(\w[\w.:-]*|\*)(?:\[(.*)\])?", re.S)
_CONDITION = re.compile(r"""\s*(args\.)?([A-Za-z_][\w.-]*)\s*(?:(!=|=)\s*("[^"]*"|'[^']*'|[^,"'\s\]]+))?\s*(,|$)""")
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


def _atom(token: str):
    """("atom", name) for a plain name (matched by name, as before predicates existed), ("atom", name, Atom) else"""
    prefix, name, body = _ATOM.fullmatch(token).groups()
    if not prefix and body is None and name != "*":
        return ("atom", name)
    conditions, pos = [], 0
    while body is not None:
        match = _CONDITION.match(body, pos)
        if not match or match.end() == pos:
            raise error_msg(value=token, message="atom predicate: expected arg, arg=value, arg!=value or status=value")
        args_prefix, key, op, value = match.group(1, 2, 3, 4)
        where = "args" if args_prefix or key not in ("status", "kind") else key
        if op is None and where != "args":
            raise error_msg(value=token, message=f"atom predicate: {key} needs a value ({key}=...)")
        if value is not None and value[0] in "\"'":
            value = value[1:-1]
        conditions.append((where, key, op or "has", value))
        pos = match.end()
        if match.group(5) == "":
            break
    return ("atom", name, Atom(name, bool(prefix), tuple(conditions)))


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
        return _atom(token)


def parse(formula: str):
    return _Parser(formula).parse()


def holds(node, trace: Sequence, i: int = 0, memo: Optional[dict] = None) -> Optional[bool]:
    """Finite-trace semantics at position i (X is the strong next: false at the last position)

    Elements are names or rich mappings (matched by name). Three-valued (Kleene): None when the verdict depends on
    data the trace lacks (an atom predicate on an element without arguments or status); True or False otherwise."""
    memo = {} if memo is None else memo
    key = (id(node), i)
    if key in memo:
        return memo[key]
    n, op = len(trace), node[0]
    if op == "atom":
        if i >= n:
            value = False
        elif len(node) == 2:
            value = _name(trace[i]) == node[1]
        else:
            value = node[2].match(trace[i])
    elif op == "true":
        value = True
    elif op == "false":
        value = False
    elif op == "!":
        value = _not3(holds(node[1], trace, i, memo))
    elif op == "&":
        value = holds(node[1], trace, i, memo)
        if value is not False:
            value = _and3(value, holds(node[2], trace, i, memo))
    elif op == "|":
        value = holds(node[1], trace, i, memo)
        if value is not True:
            value = _or3(value, holds(node[2], trace, i, memo))
    elif op == "->":
        value = _not3(holds(node[1], trace, i, memo))
        if value is not True:
            value = _or3(value, holds(node[2], trace, i, memo))
    elif op == "X":
        value = i + 1 < n and holds(node[1], trace, i + 1, memo)
    elif op == "F":
        value = False
        for j in range(i, n):
            value = _or3(value, holds(node[1], trace, j, memo))
            if value is True:
                break
    elif op == "G":
        value = True
        for j in range(i, n):
            value = _and3(value, holds(node[1], trace, j, memo))
            if value is False:
                break
    elif op == "U":  # OR over j of (b at j AND a at every k < j)
        value, prefix = False, True
        for j in range(i, n):
            value = _or3(value, _and3(prefix, holds(node[2], trace, j, memo)))
            prefix = _and3(prefix, holds(node[1], trace, j, memo))
            if value is True or prefix is False:
                break
    else:  # pragma: no cover - the parser builds only the nodes above
        raise error_msg(value=op, message="unknown LTLf operator")
    memo[key] = value
    return value


def _atoms(node):
    if node[0] == "atom":
        yield node
    else:
        for child in node[1:]:
            yield from _atoms(child)


def _bind(node, keys, values):
    """The formula with every tool atom restricted to the elements whose arguments `keys` equal `values`"""
    if node[0] == "atom":
        atom = node[2] if len(node) == 3 else Atom(node[1])
        if atom.event:
            return node
        extra = tuple(("args", k, "bind", v) for k, v in zip(keys, values))
        return ("atom", node[1], Atom(atom.name, False, atom.conditions + extra))
    return (node[0],) + tuple(_bind(child, keys, values) for child in node[1:])


# ----------------------------------------------------------------------------------------------- rules
_WEAK_NEXT = "!X true | X"  # "(!X true | X phi)": phi at the next position, or the trace ends here
TEMPLATES = {
    "existence": lambda a: f"F {a}",
    "absence": lambda a: f"!F {a}",
    "init": lambda a: a,
    "precedence": lambda a, b: f"(!{b} U {a}) | G !{b}",  # b only after a
    "response": lambda a, b: f"G ({a} -> F {b})",  # every a is followed by b
    "chain_response": lambda a, b: f"G ({a} -> X {b})",  # every a is immediately followed by b
    "not_succession": lambda a, b: f"G ({a} -> !F {b})",  # no b after a
    # every b preceded by an a since the previous b (e.g. a user confirmation before each write)
    "alternate_precedence": lambda a, b: f"((!{b} U {a}) | G !{b}) & G ({b} -> ({_WEAK_NEXT} ((!{b} U {a}) | G !{b})))",
    "alternate_response": lambda a, b: f"G ({a} -> X (!{a} U {b}))",  # every a followed by b before the next a
}


def _event_text(event) -> str:
    """A template event: an atom (str), or a list of atoms meaning any of them ("(a | b | c)")"""
    if isinstance(event, (list, tuple)):
        return "(" + " | ".join(str(e) for e in event) + ")"
    return event


@dataclass
class Rule:
    """A behavioural rule: an LTLf formula, or a Declare template with its events, or the list of allowed tools

    `bind` (argument names, e.g. ["order_id"]) makes the rule hold separately for every value of these arguments,
    each tool atom being restricted to the elements carrying that value: precedence(get_order_details,
    cancel_pending_order) with bind [order_id] means each cancel of order v follows a lookup of order v. Event atoms
    ("@name") are not bound. An element matched by a bound atom that does not carry every bound argument makes the
    run not evaluable for the rule."""

    id: str
    description: str = ""
    formula: Optional[str] = None
    template: Optional[str] = None
    events: Sequence = field(default_factory=tuple)
    tools: Sequence[str] = field(default_factory=tuple)  # template "allowed_tools"
    severity: str = "major"
    bind: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self):
        if str(self.severity).lower() not in SEVERITIES:
            warnings.warn(f"rule {self.id}: unknown severity {self.severity!r} (known: {SEVERITIES}); ranked as major")
        self.bind = (self.bind,) if isinstance(self.bind, str) else tuple(self.bind or ())
        if self.template == "allowed_tools":
            if not self.tools:
                raise error_msg(value=self.id, message="allowed_tools needs a list of tools")
            if self.bind:
                raise error_msg(value=self.id, message="allowed_tools takes no bind")
            self._allowed = set(self.tools)
            self.formula = "G (" + " | ".join(self.tools) + ")"
            self._node, self._needs_rich, self._event_names, self._all_events = None, False, set(), False
            return
        if self.template is not None:
            if self.template not in TEMPLATES:
                raise error_msg(value=self.template, message=f"unknown template; known: {sorted(TEMPLATES)} + allowed_tools")
            try:
                self.formula = TEMPLATES[self.template](*[_event_text(e) for e in self.events])
            except TypeError as exc:
                raise error_msg(value=self.id, message=f"wrong number of events for {self.template}") from exc
        if not self.formula:
            raise error_msg(value=self.id, message="a rule needs a formula or a template")
        self._node = parse(self.formula)
        atoms = list(_atoms(self._node))
        self._needs_rich = bool(self.bind) or any(len(a) == 3 for a in atoms)
        # the events a rule does not name are invisible to it: a tool-only rule judges a rich trace as its tool sequence
        self._event_names = {a[1] for a in atoms if a[1] != "*"}
        self._all_events = any(len(a) == 3 and a[2].event and a[1] == "*" for a in atoms)
        self._bound_names = {a[1] for a in atoms if not (len(a) == 3 and a[2].event)}

    @property
    def needs_rich_trace(self) -> bool:
        """True when the rule reads arguments, statuses or events (atom predicates, "@" atoms, bind)"""
        return self._needs_rich

    def _view(self, trace: list) -> list:
        if self._all_events:
            return trace
        return [e for e in trace if not _is_event(e) or _name(e) in self._event_names]

    def verdict(self, trace: Sequence) -> Optional[bool]:
        """True (complies), False (violates), or None (not evaluable: the trace lacks what the rule reads)"""
        name_only = _name_only(trace)
        if name_only and self._needs_rich:
            return None
        trace = list(trace) if name_only else self._view(list(trace))
        if self.template == "allowed_tools":
            return all(_name(e) in self._allowed for e in trace if not _is_event(e))
        if not self.bind:
            return holds(self._node, trace)
        values, incomplete = set(), False
        for e in trace:
            if _is_event(e) or not ("*" in self._bound_names or _name(e) in self._bound_names):
                continue
            args = e.get("args") if isinstance(e, Mapping) else None
            if isinstance(args, Mapping) and all(k in args for k in self.bind):
                values.add(tuple(_text(args[k]) for k in self.bind))
            else:
                incomplete = True
        if incomplete or not values:  # an element without the bound arguments may carry any other value
            values.add((_NO_VALUE,) * len(self.bind))
        value = True
        for v in values:
            value = _and3(value, holds(_bind(self._node, self.bind, v), trace))
            if value is False:
                break
        return value

    def check(self, trace: Sequence) -> bool:
        """True when the run complies; raises RuleNotEvaluable when the trace lacks what the rule reads"""
        value = self.verdict(trace)
        if value is None:
            raise RuleNotEvaluable(
                value=self.id, message=f"rule {self.id} needs arguments, statuses or events this trace does not carry"
            )
        return value


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


def _as_trace(trace):
    return trace if isinstance(trace, NameTrace) else list(trace)


class ComplianceMonitor:
    """Check agent trajectories against rules; per run: the violated rules and the alert of the severe ones; per rule:
    the compliance rate, absolute or against a reference period

    Traces are lists of names or rich traces (mappings with name, kind, args, status). Rules that need a rich trace
    are not evaluable on a name-only one: `violations` and `alerts` leave them out with a NotEvaluableWarning,
    `run_record` lists them in `not_evaluable_rules`, `evaluate` counts only the runs they could be judged on."""

    accepts_rich_trace = True  # the trajectory collector passes rich traces (args, status, conversation events)

    def __init__(self, rules: Iterable, version: str = "unversioned"):
        self.rules = [r if isinstance(r, Rule) else Rule(**r) for r in rules]
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise error_msg(value=None, message="rule ids must be unique")
        self.version = version

    def _verdicts(self, trace: Sequence) -> list:
        trace = _as_trace(trace)
        return [(r, r.verdict(trace)) for r in self.rules]

    @staticmethod
    def _warn(verdicts: list) -> None:
        skipped = [r.id for r, v in verdicts if v is None]
        if skipped:
            warnings.warn(
                f"rules not evaluable on this trace (they need arguments, statuses or events): {skipped}",
                NotEvaluableWarning,
                stacklevel=3,
            )

    def not_evaluable(self, trace: Sequence) -> list:
        """Ids of the rules this trace cannot be judged on (name-only trace, or elements without the data they read)"""
        return [r.id for r, v in self._verdicts(trace) if v is None]

    def violations(self, trace: Sequence) -> list:
        """Ids of the violated rules; the rules not evaluable on the trace are left out, with a NotEvaluableWarning"""
        verdicts = self._verdicts(trace)
        self._warn(verdicts)
        return [r.id for r, v in verdicts if v is False]

    def alerts(self, trace: Sequence, min_severity: str = "critical") -> list:
        """Per-run alert: the violated rules of severity `min_severity` or above (default: critical), most severe first

        One run violating a critical rule (e.g. a file created before authentication) is an incident in itself; it
        must not wait for a window's compliance rate. Each alert is a dict: rule_id, severity, description, formula,
        rules_version. An empty list means no alert (rules not evaluable on the trace: NotEvaluableWarning)."""
        if str(min_severity).lower() not in SEVERITIES:
            raise error_msg(value=min_severity, message=f"min_severity must be one of {SEVERITIES}")
        floor = severity_rank(min_severity)
        verdicts = [(r, v) for r, v in self._verdicts(trace) if severity_rank(r.severity) >= floor]
        self._warn(verdicts)
        violated = [r for r, v in verdicts if v is False]
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

    def run_record(self, trace: Sequence) -> dict:
        """Fields to store with the run (e.g. in the agent_run record): violated_rules (comma-separated ids),
        critical_violations (count), max_severity (None when no violation), rules_version; plus not_evaluable_rules
        (comma-separated ids) only when some rules could not be judged on the trace"""
        verdicts = self._verdicts(trace)
        violated = [r for r, v in verdicts if v is False]
        worst = max(violated, key=lambda r: severity_rank(r.severity), default=None)
        record = {
            "violated_rules": ",".join(r.id for r in violated),
            "critical_violations": sum(severity_rank(r.severity) == SEVERITIES.index("critical") for r in violated),
            "max_severity": None if worst is None else str(worst.severity).lower(),
            "rules_version": self.version,
        }
        skipped = [r.id for r, v in verdicts if v is None]
        if skipped:
            record["not_evaluable_rules"] = ",".join(skipped)
        return record

    def score(self, trace: Sequence) -> float:
        """Share of the rules a run complies with, among the rules it can be judged on (NaN when none)"""
        if not self.rules:
            return 1.0
        judged = [v for _, v in self._verdicts(trace) if v is not None]
        return sum(judged) / len(judged) if judged else float("nan")

    def evaluate(
        self,
        traces: Sequence[Sequence],
        min_compliance: float = 1.0,
        confidence: float = 0.95,
        reference: Optional[Sequence[Sequence]] = None,
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

        Each rule is judged on the runs it is evaluable on (`n`), the method says how many were not. A rule evaluable
        on no run gets metric_value and drift_status None and status "not_evaluable"; one evaluable on no reference
        run keeps the absolute verdict.
        """
        traces = [_as_trace(t) for t in traces]
        if not traces:
            raise error_msg(value=None, message="no trajectory to evaluate")
        if reference is not None:
            reference = [_as_trace(t) for t in reference]
            if not reference:
                raise error_msg(value=None, message="the reference holds no trajectory")
        floor = None if absolute_from is None else severity_rank(absolute_from)
        rows = []
        for rule in self.rules:
            judged = [v for v in (rule.verdict(t) for t in traces) if v is not None]
            n, ok = len(judged), sum(judged)
            relative = reference is not None and n > 0 and (floor is None or severity_rank(rule.severity) < floor)
            p_value, ref_ok, ref_n = None, None, None
            if relative:
                ref_judged = [v for v in (rule.verdict(t) for t in reference) if v is not None]
                ref_n, ref_ok = len(ref_judged), sum(ref_judged)
                relative = ref_n > 0
            if relative:
                table = [[n - ok, ok], [ref_n - ref_ok, ref_ok]]  # violations first: "greater" = more now
                p_value = float(stats.fisher_exact(table, alternative="greater").pvalue)
            rows.append((rule, n, ok, relative, p_value, ref_ok, ref_n))
        tested = [r[4] for r in rows if r[3]]
        rejected, adjusted = benjamini_hochberg(tested, alpha) if tested else ([], [])
        verdicts = iter(zip(rejected, adjusted))

        out = []
        for rule, n, ok, relative, p_value, ref_ok, ref_n in rows:
            skipped = len(traces) - n
            note = f"; {skipped} runs not evaluable (they lack the arguments, statuses or events the rule reads)"
            if n == 0:
                out.append(
                    MetricResults(
                        metric_type="agents",
                        metric_name="rule_compliance",
                        feature_name=rule.id,
                        n=0,
                        threshold=min_compliance,
                        method=f"{rule.formula} [{rule.severity}; rules {self.version}]" + note,
                        test_type="fixed",
                        status=NOT_EVALUABLE,
                    )
                )
                continue
            ci = stats.binomtest(ok, n).proportion_ci(confidence_level=confidence, method="wilson")
            rate = ok / n
            base = f"{rule.formula} [{rule.severity}; rules {self.version}]; {n - ok} violating runs; Wilson"
            if relative:
                drift, p_adj = next(verdicts)
                ref_rate = ref_ok / ref_n
                status, threshold, adjusted_p = bool(drift), ref_rate, float(p_adj)
                method = (
                    f"{base}; against the reference compliance {ref_rate:.3f} ({ref_n} runs): "
                    f"Fisher one-sided p = {p_value:.3g}, BH-adjusted {p_adj:.3g} over {len(tested)} rules"
                )
            else:
                status, threshold, adjusted_p = rate < min_compliance, min_compliance, None
                method = base + ("; absolute verdict (severity)" if reference is not None else "")
            if skipped:
                method += note
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
    """Name-only trajectory (a NameTrace, a list) from the `tool_sequence` field of the agent_run records
    ("search_programs>check_eligibility"): rules that need arguments, statuses or events are not evaluable on it"""
    if not tool_sequence or tool_sequence == "none":
        return NameTrace()
    return NameTrace(tool_sequence.split(">"))
