"""Recommended drift trigger of a RAG agent: three per-run signals, one Bonferroni decision, confirmed over 2 windows.

A governance layer re-evaluates a compliance control when Pulsar reports a measured drift of the agent: every false
alarm flips a control and sends a reviewer a bogus alert, and a late alert leaves a degraded agent in service.
`RAGDriftTrigger` packages the configuration measured on the replay of the night-lab campaign LR-03 (1,800 runs of a
RAG agent on the PIAF corpus, llama3.2:3b, BM25 retriever; pulsar_demo, `lab/replay/declencheur_derive.py`, trigger
"score"):

- three signals per run, each tested between the current window and the reference, one-sided in the direction of a
  degradation: the faithfulness of the answer to the passages (Mann-Whitney, window lower), the share of empty
  retrievals (exact Fisher test, window higher, `rate_increase`) and the retriever's score of the top passage
  (Mann-Whitney, window lower);
- one decision per window: the smallest p-value times the number of signals (Bonferroni), below alpha 0.05;
- confirmation over consecutive windows: `DriftPersistence(windows=2)`.

Measured on the demo agent (windows of 100 runs against a reference of 300, one window per hour): 0.05 % [0.01;
0.28] false alarms per window at k = 2, about 0.08 per week [0.02; 0.47]; a loss of 40 % of the corpus is confirmed
after 2 windows in median (90th percentile 5), 60 % and more after 2. A 10 % loss is not detected, 20 % after 11
windows in median. These figures hold for that agent and that kind of degradation (passages removed): measure the
false alarm rate on a reference period of your own traffic before relying on them.

The rates of abstentions and of answers without support (`det_abstention`, `det_unsupported_answer`) are governance
indicators, tested on their own with `rate_increase`: added to the trigger they cost power (a Bonferroni factor of 4
or 5) without shortening the delay (pulsar_demo, `lab/rapports/rapport_signal_abstention.md`).
"""

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from ..analyzers.persistence import DriftPersistence, PersistentDrift
from ..evidence import Evidence, evidence_from_result
from ..exceptions import CustomExceptionPulsarMetric as error_msg
from ..exceptions import MetricEvaluationError
from ..metrics.base import MetricResults
from ..metrics.drift import RATE_METRICS, DriftTestMetric
from ..metrics.enums import DriftTestMetricsFuncs, MetricsType

TRIGGER_METRIC = "rag_drift_trigger"
# status of a signal (or of the whole trigger) that cannot be evaluated; reliability reason codes of the evidence
MISSING_SIGNAL, FEW_VALUES, NO_SIGNAL = "missing_signal", "few_values", "no_signal"
MIN_WINDOW_RUNS = 20  # below, the window carries too few affected runs (night lab, register of AI, windows of 8)

Records = Union[pd.DataFrame, Iterable[Mapping]]


@dataclass(frozen=True)
class TriggerSignal:
    """One per-run signal of the trigger and its test against the reference

    Parameters
    ----------
    name : str
        Name of the signal in the results (e.g. "faithfulness")
    column : str
        Column of the run records holding it (e.g. "det_faithfulness", as written by pulsar-data-collection)
    test : str, optional
        A DriftTestMetric test (default "manwu"); "rate_increase" for a 0 / 1 indicator (one-sided by construction)
    alternative : str, optional
        Passed to the test (scipy convention, the window first): "less" alerts when the window is lower, "greater"
        when it is higher; None (default) keeps the test's own default (two-sided for manwu). Not allowed with
        rate_increase
    min_values : int, optional
        Fewest non-missing values in the window and in the reference (default 5, as in the replay for the rank tests);
        below, the signal is not evaluable (status "few_values")
    """

    name: str
    column: str
    test: str = "manwu"
    alternative: Optional[str] = None
    min_values: int = 5

    def __post_init__(self):
        if self.test not in DriftTestMetricsFuncs._member_names_:
            raise error_msg(value=self.test, message=f"test must be one of {DriftTestMetricsFuncs._member_names_}")
        if self.test in RATE_METRICS and self.alternative is not None:
            raise error_msg(value=self.alternative, message=f"{self.test} is one-sided (window higher): no alternative")
        if not isinstance(self.min_values, int) or isinstance(self.min_values, bool) or self.min_values < 1:
            raise error_msg(value=self.min_values, message="min_values must be a positive integer")


# The signals of the replay: faithfulness (as LR-03), empty retrieval (empty search or no search; "empty_retrieval"),
# retriever score of the top passage (BM25; "relevance_bm25" in the replay, `det_retrieval_top` in the records)
RAG_TRIGGER_SIGNALS = (
    TriggerSignal("faithfulness", "det_faithfulness", "manwu", "less", 5),
    TriggerSignal("empty_retrieval", "det_retrieval_empty", "rate_increase", None, 1),
    TriggerSignal("retrieval_top", "det_retrieval_top", "manwu", "less", 5),
)

# Defaults of RAGDriftTrigger and the figures measured with them (LR-03 replay, alpha 0.05, k = 2)
RECOMMENDED_RAG_TRIGGER = {
    "signals": tuple(s.name for s in RAG_TRIGGER_SIGNALS),
    "alpha": 0.05,
    "correction": "bonferroni",
    "persistence": 2,
    "window_runs": 100,
    "reference_runs": 300,
    "min_window_runs": MIN_WINDOW_RUNS,
    "measured": {
        "source": "pulsar_demo lab/replay/declencheur_derive.py, LR-03 (1,800 runs, PIAF corpus, llama3.2:3b, BM25)",
        "false_alarms_per_window": 0.0005,
        "false_alarms_per_window_ci": (0.0001, 0.0028),
        "false_alarms_per_week": 0.08,  # one window of 100 runs per hour
        "false_alarms_per_week_ci": (0.02, 0.47),
        "power_per_window_40pct_loss": 0.79,
        "median_delay_windows_40pct_loss": 2,
        "p90_delay_windows_40pct_loss": 5,
    },
}


def _frame(records: Records, what: str) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    try:
        return pd.DataFrame(list(records))
    except Exception as e:  # e.g. not an iterable of mappings
        raise error_msg(value=what, message=f"{what} must be a DataFrame or an iterable of run records: {e}") from e


def _resolve_signals(signals, columns: Optional[Mapping[str, str]]) -> tuple:
    defaults = {s.name: s for s in RAG_TRIGGER_SIGNALS}
    out = []
    for s in RAG_TRIGGER_SIGNALS if signals is None else signals:
        if isinstance(s, str):
            if s not in defaults:
                raise error_msg(value=s, message=f"unknown signal: one of {sorted(defaults)}, or a TriggerSignal")
            s = defaults[s]
        if not isinstance(s, TriggerSignal):
            raise error_msg(value=s, message="signals must be names of the default signals or TriggerSignal objects")
        out.append(s)
    if not out:
        raise error_msg(value=signals, message="the trigger needs at least one signal")
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise error_msg(value=names, message="signal names must be unique")
    unknown = set(columns or {}) - set(names)
    if unknown:
        raise error_msg(value=sorted(unknown), message="columns names signals the trigger does not have")
    return tuple(TriggerSignal(s.name, (columns or {}).get(s.name, s.column), s.test, s.alternative, s.min_values) for s in out)


def _direction(signal: TriggerSignal) -> str:
    if signal.test in RATE_METRICS:
        return "window higher"
    return {"less": "window lower", "greater": "window higher"}.get(signal.alternative, "two-sided")


@dataclass
class TriggerDecision:
    """Decision of the trigger on one window

    Attributes
    ----------
    signals : list of MetricResults
        One per signal (metric_name the test, feature_name the column): metric_value is the raw p-value, threshold
        the per-signal level alpha / m, drift_status whether m x p < alpha (None when not evaluable, see status)
    result : MetricResults
        The combined decision (metric_name "rag_drift_trigger"): metric_value and p_value min(1, m x smallest
        p-value), threshold alpha, drift_status the raw verdict of the window
    persistence : PersistentDrift
        The combined verdict after DriftPersistence (confirmed_drift, changed, streaks)
    """

    signals: list
    result: MetricResults
    persistence: PersistentDrift
    names: tuple = field(default=())
    settings: dict = field(default_factory=dict)

    @property
    def p_values(self) -> dict:
        """Raw p-value of each signal (None when not evaluable)"""
        return {name: r.metric_value for name, r in zip(self.names, self.signals)}

    @property
    def p_value(self) -> Optional[float]:
        """Bonferroni-adjusted p-value of the window: min(1, m x smallest p-value); None when no signal is evaluable"""
        return self.result.metric_value

    @property
    def drift(self) -> Optional[bool]:
        """Raw verdict of this window (None when no signal is evaluable)"""
        return self.result.drift_status

    @property
    def confirmed(self) -> bool:
        """Verdict after the persistence rule: the alert to act upon"""
        return self.persistence.confirmed_drift

    @property
    def changed(self) -> bool:
        """The confirmed verdict switched in this window (the event to send to a governance layer)"""
        return self.persistence.changed

    @property
    def not_evaluable(self) -> dict:
        """Signals that could not be evaluated -> reason code (missing_signal, few_values) or the error"""
        return {name: r.status or r.error for name, r in zip(self.names, self.signals) if r.metric_value is None and r.error}

    def to_pandas(self) -> pd.DataFrame:
        """One row per signal and one for the combined decision"""
        rows = [{"signal": n, **r.model_dump()} for n, r in zip(self.names, self.signals)]
        rows.append({"signal": TRIGGER_METRIC, **self.result.model_dump(), "confirmed_drift": self.confirmed})
        return pd.DataFrame(rows)

    def to_evidence(
        self,
        system_id: str,
        window: Optional[dict] = None,
        reference: Optional[dict] = None,
        include_signals: bool = True,
        **kwargs,
    ) -> list:
        """Evidence records ("pulsar.evidence/1") of the window: one per signal, then the combined decision

        Parameters
        ----------
        system_id : str
            External identifier of the monitored system
        window, reference : dict, optional
            Window and reference blocks (id, start, end); n_rows is filled from the frames when absent. Pass a window
            id (or start and end) so that the same finding sent again keeps its evidence_id
        include_signals : bool, optional
            Also export the per-signal records (default True); False keeps the combined decision only
        kwargs :
            model_id, model_version, run_name, produced_at, passed to evidence_from_result

        The combined record is in the genai family, decided on its p_value with threshold alpha; its reliability
        carries the persistence block (windows, streaks, confirmed, changed), the note drift_not_confirmed for an
        alert not yet confirmed, partial_signals when some signal was not evaluable and small_window below 20 runs;
        details.trigger holds the settings and the p-value of every signal. A signal that cannot be evaluated is
        exported as not_evaluable with its reason code (missing_signal, few_values).
        """
        window = {"n_rows": self.settings.get("n_current"), **(window or {})}
        reference = {"n_rows": self.settings.get("n_reference"), **(reference or {})}
        records = []
        if include_signals:
            for name, result in zip(self.names, self.signals):
                records.append(
                    evidence_from_result(
                        result,
                        system_id,
                        window=window,
                        reference=reference,
                        details={"trigger": TRIGGER_METRIC, "trigger_signal": name},
                        **kwargs,
                    )
                )
        summary = {
            **{k: v for k, v in self.settings.items() if k not in ("n_current", "n_reference")},
            "signals": [
                {
                    "name": name,
                    "column": r.feature_name,
                    "test": r.metric_name,
                    "p_value": r.metric_value,
                    "alert": r.drift_status,
                    "status": r.status,
                }
                for name, r in zip(self.names, self.signals)
            ],
        }
        record = evidence_from_result(
            self.result,
            system_id,
            window=window,
            reference=reference,
            persistence=[self.persistence],
            details={"trigger": summary},
            **kwargs,
        )
        record.reliability.persistence.windows = self.settings.get("persistence")
        notes = record.reliability.notes
        if self.not_evaluable and self.result.metric_value is not None:
            notes.append("partial_signals")
        n_current = self.settings.get("n_current")
        if n_current is not None and n_current < MIN_WINDOW_RUNS:
            notes.append("small_window")
        records.append(record)
        return records


class RAGDriftTrigger:
    """Recommended drift trigger of a RAG agent: faithfulness, empty retrievals and top retrieval score, Bonferroni,
    confirmed over consecutive windows

    Parameters
    ----------
    signals : sequence, optional
        Names of default signals ("faithfulness", "empty_retrieval", "retrieval_top") and / or TriggerSignal objects;
        default RAG_TRIGGER_SIGNALS (all three). The Bonferroni factor is the number of signals
    alpha : float, optional
        Level of the combined decision (default 0.05): an alert when m x the smallest p-value is below it
    persistence : int, optional
        Consecutive windows in alert needed to confirm the drift, and without alert to clear it (default 2;
        DriftPersistence). 1 keeps the raw verdicts
    columns : dict, optional
        Signal name -> column of the run records, to rename the columns of the default signals (defaults:
        det_faithfulness, det_retrieval_empty, det_retrieval_top)

    The defaults equal the configuration measured on the LR-03 replay (see the module docstring and
    RECOMMENDED_RAG_TRIGGER). Keep one instance for the whole monitoring: `evaluate` feeds each window to the
    persistence rule in turn; `reset()` after a new model version, corpus or reference.

    A signal whose column is absent from the window or the reference, or with fewer than `min_values` non-missing
    values, is not evaluable (status missing_signal or few_values): the decision rests on the other signals with the
    same Bonferroni factor (conservative, as in the replay). With no evaluable signal, the window has no verdict and
    the persistence streaks are kept.
    """

    def __init__(
        self,
        signals: Optional[Sequence[Union[str, TriggerSignal]]] = None,
        alpha: float = RECOMMENDED_RAG_TRIGGER["alpha"],
        persistence: int = RECOMMENDED_RAG_TRIGGER["persistence"],
        columns: Optional[Mapping[str, str]] = None,
    ):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
            raise error_msg(value=alpha, message="alpha must be in (0, 1)")
        self.signals = _resolve_signals(signals, columns)
        self.alpha = float(alpha)
        self._persistence = DriftPersistence(windows=persistence, metric_types=(MetricsType.drift.value,))

    @property
    def persistence(self) -> int:
        return self._persistence.windows

    @property
    def is_recommended(self) -> bool:
        """True when the configuration is the measured one (signals, tests, alpha and persistence)"""
        return (
            tuple((s.name, s.test, s.alternative, s.min_values) for s in self.signals)
            == tuple((s.name, s.test, s.alternative, s.min_values) for s in RAG_TRIGGER_SIGNALS)
            and self.alpha == RECOMMENDED_RAG_TRIGGER["alpha"]
            and self.persistence == RECOMMENDED_RAG_TRIGGER["persistence"]
        )

    def reset(self):
        """Forget the persistence history"""
        self._persistence.reset()

    def _signal_result(self, signal: TriggerSignal, current: pd.DataFrame, reference: pd.DataFrame) -> MetricResults:
        m = len(self.signals)
        level = self.alpha / m
        common = dict(metric_type=MetricsType.drift.value, metric_name=signal.test, feature_name=signal.column, threshold=level)
        absent = [what for what, frame in (("window", current), ("reference", reference)) if signal.column not in frame]
        if absent:
            return MetricResults(
                **common,
                n=len(current),
                status=MISSING_SIGNAL,
                reliable=False,
                error=f"signal {signal.name}: column {signal.column} absent from the {' and the '.join(absent)}",
            )
        counts = {"window": int(current[signal.column].notna().sum()), "reference": int(reference[signal.column].notna().sum())}
        few = {what: n for what, n in counts.items() if n < signal.min_values}
        if few:
            return MetricResults(
                **common,
                n=len(current),
                status=FEW_VALUES,
                reliable=False,
                error=f"signal {signal.name}: "
                + ", ".join(f"{n} non-missing values in the {what}" for what, n in few.items())
                + f", fewer than min_values={signal.min_values}",
            )
        kwargs = {"alternative": signal.alternative} if signal.alternative is not None else {}
        try:
            result = DriftTestMetric(signal.test, signal.column).evaluate(current, reference, alpha=level, **kwargs)
        except MetricEvaluationError as e:
            return MetricResults(**common, n=len(current), error=f"signal {signal.name}: {e}")
        if result.metric_value is not None:
            # same comparison as the combined decision: m x p < alpha (p < alpha / m up to rounding)
            result.drift_status = min(1.0, m * result.metric_value) < self.alpha
        rule = f"Bonferroni over {m} signal{'s' if m > 1 else ''}: alert when {m} x p < {self.alpha:g}"
        if signal.test in RATE_METRICS:
            result.method = f"{result.method}; {rule}"  # keeps the rates read back by pulsar_metrics.evidence
        else:
            result.method = f"{signal.test}, one-sided {_direction(signal)}" if kwargs else f"{signal.test}, two-sided"
            result.method += f"; {rule}"
        return result

    def evaluate(self, current: Records, reference: Records) -> TriggerDecision:
        """Test one window against the reference and feed the decision to the persistence rule

        Parameters
        ----------
        current : DataFrame or iterable of dict
            Run records of the window (one row per agent run), with the columns of the signals
        reference : DataFrame or iterable of dict
            Run records of the reference period (at least 300 runs in the measured configuration)

        Returns
        -------
        TriggerDecision
            Per-signal results, the combined decision (raw and confirmed) and to_evidence()
        """
        current, reference = _frame(current, "current"), _frame(reference, "reference")
        m = len(self.signals)
        results = [self._signal_result(s, current, reference) for s in self.signals]
        evaluable = [r.metric_value for r in results if r.metric_value is not None and not math.isnan(float(r.metric_value))]
        names = tuple(s.name for s in self.signals)
        p_value = min(1.0, m * min(evaluable)) if evaluable else None
        parts = ", ".join(
            (
                f"{n} ({r.feature_name}) p = {r.metric_value:.4g}"
                if r.metric_value is not None
                else f"{n} ({r.feature_name}) " f"{r.status or 'error'}"
            )
            for n, r in zip(names, results)
        )
        method = f"Bonferroni over {m} signals: min(1, {m} x smallest p-value) < {self.alpha:g}; {parts}"
        if p_value is None:
            combined = MetricResults(
                metric_type=MetricsType.drift.value,
                metric_name=TRIGGER_METRIC,
                threshold=self.alpha,
                n=len(current),
                status=NO_SIGNAL,
                reliable=False,
                error="no signal of the trigger could be evaluated: no verdict",
                method=method,
            )
        else:
            combined = MetricResults(
                metric_type=MetricsType.drift.value,
                metric_name=TRIGGER_METRIC,
                metric_value=p_value,
                p_value=p_value,
                threshold=self.alpha,
                drift_status=p_value < self.alpha,
                n=len(current),
                method=method,
            )
        verdict = self._persistence.update([combined])[0]
        settings = {
            "preset": "recommended" if self.is_recommended else "custom",
            "alpha": self.alpha,
            "correction": "bonferroni",
            "factor": m,
            "persistence": self.persistence,
            "n_current": len(current),
            "n_reference": len(reference),
        }
        return TriggerDecision(signals=results, result=combined, persistence=verdict, names=names, settings=settings)

    def evaluate_to_evidence(self, current: Records, reference: Records, system_id: str, **kwargs) -> list[Evidence]:
        """evaluate() then TriggerDecision.to_evidence(system_id, **kwargs)"""
        return self.evaluate(current, reference).to_evidence(system_id, **kwargs)
