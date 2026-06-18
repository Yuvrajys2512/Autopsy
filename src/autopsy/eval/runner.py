"""`evaluate(trace)` — run a suite of evaluators and aggregate into a report.

This is the Phase 1 gate: one call, a structured score report for a single
trace. Rule metrics always run; the LLM-judge metrics run only when a `judge`
is supplied (otherwise they report as not-applicable), so the default path works
fully offline.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

from ..schema import Trace
from .base import Evaluator
from .judge import Judge
from .llm_metrics import Faithfulness, GoalCompletion
from .models import EvalReport, EvalResult, Expectations
from .rules import RULE_EVALUATORS


def default_evaluators(judge: Optional[Judge] = None) -> list[Evaluator]:
    """The standard suite: all rule metrics, plus the judge metrics if a judge
    is available."""
    evaluators: list[Evaluator] = list(RULE_EVALUATORS)
    if judge is not None:
        evaluators += [GoalCompletion(judge), Faithfulness(judge)]
    return evaluators


def evaluate(
    trace: Trace,
    *,
    expected: Optional[Union[Expectations, dict]] = None,
    judge: Optional[Judge] = None,
    evaluators: Optional[Sequence[Evaluator]] = None,
) -> EvalReport:
    """Score a single trace.

    Args:
        trace: the run to evaluate.
        expected: optional reference data (``Expectations`` or a plain dict) —
            enables reference-based metrics (expected tools, gold answer, budgets).
        judge: optional `Judge` to enable the LLM-as-judge metrics.
        evaluators: override the evaluator suite entirely. Defaults to
            ``default_evaluators(judge)``.

    Returns:
        An `EvalReport` with one `EvalResult` per evaluator plus aggregates.
        A crash in any single evaluator is captured as an errored result rather
        than failing the whole report.
    """
    if isinstance(expected, dict):
        expected = Expectations(**expected)
    expected = expected or Expectations()

    suite = list(evaluators) if evaluators is not None else default_evaluators(judge)

    results: list[EvalResult] = []
    for ev in suite:
        try:
            results.append(ev.evaluate(trace, expected))
        except Exception as exc:  # an evaluator bug must not sink the report
            results.append(EvalResult.errored(getattr(ev, "name", repr(ev)), getattr(ev, "kind", "rule"), str(exc)))

    return EvalReport(trace_id=trace.trace_id, results=results)
