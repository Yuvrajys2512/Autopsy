"""The Evaluator contract and shared helpers.

An **Evaluator** is the core abstraction (project brief §4.3): a function from a
trace (plus optional reference `Expectations`) to an `EvalResult` —
``score + optional failure_label + rationale``. Everything in Phase 1, rule or
judge, implements this one interface, so the runner treats them uniformly.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import ClassVar

from ..schema import Span, SpanKind, SpanStatus, Trace
from .models import EvalResult, Expectations, MetricKind


class Evaluator(ABC):
    """Base class for all metrics. Subclasses set ``name``/``kind`` and implement
    ``evaluate``."""

    name: ClassVar[str]
    kind: ClassVar[MetricKind]

    @abstractmethod
    def evaluate(self, trace: Trace, expected: Expectations) -> EvalResult:
        """Assess ``trace`` and return a single result. Must not raise on a
        well-formed trace — return a ``not_applicable`` result instead when there
        is nothing to measure."""

    # convenience constructors so subclasses don't repeat self.name/self.kind
    def _na(self, reason: str) -> EvalResult:
        return EvalResult.not_applicable(self.name, self.kind, reason)

    def _result(self, **kwargs) -> EvalResult:
        return EvalResult(name=self.name, kind=self.kind, **kwargs)


# --- span helpers (shared across evaluators) --------------------------------


def spans_of(trace: Trace, *kinds: SpanKind) -> list[Span]:
    return [s for s in trace.spans if s.kind in kinds]


def tool_spans(trace: Trace) -> list[Span]:
    return spans_of(trace, SpanKind.TOOL_CALL)


def llm_spans(trace: Trace) -> list[Span]:
    return spans_of(trace, SpanKind.LLM_CALL)


def retrieval_spans(trace: Trace) -> list[Span]:
    return spans_of(trace, SpanKind.RETRIEVAL)


def errored_spans(trace: Trace) -> list[Span]:
    return [s for s in trace.spans if s.status is SpanStatus.ERROR]


def in_order(spans: list[Span]) -> list[Span]:
    """Spans sorted by start time — the temporal order the agent executed them."""
    return sorted(spans, key=lambda s: s.start_time)


def tool_signature(span: Span) -> str:
    """A stable identity for a tool call: name + canonicalized inputs.

    Two calls with the same signature are the *same* call repeated — the basis
    for redundancy / loop detection.
    """
    args = json.dumps(span.inputs, sort_keys=True, default=str)
    return f"{span.name}({args})"
