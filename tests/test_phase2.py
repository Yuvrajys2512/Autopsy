"""Phase 2 tests: failure classifier (rule-based and LLM-driven)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from autopsy import Expectations, SpanKind, SpanStatus, evaluate
from autopsy.eval import (
    ErrorRecovery,
    LLMFailureClassifier,
    RuleBasedFailureClassifier,
    ResourceBudget,
    ToolSelectionAccuracy,
    TrajectoryEfficiency,
)
from autopsy.schema import Span, Trace
from autopsy.taxonomy import FailureLabel

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mk_span(name, kind, *, status=SpanStatus.OK, inputs=None, outputs=None, order=0):
    return Span(
        trace_id="t" * 32,
        name=name,
        kind=kind,
        status=status,
        inputs=inputs or {},
        outputs=outputs or {},
        start_time=_BASE + timedelta(seconds=order),
        end_time=_BASE + timedelta(seconds=order, milliseconds=100),
    )


def mk_trace(spans, *, status=SpanStatus.OK, input=None, output=None):
    t = Trace(name="t", status=status, input=input, output=output)
    for s in spans:
        s.trace_id = t.trace_id
    t.spans = spans
    t.start_time = _BASE
    t.end_time = _BASE + timedelta(seconds=len(spans) + 1)
    return t


# --- StubJudge for testing ---------------------------------------------------


class StubJudge:
    """A judge that returns a fixed classification for testing."""

    def __init__(self, root_cause: str = FailureLabel.WRONG_TOOL_SELECTED.value):
        self.root_cause = root_cause

    def score(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a fixed stub response."""
        return {
            "root_cause": self.root_cause,
            "rationale": "stub rationale",
            "culprit_span_index": 0,
            "confidence": 0.95,
        }


# --- RuleBasedFailureClassifier -----------------------------------------------


def test_rule_classifier_single_label():
    """Single failure label → high confidence."""
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=0),
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=1),
    ])
    report = evaluate(trace, expected=Expectations())

    classifier = RuleBasedFailureClassifier()
    diagnosis = classifier.classify(report, trace)

    assert diagnosis.root_cause == FailureLabel.REDUNDANT_CALLS.value
    assert diagnosis.confidence == 0.8
    assert len(diagnosis.evidence_labels) == 1


def test_rule_classifier_multiple_labels_wrong_tool_priority():
    """Multiple labels: WRONG_TOOL_SELECTED has priority."""
    trace = mk_trace([
        mk_span("wrong_tool", SpanKind.TOOL_CALL, order=0),  # Not in allowed set
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=1),
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=2),  # Redundant
    ])
    report = evaluate(
        trace,
        expected=Expectations(
            allowed_tools=["search", "fetch"],
            tool_schemas={"search": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"], "additionalProperties": False}},
        ),
    )

    classifier = RuleBasedFailureClassifier()
    diagnosis = classifier.classify(report, trace)

    # Should pick wrong_tool (higher priority) over redundant_calls.
    assert diagnosis.root_cause == FailureLabel.WRONG_TOOL_SELECTED.value
    assert diagnosis.confidence == 0.6


def test_rule_classifier_no_failures():
    """No failures in report → defaults gracefully."""
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "unique"}, order=0),
    ])
    report = evaluate(trace, expected=Expectations())

    classifier = RuleBasedFailureClassifier()
    diagnosis = classifier.classify(report, trace)

    # Should pick GOAL_NOT_COMPLETED (default).
    assert diagnosis.root_cause == FailureLabel.GOAL_NOT_COMPLETED.value
    assert len(diagnosis.evidence_labels) == 0


def test_rule_classifier_extracts_culprit_span():
    """Culprit span_id extraction (when details have span_ids)."""
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=0),
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=1),
    ])
    report = evaluate(trace, expected=Expectations())

    # The report should have REDUNDANT_CALLS failure.
    eff_result = report.get(TrajectoryEfficiency().name)
    assert eff_result is not None
    assert eff_result.failure_label == FailureLabel.REDUNDANT_CALLS.value

    classifier = RuleBasedFailureClassifier()
    diagnosis = classifier.classify(report, trace)

    # Should classify REDUNDANT_CALLS with high confidence.
    assert diagnosis.root_cause == FailureLabel.REDUNDANT_CALLS.value
    assert diagnosis.confidence == 0.8


# --- LLMFailureClassifier with StubJudge -----------------------------------


def test_llm_classifier_with_stub_judge():
    """LLM classifier uses judge when available."""
    trace = mk_trace([
        mk_span("wrong", SpanKind.TOOL_CALL, order=0),
    ])
    report = evaluate(
        trace,
        expected=Expectations(allowed_tools=["search", "fetch"]),
    )

    judge = StubJudge(root_cause=FailureLabel.WRONG_TOOL_SELECTED.value)
    classifier = LLMFailureClassifier(judge=judge)
    diagnosis = classifier.classify(report, trace)

    assert diagnosis.root_cause == FailureLabel.WRONG_TOOL_SELECTED.value
    assert diagnosis.rationale == "stub rationale"
    assert diagnosis.confidence == 0.95


def test_llm_classifier_fallback_no_judge():
    """LLM classifier falls back to rule-based when no judge."""
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=0),
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=1),
    ])
    report = evaluate(trace, expected=Expectations())

    classifier = LLMFailureClassifier(judge=None)
    diagnosis = classifier.classify(report, trace)

    # Should fall back to rule-based.
    assert diagnosis.root_cause == FailureLabel.REDUNDANT_CALLS.value
    assert diagnosis.confidence == 0.8  # Rule-based confidence


def test_llm_classifier_span_index_mapping():
    """Culprit span index is mapped to span_id."""
    trace = mk_trace([
        mk_span("a", SpanKind.TOOL_CALL, order=0),
        mk_span("bad_tool", SpanKind.TOOL_CALL, order=1),  # Not allowed
        mk_span("c", SpanKind.TOOL_CALL, order=2),
    ])
    report = evaluate(
        trace,
        expected=Expectations(allowed_tools=["a", "c"]),  # bad_tool not allowed
    )

    # Report should have WRONG_TOOL_SELECTED failure.
    assert FailureLabel.WRONG_TOOL_SELECTED.value in report.failure_labels

    # Judge says culprit is span index 1 (the bad_tool).
    class IndexedStubJudge(StubJudge):
        def score(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
            return {
                "root_cause": FailureLabel.WRONG_TOOL_SELECTED.value,
                "rationale": "test",
                "culprit_span_index": 1,
                "confidence": 0.9,
            }

    classifier = LLMFailureClassifier(judge=IndexedStubJudge())
    diagnosis = classifier.classify(report, trace)

    # Should map index 1 to trace.spans[1].span_id.
    assert diagnosis.culprit_span_id == trace.spans[1].span_id


def test_llm_classifier_judge_failure_fallback():
    """Judge failure → gracefully falls back to rule-based."""

    class FailingJudge:
        def score(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("Judge exploded!")

    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=0),
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=1),
    ])
    report = evaluate(trace, expected=Expectations())

    classifier = LLMFailureClassifier(judge=FailingJudge())
    diagnosis = classifier.classify(report, trace)

    # Should gracefully fall back to rule-based (REDUNDANT_CALLS).
    assert diagnosis.root_cause == FailureLabel.REDUNDANT_CALLS.value


def test_llm_classifier_invalid_span_index():
    """Span index out of bounds → culprit_span_id is None."""
    trace = mk_trace([
        mk_span("a", SpanKind.TOOL_CALL, order=0),
    ])
    report = evaluate(trace, expected=Expectations())

    class OutOfBoundsJudge(StubJudge):
        def score(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
            return {
                "root_cause": FailureLabel.WRONG_TOOL_SELECTED.value,
                "rationale": "test",
                "culprit_span_index": 999,  # Out of bounds
                "confidence": 0.9,
            }

    classifier = LLMFailureClassifier(judge=OutOfBoundsJudge())
    diagnosis = classifier.classify(report, trace)

    # culprit_span_id should be None (out of bounds).
    assert diagnosis.culprit_span_id is None


# --- Integration: evaluate + classify -----------------------------------------------


def test_classify_a_real_failure():
    """End-to-end: evaluate a flawed trace, then classify it."""
    # A trace with multiple issues: wrong tool + error.
    trace = mk_trace([
        mk_span("wrong_tool", SpanKind.TOOL_CALL, status=SpanStatus.OK, order=0),
        mk_span("search", SpanKind.TOOL_CALL, status=SpanStatus.ERROR, order=1),
    ], status=SpanStatus.ERROR)

    expected = Expectations(
        allowed_tools=["search", "fetch"],
    )
    report = evaluate(trace, expected=expected)

    # Should have multiple failure signals.
    assert len(report.failure_labels) > 0

    # Rule classifier should identify wrong_tool as root (higher priority).
    classifier = RuleBasedFailureClassifier()
    diagnosis = classifier.classify(report, trace)

    assert diagnosis.root_cause == FailureLabel.WRONG_TOOL_SELECTED.value
    # Should have collected evidence from multiple metrics.
    assert len(diagnosis.evidence_labels) > 0
