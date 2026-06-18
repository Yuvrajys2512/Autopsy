"""Phase 1 tests: rule evaluators, judge metrics (via a fake judge), and the runner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from autopsy import Expectations, SpanKind, SpanStatus, evaluate
from autopsy.eval import (
    ErrorRecovery,
    Faithfulness,
    GoalCompletion,
    ResourceBudget,
    ToolArgumentValidity,
    ToolSelectionAccuracy,
    TrajectoryEfficiency,
)
from autopsy.schema import Span, TokenUsage, Trace
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


# --- TrajectoryEfficiency -----------------------------------------------------


def test_efficiency_all_unique_passes():
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=0),
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "b"}, order=1),
    ])
    r = TrajectoryEfficiency().evaluate(trace, Expectations())
    assert r.passed is True and r.score == 1.0 and r.failure_label is None


def test_efficiency_flags_redundant_calls():
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=0),
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=1),
    ])
    r = TrajectoryEfficiency().evaluate(trace, Expectations())
    assert r.passed is False
    assert r.failure_label == FailureLabel.REDUNDANT_CALLS.value
    assert r.score == 0.5  # 1 unique / 2 calls


def test_efficiency_flags_loop():
    spans = [mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=i) for i in range(3)]
    r = TrajectoryEfficiency().evaluate(mk_trace(spans), Expectations())
    assert r.failure_label == FailureLabel.INFINITE_LOOP.value


def test_efficiency_na_without_tool_calls():
    trace = mk_trace([mk_span("llm", SpanKind.LLM_CALL)])
    r = TrajectoryEfficiency().evaluate(trace, Expectations())
    assert r.applicable is False and r.score is None


# --- ErrorRecovery ------------------------------------------------------------


def test_recovery_no_errors_is_perfect():
    trace = mk_trace([mk_span("t", SpanKind.TOOL_CALL)])
    r = ErrorRecovery().evaluate(trace, Expectations())
    assert r.score == 1.0 and r.passed is True


def test_recovery_recovered_after_error():
    trace = mk_trace([
        mk_span("fetch", SpanKind.TOOL_CALL, status=SpanStatus.ERROR, order=0),
        mk_span("fetch", SpanKind.TOOL_CALL, status=SpanStatus.OK, order=1),
    ])
    r = ErrorRecovery().evaluate(trace, Expectations())
    assert r.score == 1.0 and r.passed is True and r.failure_label is None


def test_recovery_gave_up_early():
    trace = mk_trace(
        [
            mk_span("a", SpanKind.TOOL_CALL, status=SpanStatus.OK, order=0),
            mk_span("fetch", SpanKind.TOOL_CALL, status=SpanStatus.ERROR, order=1),
        ],
        status=SpanStatus.ERROR,
    )
    r = ErrorRecovery().evaluate(trace, Expectations())
    assert r.passed is False
    assert r.failure_label == FailureLabel.GAVE_UP_EARLY.value


# --- ToolArgumentValidity -----------------------------------------------------


def test_arg_validity_detects_violations():
    schema = {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
        "additionalProperties": False,
    }
    trace = mk_trace([
        mk_span("weather", SpanKind.TOOL_CALL, inputs={"location": "Paris"}, order=0),
        mk_span("weather", SpanKind.TOOL_CALL, inputs={"city": "Paris"}, order=1),  # wrong key
    ])
    r = ToolArgumentValidity().evaluate(trace, Expectations(tool_schemas={"weather": schema}))
    assert r.score == 0.5
    assert r.failure_label == FailureLabel.HALLUCINATED_TOOL_ARGS.value


def test_arg_validity_na_without_schema():
    trace = mk_trace([mk_span("weather", SpanKind.TOOL_CALL, inputs={"location": "Paris"})])
    r = ToolArgumentValidity().evaluate(trace, Expectations())
    assert r.applicable is False


# --- ToolSelectionAccuracy ----------------------------------------------------


def test_selection_allowed_tools_flags_out_of_registry():
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, order=0),
        mk_span("delete_db", SpanKind.TOOL_CALL, order=1),
    ])
    r = ToolSelectionAccuracy().evaluate(trace, Expectations(allowed_tools=["search", "fetch"]))
    assert r.score == 0.5
    assert r.failure_label == FailureLabel.WRONG_TOOL_SELECTED.value
    assert r.details["out_of_registry"] == ["delete_db"]


def test_selection_expected_tools_precision():
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, order=0),
        mk_span("fetch", SpanKind.TOOL_CALL, order=1),
    ])
    r = ToolSelectionAccuracy().evaluate(trace, Expectations(expected_tools=["search"]))
    assert r.score == 0.5  # 1 of 2 calls was expected
    assert "fetch" in r.details["unexpected"]


# --- ResourceBudget -----------------------------------------------------------


def test_budget_exceeded_steps_and_cost():
    spans = [mk_span(f"s{i}", SpanKind.TOOL_CALL, order=i) for i in range(5)]
    trace = mk_trace(spans)
    r = ResourceBudget().evaluate(trace, Expectations(max_steps=3))
    assert r.passed is False
    assert r.failure_label == FailureLabel.BUDGET_EXCEEDED.value


def test_budget_na_without_limits():
    r = ResourceBudget().evaluate(mk_trace([mk_span("s", SpanKind.TOOL_CALL)]), Expectations())
    assert r.applicable is False


# --- LLM judge metrics (fake judge) ------------------------------------------


class FakeJudge:
    """Returns a canned judgement, adapting to whichever boolean key the rubric's
    schema asks for (completed/faithful)."""

    def __init__(self, score=0.9, verdict=True, rationale="looks good"):
        self.score_ = score
        self.verdict = verdict
        self.rationale = rationale
        self.calls = []

    def score(self, *, system, prompt, schema):
        self.calls.append({"system": system, "prompt": prompt})
        bool_key = next(
            k for k, v in schema["properties"].items() if v.get("type") == "boolean"
        )
        return {"score": self.score_, bool_key: self.verdict, "rationale": self.rationale}


def test_goal_completion_with_fake_judge():
    trace = mk_trace([mk_span("llm", SpanKind.LLM_CALL)], input="q", output="an answer")
    judge = FakeJudge(score=0.95, verdict=True)
    r = GoalCompletion(judge).evaluate(trace, Expectations(goal="do the thing"))
    assert r.passed is True and r.score == 0.95
    assert "Goal" in judge.calls[0]["prompt"]


def test_goal_completion_failure_label():
    trace = mk_trace([mk_span("llm", SpanKind.LLM_CALL)], input="q", output="wrong")
    r = GoalCompletion(FakeJudge(score=0.1, verdict=False)).evaluate(trace, Expectations())
    assert r.passed is False
    assert r.failure_label == FailureLabel.GOAL_NOT_COMPLETED.value


def test_goal_completion_na_without_judge():
    trace = mk_trace([mk_span("llm", SpanKind.LLM_CALL)], output="x")
    r = GoalCompletion(judge=None).evaluate(trace, Expectations(goal="g"))
    assert r.applicable is False


def test_faithfulness_grounded():
    trace = mk_trace(
        [mk_span("fetch", SpanKind.RETRIEVAL, outputs={"result": "Paris is in France."}, order=0)],
        output="Paris is in France.",
    )
    r = Faithfulness(FakeJudge(verdict=True, score=1.0)).evaluate(trace, Expectations())
    assert r.passed is True


def test_faithfulness_ungrounded_label():
    trace = mk_trace(
        [mk_span("fetch", SpanKind.RETRIEVAL, outputs={"result": "Paris is in France."}, order=0)],
        output="Paris is in Italy.",
    )
    r = Faithfulness(FakeJudge(verdict=False, score=0.0)).evaluate(trace, Expectations())
    assert r.failure_label == FailureLabel.CORRECT_RETRIEVAL_WRONG_ANSWER.value


def test_faithfulness_retrieval_miss_without_judge():
    # retrieval ran but produced nothing — decided without a judge call
    trace = mk_trace([mk_span("fetch", SpanKind.RETRIEVAL, outputs={"result": ""}, order=0)],
                     output="some answer")
    r = Faithfulness(judge=None).evaluate(trace, Expectations())
    assert r.failure_label == FailureLabel.RETRIEVAL_MISS.value


# --- runner / report ----------------------------------------------------------


def test_evaluate_returns_structured_report_offline():
    trace = mk_trace([
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=0),
        mk_span("search", SpanKind.TOOL_CALL, inputs={"q": "a"}, order=1),  # redundant
    ])
    report = evaluate(trace, expected={"allowed_tools": ["search"], "max_steps": 1})

    assert report.trace_id == trace.trace_id
    # rule metrics ran; judge metrics absent (no judge) → not in default suite
    eff = report.get("trajectory_efficiency")
    assert eff.failure_label == FailureLabel.REDUNDANT_CALLS.value
    assert FailureLabel.REDUNDANT_CALLS.value in report.failure_labels
    assert FailureLabel.BUDGET_EXCEEDED.value in report.failure_labels
    assert report.overall_score is not None
    assert report.passed is False
    assert isinstance(report.render(), str)


def test_evaluate_with_judge_includes_llm_metrics():
    trace = mk_trace([mk_span("llm", SpanKind.LLM_CALL)], input="q", output="answer")
    report = evaluate(trace, judge=FakeJudge(verdict=True, score=0.8))
    assert report.get("goal_completion") is not None
    assert report.get("goal_completion").passed is True


def test_evaluate_dict_expectations_coerced():
    trace = mk_trace([mk_span("s", SpanKind.TOOL_CALL)])
    report = evaluate(trace, expected={"max_steps": 10})
    assert report.get("resource_budget").passed is True
