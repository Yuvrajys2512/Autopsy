"""Phase 3 tests: datasets, batch runs, and aggregation."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from autopsy import (
    Dataset,
    Expectations,
    RuleBasedFailureClassifier,
    SpanKind,
    SpanStatus,
    TestCase,
    Trace,
    batch_evaluate,
)
from autopsy.eval.aggregation import BatchCaseResult, BatchResultSet, aggregate_diagnoses
from autopsy.eval.batch import BatchResult
from autopsy.eval.classifier import FailureClassification
from autopsy.eval.models import EvalReport
from autopsy.schema import Span

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


# --- Dataset Tests ---


def test_testcase_basic():
    tc = TestCase(
        input="What is 2+2?",
        expected_output="4",
        expected_tools=["calculator"],
    )
    assert tc.input == "What is 2+2?"
    assert tc.expected_output == "4"
    assert tc.expected_tools == ["calculator"]


def test_testcase_to_expectations():
    tc = TestCase(
        input="What is 2+2?",
        expected_output="4",
        expected_tools=["calculator"],
        allowed_tools=["calculator", "search"],
        max_steps=10,
    )
    exp = tc.to_expectations(goal="Math question")
    assert exp.goal == "Math question"
    assert exp.reference_output == "4"
    assert exp.expected_tools == ["calculator"]
    assert exp.allowed_tools == ["calculator", "search"]
    assert exp.max_steps == 10


def test_testcase_optional_fields():
    tc = TestCase(input="Do something")
    assert tc.expected_output is None
    assert tc.expected_tools is None
    assert tc.metadata == {}


def test_testcase_with_metadata():
    tc = TestCase(
        input="Question",
        metadata={"source": "manual", "difficulty": "hard"}
    )
    assert tc.metadata["source"] == "manual"
    assert tc.metadata["difficulty"] == "hard"


def test_dataset_empty():
    ds = Dataset(name="empty", version="1.0.0")
    assert len(ds) == 0
    assert list(ds) == []


def test_dataset_with_cases():
    cases = [
        TestCase(input="Q1", expected_output="A1"),
        TestCase(input="Q2", expected_output="A2"),
    ]
    ds = Dataset(name="test", cases=cases)
    assert len(ds) == 2
    assert ds[0].input == "Q1"
    assert ds[1].input == "Q2"


def test_dataset_save_and_load():
    cases = [
        TestCase(
            input="Q1",
            expected_output="A1",
            expected_tools=["search"],
            metadata={"id": 1},
        ),
        TestCase(
            input="Q2",
            expected_output="A2",
        ),
    ]
    ds = Dataset(name="test", version="1.0.0", description="Test dataset", cases=cases)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dataset.json"
        ds.save(path)
        assert path.exists()

        loaded = Dataset.from_file(path)
        assert loaded.name == "test"
        assert loaded.version == "1.0.0"
        assert loaded.description == "Test dataset"
        assert len(loaded) == 2
        assert loaded[0].input == "Q1"
        assert loaded[0].expected_tools == ["search"]
        assert loaded[0].metadata["id"] == 1


def test_dataset_json_format():
    ds = Dataset(
        name="test",
        version="2.0.0",
        description="A test",
        source="manual",
        cases=[TestCase(input="Q")],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ds.json"
        ds.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "test"
        assert data["version"] == "2.0.0"
        assert len(data["cases"]) == 1


# --- Batch Runner Tests ---


class StubJudge:
    def score(self, *, system, prompt, schema):
        return {
            "root_cause": "wrong_tool_selected",
            "rationale": "Test",
            "culprit_span_index": 0,
            "confidence": 0.9,
        }


def simple_agent(input_text: str) -> Trace:
    """A mock agent that succeeds on most inputs."""
    trace = mk_trace(
        [mk_span("llm", SpanKind.LLM_CALL, outputs={"text": f"Response to: {input_text}"})],
        input=input_text,
        output=f"Response to: {input_text}",
    )
    return trace


def failing_agent(input_text: str) -> Trace:
    """A mock agent that always fails."""
    trace = mk_trace(
        [mk_span("bad_tool", SpanKind.TOOL_CALL, status=SpanStatus.ERROR, inputs={"q": "xyz"})],
        status=SpanStatus.ERROR,
        input=input_text,
    )
    return trace


def crashing_agent(input_text: str) -> Trace:
    """An agent that crashes."""
    raise RuntimeError("Agent crash!")


def test_batch_evaluate_success():
    ds = Dataset(
        name="test",
        cases=[
            TestCase(input="Q1", allowed_tools=[]),
            TestCase(input="Q2", allowed_tools=[]),
        ],
    )
    results = batch_evaluate(simple_agent, ds, judge=None)
    assert len(results) == 2
    assert all(isinstance(r, BatchResult) for r in results)
    assert results[0].trace is not None
    assert results[1].trace is not None


def test_batch_evaluate_with_failures():
    ds = Dataset(
        name="test",
        cases=[
            TestCase(input="Q1", allowed_tools=["good_tool"]),
            TestCase(input="Q2", allowed_tools=["good_tool"]),
        ],
    )
    results = batch_evaluate(failing_agent, ds, judge=None)
    assert len(results) == 2


def test_batch_evaluate_handles_crashes():
    ds = Dataset(
        name="test",
        cases=[TestCase(input="Q1"), TestCase(input="Q2")],
    )
    results = batch_evaluate(crashing_agent, ds, judge=None)
    assert len(results) == 2
    # Both should have errors
    assert results[0].error()
    assert results[1].error()
    # Traces should be None
    assert results[0].trace is None
    assert results[1].trace is None
    # But reports should exist
    assert results[0].report is not None
    assert results[1].report is not None


def test_batch_result_status():
    trace = mk_trace([mk_span("s", SpanKind.LLM_CALL)])
    report = EvalReport(trace_id=trace.trace_id, results=[])
    report._results_cache = ([], True)  # Override passed prop
    classification = FailureClassification(root_cause="wrong_tool_selected", rationale="test")

    br = BatchResult(trace=trace, report=report, classification=classification)
    assert not br.error()  # Trace exists
    # Note: passed/failed depend on report.passed which is None by default


def test_batch_result_error_case():
    report = EvalReport(trace_id="error", results=[])
    classification = FailureClassification(root_cause="goal_not_completed", rationale="")

    br = BatchResult(trace=None, report=report, classification=classification)
    assert br.error()


# --- Aggregation Tests ---


def test_batch_result_set_empty():
    result_set = BatchResultSet(
        dataset_name="empty",
        dataset_version="1.0.0",
        total_cases=0,
        metrics=pytest.importorskip("autopsy.eval.aggregation").AggregateMetrics(total=0),
        results=[],
    )
    assert result_set.total_cases == 0
    assert result_set.metrics.pass_rate == 0.0


def test_batch_result_set_from_batch_results():
    # Create a few mock batch results
    trace1 = mk_trace([mk_span("s", SpanKind.LLM_CALL)], output="Good")
    report1 = EvalReport(trace_id=trace1.trace_id, results=[])
    # Manually set passed since we're not running the full eval
    report1.results = []

    classification1 = FailureClassification(
        root_cause="wrong_tool_selected",
        rationale="test",
        confidence=0.9,
    )
    br1 = BatchResult(trace=trace1, report=report1, classification=classification1)

    trace2 = mk_trace([mk_span("s", SpanKind.TOOL_CALL, status=SpanStatus.ERROR)])
    report2 = EvalReport(trace_id=trace2.trace_id, results=[])
    classification2 = FailureClassification(
        root_cause="infinite_loop",
        rationale="test",
        confidence=0.85,
    )
    br2 = BatchResult(trace=trace2, report=report2, classification=classification2)

    batch_results = [br1, br2]
    result_set = BatchResultSet.from_batch_results("test", "1.0.0", batch_results)

    assert result_set.dataset_name == "test"
    assert result_set.dataset_version == "1.0.0"
    assert result_set.total_cases == 2
    # Both failed (default when report has no passing results)
    assert result_set.metrics.failed == 2
    assert result_set.metrics.failure_distribution["wrong_tool_selected"] == 1
    assert result_set.metrics.failure_distribution["infinite_loop"] == 1


def test_failure_distribution_excludes_passing_cases():
    """A passing case still carries a classification (the classifier always
    returns one); it must not pollute the failure distribution."""
    from autopsy.eval.models import EvalResult

    # A passing case: a metric rendered a True verdict.
    passing_report = EvalReport(
        trace_id="t" * 32,
        results=[EvalResult(name="m", kind="rule", score=1.0, passed=True)],
    )
    passing = BatchResult(
        trace=mk_trace([mk_span("s", SpanKind.TOOL_CALL)]),
        report=passing_report,
        # the no-failure fallback the rule classifier returns for clean runs
        classification=FailureClassification(root_cause="goal_not_completed", rationale=""),
    )
    # A failing case.
    failing_report = EvalReport(
        trace_id="u" * 32,
        results=[EvalResult(name="m", kind="rule", score=0.0, passed=False,
                            failure_label="redundant_calls")],
    )
    failing = BatchResult(
        trace=mk_trace([mk_span("s", SpanKind.TOOL_CALL)]),
        report=failing_report,
        classification=FailureClassification(root_cause="redundant_calls", rationale=""),
    )

    rs = BatchResultSet.from_batch_results("test", "1.0.0", [passing, failing])

    assert rs.metrics.passed == 1
    assert rs.metrics.failed == 1
    # Only the failing case contributes; the passing fallback label is absent.
    assert rs.metrics.failure_distribution == {"redundant_calls": 1}
    assert "goal_not_completed" not in rs.metrics.failure_distribution


def test_aggregate_diagnoses():
    trace = mk_trace([mk_span("s", SpanKind.LLM_CALL)])
    report = EvalReport(trace_id=trace.trace_id, results=[])
    classification = FailureClassification(
        root_cause="wrong_tool_selected",
        rationale="test",
        confidence=0.9,
    )
    br = BatchResult(trace=trace, report=report, classification=classification)

    summary = aggregate_diagnoses([br])
    assert summary["total"] == 1
    assert "pass_rate" in summary
    assert "failure_distribution" in summary


def test_batch_case_result_serialization():
    case = BatchCaseResult(
        case_index=0,
        trace_id="abc123",
        passed=True,
        failed=False,
        error=False,
        overall_score=0.95,
        failure_labels=[],
        root_cause="none",
        diagnosis_rationale="Passed",
        confidence=1.0,
    )
    data = case.model_dump()
    assert data["case_index"] == 0
    assert data["passed"] is True
    assert data["overall_score"] == 0.95


# --- Round-trip Tests ---


def test_batch_result_set_save_and_load():
    trace = mk_trace([mk_span("s", SpanKind.LLM_CALL)])
    report = EvalReport(trace_id=trace.trace_id, results=[])
    classification = FailureClassification(
        root_cause="wrong_tool_selected",
        rationale="Test root cause",
        confidence=0.9,
    )
    br = BatchResult(trace=trace, report=report, classification=classification)

    result_set = BatchResultSet.from_batch_results("test_ds", "1.0.0", [br])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.json"
        result_set.save(path)
        assert path.exists()

        loaded = BatchResultSet.from_file(path)
        assert loaded.dataset_name == "test_ds"
        assert loaded.dataset_version == "1.0.0"
        assert loaded.total_cases == 1
        assert len(loaded.results) == 1
        assert loaded.results[0].root_cause == "wrong_tool_selected"


# --- Integration Test ---


def test_full_phase3_pipeline():
    """End-to-end: dataset → batch → aggregation → save/load."""
    ds = Dataset(
        name="Q&A Basic",
        version="1.0.0",
        description="Simple questions",
        cases=[
            TestCase(input="What is 2+2?", expected_output="4"),
            TestCase(input="What is Paris?", expected_output="A city in France"),
        ],
    )

    # Run batch
    results = batch_evaluate(simple_agent, ds, judge=None)
    assert len(results) == 2

    # Aggregate
    result_set = BatchResultSet.from_batch_results(ds.name, ds.version, results)
    assert result_set.dataset_name == "Q&A Basic"
    assert result_set.total_cases == 2

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save dataset
        ds_path = Path(tmpdir) / "dataset.json"
        ds.save(ds_path)

        # Save results
        results_path = Path(tmpdir) / "results.json"
        result_set.save(results_path)

        # Load back
        loaded_ds = Dataset.from_file(ds_path)
        loaded_results = BatchResultSet.from_file(results_path)

        assert loaded_ds.name == ds.name
        assert loaded_results.dataset_name == "Q&A Basic"
        assert loaded_results.total_cases == 2
