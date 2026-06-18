"""Batch evaluation runner for Phase 3 — run many test cases against an agent.

`batch_evaluate` takes an agent function, a dataset, and an optional judge, and
runs every test case through the full evaluate → classify pipeline. It handles
agent crashes gracefully, logging errors and moving to the next case.

Each batch run produces a list of (Trace, EvalReport, FailureClassification) tuples.
"""

from __future__ import annotations

from typing import Callable, Optional, TypeAlias

from ..schema import Trace
from .classifier import FailureClassification, FailureClassifier, LLMFailureClassifier
from .dataset import Dataset
from .judge import Judge
from .models import EvalReport, EvalResult
from .runner import evaluate

AgentFn: TypeAlias = Callable[[str], Trace]


class BatchRunner:
    """Runs a full evaluation suite over a dataset of test cases.

    Attributes:
        agent_fn: A callable that takes a string input and returns a Trace.
        dataset: The Dataset of labeled test cases.
        judge: Optional Judge for LLM-driven evaluation.
        classifier: Failure classifier (defaults to LLMFailureClassifier if judge provided).
    """

    def __init__(
        self,
        agent_fn: AgentFn,
        dataset: Dataset,
        judge: Optional[Judge] = None,
        classifier: Optional[FailureClassifier] = None,
    ) -> None:
        self.agent_fn = agent_fn
        self.dataset = dataset
        self.judge = judge
        self.classifier = classifier or LLMFailureClassifier(judge=judge)

    def run(self) -> list[BatchResult]:
        """Execute the full evaluation suite for every test case.

        Runs the agent on each test case's input, evaluates the trace, and
        classifies any failures. Handles exceptions gracefully — if the agent
        crashes on one case, the error is logged and evaluation continues.

        Returns:
            List of BatchResult tuples (trace, report, classification) in order.
            If a case crashes, the trace is None and the report captures the error.
        """
        results: list[BatchResult] = []

        for i, case in enumerate(self.dataset.cases):
            try:
                # Run the agent.
                trace = self.agent_fn(case.input)
                assert isinstance(trace, Trace), f"agent_fn must return a Trace, got {type(trace)}"

                # Evaluate the trace.
                report = evaluate(trace, expected=case.to_expectations(), judge=self.judge)

                # Classify any failure.
                classification = self.classifier.classify(report, trace)

                results.append(BatchResult(trace=trace, report=report, classification=classification))
            except Exception as exc:
                # Agent crashed; log the error and continue.
                error_result = EvalResult.errored("agent_execution", "error", str(exc))
                error_report = EvalReport(trace_id=f"error-case-{i}", results=[error_result])
                error_classification = FailureClassification(
                    root_cause="goal_not_completed",
                    rationale=f"Agent failed with exception: {type(exc).__name__}: {exc}",
                    confidence=1.0,
                )
                results.append(BatchResult(trace=None, report=error_report, classification=error_classification))

        return results


class BatchResult:
    """One result from batch evaluation: trace, report, and diagnosis.

    Attributes:
        trace: The captured Trace (None if agent crashed).
        report: The EvalReport from Phase 1 evaluation.
        classification: The FailureClassification from Phase 2 diagnosis.
    """

    __slots__ = ("trace", "report", "classification")

    def __init__(
        self,
        trace: Optional[Trace],
        report: EvalReport,
        classification: FailureClassification,
    ) -> None:
        self.trace = trace
        self.report = report
        self.classification = classification

    def passed(self) -> bool:
        """Return True if the trace passed evaluation."""
        return self.report.passed is True

    def failed(self) -> bool:
        """Return True if the trace failed evaluation."""
        return self.report.passed is False

    def error(self) -> bool:
        """Return True if the agent crashed (trace is None)."""
        return self.trace is None


def batch_evaluate(
    agent_fn: AgentFn,
    dataset: Dataset,
    judge: Optional[Judge] = None,
    classifier: Optional[FailureClassifier] = None,
) -> list[BatchResult]:
    """Convenience function: run batch evaluation over a dataset.

    Args:
        agent_fn: Callable that takes a string input and returns a Trace.
        dataset: Dataset of labeled test cases.
        judge: Optional Judge for LLM-driven evaluation and classification.
        classifier: Optional FailureClassifier (defaults to LLMFailureClassifier).

    Returns:
        List of BatchResult tuples (trace, report, classification) in order.
    """
    runner = BatchRunner(agent_fn, dataset, judge=judge, classifier=classifier)
    return runner.run()
