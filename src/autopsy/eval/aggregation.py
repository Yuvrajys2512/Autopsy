"""Aggregation and metrics for batch results — Phase 3.

Takes a list of batch results and computes overall statistics: pass rate,
failure distribution, top failure causes, etc. The BatchResultSet is a
serializable summary that the dashboard loads and displays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..schema import Trace
from .batch import BatchResult
from .classifier import FailureClassification
from .models import EvalReport


class AggregateMetrics(BaseModel):
    """Summary statistics over a batch of results.

    Attributes:
        total: Total number of test cases run.
        passed: Number that passed.
        failed: Number that failed.
        errors: Number where the agent crashed.
        pass_rate: Fraction that passed (passed / total).
        failure_distribution: Dict of {failure_label: count}.
    """

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    pass_rate: float = 0.0
    failure_distribution: dict[str, int] = Field(default_factory=dict)

    @property
    def top_failure_cause(self) -> Optional[str]:
        """The most common failure root cause, or None if no failures."""
        if not self.failure_distribution:
            return None
        return max(self.failure_distribution, key=self.failure_distribution.get)

    @property
    def top_failure_count(self) -> int:
        """Count for the top failure cause."""
        if not self.failure_distribution:
            return 0
        return max(self.failure_distribution.values())

    def summary_dict(self) -> dict[str, Any]:
        """A dict suitable for display in the dashboard."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": f"{self.pass_rate:.1%}",
            "top_failure_cause": self.top_failure_cause,
            "top_failure_count": self.top_failure_count,
            "failure_distribution": self.failure_distribution,
        }


class BatchResultSet(BaseModel):
    """A serialized batch of results with metadata and aggregates.

    This is what gets saved to disk and loaded by the dashboard. It omits
    the full Trace objects (keep separately if needed) but preserves all
    evaluation results and classifications for drill-down.

    Attributes:
        dataset_name: Name of the dataset that was evaluated.
        dataset_version: Version of the dataset.
        total_cases: Number of test cases in the batch.
        metrics: Aggregate statistics.
        results: List of per-case results (serialized).
    """

    dataset_name: str
    dataset_version: str
    total_cases: int
    metrics: AggregateMetrics
    results: list[BatchCaseResult] = Field(default_factory=list)

    @classmethod
    def from_batch_results(
        cls,
        dataset_name: str,
        dataset_version: str,
        batch_results: list[BatchResult],
    ) -> BatchResultSet:
        """Create a BatchResultSet from a list of batch results.

        Args:
            dataset_name: Name of the dataset.
            dataset_version: Version of the dataset.
            batch_results: List of BatchResult tuples.

        Returns:
            A BatchResultSet with aggregated metrics.
        """
        metrics = AggregateMetrics(total=len(batch_results))
        results = []

        for i, br in enumerate(batch_results):
            error = br.error()

            if error:
                metrics.errors += 1
            elif br.report.passed is True:
                metrics.passed += 1
            elif br.report.passed is False:
                metrics.failed += 1
            else:
                # Neither passed nor failed (e.g., no applicable metrics)
                # Treat as failed for aggregation purposes
                metrics.failed += 1

            # Track failure causes.
            if br.classification.root_cause:
                label = br.classification.root_cause
                metrics.failure_distribution[label] = metrics.failure_distribution.get(label, 0) + 1

            # Serialize the result.
            passed_value = br.report.passed is True
            failed_value = br.report.passed is False or (br.report.passed is None and not error)
            case_result = BatchCaseResult(
                case_index=i,
                trace_id=br.report.trace_id,
                passed=passed_value,
                failed=failed_value,
                error=error,
                overall_score=br.report.overall_score,
                failure_labels=br.report.failure_labels,
                root_cause=br.classification.root_cause,
                diagnosis_rationale=br.classification.rationale,
                culprit_span_id=br.classification.culprit_span_id,
                confidence=br.classification.confidence,
                eval_results=[r.model_dump() for r in br.report.results],
            )
            results.append(case_result)

        # Compute pass rate.
        if metrics.total > 0:
            metrics.pass_rate = metrics.passed / metrics.total

        return cls(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            total_cases=len(batch_results),
            metrics=metrics,
            results=results,
        )

    def save(self, path: str | Path) -> None:
        """Save the result set to a JSON file.

        Args:
            path: Path where the file will be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def from_file(cls, path: str | Path) -> BatchResultSet:
        """Load a result set from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A BatchResultSet instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class BatchCaseResult(BaseModel):
    """Serialized result for one test case in the batch.

    Attributes:
        case_index: Index in the batch (for reference).
        trace_id: The trace ID (may be None if error).
        passed: True if passed evaluation.
        failed: True if failed evaluation.
        error: True if agent crashed.
        overall_score: Overall evaluation score.
        failure_labels: List of failure labels from evaluation.
        root_cause: The classified root cause (FailureLabel value).
        diagnosis_rationale: Why this is the root cause.
        culprit_span_id: ID of the span responsible (if available).
        confidence: Confidence in the diagnosis.
        eval_results: List of all EvalResult dicts from the report.
    """

    case_index: int
    trace_id: str
    passed: bool
    failed: bool
    error: bool
    overall_score: Optional[float] = None
    failure_labels: list[str] = Field(default_factory=list)
    root_cause: str
    diagnosis_rationale: str
    culprit_span_id: Optional[str] = None
    confidence: Optional[float] = None
    eval_results: list[dict[str, Any]] = Field(default_factory=list)


def aggregate_diagnoses(batch_results: list[BatchResult]) -> dict[str, Any]:
    """Quick aggregation of failure diagnoses from a batch.

    Args:
        batch_results: List of BatchResult tuples.

    Returns:
        Dict with keys:
          - total, passed, failed, errors
          - pass_rate
          - failure_distribution: {label: count}
          - top_failure_cause and top_failure_count
    """
    result_set = BatchResultSet.from_batch_results(
        dataset_name="(unnamed)",
        dataset_version="1.0.0",
        batch_results=batch_results,
    )
    return result_set.metrics.summary_dict()
