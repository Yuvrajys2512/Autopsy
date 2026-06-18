"""Result comparison module for Phase 4 — compare batch results across time or variants.

Compares two BatchResultSet objects to detect regressions, improvements, and
failure cause shifts. Produces a ComparisonMetrics report suitable for experiment
tracking and A/B testing workflows.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .aggregation import BatchResultSet


class CaseDiff(BaseModel):
    """One case's change between baseline and current.

    Attributes:
        case_index: Index of the case in both result sets.
        trace_id_baseline: Trace ID from baseline run (or "error" if crashed).
        trace_id_current: Trace ID from current run (or "error" if crashed).
        baseline_passed: Whether case passed in baseline.
        current_passed: Whether case passed in current.
        baseline_score: Overall score from baseline.
        current_score: Overall score from current.
        baseline_root_cause: Root cause classified in baseline.
        current_root_cause: Root cause classified in current.
        change: Category of change ("improved" | "regressed" | "same" | "baseline_error" | "current_error").
    """

    case_index: int
    trace_id_baseline: str
    trace_id_current: str
    baseline_passed: bool
    current_passed: bool
    baseline_score: Optional[float] = None
    current_score: Optional[float] = None
    baseline_root_cause: str
    current_root_cause: str
    change: str  # "improved", "regressed", "same", "baseline_error", "current_error"


class ComparisonMetrics(BaseModel):
    """Metrics comparing baseline vs. current batch result sets.

    Attributes:
        baseline_pass_rate: Pass rate in baseline.
        current_pass_rate: Pass rate in current.
        pass_rate_delta: Difference (current - baseline). Positive = improvement.
        regression: True if any case went from pass to fail.
        improved: True if any case went from fail to pass.
        new_failures: Dict of {root_cause: count} appearing in current but not baseline.
        fixed_failures: Dict of {root_cause: count} that disappeared.
        case_diffs: Per-case changes.
        improvement_count: Count of improved cases.
        regression_count: Count of regressed cases.
    """

    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float
    regression: bool
    improved: bool
    new_failures: dict[str, int] = Field(default_factory=dict)
    fixed_failures: dict[str, int] = Field(default_factory=dict)
    case_diffs: list[CaseDiff] = Field(default_factory=list)
    improvement_count: int = 0
    regression_count: int = 0

    def summary_dict(self) -> dict:
        """Dict suitable for dashboard display."""
        return {
            "baseline_pass_rate": f"{self.baseline_pass_rate:.1%}",
            "current_pass_rate": f"{self.current_pass_rate:.1%}",
            "pass_rate_delta": f"{self.pass_rate_delta:+.1%}",
            "regression": self.regression,
            "improved": self.improved,
            "improvement_count": self.improvement_count,
            "regression_count": self.regression_count,
            "new_failures": self.new_failures,
            "fixed_failures": self.fixed_failures,
        }


def compare(baseline: BatchResultSet, current: BatchResultSet) -> ComparisonMetrics:
    """Compare two batch result sets.

    Args:
        baseline: Previous run (baseline).
        current: New run (experiment).

    Returns:
        ComparisonMetrics with diffs and summary.

    Raises:
        ValueError: If datasets don't match (different case counts).
    """
    if baseline.total_cases != current.total_cases:
        raise ValueError(
            f"Cannot compare: baseline has {baseline.total_cases} cases, "
            f"current has {current.total_cases} cases"
        )

    if len(baseline.results) != len(current.results):
        raise ValueError(
            f"Result count mismatch: baseline has {len(baseline.results)} results, "
            f"current has {len(current.results)} results"
        )

    case_diffs: list[CaseDiff] = []
    improvement_count = 0
    regression_count = 0
    new_failures_dict: dict[str, int] = {}
    fixed_failures_dict: dict[str, int] = {}

    # Track which failure causes appear in each set
    baseline_causes = set(baseline.metrics.failure_distribution.keys())
    current_causes = set(current.metrics.failure_distribution.keys())

    # Identify new and fixed failures
    new_causes = current_causes - baseline_causes
    fixed_causes = baseline_causes - current_causes

    for cause in new_causes:
        new_failures_dict[cause] = current.metrics.failure_distribution[cause]

    for cause in fixed_causes:
        fixed_failures_dict[cause] = baseline.metrics.failure_distribution[cause]

    # Compare case by case
    for i, (b_result, c_result) in enumerate(zip(baseline.results, current.results)):
        b_passed = b_result.passed
        c_passed = c_result.passed
        b_error = b_result.error
        c_error = c_result.error

        # Determine change type
        if b_error and c_error:
            change = "baseline_error"  # Both errored
        elif b_error:
            change = "improved"  # Was error, now something else
            improvement_count += 1
        elif c_error:
            change = "current_error"  # Was working, now errored
            regression_count += 1
        elif b_passed and not c_passed:
            change = "regressed"
            regression_count += 1
        elif not b_passed and c_passed:
            change = "improved"
            improvement_count += 1
        else:
            change = "same"

        diff = CaseDiff(
            case_index=i,
            trace_id_baseline=b_result.trace_id,
            trace_id_current=c_result.trace_id,
            baseline_passed=b_passed,
            current_passed=c_passed,
            baseline_score=b_result.overall_score,
            current_score=c_result.overall_score,
            baseline_root_cause=b_result.root_cause,
            current_root_cause=c_result.root_cause,
            change=change,
        )
        case_diffs.append(diff)

    has_regression = regression_count > 0
    has_improvement = improvement_count > 0

    return ComparisonMetrics(
        baseline_pass_rate=baseline.metrics.pass_rate,
        current_pass_rate=current.metrics.pass_rate,
        pass_rate_delta=current.metrics.pass_rate - baseline.metrics.pass_rate,
        regression=has_regression,
        improved=has_improvement,
        new_failures=new_failures_dict,
        fixed_failures=fixed_failures_dict,
        case_diffs=case_diffs,
        improvement_count=improvement_count,
        regression_count=regression_count,
    )
