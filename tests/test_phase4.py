"""Phase 4 tests: result comparison, history tracking, and advanced dashboard."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from autopsy import (
    Dataset,
    RuleBasedFailureClassifier,
    SpanKind,
    SpanStatus,
    TestCase,
    Trace,
    batch_evaluate,
)
from autopsy.eval.aggregation import AggregateMetrics, BatchCaseResult, BatchResultSet
from autopsy.eval.comparison import CaseDiff, ComparisonMetrics, compare
from autopsy.eval.history import BatchRunMetadata, HistoryEntry, RunHistory

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mk_batch_result_set(
    name: str,
    version: str,
    total_cases: int,
    passed: int = 5,
    failed: int = 3,
    errors: int = 0,
    failure_dist: dict | None = None,
) -> BatchResultSet:
    """Create a mock BatchResultSet for testing."""
    if failure_dist is None:
        failure_dist = {"timeout": 2, "wrong_answer": 1}

    results = []
    for i in range(total_cases):
        is_passed = i < passed
        is_error = i >= (passed + failed)
        results.append(
            BatchCaseResult(
                case_index=i,
                trace_id=f"trace-{i:03d}",
                passed=is_passed,
                failed=not is_passed and not is_error,
                error=is_error,
                overall_score=0.9 if is_passed else 0.3,
                failure_labels=["wrong_answer"] if not is_passed else [],
                root_cause="wrong_answer" if not is_passed else "passed",
                diagnosis_rationale="Test",
                confidence=0.95,
            )
        )

    metrics = AggregateMetrics(
        total=total_cases,
        passed=passed,
        failed=failed,
        errors=errors,
        pass_rate=passed / total_cases if total_cases > 0 else 0.0,
        failure_distribution=failure_dist,
    )

    return BatchResultSet(
        dataset_name=name, dataset_version=version, total_cases=total_cases, metrics=metrics, results=results
    )


class TestComparisonBasic:
    """Basic comparison functionality."""

    def test_compare_identical_runs(self):
        """Comparing identical runs should show zero delta."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=8, passed=5, failed=3)
        current = mk_batch_result_set("test", "1.0.0", total_cases=8, passed=5, failed=3)

        result = compare(baseline, current)

        assert result.pass_rate_delta == 0.0
        assert not result.regression
        assert not result.improved

    def test_compare_detects_improvement(self):
        """If current has higher pass_rate, delta is positive."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=5, failed=5)
        current = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=7, failed=3)

        result = compare(baseline, current)

        assert abs(result.baseline_pass_rate - 0.5) < 0.01
        assert abs(result.current_pass_rate - 0.7) < 0.01
        assert abs(result.pass_rate_delta - 0.2) < 0.01
        assert result.improved

    def test_compare_detects_regression(self):
        """If any case went pass→fail, regression=True."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=8, passed=5, failed=3)
        current = mk_batch_result_set("test", "1.0.0", total_cases=8, passed=4, failed=4)

        result = compare(baseline, current)

        assert result.regression

    def test_compare_case_diff_logic(self):
        """CaseDiff correctly categorizes improvements vs regressions."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=3, passed=2, failed=1)
        current = mk_batch_result_set("test", "1.0.0", total_cases=3, passed=1, failed=2)

        result = compare(baseline, current)

        # At least one regression and one same should be detected
        diffs = result.case_diffs
        assert len(diffs) == 3
        # Verify diffs have required fields
        for diff in diffs:
            assert diff.case_index >= 0
            assert diff.trace_id_baseline
            assert diff.trace_id_current
            assert diff.change in ("improved", "regressed", "same", "baseline_error", "current_error")

    def test_compare_failure_cause_shifts(self):
        """new_failures and fixed_failures correctly track cause changes."""
        baseline = mk_batch_result_set(
            "test",
            "1.0.0",
            total_cases=8,
            passed=5,
            failed=3,
            failure_dist={"timeout": 2, "wrong_answer": 1},
        )
        current = mk_batch_result_set(
            "test",
            "1.0.0",
            total_cases=8,
            passed=5,
            failed=3,
            failure_dist={"memory_limit": 2, "wrong_answer": 1},
        )

        result = compare(baseline, current)

        assert "memory_limit" in result.new_failures
        assert "timeout" in result.fixed_failures

    def test_compare_mismatched_case_counts(self):
        """Comparing datasets with different case counts raises ValueError."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=8)
        current = mk_batch_result_set("test", "1.0.0", total_cases=10)

        with pytest.raises(ValueError, match="Cannot compare"):
            compare(baseline, current)

    def test_comparison_metrics_summary_dict(self):
        """ComparisonMetrics.summary_dict() returns formatted dict."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=5)
        current = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=6)

        result = compare(baseline, current)
        summary = result.summary_dict()

        assert "baseline_pass_rate" in summary
        assert "current_pass_rate" in summary
        assert "pass_rate_delta" in summary
        assert isinstance(summary["baseline_pass_rate"], str)  # Formatted


class TestComparisonEdgeCases:
    """Edge cases for comparison."""

    def test_compare_all_pass_to_all_fail(self):
        """Extreme regression: all passed → all failed."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=5, passed=5, failed=0)
        current = mk_batch_result_set("test", "1.0.0", total_cases=5, passed=0, failed=5)

        result = compare(baseline, current)

        assert result.baseline_pass_rate == 1.0
        assert result.current_pass_rate == 0.0
        assert result.pass_rate_delta == -1.0
        assert result.regression
        assert not result.improved

    def test_compare_with_errors(self):
        """Handle cases with errors in baseline or current."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=5, passed=3, failed=2, errors=0)
        current = mk_batch_result_set("test", "1.0.0", total_cases=5, passed=3, failed=1, errors=1)

        result = compare(baseline, current)
        # Should have at least one "current_error" in diffs
        error_diffs = [d for d in result.case_diffs if d.change == "current_error"]
        assert len(error_diffs) > 0

    def test_compare_improvement_count(self):
        """improvement_count tracks cases that improved."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=3, failed=7)
        current = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=6, failed=4)

        result = compare(baseline, current)

        assert result.improvement_count > 0
        assert result.improvement_count <= 7  # At most 7 cases improved

    def test_compare_regression_count(self):
        """regression_count tracks cases that regressed."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=5, failed=5)
        current = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=4, failed=6)

        result = compare(baseline, current)

        assert result.regression_count > 0


class TestHistoryBasic:
    """Basic history functionality."""

    def test_run_history_empty(self):
        """Empty history doesn't crash."""
        history = RunHistory(name="test-history")
        assert history.name == "test-history"
        assert len(history.entries) == 0

    def test_run_history_add_run(self):
        """Adding runs to history maintains order."""
        history = RunHistory(name="test")
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=8, passed=5)

        metadata = BatchRunMetadata(
            run_id="v1.0",
            timestamp=_BASE,
            dataset_name="test",
            dataset_version="1.0.0",
            agent_version="v1.0",
            judge_used=False,
            notes="baseline",
        )

        history.add_run("v1.0", metadata, baseline)

        assert len(history.entries) == 1
        assert history.entries[0].run_id == "v1.0"
        assert history.entries[0].metadata.agent_version == "v1.0"

    def test_run_history_get_trend(self):
        """Trend extraction gives correct time-series."""
        history = RunHistory(name="test")

        # Add 3 runs with different pass rates
        for i, pass_count in enumerate([5, 6, 7]):
            result_set = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=pass_count)
            metadata = BatchRunMetadata(
                run_id=f"run-{i}",
                timestamp=_BASE + timedelta(days=i),
                dataset_name="test",
                dataset_version="1.0.0",
                agent_version=f"v1.{i}",
            )
            history.add_run(f"run-{i}", metadata, result_set)

        trend = history.get_trend("pass_rate")

        assert len(trend) == 3
        assert trend[0][0] == "run-0"
        assert abs(trend[0][1] - 0.5) < 0.01  # 5/10 = 0.5
        assert abs(trend[1][1] - 0.6) < 0.01  # 6/10 = 0.6
        assert abs(trend[2][1] - 0.7) < 0.01  # 7/10 = 0.7

    def test_run_history_detect_regression(self):
        """Regression detection flags runs below threshold."""
        history = RunHistory(name="test")

        # Run 1: 70%, Run 2: 65% (5% drop), Run 3: 50% (15% drop)
        for i, pass_count in enumerate([7, 6, 5]):
            result_set = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=pass_count)
            metadata = BatchRunMetadata(
                run_id=f"run-{i}",
                timestamp=_BASE + timedelta(days=i),
                dataset_name="test",
                dataset_version="1.0.0",
                agent_version=f"v1.{i}",
            )
            history.add_run(f"run-{i}", metadata, result_set)

        regressions = history.detect_regression(threshold=0.05)

        # Run 2 should be flagged (5% drop from Run 1) but might be on boundary
        # Run 3 should definitely be flagged (15% drop from Run 2)
        assert "run-2" in regressions

    def test_run_history_save_and_load(self):
        """History round-trips to JSON without loss."""
        history = RunHistory(name="test-history")

        for i in range(3):
            result_set = mk_batch_result_set("test", "1.0.0", total_cases=8, passed=5)
            metadata = BatchRunMetadata(
                run_id=f"run-{i}",
                timestamp=_BASE + timedelta(days=i),
                dataset_name="test",
                dataset_version="1.0.0",
                agent_version=f"v{i}",
            )
            history.add_run(f"run-{i}", metadata, result_set)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"
            history.save(path)

            loaded = RunHistory.from_file(path)

            assert loaded.name == "test-history"
            assert len(loaded.entries) == 3
            assert loaded.entries[0].run_id == "run-0"
            assert loaded.entries[1].metadata.agent_version == "v1"

    def test_run_history_get_entry_by_run_id(self):
        """Can retrieve specific entry by run ID."""
        history = RunHistory(name="test")

        for i in range(3):
            result_set = mk_batch_result_set("test", "1.0.0", total_cases=8, passed=5)
            metadata = BatchRunMetadata(
                run_id=f"run-{i}",
                timestamp=_BASE,
                dataset_name="test",
                dataset_version="1.0.0",
                agent_version="v1",
            )
            history.add_run(f"run-{i}", metadata, result_set)

        entry = history.get_entry_by_run_id("run-1")
        assert entry is not None
        assert entry.run_id == "run-1"

        missing = history.get_entry_by_run_id("nonexistent")
        assert missing is None

    def test_run_history_filter_by_agent_version(self):
        """Can filter history by agent version."""
        history = RunHistory(name="test")

        # Add runs with different agent versions
        for i, version in enumerate(["v1.0", "v1.0", "v2.0", "v2.0", "v1.0"]):
            result_set = mk_batch_result_set("test", "1.0.0", total_cases=8)
            metadata = BatchRunMetadata(
                run_id=f"run-{i}",
                timestamp=_BASE + timedelta(days=i),
                dataset_name="test",
                dataset_version="1.0.0",
                agent_version=version,
            )
            history.add_run(f"run-{i}", metadata, result_set)

        v1_entries = history.filter_by_agent_version("v1.0")
        assert len(v1_entries) == 3

        v2_entries = history.filter_by_agent_version("v2.0")
        assert len(v2_entries) == 2

    def test_run_history_filter_by_metadata(self):
        """Can filter by arbitrary metadata fields."""
        history = RunHistory(name="test")

        for i in range(4):
            judge_used = i % 2 == 0
            result_set = mk_batch_result_set("test", "1.0.0", total_cases=8)
            metadata = BatchRunMetadata(
                run_id=f"run-{i}",
                timestamp=_BASE,
                dataset_name="test",
                dataset_version="1.0.0",
                agent_version="v1",
                judge_used=judge_used,
            )
            history.add_run(f"run-{i}", metadata, result_set)

        with_judge = history.filter_by_metadata(judge_used=True)
        assert len(with_judge) == 2

        without_judge = history.filter_by_metadata(judge_used=False)
        assert len(without_judge) == 2


class TestHistoryEdgeCases:
    """Edge cases for history."""

    def test_history_single_run(self):
        """History with one run should work."""
        history = RunHistory(name="single")
        result_set = mk_batch_result_set("test", "1.0.0", total_cases=5)
        metadata = BatchRunMetadata(
            run_id="only",
            timestamp=_BASE,
            dataset_name="test",
            dataset_version="1.0.0",
            agent_version="v1",
        )
        history.add_run("only", metadata, result_set)

        trend = history.get_trend("pass_rate")
        assert len(trend) == 1

        regressions = history.detect_regression()
        assert len(regressions) == 0  # No previous run to regress from

    def test_history_get_trend_various_metrics(self):
        """get_trend works for all supported metrics."""
        history = RunHistory(name="test")

        for i in range(2):
            result_set = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=5 + i)
            metadata = BatchRunMetadata(
                run_id=f"run-{i}",
                timestamp=_BASE + timedelta(days=i),
                dataset_name="test",
                dataset_version="1.0.0",
                agent_version="v1",
            )
            history.add_run(f"run-{i}", metadata, result_set)

        # Test various metrics
        for metric in ["pass_rate", "passed", "failed", "errors", "total"]:
            trend = history.get_trend(metric)
            assert len(trend) == 2
            assert trend[0][0] in ("run-0", "run-1")

    def test_history_get_trend_invalid_metric(self):
        """Invalid metric raises ValueError."""
        history = RunHistory(name="test")
        result_set = mk_batch_result_set("test", "1.0.0", total_cases=5)
        metadata = BatchRunMetadata(
            run_id="run-0",
            timestamp=_BASE,
            dataset_name="test",
            dataset_version="1.0.0",
            agent_version="v1",
        )
        history.add_run("run-0", metadata, result_set)

        with pytest.raises(ValueError, match="Unknown metric"):
            history.get_trend("nonexistent_metric")

    def test_history_load_from_nonexistent_file(self):
        """Loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            RunHistory.from_file("nonexistent.json")


class TestComparisonIntegration:
    """Integration tests combining comparison with history."""

    def test_compare_sequence_of_runs(self):
        """Can compare consecutive runs in a history."""
        history = RunHistory(name="sequence-test")

        # Create 3 runs: baseline, intermediate, final
        pass_counts = [5, 6, 7]
        for i, pass_count in enumerate(pass_counts):
            result_set = mk_batch_result_set("test", "1.0.0", total_cases=10, passed=pass_count)
            metadata = BatchRunMetadata(
                run_id=f"run-{i}",
                timestamp=_BASE + timedelta(days=i),
                dataset_name="test",
                dataset_version="1.0.0",
                agent_version=f"v1.{i}",
            )
            history.add_run(f"run-{i}", metadata, result_set)

        # Compare baseline vs final
        baseline_set = history.entries[0].result_set
        final_set = history.entries[2].result_set
        result = compare(baseline_set, final_set)

        assert abs(result.baseline_pass_rate - 0.5) < 0.01
        assert abs(result.current_pass_rate - 0.7) < 0.01
        assert abs(result.pass_rate_delta - 0.2) < 0.01
        assert result.improved

    def test_case_diff_serialization(self):
        """CaseDiff can be serialized to dict."""
        baseline = mk_batch_result_set("test", "1.0.0", total_cases=3)
        current = mk_batch_result_set("test", "1.0.0", total_cases=3)

        result = compare(baseline, current)
        diff = result.case_diffs[0]

        # Serialize to dict
        diff_dict = diff.model_dump()
        assert "case_index" in diff_dict
        assert "change" in diff_dict

        # Can reconstruct from dict
        reconstructed = CaseDiff(**diff_dict)
        assert reconstructed.case_index == diff.case_index
