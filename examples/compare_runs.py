"""Compare two batch runs: baseline vs. experiment.

This example demonstrates:
  1. Running two different agents (or same agent with different configs) on a dataset
  2. Comparing their BatchResultSet outputs
  3. Interpreting the ComparisonMetrics report
  4. Detecting regressions and improvements

Usage:
    python examples/compare_runs.py
"""

from datetime import datetime, timezone

from autopsy import Dataset, TestCase, Trace
from autopsy.eval.aggregation import BatchResultSet
from autopsy.eval.batch import batch_evaluate
from autopsy.eval.comparison import compare
from autopsy.eval.dataset import Dataset
from autopsy.schema import Span, SpanKind, SpanStatus

# Create a simple dataset for testing
dataset = Dataset(
    name="simple-math",
    version="1.0.0",
    description="Simple math problem solving",
    cases=[
        TestCase(
            input="What is 2 + 2?",
            expected_output="4",
            metadata={"category": "basic_arithmetic"},
        ),
        TestCase(
            input="What is 10 * 5?",
            expected_output="50",
            metadata={"category": "basic_arithmetic"},
        ),
        TestCase(
            input="What is 100 / 2?",
            expected_output="50",
            metadata={"category": "basic_arithmetic"},
        ),
        TestCase(
            input="What is 7 - 3?",
            expected_output="4",
            metadata={"category": "basic_arithmetic"},
        ),
        TestCase(
            input="What is 12 * 12?",
            expected_output="144",
            metadata={"category": "basic_arithmetic"},
        ),
    ],
)


def agent_v1(query: str) -> Trace:
    """Simple agent that tries to solve math problems (baseline)."""
    trace = Trace(name="math-solver-v1", input=query)

    # Simulate solving the problem
    output = None
    if "2 + 2" in query:
        output = "4"
    elif "10 * 5" in query:
        output = "50"
    elif "100 / 2" in query:
        output = "50"
    elif "7 - 3" in query:
        output = "4"
    elif "12 * 12" in query:
        output = "100"  # Wrong! Should be 144
    else:
        output = "unknown"

    span = Span(
        trace_id=trace.trace_id,
        name="solve",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        inputs={"query": query},
        outputs={"answer": output},
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
    )
    trace.spans = [span]
    trace.output = output
    return trace


def agent_v2(query: str) -> Trace:
    """Improved agent with better accuracy (experiment)."""
    trace = Trace(name="math-solver-v2", input=query)

    # Improved logic with fewer mistakes
    output = None
    if "2 + 2" in query:
        output = "4"
    elif "10 * 5" in query:
        output = "50"
    elif "100 / 2" in query:
        output = "50"
    elif "7 - 3" in query:
        output = "4"
    elif "12 * 12" in query:
        output = "144"  # Fixed!
    else:
        output = "unknown"

    span = Span(
        trace_id=trace.trace_id,
        name="solve",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        inputs={"query": query},
        outputs={"answer": output},
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
    )
    trace.spans = [span]
    trace.output = output
    return trace


def main():
    print("=" * 80)
    print("PHASE 4: COMPARING TWO BATCH RUNS")
    print("=" * 80)

    # Run baseline
    print("\n1. Running BASELINE (agent_v1)...")
    baseline_results = batch_evaluate(agent_v1, dataset)
    baseline_set = BatchResultSet.from_batch_results(
        dataset_name="simple-math", dataset_version="1.0.0", batch_results=baseline_results
    )
    print(f"   Baseline: {baseline_set.metrics.passed} passed, {baseline_set.metrics.failed} failed")
    print(f"   Pass rate: {baseline_set.metrics.pass_rate:.1%}")

    # Run experiment
    print("\n2. Running EXPERIMENT (agent_v2)...")
    exp_results = batch_evaluate(agent_v2, dataset)
    exp_set = BatchResultSet.from_batch_results(
        dataset_name="simple-math", dataset_version="1.0.0", batch_results=exp_results
    )
    print(f"   Experiment: {exp_set.metrics.passed} passed, {exp_set.metrics.failed} failed")
    print(f"   Pass rate: {exp_set.metrics.pass_rate:.1%}")

    # Compare
    print("\n3. COMPARING RESULTS...")
    comparison = compare(baseline_set, exp_set)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Baseline pass rate:  {comparison.baseline_pass_rate:.1%}")
    print(f"Current pass rate:   {comparison.current_pass_rate:.1%}")
    print(f"Delta:               {comparison.pass_rate_delta:+.1%}")

    if comparison.improved:
        print(f"✓ IMPROVED: {comparison.improvement_count} case(s) improved")

    if comparison.regression:
        print(f"✗ REGRESSION: {comparison.regression_count} case(s) regressed")

    if comparison.new_failures:
        print(f"\nNew failure causes:")
        for cause, count in comparison.new_failures.items():
            print(f"  - {cause}: {count}")

    if comparison.fixed_failures:
        print(f"\nFixed failure causes:")
        for cause, count in comparison.fixed_failures.items():
            print(f"  - {cause}: {count}")

    print(f"\nDetailed case-by-case changes:")
    for diff in comparison.case_diffs:
        status = "✓" if diff.change == "same" else "↑" if diff.change == "improved" else "↓"
        print(
            f"  Case {diff.case_index}: {diff.change:15s} "
            f"(baseline: {diff.baseline_passed}, current: {diff.current_passed})"
        )

    # Save results for dashboard
    print("\n4. SAVING RESULTS...")
    baseline_set.save("baseline.json")
    exp_set.save("experiment.json")
    print("   ✓ baseline.json")
    print("   ✓ experiment.json")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. View in dashboard:")
    print("   streamlit run src/autopsy/dashboard/app.py -- --mode compare --baseline baseline.json --current experiment.json")
    print("\n2. Or use the history tracking example:")
    print("   python examples/track_trends.py")


if __name__ == "__main__":
    main()
