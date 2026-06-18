"""Track improvement trends across multiple runs using RunHistory.

This example demonstrates:
  1. Creating a RunHistory to track multiple batch runs over time
  2. Adding runs with metadata (version, timestamp, notes)
  3. Extracting pass-rate trends
  4. Detecting regressions
  5. Filtering by agent version or custom metadata

Usage:
    python examples/track_trends.py
"""

from datetime import datetime, timedelta, timezone

from autopsy import Dataset, TestCase, Trace
from autopsy.eval.aggregation import BatchResultSet
from autopsy.eval.batch import batch_evaluate
from autopsy.eval.history import BatchRunMetadata, RunHistory
from autopsy.schema import Span, SpanKind, SpanStatus

# Create a test dataset
dataset = Dataset(
    name="qa-basic",
    version="1.0.0",
    description="Basic Q&A dataset for tracking agent improvements",
    cases=[
        TestCase(input="What is the capital of France?", expected_output="Paris"),
        TestCase(input="What is 2+2?", expected_output="4"),
        TestCase(input="Who wrote Romeo and Juliet?", expected_output="Shakespeare"),
        TestCase(input="What is the largest planet?", expected_output="Jupiter"),
        TestCase(input="What is H2O?", expected_output="Water"),
        TestCase(input="What is photosynthesis?", expected_output="Plant process"),
        TestCase(input="What is the speed of light?", expected_output="3e8 m/s"),
        TestCase(input="What is quantum mechanics?", expected_output="Physics"),
        TestCase(input="What is gravity?", expected_output="Force"),
        TestCase(input="What is DNA?", expected_output="Genetic material"),
    ],
)


class SimpleQAAgent:
    """A simple Q&A agent whose accuracy improves over versions."""

    def __init__(self, version: str, accuracy_level: float = 0.5):
        self.version = version
        self.accuracy_level = accuracy_level

    def __call__(self, query: str) -> Trace:
        trace = Trace(name=f"qa-agent-{self.version}", input=query)

        # Simulate varying accuracy based on version
        answers = {
            "France": "Paris",
            "2+2": "4",
            "Romeo": "Shakespeare",
            "largest": "Jupiter",
            "H2O": "Water",
            "photosynthesis": "Plant process",
            "speed": "3e8 m/s",
            "quantum": "Quantum mechanics",
            "gravity": "Force",
            "DNA": "DNA",
        }

        # Find best matching answer
        output = "Unknown"
        for key, answer in answers.items():
            if key.lower() in query.lower():
                # Version-dependent accuracy: higher versions get more correct
                import random

                random.seed(hash(query + self.version))
                if random.random() < self.accuracy_level:
                    output = answer
                else:
                    output = f"Wrong answer for {key}"
                break

        span = Span(
            trace_id=trace.trace_id,
            name="answer",
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
    print("PHASE 4: TRACKING TRENDS ACROSS MULTIPLE RUNS")
    print("=" * 80)

    history = RunHistory(name="qa-agent-evolution")
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Simulate multiple runs with improving accuracy over time
    versions = [
        ("v1.0", 0.50, "Initial baseline"),
        ("v1.1", 0.60, "Bug fixes in answer matching"),
        ("v1.2", 0.65, "Improved prompt engineering"),
        ("v2.0", 0.75, "New model backbone"),
        ("v2.1", 0.78, "Fine-tuning on domain data"),
    ]

    print(f"\nRunning {len(versions)} versions of the agent...")
    for i, (version, accuracy, notes) in enumerate(versions):
        print(f"\n{i + 1}. Running {version} (accuracy: {accuracy:.0%})...")

        agent = SimpleQAAgent(version=version, accuracy_level=accuracy)
        results = batch_evaluate(agent, dataset)
        result_set = BatchResultSet.from_batch_results(
            dataset_name="qa-basic", dataset_version="1.0.0", batch_results=results
        )

        metadata = BatchRunMetadata(
            run_id=version,
            timestamp=base_time + timedelta(days=i * 7),
            dataset_name="qa-basic",
            dataset_version="1.0.0",
            agent_version=version,
            judge_used=False,
            notes=notes,
        )

        history.add_run(version, metadata, result_set)
        print(f"   Pass rate: {result_set.metrics.pass_rate:.1%} " f"({result_set.metrics.passed}/{result_set.total_cases})")

    # Extract and display trends
    print("\n" + "=" * 80)
    print("TREND ANALYSIS")
    print("=" * 80)

    trend = history.get_trend("pass_rate")
    print("\nPass rate trend over time:")
    print("Version  | Pass Rate | Status")
    print("-" * 40)
    for run_id, pass_rate in trend:
        status = "→" if pass_rate > 0.5 else "=" if pass_rate == 0.5 else "↓"
        print(f"{run_id:7s} | {pass_rate:8.1%} | {status}")

    # Detect regressions
    print("\nRegression detection (threshold: 5%):")
    regressions = history.detect_regression(threshold=0.05)
    if regressions:
        print(f"  ✗ Regressions detected in: {', '.join(regressions)}")
    else:
        print("  ✓ No regressions detected")

    # Filter by agent version
    print("\n" + "=" * 80)
    print("FILTERING AND QUERIES")
    print("=" * 80)

    v1_runs = history.filter_by_agent_version("v1.1")
    print(f"\nRuns with agent_version='v1.1': {len(v1_runs)}")
    for entry in v1_runs:
        print(f"  - {entry.run_id}: {entry.metadata.notes}")

    # Get specific entry
    v2_run = history.get_entry_by_run_id("v2.0")
    if v2_run:
        print(f"\nDetails for v2.0:")
        print(f"  Timestamp: {v2_run.metadata.timestamp}")
        print(f"  Pass rate: {v2_run.result_set.metrics.pass_rate:.1%}")
        print(f"  Notes: {v2_run.metadata.notes}")

    # Save history
    print("\n" + "=" * 80)
    print("SAVING HISTORY")
    print("=" * 80)
    history.save("qa_agent_history.json")
    print("✓ Saved to qa_agent_history.json")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. View history in dashboard:")
    print("   streamlit run src/autopsy/dashboard/app.py -- --mode history --history qa_agent_history.json")
    print("\n2. Compare specific runs from history:")
    print("   from autopsy.eval.history import RunHistory")
    print("   from autopsy.eval.comparison import compare")
    print("   history = RunHistory.from_file('qa_agent_history.json')")
    print("   v1 = history.get_entry_by_run_id('v1.0').result_set")
    print("   v2 = history.get_entry_by_run_id('v2.0').result_set")
    print("   result = compare(v1, v2)")
    print("   print(result.summary_dict())")


if __name__ == "__main__":
    main()
