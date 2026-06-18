"""Phase 3 gate demo — batch evaluation, aggregation, and results export.

Shows the full Phase 3 pipeline:
1. Create or load a Dataset of test cases
2. Run batch_evaluate() against an agent
3. Aggregate results with BatchResultSet
4. Save for dashboard exploration
5. Display summary stats

Run with:
    python examples/batch_run.py

(Uses a mock agent and rule-based classifier so it works without API keys.)
"""

from __future__ import annotations

from autopsy import (
    Dataset,
    RuleBasedFailureClassifier,
    SpanKind,
    TestCase,
    batch_evaluate,
    span,
    start_run,
)
from autopsy.eval.aggregation import BatchResultSet


# --- Mock agent that shows different behaviors ---


def simple_qa_agent(question: str):
    """A simple QA agent with different behaviors per input."""
    with start_run("qa-agent", input=question, save=False) as run:
        # Simulate a tool call
        with span("search", kind=SpanKind.TOOL_CALL, inputs={"query": question}) as s:
            s.outputs = {"results": [f"result for {question}"]}

        # Simulate LLM synthesis
        with span("synthesize", kind=SpanKind.LLM_CALL) as s:
            s.outputs = {"answer": f"Answer to: {question}"}

        run.output = f"Answer to: {question}"
    return run


def main() -> None:
    print("=" * 70)
    print("PHASE 3: BATCH EVALUATION & AGGREGATION")
    print("=" * 70)
    print()

    # --- 1. Create a dataset ---
    print("Step 1: Creating a test dataset...")
    dataset = Dataset(
        name="QA Basics",
        version="1.0.0",
        description="Simple question-answering test cases",
        source="hand-crafted",
        cases=[
            TestCase(
                input="What is the capital of France?",
                expected_output="Paris",
                expected_tools=["search"],
                allowed_tools=["search"],
                metadata={"difficulty": "easy"},
            ),
            TestCase(
                input="Who wrote Romeo and Juliet?",
                expected_output="William Shakespeare",
                expected_tools=["search"],
                allowed_tools=["search"],
                metadata={"difficulty": "easy"},
            ),
            TestCase(
                input="What is 2+2?",
                expected_output="4",
                expected_tools=["calculator"],
                allowed_tools=["calculator"],
                metadata={"difficulty": "trivial"},
            ),
            TestCase(
                input="Explain quantum entanglement.",
                expected_output="A quantum phenomenon where particles are correlated",
                expected_tools=["search"],
                allowed_tools=["search"],
                metadata={"difficulty": "hard"},
            ),
        ],
    )
    print(f"  [ok] Created dataset with {len(dataset)} test cases")
    print()

    # --- 2. Run batch evaluation ---
    print("Step 2: Running batch evaluation...")
    classifier = RuleBasedFailureClassifier()
    results = batch_evaluate(
        agent_fn=simple_qa_agent,
        dataset=dataset,
        judge=None,
        classifier=classifier,
    )
    print(f"  [ok] Evaluated {len(results)} cases")
    print()

    # --- 3. Aggregate results ---
    print("Step 3: Aggregating results...")
    result_set = BatchResultSet.from_batch_results(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        batch_results=results,
    )
    print(f"  [ok] Aggregated metrics:")
    print(f"    - Total: {result_set.metrics.total}")
    print(f"    - Passed: {result_set.metrics.passed}")
    print(f"    - Failed: {result_set.metrics.failed}")
    print(f"    - Errors: {result_set.metrics.errors}")
    print(f"    - Pass Rate: {result_set.metrics.pass_rate:.0%}")
    if result_set.metrics.top_failure_cause:
        print(f"    - Top Failure: {result_set.metrics.top_failure_cause} ({result_set.metrics.top_failure_count} cases)")
    print()

    # --- 4. Save results ---
    print("Step 4: Saving results...")
    output_file = "batch_run_results.json"
    result_set.save(output_file)
    print(f"  [ok] Saved to {output_file}")
    print()

    # --- 5. Display per-case breakdown ---
    print("Step 5: Per-case breakdown")
    print()
    print("  Case | Status  | Score | Root Cause              | Confidence")
    print("  " + "-" * 65)
    for i, case_result in enumerate(result_set.results):
        status = "[PASS]" if case_result.passed else "[FAIL]" if case_result.failed else "[ERR]"
        score = f"{case_result.overall_score:.2f}" if case_result.overall_score else "   —"
        root = case_result.root_cause or "—"
        conf = f"{case_result.confidence:.0%}" if case_result.confidence else "—"
        print(f"  {i:2d}  | {status} | {score:5} | {root:22} | {conf:10}")
    print()

    # --- 6. Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Dataset: {dataset.name} (v{dataset.version})
  Cases: {result_set.total_cases}
  Pass Rate: {result_set.metrics.pass_rate:.0%}

Failure Distribution:
{chr(10).join(f'  {label}: {count} cases' for label, count in result_set.metrics.failure_distribution.items()) if result_set.metrics.failure_distribution else '  (none)'}

To explore the results in a dashboard:
  streamlit run src/autopsy/dashboard/app.py -- --data {output_file}

To use these results programmatically:
  from autopsy.eval.aggregation import BatchResultSet
  rs = BatchResultSet.from_file("{output_file}")
  print(rs.metrics.pass_rate)
""")


if __name__ == "__main__":
    main()
