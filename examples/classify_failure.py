"""Phase 2 gate demo — evaluate a trace, then classify its failure.

Shows the full evaluate + classify pipeline:
1. capture + evaluate → EvalReport with failure_labels and details
2. classify → root-cause diagnosis with rationale and culprit span

Run with:
    python examples/classify_failure.py

(Uses a stub judge so it works without an API key.)
"""

from __future__ import annotations

from autopsy import (
    Expectations,
    LLMFailureClassifier,
    RuleBasedFailureClassifier,
    SpanKind,
    evaluate,
    span,
    start_run,
    trace,
)


# --- A deliberately flawed agent: wrong tool + redundant calls -------


@trace(kind=SpanKind.TOOL_CALL)
def search(query: str) -> list[str]:
    return [f"https://example.com/{query.replace(' ', '-')}"]


@trace(kind=SpanKind.TOOL_CALL)
def fetch(url: str) -> str:
    return "Paris is the capital of France."


class StubJudge:
    """Stand-in for AnthropicJudge for demo purposes."""

    def score(self, *, system, prompt, schema):
        # In a real scenario, the judge would reason over the failure signals.
        # For this demo, we identify the primary issue from the signals.
        # In practice, the LLM would see the trace and outputs and reason holistically.
        return {
            "root_cause": "wrong_tool_selected",
            "rationale": "The agent called 'delete_db' which is not in the allowed tool set.",
            "culprit_span_index": 0,
            "confidence": 0.92,
        }


def main() -> None:
    question = "What is the capital of France?"

    # Capture a trace with multiple issues:
    # 1. Wrong tool call (delete_db is not allowed)
    # 2. Redundant search calls
    # 3. Overall goal is completed (so it's not a complete failure)
    with start_run("flawed-agent", input=question, save=False) as run:
        # First issue: calling a tool not in the allowed set
        with span("delete_db", kind=SpanKind.TOOL_CALL, inputs={"table": "users"}):
            pass

        # Second issue: redundant search calls
        search("capital of France")
        search("capital of France")  # same query again

        # Good step: fetch the result
        ctx = fetch("https://example.com/capital-of-France")
        with span("synthesize", kind=SpanKind.LLM_CALL, inputs={"ctx": ctx}) as sp:
            sp.outputs = {"text": "Paris."}
        run.output = "Paris."

    # Define expectations: what tools should be available, what's the goal?
    expected = Expectations(
        goal=question,
        allowed_tools=["search", "fetch"],  # delete_db is NOT allowed → wrong_tool
        expected_tools=["search", "fetch"],
        tool_schemas={
            "search": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            }
        },
    )

    # --- Phase 1: Evaluate the trace -------
    report = evaluate(run, expected=expected, judge=None)
    print("=" * 70)
    print("PHASE 1: EVALUATION REPORT")
    print("=" * 70)
    print(report.render())
    print()

    # --- Phase 2: Classify the failure -------
    print("=" * 70)
    print("PHASE 2: FAILURE CLASSIFICATION")
    print("=" * 70)

    # Use rule-based classifier (no LLM call needed, fast).
    print("\n1. Rule-based classifier (heuristic):")
    rule_classifier = RuleBasedFailureClassifier()
    rule_diagnosis = rule_classifier.classify(report, run)
    print(f"   Root cause: {rule_diagnosis.root_cause}")
    print(f"   Rationale: {rule_diagnosis.rationale}")
    print(f"   Confidence: {rule_diagnosis.confidence:.0%}")
    print(f"   Evidence labels: {rule_diagnosis.evidence_labels}")
    if rule_diagnosis.culprit_span_id:
        print(f"   Culprit span: {rule_diagnosis.culprit_span_id}")

    # Alternatively: use LLM classifier with a stub judge.
    print("\n2. LLM classifier (with stub judge for demo):")
    judge = StubJudge()
    llm_classifier = LLMFailureClassifier(judge=judge)
    llm_diagnosis = llm_classifier.classify(report, run)
    print(f"   Root cause: {llm_diagnosis.root_cause}")
    print(f"   Rationale: {llm_diagnosis.rationale}")
    print(f"   Confidence: {llm_diagnosis.confidence:.0%}")
    print(f"   Evidence labels: {llm_diagnosis.evidence_labels}")
    if llm_diagnosis.culprit_span_id:
        # Find the span by ID and print its details.
        for sp in run.spans:
            if sp.span_id == llm_diagnosis.culprit_span_id:
                print(f"   Culprit span: {sp.name} (step with wrong tool)")
                break

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
The agent ran and produced an output, but we detected:
  * {len(report.failure_labels)} distinct failure signals
  * Root cause: {rule_diagnosis.root_cause}

This run demonstrates the Phase 2 differentiator:
Instead of just a failing score, we provide:
  [+] A labeled failure cause (from the taxonomy)
  [+] Evidence: which metrics failed and why
  [+] A culprit span: the exact step responsible

For a human (or downstream tooling), this is far more actionable than
"score: 0.73, passed: false" — you now know exactly what to fix.
""")


if __name__ == "__main__":
    main()
