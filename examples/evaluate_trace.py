"""Phase 1 gate demo — capture a trace, then evaluate it into a score report.

Runs fully offline: a tiny stub judge stands in for a real LLM judge so you can
see both the rule-based and the LLM-as-judge metrics without a key. Swap in
``autopsy.AnthropicJudge()`` (with ANTHROPIC_API_KEY set) for real judging.

    python examples/evaluate_trace.py
"""

from __future__ import annotations

from autopsy import (
    Expectations,
    SpanKind,
    evaluate,
    span,
    start_run,
    trace,
)


# --- a deliberately flawed agent: it searches twice with the SAME query -------


@trace(kind=SpanKind.TOOL_CALL)
def search(query: str) -> list[str]:
    return [f"https://example.com/{query.replace(' ', '-')}"]


@trace(kind=SpanKind.RETRIEVAL)
def fetch(url: str) -> str:
    return "Paris is the capital of France."


class StubJudge:
    """Stand-in for AnthropicJudge. Says 'goal met, answer grounded'."""

    def score(self, *, system, prompt, schema):
        bool_key = next(k for k, v in schema["properties"].items() if v.get("type") == "boolean")
        return {"score": 0.9, bool_key: True, "rationale": "(stub) supported by context"}


def main() -> None:
    question = "What is the capital of France?"

    with start_run("flawed-agent", input=question, save=False) as run:
        search("capital of France")
        search("capital of France")  # redundant — same args (efficiency will flag)
        ctx = fetch("https://example.com/capital-of-France")
        with span("synthesize", kind=SpanKind.LLM_CALL, inputs={"ctx": ctx}) as sp:
            sp.outputs = {"text": "Paris."}
        run.output = "Paris."

    expected = Expectations(
        goal=question,
        allowed_tools=["search", "fetch"],  # both are registered → no wrong-tool
        expected_tools=["search", "fetch"],
        max_steps=3,                          # we took 4 → budget exceeded
        tool_schemas={
            "search": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            }
        },
    )

    report = evaluate(run, expected=expected, judge=StubJudge())
    print(report.render())
    print("\nfailure labels:", report.failure_labels)


if __name__ == "__main__":
    main()
