"""Phase 0 gate demo — wrap a (fake) agent, run it once, save a trace, read it back.

Runs with no API key and no network: the "LLM call" is simulated so you can see
the full trajectory — tool calls, a retry, a nested LLM span with token usage and
cost — captured and persisted as a structured trace on disk.

    python examples/mock_agent.py
"""

from __future__ import annotations

from autopsy import (
    SpanKind,
    TokenUsage,
    list_traces,
    load_trace,
    span,
    start_run,
    trace,
)
from autopsy.pricing import compute_cost


@trace(kind=SpanKind.TOOL_CALL)
def search_web(query: str) -> list[str]:
    """A tool the agent calls. Decorated → becomes a tool_call span."""
    return [f"https://example.com/{query.replace(' ', '-')}"]


@trace(kind=SpanKind.RETRIEVAL)
def fetch_page(url: str) -> str:
    return f"<contents of {url}>"


def fake_llm_call(prompt: str) -> str:
    """Stand-in for a real LLM call, recorded as an llm_call span with usage/cost."""
    with span("mock.llm", kind=SpanKind.LLM_CALL, inputs={"prompt": prompt}) as sp:
        sp.model = "claude-opus-4-8"
        sp.usage = TokenUsage(input_tokens=1200, output_tokens=180)
        sp.cost_usd = compute_cost(sp.model, sp.usage)
        answer = "France won the 2018 FIFA World Cup."
        sp.outputs = {"text": answer}
        return answer


def main() -> None:
    question = "who won the 2018 world cup?"

    with start_run("mock-research-agent", input=question) as run:
        hits = search_web("2018 world cup winner")

        # Simulate a transient failure + retry to exercise error capture.
        with span("flaky-fetch", kind=SpanKind.RETRY) as retry_span:
            try:
                raise ConnectionError("upstream timed out")
            except ConnectionError as exc:
                retry_span.add_event("retrying", reason=str(exc))

        page = fetch_page(hits[0])
        answer = fake_llm_call(f"Using {page}, answer: {question}")
        run.output = answer

    trace_id = run.trace_id
    print(f"\nSaved trace {trace_id}")
    print(f"  spans:       {len(run.spans)}")
    print(f"  latency_ms:  {run.latency_ms:.1f}")
    print(f"  tokens:      {run.total_tokens}")
    print(f"  cost (usd):  ${run.total_cost_usd:.6f}")

    print("\nReading it back from disk:")
    reloaded = load_trace(trace_id)
    for sp in reloaded.spans:
        status = sp.status.value
        ms = f"{sp.latency_ms:.1f}ms" if sp.latency_ms is not None else "?"
        print(f"  [{status:5}] {sp.kind.value:10} {sp.name:24} {ms}")

    print(f"\nTotal traces on disk: {len(list_traces())}")


if __name__ == "__main__":
    main()
