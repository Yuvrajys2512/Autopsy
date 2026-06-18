"""Real LLM capture with the Anthropic SDK.

Requires ``ANTHROPIC_API_KEY`` and ``pip install 'autopsy[anthropic]'``.

    python examples/anthropic_agent.py
"""

from __future__ import annotations

import anthropic

from autopsy import SpanKind, load_trace, start_run, trace, wrap_anthropic

client = wrap_anthropic(anthropic.Anthropic())


@trace(kind=SpanKind.TOOL_CALL)
def lookup(topic: str) -> str:
    # A real tool would do something here; this is a placeholder.
    return f"reference notes about {topic}"


def main() -> None:
    question = "In one sentence, what is quantum tunnelling?"

    with start_run("anthropic-agent", input=question) as run:
        notes = lookup("quantum tunnelling")
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=256,
            messages=[
                {"role": "user", "content": f"{notes}\n\n{question}"},
            ],
        )
        run.output = resp.content[0].text

    reloaded = load_trace(run.trace_id)
    print(f"Saved trace {run.trace_id}")
    print(f"  tokens: {reloaded.total_tokens}  cost: ${reloaded.total_cost_usd:.6f}")
    for sp in reloaded.spans:
        print(f"  {sp.kind.value:10} {sp.name:32} {sp.status.value}")
    print(f"\nAnswer: {run.output}")


if __name__ == "__main__":
    main()
