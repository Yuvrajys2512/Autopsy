# Autopsy

> Capture an agent's full execution trace and find out not just **whether** it failed, but **how and why** — at the trajectory level, not just the final answer.

Autopsy is a lightweight, framework-agnostic Python library for evaluating and diagnosing LLM agents. Most eval tools score the *output* ("is this answer good?"). Autopsy looks at the *trajectory* — did the agent pick the right tool, hallucinate tool arguments, recover from an error, loop, or blow its step budget — and is being built toward **automatic failure classification**: handing you *why it broke*, with evidence, instead of a lonely number.

This is an early, in-progress build.
let
## Status

| Phase | What | State |
|---|---|---|
| **0 — Capture** | Trace/Span schema, instrumentation, client wrappers, local persistence | ✅ in progress |
| 1 — Evaluators | rule-based + LLM-as-judge metrics | ⏳ |
| 2 — Failure classifier | the differentiator | ⏳ |
| 3 — Datasets, batch runs & dashboard | Streamlit dashboard | ⏳ |
| 4 — Regression diff, polish & ship | PyPI, docs | ⏳ |

## Install (dev)

```bash
uv sync --extra dev
```

## Quickstart

Wrap any agent run in `start_run`, decorate the steps you care about, and you get a complete structured trace saved to disk.

```python
from autopsy import start_run, trace, SpanKind

@trace(kind=SpanKind.TOOL_CALL)
def search(query: str) -> list[str]:
    return ["result for " + query]

with start_run("demo-agent", input="who won the 2018 world cup?") as run:
    hits = search("2018 world cup winner")
    run.output = hits[0]

# -> trace saved under .autopsy/traces/<trace_id>.json
```

Automatically capture LLM calls by wrapping your client:

```python
import anthropic
from autopsy import start_run, wrap_anthropic

client = wrap_anthropic(anthropic.Anthropic())

with start_run("llm-agent", input="explain quantum tunnelling") as run:
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": "explain quantum tunnelling"}],
    )
    run.output = resp.content[0].text
```

The saved trace records every span with inputs, outputs, latency, token usage, cost, and any error — in an OpenTelemetry-compatible shape.

## Reading a trace

```python
from autopsy import load_trace, list_traces

for meta in list_traces():
    print(meta["trace_id"], meta["name"], meta["status"])

trace = load_trace("<trace_id>")
print(trace.latency_ms, trace.total_tokens, trace.total_cost_usd)
for span in trace.spans:
    print(span.kind, span.name, span.latency_ms, span.status)
```
