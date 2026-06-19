# Autopsy

> Capture an agent's full execution trace and find out not just **whether** it failed, but **how and why** — at the trajectory level, not just the final answer.

Autopsy is a lightweight, framework-agnostic Python library for evaluating and diagnosing LLM agents. Most eval tools score the *output* ("is this answer good?"). Autopsy looks at the *trajectory* — did the agent pick the right tool, hallucinate tool arguments, recover from an error, loop, or blow its step budget — and is being built toward **automatic failure classification**: handing you *why it broke*, with evidence, instead of a lonely number.

## Status

| Phase | What | State |
|---|---|---|
| **0 — Capture** | Trace/Span schema, instrumentation, client wrappers, local persistence | ✅ done |
| **1 — Evaluators** | rule-based + LLM-as-judge metrics, `evaluate()` report | ✅ done |
| **2 — Failure classifier** | hybrid LLM + rules root-cause diagnosis with evidence span — *the differentiator* | ✅ done |
| **3 — Datasets, batch runs & dashboard** | labeled test cases, batch runner, aggregates, Streamlit dashboard | ✅ done |
| **4 — Comparison & history** | run-to-run diff, regression detection, trend tracking, compare/history dashboard modes | ✅ done |
| 5 — Dogfood & ship | PyPI publish, run on a real public agent, launch writeup | ⏳ next |

86 tests passing. The core library is feature-complete; what remains is packaging and a real-world dogfood run.

## Install

```bash
pip install autopsy            # core (pydantic only)
pip install "autopsy[anthropic]"   # + Anthropic client wrapping & built-in judge
pip install "autopsy[openai]"      # + OpenAI client wrapping
pip install "autopsy[dashboard]"   # + Streamlit dashboard
```

Development:

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

## Evaluate a trace

`evaluate(trace)` scores the *trajectory*, not just the answer — and tells you what kind of failure it was, with evidence. Rule-based metrics (tool selection, argument validity, efficiency/loops, error recovery, resource budgets) run with no API calls. Pass a `judge` to add LLM-as-judge metrics (goal completion, faithfulness/grounding).

```python
from autopsy import evaluate, Expectations

report = evaluate(
    trace,
    expected=Expectations(
        allowed_tools=["search", "fetch"],
        expected_tools=["search", "fetch"],
        max_steps=3,
    ),
)

print(report.render())
print(report.overall_score, report.passed, report.failure_labels)
# e.g. failure_labels -> ['redundant_calls', 'budget_exceeded']
```

Add model-graded metrics by supplying a judge (bring your own, or use the built-in Anthropic one):

```python
from autopsy import evaluate, AnthropicJudge

report = evaluate(trace, judge=AnthropicJudge())  # adds goal_completion + faithfulness
```

A `Judge` is any object with a `score(*, system, prompt, schema) -> dict` method, so you can plug in any provider or a local model.

## Diagnose *why* it failed (the differentiator)

A score tells you a run was bad. A **classification** tells you *what broke*, *why*, and *which span did it*. Feed an `EvalReport` and its trace to a classifier:

```python
from autopsy import RuleBasedFailureClassifier, evaluate

report = evaluate(trace, expected=expectations)
diagnosis = RuleBasedFailureClassifier().classify(report, trace)

print(diagnosis.root_cause)      # e.g. 'hallucinated_tool_args'
print(diagnosis.rationale)       # plain-English explanation
print(diagnosis.culprit_span_id) # the exact span that caused it
```

`RuleBasedFailureClassifier` is deterministic and free (no API calls). For nuanced cases — `correct_retrieval_wrong_answer`, `gave_up_early` — use `LLMFailureClassifier(judge=AnthropicJudge())`, which feeds the judge the trace plus the failure taxonomy and returns the same `FailureClassification` shape.

## Datasets, batch runs & dashboard

Define a suite of cases, run your agent over all of them, and aggregate into a saved result set:

```python
from autopsy import Dataset, TestCase, batch_evaluate, BatchResultSet

dataset = Dataset(name="benchmark", cases=[
    TestCase(input="who won the 2018 world cup?", expected_output="France"),
    # ...
])

results = batch_evaluate(my_agent, dataset)
result_set = BatchResultSet.from_batch_results("benchmark", "1.0.0", results)
result_set.save("results.json")

print(result_set.metrics.pass_rate)            # 0.78
print(result_set.metrics.failure_distribution) # {'redundant_calls': 3, ...}
```

Explore it visually with the Streamlit dashboard (`pip install "autopsy[dashboard]"`):

```bash
streamlit run src/autopsy/dashboard/app.py -- --mode single --data results.json
```

You get aggregate pass rate, a failure-cause breakdown, and drill-down into any individual trace.

## Compare runs & track regressions

Changed a prompt or model? Diff two batch runs to catch regressions before they ship:

```python
from autopsy import compare

cmp = compare(baseline_set, experiment_set)
print(f"Pass rate: {cmp.baseline_pass_rate:.0%} → {cmp.current_pass_rate:.0%} ({cmp.pass_rate_delta:+.1%})")
if cmp.regression:
    print("Regressions:", [d.case_index for d in cmp.case_diffs if d.change == "regressed"])
print("New failure modes:", cmp.new_failures)   # causes that appeared
print("Fixed:", cmp.fixed_failures)             # causes that disappeared
```

Track many runs over time and detect drops automatically:

```python
from autopsy import RunHistory, BatchRunMetadata
from datetime import datetime

history = RunHistory(name="agent-experiments")
history.add_run("v1.0", BatchRunMetadata(run_id="v1.0", timestamp=datetime.now(),
                dataset_name="benchmark", dataset_version="1.0.0",
                agent_version="v1.0", judge_used=False, notes="baseline"), baseline_set)
# ... add more runs ...

print(history.get_trend("pass_rate"))      # [('v1.0', 0.72), ('v2.0', 0.81), ...]
print(history.detect_regression(threshold=0.05))  # run_ids that dropped > 5%
history.save("experiment_history.json")
```

Both have dashboard views:

```bash
streamlit run src/autopsy/dashboard/app.py -- --mode compare --baseline baseline.json --current current.json
streamlit run src/autopsy/dashboard/app.py -- --mode history --history experiment_history.json
```

## Examples

Runnable end-to-end demos live in [`examples/`](examples/) — each works with a mock agent and rule-based classifier, so no API key is required:

| File | Shows |
|---|---|
| `mock_agent.py` / `anthropic_agent.py` | Capturing a trace (Phase 0) |
| `evaluate_trace.py` | Scoring a trajectory (Phase 1) |
| `classify_failure.py` | Root-cause diagnosis with evidence (Phase 2) |
| `batch_run.py` | Dataset + batch run + aggregation (Phase 3) |
| `compare_runs.py` / `track_trends.py` | Regression diff & trend tracking (Phase 4) |

## Documentation

Deep-dives for each phase — what was built and why, the theory, the architecture, and a code walkthrough — live under [`documentation/`](documentation/).
