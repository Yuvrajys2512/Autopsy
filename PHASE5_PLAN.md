# Phase 5 — Dogfood & Launch (Plan)

**Goal (the actual point of the project):** run Autopsy on a *real* agent that isn't ours, produce a writeup with concrete numbers and a failure breakdown, and publish it. The gate is met when a launch post is live with a concrete finding **and** someone other than us stars the repo or opens an issue.

This plan is the bridge from "feature-complete library" (Phases 0–4, 86 tests) to "used and visible tool."

---

## 1. Strategy: two targets, in order

| Order | Target | Why | Risk |
|---|---|---|---|
| **A. Warm-up** | **PaperMind** (ours) | We control it → fast iteration, validates the full capture→evaluate→classify→batch→compare pipeline on a real agent before touching anything public. | Low. "It's mine" doesn't count for the gate, so this is rehearsal only. |
| **B. Headline** | **One public, well-known open-source agent** | This is the launch. A finding about *someone else's* agent is the credible, shareable result. | Higher — must be runnable and instrumentable. Target selection is the key decision (see §2). |

Do A first to de-risk the toolchain, then B for the post.

---

## 2. Choosing the public target (the one real decision)

**Requirements for a good target:**
- Runs on a raw OpenAI/Anthropic SDK loop **or** is thin enough to wrap — Autopsy instruments via `wrap_anthropic` / `wrap_openai` and the `@trace` decorator, not via LangChain callbacks. The less framework magic, the cleaner the trace.
- **Tool-using** (search, fetch, code-exec) — trajectory failures (`wrong_tool_selected`, `hallucinated_tool_args`, `retrieval_miss`, `correct_retrieval_wrong_answer`) only show up when there are tools and multiple steps. A single-shot chat agent has nothing interesting to autopsy.
- Public, popular enough that "I analyzed X" carries weight, and cheap enough to run ~20–30 cases.

**Candidate shortlist:**
1. **smolagents (Hugging Face)** sample agents — small, raw-ish, tool-using, popular. Easy to wrap the underlying model client. Strong headline ("I ran Autopsy on a smolagents agent…").
2. **A raw-SDK ReAct/tool-use agent from the Anthropic cookbook** — minimal, instruments trivially, but less of a "famous third party" name.
3. **gpt-researcher / GPT-Researcher** — well-known, very tool/retrieval-heavy (great for `retrieval_miss` and `correct_retrieval_wrong_answer` findings), but heavier to set up and run.
4. **A LangChain/LangGraph tool agent** — most popular ecosystem, but instrumenting means wrapping the model client beneath LangChain; more friction, validates "framework-agnostic" claim hardest.

**Recommendation:** start with **smolagents** (best runnable-vs-recognizable tradeoff), keep **gpt-researcher** as the stretch target if its retrieval failures produce a juicier headline.

---

## 3. Instrumentation approach

No changes to the target's source beyond wrapping its model client:

```python
from autopsy import start_run, wrap_anthropic   # or wrap_openai

client = wrap_anthropic(anthropic.Anthropic())  # hand this to the agent
# point the public agent at `client`, then:
with start_run("smolagents-search", input=case.input) as run:
    run.output = agent.run(case.input)
# trace auto-saved under .autopsy/traces/
```

If the agent hides its client, fall back to the `@trace(kind=SpanKind.TOOL_CALL)` decorator on its tool functions plus a manual `span()` around each LLM call. Document whichever path we use — that *is* the "framework-agnostic" proof.

---

## 4. Dataset design

Build a small **golden dataset** (~20–30 cases) — enough for a credible distribution, small enough to label by hand and run cheaply.

- **Source tasks** from a public benchmark so they're defensible and reproducible: a subset of **HotpotQA** (multi-hop retrieval → exposes `retrieval_miss` / `correct_retrieval_wrong_answer`) or **GAIA**-style tool tasks. Avoid inventing tasks from scratch — borrowed benchmark tasks are more convincing.
- Each `TestCase` gets: `input`, `expected_output` (or a reference), `expected_tools`, `allowed_tools`, and budgets (`max_steps`, `max_cost_usd`). These drive the rule-based metrics with zero judge cost.
- Save as a `Dataset` JSON, version it `1.0.0`, commit it under `examples/` so the run is reproducible.

## 5. Run & evaluate

```python
from autopsy import batch_evaluate, Dataset, BatchResultSet, AnthropicJudge

dataset = Dataset.from_file("examples/hotpot_subset.json")
results = batch_evaluate(public_agent, dataset, judge=AnthropicJudge())  # judge adds goal_completion + faithfulness
rs = BatchResultSet.from_batch_results("hotpot_subset", "1.0.0", results)
rs.save("public_agent_run.json")
```

Then classify every failing case (`LLMFailureClassifier` for the nuanced ones) and read off `rs.metrics.failure_distribution`. Open the dashboard (`--mode single`) for the screenshot/GIF the README needs.

**Bonus comparison angle:** run the same dataset against the agent with a tweaked prompt or a different model, then `compare()` the two BatchResultSets — a regression/improvement diff is a second, stronger story ("changing X fixed 40% of the arg-hallucinations").

## 6. The writeup

Lead with the *finding*, not the features. Target sentence (fill in real numbers):

> *"I ran Autopsy on \<smolagents / gpt-researcher\> over 25 HotpotQA tasks and found N% of its failures were \<silent tool-arg hallucinations / correct-retrieval-wrong-answer\> that a plain pass/fail score never surfaces."*

Structure: the headline finding → the failure-distribution chart → 1–2 annotated trace drill-downs (show the culprit span) → the `pip install` + 10-line quickstart → link to repo. Publish to LinkedIn first (existing audience), then Show HN / the target framework's community.

---

## 7. Gate checklist (from the brief)

- [ ] `pip install autopsy` works for a stranger — **ready** (built, `twine check` passes, clean-room install verified). Needs the actual PyPI upload.
- [ ] A stranger can wrap their agent → eval report in <15 min from the README alone.
- [ ] README leads with a GIF/screenshot of the dashboard failure breakdown — **produce during the run**.
- [ ] Run on ≥1 real agent that isn't ours, with published numbers.
- [ ] Launch post live with a concrete finding.
- [ ] **True signal:** someone else stars it or opens an issue, unprompted.

## 8. Suggested sequence

1. Publish to PyPI (one `twine upload`, see README/packaging).
2. Warm-up dogfood on PaperMind → fix any rough edges the pipeline exposes.
3. Stand up the golden dataset (HotpotQA subset).
4. Instrument + run the public target; classify failures; capture the dashboard GIF.
5. (Optional) prompt/model comparison run for a second story.
6. Write + publish the post, leading with the finding.
