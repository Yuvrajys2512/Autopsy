"""Phase 5 — dogfood Autopsy on a smolagents web-search agent.

This is the headline run: point Autopsy at a real, third-party tool-using agent
(Hugging Face's smolagents) and produce a failure breakdown over the multihop_qa
dataset.

Two modes, chosen automatically:

  REAL  — if `smolagents` is importable AND ANTHROPIC_API_KEY is set, build a
          smolagents ToolCallingAgent with the default web_search + visit_webpage
          tools, instrument its tools (and, best-effort, its model) so every run
          is captured as an Autopsy Trace, then evaluate the whole dataset with
          the LLM judge + LLM classifier.

  MOCK   — otherwise, run a built-in mock agent that emulates the smolagents
          toolset and a realistic spread of trajectory failures, so the full
          capture -> evaluate -> classify -> aggregate pipeline runs end-to-end
          with no dependencies and produces a saved BatchResultSet for the
          dashboard. Use --mock to force this even when smolagents is present.

Usage:
    # real run
    pip install "smolagents[toolkit]" anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/dogfood_smolagents.py

    # offline pipeline validation (no deps, no key)
    python examples/dogfood_smolagents.py --mock

Output:
    examples/datasets/runs/smolagents_run.json   (a BatchResultSet)
    Explore it:  streamlit run src/autopsy/dashboard/app.py -- --mode single --data examples/datasets/runs/smolagents_run.json
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autopsy import (
    AnthropicJudge,
    BatchResultSet,
    Dataset,
    LLMFailureClassifier,
    RuleBasedFailureClassifier,
    Span,
    SpanKind,
    SpanStatus,
    Trace,
    batch_evaluate,
    span,
    start_run,
)

DATASET_PATH = Path(__file__).parent / "datasets" / "multihop_qa.json"
OUT_PATH = Path(__file__).parent / "datasets" / "runs" / "smolagents_run.json"
MODEL_ID = "anthropic/claude-opus-4-8"


# --------------------------------------------------------------------------- #
# REAL: instrument a smolagents agent so each run becomes an Autopsy Trace.
# --------------------------------------------------------------------------- #
def _instrument_tool(tool) -> None:
    """Wrap a smolagents Tool so each invocation emits a TOOL_CALL span.

    smolagents Tool.__call__ delegates to Tool.forward, so wrapping the bound
    `forward` captures the call regardless of how the agent invokes it. The span
    name is the tool's registered name (e.g. "web_search"), which is what the
    rule evaluators key on for tool-selection and redundancy.
    """
    name = getattr(tool, "name", tool.__class__.__name__)
    inner = tool.forward

    @functools.wraps(inner)
    def wrapped(*args, **kwargs):
        inputs = dict(kwargs)
        if args:
            inputs["args"] = list(args)
        with span(name, kind=SpanKind.TOOL_CALL, inputs=inputs) as sp:
            result = inner(*args, **kwargs)
            sp.outputs = {"result": str(result)[:4000]}
            return result

    tool.forward = wrapped


def _instrument_model(model) -> None:
    """Best-effort: wrap the model's `generate` so LLM calls emit LLM_CALL spans.

    Dunder `__call__` can't be overridden per-instance (Python looks it up on the
    type), so we wrap `generate`, which recent smolagents models route through.
    If the attribute isn't present we skip it — tool spans alone already drive
    every rule metric; LLM spans only add token/cost enrichment.
    """
    inner = getattr(model, "generate", None)
    if inner is None:
        return

    @functools.wraps(inner)
    def wrapped(*args, **kwargs):
        with span("llm_call", kind=SpanKind.LLM_CALL) as sp:
            msg = inner(*args, **kwargs)
            sp.model = getattr(model, "model_id", None)
            usage = getattr(msg, "token_usage", None)
            if usage is not None:
                from autopsy import TokenUsage

                sp.usage = TokenUsage(
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                )
            return msg

    model.generate = wrapped


def build_real_agent_fn():
    """Build an instrumented smolagents agent; return an agent_fn(input)->Trace."""
    from smolagents import (  # type: ignore
        LiteLLMModel,
        ToolCallingAgent,
        VisitWebpageTool,
        WebSearchTool,
    )

    model = LiteLLMModel(model_id=MODEL_ID)
    _instrument_model(model)

    tools = [WebSearchTool(), VisitWebpageTool()]
    for t in tools:
        _instrument_tool(t)

    agent = ToolCallingAgent(tools=tools, model=model, max_steps=6)

    def agent_fn(question: str) -> Trace:
        with start_run("smolagents", input=question) as run:
            # reset=True clears memory between cases so runs are independent.
            run.output = str(agent.run(question, reset=True))
        return run

    return agent_fn


# --------------------------------------------------------------------------- #
# MOCK: emulate the smolagents toolset + a realistic failure distribution.
# Builds Trace objects directly for full control over the trajectory shape.
# --------------------------------------------------------------------------- #
_T0 = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)


def _mk_span(trace_id: str, idx: int, name: str, kind: SpanKind, inputs: dict,
             output: str | None = None, error: bool = False) -> Span:
    start = _T0 + timedelta(seconds=idx)
    sp = Span(
        trace_id=trace_id,
        name=name,
        kind=kind,
        inputs=inputs,
        start_time=start,
        end_time=start + timedelta(milliseconds=300),
        status=SpanStatus.ERROR if error else SpanStatus.OK,
    )
    if output is not None:
        sp.outputs = {"result": output}
    if error:
        from autopsy import ErrorInfo

        sp.error = ErrorInfo(type="ToolExecutionError", message=f"{name} failed: no results")
    return sp


def _make_mock_plan(dataset: Dataset) -> dict[str, str]:
    """Map each question to a deterministic behavior, spread across failure modes."""
    failures = {
        2: "redundant_search",
        5: "wrong_tool",
        8: "budget",
        11: "loop",
        14: "gave_up",
        17: "no_retry",
        20: "redundant_visit",
        23: "wrong_tool",
    }
    return {case.input: failures.get(i, "clean") for i, case in enumerate(dataset.cases)}


def build_mock_agent_fn(dataset: Dataset):
    plan = _make_mock_plan(dataset)
    answers = {case.input: (case.expected_output or "") for case in dataset.cases}

    def agent_fn(question: str) -> Trace:
        behavior = plan.get(question, "clean")
        answer = answers.get(question, "")
        trace = Trace(name="smolagents", input=question, status=SpanStatus.OK, output=answer)
        tid = trace.trace_id
        q = question[:48]
        spans: list[Span] = []

        if behavior == "clean":
            spans.append(_mk_span(tid, 0, "web_search", SpanKind.TOOL_CALL, {"query": q}, "top hit"))
            spans.append(_mk_span(tid, 1, "visit_webpage", SpanKind.TOOL_CALL, {"url": "https://en.wikipedia.org/x"}, "page text"))
        elif behavior == "redundant_search":
            spans.append(_mk_span(tid, 0, "web_search", SpanKind.TOOL_CALL, {"query": q}, "top hit"))
            spans.append(_mk_span(tid, 1, "web_search", SpanKind.TOOL_CALL, {"query": q}, "top hit"))  # identical → redundant
        elif behavior == "redundant_visit":
            spans.append(_mk_span(tid, 0, "web_search", SpanKind.TOOL_CALL, {"query": q}, "top hit"))
            spans.append(_mk_span(tid, 1, "visit_webpage", SpanKind.TOOL_CALL, {"url": "https://e.org/a"}, "text"))
            spans.append(_mk_span(tid, 2, "visit_webpage", SpanKind.TOOL_CALL, {"url": "https://e.org/a"}, "text"))  # identical
        elif behavior == "wrong_tool":
            spans.append(_mk_span(tid, 0, "web_search", SpanKind.TOOL_CALL, {"query": q}, "top hit"))
            spans.append(_mk_span(tid, 1, "wikipedia", SpanKind.TOOL_CALL, {"title": q}, "article"))  # not in registry
        elif behavior == "budget":
            for k in range(9):  # 9 distinct searches → within-efficiency but over step budget (max 8)
                spans.append(_mk_span(tid, k, "web_search", SpanKind.TOOL_CALL, {"query": f"{q} {k}"}, "hit"))
        elif behavior == "loop":
            for k in range(3):  # identical call x3 → infinite_loop
                spans.append(_mk_span(tid, k, "web_search", SpanKind.TOOL_CALL, {"query": q}, "hit"))
        elif behavior == "gave_up":
            spans.append(_mk_span(tid, 0, "web_search", SpanKind.TOOL_CALL, {"query": q}, error=True))
            trace.status = SpanStatus.ERROR  # ended on error, never recovered
        elif behavior == "no_retry":
            spans.append(_mk_span(tid, 0, "web_search", SpanKind.TOOL_CALL, {"query": q}, "hit"))
            spans.append(_mk_span(tid, 1, "visit_webpage", SpanKind.TOOL_CALL, {"url": "https://e.org/a"}, error=True))
            # trace.status stays OK (degraded answer), but the error was never recovered

        trace.spans = spans
        trace.end_time = _T0 + timedelta(seconds=len(spans))
        return trace

    return agent_fn


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="force the offline mock agent")
    args = parser.parse_args()

    dataset = Dataset.from_file(DATASET_PATH)

    have_smolagents = False
    try:
        import smolagents  # noqa: F401

        have_smolagents = True
    except ImportError:
        pass
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    use_real = have_smolagents and have_key and not args.mock

    if use_real:
        print(f"REAL run: smolagents + {MODEL_ID} over {len(dataset)} cases\n")
        agent_fn = build_real_agent_fn()
        judge = AnthropicJudge()
        classifier = LLMFailureClassifier(judge=judge)
        results = batch_evaluate(agent_fn, dataset, judge=judge, classifier=classifier)
    else:
        reason = "forced --mock" if args.mock else (
            "smolagents not installed" if not have_smolagents else "ANTHROPIC_API_KEY not set")
        print(f"MOCK run ({reason}): emulated smolagents agent over {len(dataset)} cases")
        print("  (validates the pipeline end-to-end; install smolagents + set a key for the real headline)\n")
        agent_fn = build_mock_agent_fn(dataset)
        results = batch_evaluate(agent_fn, dataset, classifier=RuleBasedFailureClassifier())

    rs = BatchResultSet.from_batch_results(dataset.name, dataset.version, results)
    rs.save(OUT_PATH)

    m = rs.metrics
    print(f"  pass rate : {m.pass_rate:.0%}  ({m.passed}/{m.total} passed, {m.failed} failed, {m.errors} errored)")
    if m.failure_distribution:
        print("  failure breakdown:")
        for cause, n in sorted(m.failure_distribution.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>2}x  {cause}")
    print(f"\n  saved -> {OUT_PATH}")
    print("  dashboard: streamlit run src/autopsy/dashboard/app.py -- "
          f"--mode single --data {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
