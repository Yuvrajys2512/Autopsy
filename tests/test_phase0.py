"""Phase 0 tests: schema, instrumentation, persistence, client wrappers."""

from __future__ import annotations

import asyncio

import pytest

from autopsy import (
    JSONStorage,
    SpanKind,
    SpanStatus,
    TokenUsage,
    list_traces,
    load_trace,
    span,
    start_run,
    trace,
    wrap_anthropic,
)
from autopsy.pricing import compute_cost, get_pricing


@pytest.fixture
def store(tmp_path):
    return JSONStorage(tmp_path / "traces")


def test_start_run_saves_trace_and_roundtrips(store):
    with start_run("agent", input="hello", store=store) as run:
        with span("step", kind=SpanKind.TOOL_CALL):
            pass
        run.output = "world"

    assert run.status is SpanStatus.OK
    assert run.end_time is not None

    reloaded = load_trace(run.trace_id, store=store)
    assert reloaded.trace_id == run.trace_id
    assert reloaded.input == "hello"
    assert reloaded.output == "world"
    assert len(reloaded.spans) == 1
    assert reloaded.spans[0].kind is SpanKind.TOOL_CALL


def test_span_nesting_sets_parent_linkage(store):
    with start_run("agent", store=store) as run:
        with span("outer", kind=SpanKind.CHAIN) as outer:
            with span("inner", kind=SpanKind.TOOL_CALL) as inner:
                pass

    # Flat list, OTel-style; parent linkage via parent_span_id.
    by_name = {s.name: s for s in run.spans}
    assert by_name["outer"].parent_span_id is None
    assert by_name["inner"].parent_span_id == by_name["outer"].span_id
    assert by_name["inner"].trace_id == run.trace_id


def test_error_is_captured_and_reraised(store):
    with pytest.raises(ValueError):
        with start_run("agent", store=store) as run:
            with span("boom"):
                raise ValueError("kaboom")

    assert run.status is SpanStatus.ERROR
    assert run.error is not None and run.error.type == "ValueError"

    boom = run.spans[0]
    assert boom.status is SpanStatus.ERROR
    assert boom.error is not None
    assert "kaboom" in boom.error.message
    assert boom.error.traceback


def test_decorator_captures_io(store):
    @trace(kind=SpanKind.TOOL_CALL)
    def add(a: int, b: int) -> int:
        return a + b

    with start_run("agent", store=store) as run:
        assert add(2, 3) == 5

    sp = run.spans[0]
    assert sp.name.endswith("add")
    assert sp.inputs == {"a": 2, "b": 3}
    assert sp.outputs == {"result": 5}


def test_async_decorator(store):
    @trace(kind=SpanKind.LLM_CALL)
    async def fetch(x: int) -> int:
        await asyncio.sleep(0)
        return x * 10

    async def run_it():
        with start_run("agent", store=store) as run:
            assert await fetch(4) == 40
        return run

    run = asyncio.run(run_it())
    assert run.spans[0].outputs == {"result": 40}
    assert run.spans[0].kind is SpanKind.LLM_CALL


def test_span_without_run_is_noop():
    # No active run: block still executes, nothing is persisted, no error.
    with span("orphan") as sp:
        sp.outputs = {"ok": True}
    assert sp.trace_id == "0" * 32


def test_list_traces_returns_summaries(store):
    for i in range(3):
        with start_run(f"run-{i}", store=store):
            pass
    summaries = list_traces(store=store)
    assert len(summaries) == 3
    assert {"trace_id", "name", "status", "span_count"} <= set(summaries[0])


def test_pricing_lookup_and_cost():
    assert get_pricing("claude-opus-4-8") == (5.0, 25.0)
    # Dated/aliased variants resolve by prefix.
    assert get_pricing("claude-haiku-4-5-20251001") == (1.0, 5.0)
    assert get_pricing("totally-unknown-model") is None

    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert compute_cost("claude-opus-4-8", usage) == pytest.approx(30.0)
    assert compute_cost("unknown", usage) is None


# --- client wrapper -----------------------------------------------------------


class _FakeUsage:
    input_tokens = 100
    output_tokens = 50
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _FakeTextBlock:
    type = "text"
    text = "the answer"


class _FakeResponse:
    model = "claude-opus-4-8"
    stop_reason = "end_turn"
    usage = _FakeUsage()
    content = [_FakeTextBlock()]


class _FakeMessages:
    def create(self, **kwargs):
        return _FakeResponse()


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_wrap_anthropic_captures_llm_span(store):
    client = wrap_anthropic(_FakeAnthropicClient())

    with start_run("agent", store=store) as run:
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=256,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert resp.content[0].text == "the answer"

    assert len(run.spans) == 1
    sp = run.spans[0]
    assert sp.kind is SpanKind.LLM_CALL
    assert sp.model == "claude-opus-4-8"
    assert sp.usage.input_tokens == 100
    assert sp.usage.output_tokens == 50
    assert sp.cost_usd == pytest.approx((100 * 5.0 + 50 * 25.0) / 1_000_000)
    assert sp.outputs["text"] == "the answer"
    assert sp.outputs["stop_reason"] == "end_turn"


def test_wrap_anthropic_is_idempotent():
    client = _FakeAnthropicClient()
    wrap_anthropic(client)
    first = client.messages.create
    wrap_anthropic(client)
    assert client.messages.create is first
