"""Thin wrappers that auto-capture LLM calls as spans.

These don't import the provider SDKs themselves — you pass an already-constructed
client, and the wrapper monkeypatches the one method that makes the network call.
That keeps Autopsy's core free of heavy optional dependencies and works against
a raw SDK loop, not just a framework.

    import anthropic
    from autopsy import wrap_anthropic

    client = wrap_anthropic(anthropic.Anthropic())
    client.messages.create(...)   # now emits an llm_call span
"""

from __future__ import annotations

from typing import Any, Optional

from ._serialize import to_jsonable
from .pricing import compute_cost
from .schema import SpanKind, TokenUsage
from .tracing import span

_WRAPPED_FLAG = "_autopsy_wrapped"


def _coerce_usage(raw: Any) -> Optional[TokenUsage]:
    """Pull token counts off an SDK usage object (Anthropic-style names)."""
    if raw is None:
        return None

    def field(name: str) -> int:
        return int(getattr(raw, name, 0) or 0)

    return TokenUsage(
        input_tokens=field("input_tokens"),
        output_tokens=field("output_tokens"),
        cache_read_input_tokens=field("cache_read_input_tokens"),
        cache_creation_input_tokens=field("cache_creation_input_tokens"),
    )


def wrap_anthropic(client: Any) -> Any:
    """Instrument an ``anthropic.Anthropic`` client's ``messages.create``.

    Returns the same client (mutated in place). Safe to call once; repeated
    calls are ignored.
    """
    messages = client.messages
    if getattr(messages, _WRAPPED_FLAG, False):
        return client

    original = messages.create

    def create(*args: Any, **kwargs: Any) -> Any:
        requested_model = kwargs.get("model")
        inputs = {
            "model": requested_model,
            "messages": to_jsonable(kwargs.get("messages")),
            "system": to_jsonable(kwargs.get("system")),
            "max_tokens": kwargs.get("max_tokens"),
        }
        with span("anthropic.messages.create", kind=SpanKind.LLM_CALL, inputs=inputs) as sp:
            response = original(*args, **kwargs)
            # response.model is authoritative (fallbacks, aliases).
            sp.model = getattr(response, "model", requested_model)
            sp.usage = _coerce_usage(getattr(response, "usage", None))
            sp.cost_usd = compute_cost(sp.model, sp.usage)
            sp.outputs = {
                "stop_reason": getattr(response, "stop_reason", None),
                "content": to_jsonable(getattr(response, "content", None)),
                "text": _anthropic_text(response),
            }
            return response

    setattr(create, _WRAPPED_FLAG, True)
    messages.create = create  # type: ignore[assignment]
    setattr(messages, _WRAPPED_FLAG, True)
    return client


def _anthropic_text(response: Any) -> Optional[str]:
    """Concatenate text blocks from an Anthropic response, if any."""
    content = getattr(response, "content", None)
    if not content:
        return None
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts) if parts else None


def wrap_openai(client: Any) -> Any:
    """Instrument an ``openai.OpenAI`` client's ``chat.completions.create``.

    Returns the same client (mutated in place). Safe to call once.
    """
    completions = client.chat.completions
    if getattr(completions, _WRAPPED_FLAG, False):
        return client

    original = completions.create

    def create(*args: Any, **kwargs: Any) -> Any:
        requested_model = kwargs.get("model")
        inputs = {
            "model": requested_model,
            "messages": to_jsonable(kwargs.get("messages")),
            "max_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
        }
        with span("openai.chat.completions.create", kind=SpanKind.LLM_CALL, inputs=inputs) as sp:
            response = original(*args, **kwargs)
            sp.model = getattr(response, "model", requested_model)
            sp.usage = _coerce_openai_usage(getattr(response, "usage", None))
            sp.cost_usd = compute_cost(sp.model, sp.usage)
            sp.outputs = {
                "text": _openai_text(response),
                "finish_reason": _openai_finish_reason(response),
            }
            return response

    setattr(create, _WRAPPED_FLAG, True)
    completions.create = create  # type: ignore[assignment]
    setattr(completions, _WRAPPED_FLAG, True)
    return client


def _coerce_openai_usage(raw: Any) -> Optional[TokenUsage]:
    if raw is None:
        return None
    return TokenUsage(
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
    )


def _openai_text(response: Any) -> Optional[str]:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    return getattr(message, "content", None) if message else None


def _openai_finish_reason(response: Any) -> Optional[str]:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "finish_reason", None)
