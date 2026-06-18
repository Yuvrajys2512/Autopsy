"""Typed, serializable trace/span schema for Autopsy.

A **Trace** is one agent run on one input. **Spans** are the steps inside it
(LLM calls, tool calls, retrievals, retries). The shape is deliberately kept
OpenTelemetry-compatible: 16-byte trace IDs / 8-byte span IDs as hex, a flat
span list with `parent_span_id` linkage, a free-form `attributes` map, and a
span `kind`/`status`. That makes a later OTel export a translation, not a
redesign — while staying ergonomic to read straight off disk as JSON.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_trace_id() -> str:
    """16 random bytes as 32 lowercase hex chars (OTel trace-id shape)."""
    return os.urandom(16).hex()


def new_span_id() -> str:
    """8 random bytes as 16 lowercase hex chars (OTel span-id shape)."""
    return os.urandom(8).hex()


class SpanKind(str, Enum):
    """What a span represents. Open set — use CUSTOM when nothing fits."""

    AGENT = "agent"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    RETRY = "retry"
    CHAIN = "chain"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class TokenUsage(BaseModel):
    """Token accounting for an LLM call. Field names mirror the Anthropic SDK's
    `usage` object so capture is a direct copy."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


class ErrorInfo(BaseModel):
    """A captured exception. Carries enough to diagnose without re-running."""

    type: str
    message: str
    traceback: Optional[str] = None

    @classmethod
    def from_exception(cls, exc: BaseException, *, include_traceback: bool = True) -> "ErrorInfo":
        tb: Optional[str] = None
        if include_traceback:
            import traceback as _tb

            tb = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        return cls(type=type(exc).__name__, message=str(exc), traceback=tb)


class SpanEvent(BaseModel):
    """A timestamped point-in-time annotation within a span (OTel `events`)."""

    name: str
    timestamp: datetime = Field(default_factory=_now)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """One step inside a trace."""

    span_id: str = Field(default_factory=new_span_id)
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    kind: SpanKind = SpanKind.CUSTOM

    start_time: datetime = Field(default_factory=_now)
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET

    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)
    error: Optional[ErrorInfo] = None

    # Convenience fields for LLM-call spans (also derivable from attributes).
    model: Optional[str] = None
    usage: Optional[TokenUsage] = None
    cost_usd: Optional[float] = None

    @property
    def latency_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000.0

    def add_event(self, name: str, **attributes: Any) -> None:
        self.events.append(SpanEvent(name=name, attributes=attributes))


class Trace(BaseModel):
    """One agent run on one input — the root container for its spans."""

    trace_id: str = Field(default_factory=new_trace_id)
    name: str = "trace"
    start_time: datetime = Field(default_factory=_now)
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET

    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[ErrorInfo] = None

    spans: list[Span] = Field(default_factory=list)

    @property
    def latency_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000.0

    @property
    def total_tokens(self) -> int:
        return sum(s.usage.total_tokens for s in self.spans if s.usage)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans if s.cost_usd)

    def spans_by_kind(self, kind: SpanKind) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]

    def summary(self) -> dict[str, Any]:
        """Lightweight metadata for listings/dashboards (no full span bodies)."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "latency_ms": self.latency_ms,
            "span_count": len(self.spans),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }
