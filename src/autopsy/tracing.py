"""Instrumentation: capture an execution trace without polluting business logic.

Three low-friction entry points:

* ``start_run(...)`` — a context manager that opens a Trace, makes it the active
  run, and persists it on exit.
* ``span(...)`` — a context manager for a single step; nests automatically under
  whatever span/run is active.
* ``@trace(...)`` — a decorator that wraps a function as a span (sync or async),
  capturing its arguments and return value.

State is tracked with :mod:`contextvars`, so nesting works correctly across
threads and ``async`` tasks. When there is no active run, spans become no-ops
rather than raising — matching how OpenTelemetry behaves with no active context.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional, TypeVar

from ._serialize import to_jsonable
from .schema import ErrorInfo, Span, SpanKind, SpanStatus, Trace
from .storage import TraceStore, save_trace

_current_trace: ContextVar[Optional[Trace]] = ContextVar("autopsy_current_trace", default=None)
_current_span: ContextVar[Optional[Span]] = ContextVar("autopsy_current_span", default=None)

F = TypeVar("F", bound=Callable[..., Any])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def current_trace() -> Optional[Trace]:
    """The active Trace, or ``None`` if no run is open."""
    return _current_trace.get()


def current_span() -> Optional[Span]:
    """The innermost active Span, or ``None``."""
    return _current_span.get()


@contextmanager
def start_run(
    name: str = "trace",
    *,
    input: Any = None,
    metadata: Optional[dict[str, Any]] = None,
    store: Optional[TraceStore] = None,
    save: bool = True,
) -> Iterator[Trace]:
    """Open a Trace as the active run for the duration of the block.

    The Trace is yielded so the caller can set ``run.output`` (and read it back).
    On exit it is timestamped, marked OK/ERROR, and — unless ``save=False`` —
    persisted via the default (or supplied) store. Exceptions are recorded and
    re-raised.
    """
    trace = Trace(name=name, input=to_jsonable(input), metadata=metadata or {})
    token = _current_trace.set(trace)
    # A run is its own root: clear any inherited span so top-level spans attach
    # directly to the trace.
    span_token = _current_span.set(None)
    try:
        yield trace
        if trace.status is SpanStatus.UNSET:
            trace.status = SpanStatus.OK
    except BaseException as exc:
        trace.status = SpanStatus.ERROR
        trace.error = ErrorInfo.from_exception(exc)
        raise
    finally:
        trace.end_time = _now()
        _current_span.reset(span_token)
        _current_trace.reset(token)
        if save:
            try:
                save_trace(trace, store)
            except Exception:
                # Persistence must never crash the user's program.
                pass


@contextmanager
def span(
    name: str,
    *,
    kind: SpanKind = SpanKind.CUSTOM,
    inputs: Optional[dict[str, Any]] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Iterator[Span]:
    """Record a single step. Nests under the active span/run automatically.

    If no run is active this is a no-op recorder: the block still runs and a
    Span object is yielded, but it is not attached to any trace.
    """
    trace = _current_trace.get()
    parent = _current_span.get()
    sp = Span(
        trace_id=trace.trace_id if trace else "0" * 32,
        parent_span_id=parent.span_id if parent else None,
        name=name,
        kind=kind,
        inputs=to_jsonable(inputs) if inputs else {},
        attributes=attributes or {},
    )
    token = _current_span.set(sp)
    try:
        yield sp
        if sp.status is SpanStatus.UNSET:
            sp.status = SpanStatus.OK
    except BaseException as exc:
        sp.status = SpanStatus.ERROR
        sp.error = ErrorInfo.from_exception(exc)
        raise
    finally:
        sp.end_time = _now()
        _current_span.reset(token)
        if trace is not None:
            trace.spans.append(sp)


def trace(
    name: Optional[str] = None,
    *,
    kind: SpanKind = SpanKind.CUSTOM,
    capture_io: bool = True,
) -> Callable[[F], F]:
    """Decorator: run the wrapped function inside a span.

    Captures positional/keyword arguments as ``inputs`` and the return value as
    ``outputs["result"]`` (unless ``capture_io=False``). Works on both sync and
    async functions.
    """

    def decorator(fn: F) -> F:
        span_name = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "span"))

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                inputs = _capture_args(fn, args, kwargs) if capture_io else None
                with span(span_name, kind=kind, inputs=inputs) as sp:
                    result = await fn(*args, **kwargs)
                    if capture_io:
                        sp.outputs = {"result": to_jsonable(result)}
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            inputs = _capture_args(fn, args, kwargs) if capture_io else None
            with span(span_name, kind=kind, inputs=inputs) as sp:
                result = fn(*args, **kwargs)
                if capture_io:
                    sp.outputs = {"result": to_jsonable(result)}
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _capture_args(fn: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, Any]:
    """Bind call args to parameter names where possible; never raise."""
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        captured = dict(bound.arguments)
        # Drop a leading self/cls — noisy and often unserializable.
        for skip in ("self", "cls"):
            captured.pop(skip, None)
        return to_jsonable(captured)
    except Exception:
        return {"args": to_jsonable(args), "kwargs": to_jsonable(kwargs)}
