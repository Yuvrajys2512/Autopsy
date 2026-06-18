"""Autopsy — agent evaluation & failure-diagnosis toolkit.

Phase 0 (Capture): trace capture, @trace decorator, client wrappers.
Phase 1 (Evaluators): rule metrics + LLM-as-judge, EvalReport.
Phase 2 (Failure Classifier): hybrid LLM + rules to diagnose root cause.
"""

from __future__ import annotations

from .clients import wrap_anthropic, wrap_openai
from .eval import (
    AnthropicJudge,
    EvalReport,
    EvalResult,
    Evaluator,
    Expectations,
    FailureClassification,
    FailureClassifier,
    Judge,
    LLMFailureClassifier,
    RuleBasedFailureClassifier,
    default_evaluators,
    evaluate,
)
from .pricing import compute_cost, get_pricing
from .schema import (
    ErrorInfo,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    TokenUsage,
    Trace,
)
from .taxonomy import FailureLabel
from .storage import (
    JSONStorage,
    TraceStore,
    get_default_store,
    list_traces,
    load_trace,
    save_trace,
    set_default_store,
)
from .tracing import (
    current_span,
    current_trace,
    span,
    start_run,
    trace,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    # schema
    "Trace",
    "Span",
    "SpanEvent",
    "SpanKind",
    "SpanStatus",
    "TokenUsage",
    "ErrorInfo",
    "FailureLabel",
    # evaluation
    "evaluate",
    "default_evaluators",
    "EvalReport",
    "EvalResult",
    "Expectations",
    "Evaluator",
    "Judge",
    "AnthropicJudge",
    # classifier
    "FailureClassification",
    "FailureClassifier",
    "LLMFailureClassifier",
    "RuleBasedFailureClassifier",
    # instrumentation
    "start_run",
    "span",
    "trace",
    "current_trace",
    "current_span",
    # storage
    "JSONStorage",
    "TraceStore",
    "save_trace",
    "load_trace",
    "list_traces",
    "get_default_store",
    "set_default_store",
    # clients
    "wrap_anthropic",
    "wrap_openai",
    # pricing
    "compute_cost",
    "get_pricing",
]
