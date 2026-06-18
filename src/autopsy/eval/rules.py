"""Rule-based evaluators: deterministic, cheap, no model calls.

Each one is a focused, explainable check over the trajectory. They never need
the network, so they're the fast first pass — and they generate the evidence
(`details`) that Phase 2's classifier and a human will read.
"""

from __future__ import annotations

from collections import Counter
from typing import ClassVar

from ..schema import SpanStatus, Trace
from ..taxonomy import FailureLabel
from .base import (
    Evaluator,
    errored_spans,
    in_order,
    retrieval_spans,
    tool_signature,
    tool_spans,
)
from .models import EvalResult, Expectations


class TrajectoryEfficiency(Evaluator):
    """Did the agent take redundant or looping steps?

    Reference-free: detects repeated tool calls with identical arguments. One
    repeat is wasteful (``redundant_calls``); three or more of the same call is
    a likely loop (``infinite_loop``). Score = fraction of tool calls that were
    *not* redundant.
    """

    name: ClassVar[str] = "trajectory_efficiency"
    kind: ClassVar[str] = "rule"

    def __init__(self, loop_threshold: int = 3) -> None:
        self.loop_threshold = loop_threshold

    def evaluate(self, trace: Trace, expected: Expectations) -> EvalResult:
        calls = tool_spans(trace)
        if not calls:
            return self._na("no tool calls to assess")

        counts = Counter(tool_signature(s) for s in calls)
        repeated = {sig: n for sig, n in counts.items() if n > 1}
        redundant_calls = sum(n - 1 for n in repeated.values())
        unique = len(counts)
        score = unique / len(calls)  # 1.0 when every call is distinct

        label = None
        if any(n >= self.loop_threshold for n in repeated.values()):
            label = FailureLabel.INFINITE_LOOP.value
        elif repeated:
            label = FailureLabel.REDUNDANT_CALLS.value

        passed = not repeated
        rationale = (
            f"{len(calls)} tool calls, {unique} unique"
            + (f", {redundant_calls} redundant" if redundant_calls else "")
        )
        return self._result(
            score=score,
            passed=passed,
            failure_label=label,
            rationale=rationale,
            details={
                "total_calls": len(calls),
                "unique_calls": unique,
                "redundant_calls": redundant_calls,
                "repeated": repeated,
            },
        )


class ErrorRecovery(Evaluator):
    """When a step errored, did the agent recover?

    Recovery = a successful span occurred *after* the error (it tried again or
    kept making progress). If errors went unrecovered we raise
    ``no_retry_on_error``; if the run ended on an error we raise
    ``gave_up_early``. No errors at all → vacuously perfect.
    """

    name: ClassVar[str] = "error_recovery"
    kind: ClassVar[str] = "rule"

    def evaluate(self, trace: Trace, expected: Expectations) -> EvalResult:
        errors = errored_spans(trace)
        if not errors:
            return self._result(
                score=1.0, passed=True, rationale="no errored spans"
            )

        ordered = in_order(trace.spans)
        recovered = 0
        for err in errors:
            later = [s for s in ordered if s.start_time > err.start_time]
            if any(s.status is SpanStatus.OK for s in later):
                recovered += 1
        rate = recovered / len(errors)

        ended_on_error = bool(ordered) and ordered[-1].status is SpanStatus.ERROR
        label = None
        if ended_on_error and trace.status is SpanStatus.ERROR:
            label = FailureLabel.GAVE_UP_EARLY.value
        elif rate < 1.0:
            label = FailureLabel.NO_RETRY_ON_ERROR.value

        return self._result(
            score=rate,
            passed=rate == 1.0,
            failure_label=label,
            rationale=f"{recovered}/{len(errors)} errors recovered",
            details={
                "errors": len(errors),
                "recovered": recovered,
                "ended_on_error": ended_on_error,
            },
        )


class ToolArgumentValidity(Evaluator):
    """Were tool arguments well-formed, or hallucinated?

    Reference-based: needs ``expected.tool_schemas`` (JSON-schema per tool name).
    Validates each tool call's inputs against its schema. Score = fraction of
    calls with valid arguments. Raises ``hallucinated_tool_args`` on any
    violation.
    """

    name: ClassVar[str] = "tool_argument_validity"
    kind: ClassVar[str] = "rule"

    _TYPES = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    def evaluate(self, trace: Trace, expected: Expectations) -> EvalResult:
        schemas = expected.tool_schemas
        if not schemas:
            return self._na("no tool_schemas provided")
        calls = tool_spans(trace)
        checkable = [s for s in calls if s.name in schemas]
        if not checkable:
            return self._na("no tool calls matched a provided schema")

        problems: dict[str, list[str]] = {}
        valid = 0
        for span in checkable:
            issues = self._violations(span.inputs, schemas[span.name])
            if issues:
                problems[span.span_id] = issues
            else:
                valid += 1

        score = valid / len(checkable)
        return self._result(
            score=score,
            passed=not problems,
            failure_label=FailureLabel.HALLUCINATED_TOOL_ARGS.value if problems else None,
            rationale=f"{valid}/{len(checkable)} tool calls had valid arguments",
            details={"checked": len(checkable), "problems": problems},
        )

    def _violations(self, inputs: dict, schema: dict) -> list[str]:
        problems: list[str] = []
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in inputs:
                problems.append(f"missing required arg '{key}'")
        for key, value in inputs.items():
            spec = props.get(key)
            if spec and "type" in spec:
                py = self._TYPES.get(spec["type"])
                # bool is a subclass of int — guard against false negatives
                if py and (not isinstance(value, py) or (py is int and isinstance(value, bool))):
                    problems.append(f"arg '{key}' should be {spec['type']}")
        if schema.get("additionalProperties") is False:
            for key in inputs:
                if key not in props:
                    problems.append(f"unexpected arg '{key}'")
        return problems


class ToolSelectionAccuracy(Evaluator):
    """Did the agent call the right tools?

    Reference-based. With ``allowed_tools``, any call outside the registry is a
    wrong-tool error. With ``expected_tools``, score is the precision of called
    tools against the expected set (and we note expected tools that were never
    used). Raises ``wrong_tool_selected``.
    """

    name: ClassVar[str] = "tool_selection_accuracy"
    kind: ClassVar[str] = "rule"

    def evaluate(self, trace: Trace, expected: Expectations) -> EvalResult:
        calls = tool_spans(trace)
        if not calls:
            return self._na("no tool calls to assess")
        called = [s.name for s in calls]

        if expected.allowed_tools is not None:
            allowed = set(expected.allowed_tools)
            out_of_set = [n for n in called if n not in allowed]
            score = (len(called) - len(out_of_set)) / len(called)
            return self._result(
                score=score,
                passed=not out_of_set,
                failure_label=FailureLabel.WRONG_TOOL_SELECTED.value if out_of_set else None,
                rationale=(
                    f"{len(out_of_set)} call(s) used tools outside the registry"
                    if out_of_set else "all calls used registered tools"
                ),
                details={"called": called, "out_of_registry": sorted(set(out_of_set))},
            )

        if expected.expected_tools is not None:
            expected_set = set(expected.expected_tools)
            correct = [n for n in called if n in expected_set]
            score = len(correct) / len(called)
            missing = sorted(expected_set - set(called))
            unexpected = sorted(set(called) - expected_set)
            return self._result(
                score=score,
                passed=not unexpected,
                failure_label=FailureLabel.WRONG_TOOL_SELECTED.value if unexpected else None,
                rationale=(
                    f"{len(correct)}/{len(called)} calls were to expected tools"
                    + (f"; missing {missing}" if missing else "")
                ),
                details={"unexpected": unexpected, "missing": missing},
            )

        return self._na("no allowed_tools or expected_tools provided")


class ResourceBudget(Evaluator):
    """Did the run stay within step / cost / latency budgets?

    Reference-based: checks whichever of ``max_steps`` / ``max_cost_usd`` /
    ``max_latency_ms`` are set. Score is the fraction of set budgets respected.
    Raises ``budget_exceeded``.
    """

    name: ClassVar[str] = "resource_budget"
    kind: ClassVar[str] = "rule"

    def evaluate(self, trace: Trace, expected: Expectations) -> EvalResult:
        checks: list[tuple[str, float, float, bool]] = []  # (name, actual, limit, ok)
        if expected.max_steps is not None:
            steps = len(trace.spans)
            checks.append(("steps", steps, expected.max_steps, steps <= expected.max_steps))
        if expected.max_cost_usd is not None:
            cost = trace.total_cost_usd
            checks.append(("cost_usd", cost, expected.max_cost_usd, cost <= expected.max_cost_usd))
        if expected.max_latency_ms is not None:
            latency = trace.latency_ms or 0.0
            checks.append(
                ("latency_ms", latency, expected.max_latency_ms, latency <= expected.max_latency_ms)
            )

        if not checks:
            return self._na("no budgets provided")

        respected = sum(1 for *_, ok in checks if ok)
        exceeded = [name for name, *_, ok in checks if not ok]
        score = respected / len(checks)
        return self._result(
            score=score,
            passed=not exceeded,
            failure_label=FailureLabel.BUDGET_EXCEEDED.value if exceeded else None,
            rationale=(
                f"exceeded: {', '.join(exceeded)}" if exceeded else "within all budgets"
            ),
            details={
                "checks": {name: {"actual": actual, "limit": limit, "ok": ok}
                           for name, actual, limit, ok in checks}
            },
        )


# The default rule-based suite, in report order.
RULE_EVALUATORS: list[Evaluator] = [
    TrajectoryEfficiency(),
    ErrorRecovery(),
    ToolArgumentValidity(),
    ToolSelectionAccuracy(),
    ResourceBudget(),
]
