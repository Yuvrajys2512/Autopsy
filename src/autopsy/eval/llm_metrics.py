"""LLM-as-judge metrics: goal completion and faithfulness.

These grade things a rule can't: "did it actually accomplish the task?" and "is
the answer grounded in what was retrieved?". Both are *process-aware* — the
judge sees the trajectory, not just the final string — which is the trajectory-
level angle that sets Autopsy apart.

Each needs a `Judge`. With no judge configured they report themselves as not
applicable, so the default report still runs fully offline on the rule metrics.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from ..schema import SpanStatus, Trace
from ..taxonomy import FailureLabel
from .base import Evaluator, in_order, retrieval_spans, tool_spans
from .judge import Judge, judgement_schema
from .models import EvalResult, Expectations


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _final_output(trace: Trace) -> Optional[str]:
    if trace.output is not None:
        return str(trace.output)
    # fall back to the last LLM/text output we recorded
    for span in reversed(in_order(trace.spans)):
        text = span.outputs.get("text") or span.outputs.get("result")
        if text:
            return str(text)
    return None


def _trajectory_lines(trace: Trace) -> str:
    lines = []
    for s in in_order(trace.spans):
        lines.append(f"- [{s.kind.value}] {s.name} (status={s.status.value})")
    return "\n".join(lines) if lines else "(no recorded steps)"


def _retrieved_context(trace: Trace) -> str:
    """Concatenate textual outputs of retrieval (and tool) spans — the evidence
    the answer is supposed to be grounded in."""
    chunks = []
    for span in retrieval_spans(trace) + tool_spans(trace):
        for value in span.outputs.values():
            if isinstance(value, str) and value.strip():
                chunks.append(value)
            elif isinstance(value, list):
                chunks.extend(str(v) for v in value if str(v).strip())
    return "\n---\n".join(chunks)


class LLMJudgeEvaluator(Evaluator):
    """Base for judge-backed metrics: handles the no-judge and judge-error cases
    so subclasses only build a prompt and parse a result."""

    kind: ClassVar[str] = "llm_judge"

    def __init__(self, judge: Optional[Judge] = None) -> None:
        self.judge = judge

    def evaluate(self, trace: Trace, expected: Expectations) -> EvalResult:
        if self.judge is None:
            return self._na("no judge configured")
        prepared = self._prepare(trace, expected)
        if prepared is None:
            return self._na(self._na_reason)
        system, prompt, schema = prepared
        try:
            data = self.judge.score(system=system, prompt=prompt, schema=schema)
        except Exception as exc:  # a judge failure shouldn't sink the report
            return EvalResult.errored(self.name, self.kind, str(exc))
        return self._parse(data, trace, expected)

    _na_reason: ClassVar[str] = "not applicable"

    def _prepare(
        self, trace: Trace, expected: Expectations
    ) -> Optional[tuple[str, str, dict]]:
        raise NotImplementedError

    def _parse(self, data: dict, trace: Trace, expected: Expectations) -> EvalResult:
        raise NotImplementedError


class GoalCompletion(LLMJudgeEvaluator):
    """Did the agent accomplish the user's goal? End-to-end task success.

    Reference-free by default (judges against the goal); reference-based when an
    ``expected.reference_output`` is supplied. Raises ``goal_not_completed``.
    """

    name: ClassVar[str] = "goal_completion"
    _na_reason: ClassVar[str] = "no goal or output to assess"

    SYSTEM = (
        "You are a strict evaluator of AI agents. Judge ONLY whether the agent "
        "accomplished the user's goal, based on its final output and the steps it "
        "took. Do not reward effort or fluency — reward task success. Score 1.0 "
        "means fully accomplished, 0.0 means not at all."
    )

    def _prepare(self, trace, expected):
        goal = expected.goal or (str(trace.input) if trace.input is not None else None)
        output = _final_output(trace)
        if not goal or output is None:
            return None
        parts = [
            f"# Goal\n{goal}",
            f"# Steps the agent took\n{_trajectory_lines(trace)}",
            f"# Final output\n{output}",
        ]
        if expected.reference_output:
            parts.append(f"# Reference (gold) answer\n{expected.reference_output}")
        return self.SYSTEM, "\n\n".join(parts), judgement_schema(verdict_key="completed")

    def _parse(self, data, trace, expected):
        completed = bool(data.get("completed"))
        return self._result(
            score=_clamp(data.get("score")),
            passed=completed,
            failure_label=None if completed else FailureLabel.GOAL_NOT_COMPLETED.value,
            rationale=str(data.get("rationale", "")),
            details={"reference_based": expected.reference_output is not None},
        )


class Faithfulness(LLMJudgeEvaluator):
    """Is the final answer grounded in what was retrieved? (RAGAS territory.)

    Applies only when the trace has retrieval/tool context. If retrieval ran but
    returned nothing usable, that's a ``retrieval_miss`` (no judge call needed).
    Otherwise the judge checks grounding; an ungrounded answer over good evidence
    is the silent killer — ``correct_retrieval_wrong_answer``.
    """

    name: ClassVar[str] = "faithfulness"
    _na_reason: ClassVar[str] = "no retrieved context to ground against"

    SYSTEM = (
        "You are evaluating whether an answer is FAITHFUL to the provided context "
        "— i.e. every claim in the answer is supported by the context, with no "
        "fabrication. Judge grounding only, not whether the answer is generally "
        "correct. Score 1.0 = fully grounded, 0.0 = unsupported/contradicted."
    )

    def evaluate(self, trace, expected):
        # Special-case: retrieval happened but produced nothing → retrieval_miss,
        # decided without spending a judge call.
        if retrieval_spans(trace) and not _retrieved_context(trace).strip():
            return self._result(
                score=0.0,
                passed=False,
                failure_label=FailureLabel.RETRIEVAL_MISS.value,
                rationale="retrieval steps returned no usable context",
            )
        return super().evaluate(trace, expected)

    def _prepare(self, trace, expected):
        context = _retrieved_context(trace).strip()
        answer = _final_output(trace)
        if not context or answer is None:
            return None
        prompt = f"# Context\n{context}\n\n# Answer\n{answer}"
        return self.SYSTEM, prompt, judgement_schema(verdict_key="faithful")

    def _parse(self, data, trace, expected):
        faithful = bool(data.get("faithful"))
        return self._result(
            score=_clamp(data.get("score")),
            passed=faithful,
            failure_label=None if faithful else FailureLabel.CORRECT_RETRIEVAL_WRONG_ANSWER.value,
            rationale=str(data.get("rationale", "")),
        )
