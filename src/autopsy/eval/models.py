"""Result and input models for evaluation.

`EvalResult` is what a single metric returns. `EvalReport` aggregates many of
them for one trace. `Expectations` carries the optional *reference* data that
turns reference-free metrics into reference-based ones (expected tools, a gold
answer, resource budgets, tool schemas).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# Metric kind: a cheap deterministic rule, or a model-graded judgement.
MetricKind = str  # "rule" | "llm_judge"


class EvalResult(BaseModel):
    """The output of one evaluator on one trace.

    A metric returns ``score`` in ``[0, 1]`` (higher is better), a ``passed``
    verdict against its own threshold, an optional ``failure_label`` from the
    taxonomy, a human-readable ``rationale``, and ``details`` (the evidence —
    span ids, counts — that a classifier or a human can drill into).

    ``applicable=False`` means the metric had nothing to assess (e.g. no tool
    calls in the trace); such results are excluded from aggregate scoring.
    ``error`` is set when the evaluator itself blew up — we record that rather
    than letting one bad metric sink the whole report.
    """

    name: str
    kind: MetricKind
    score: Optional[float] = None
    passed: Optional[bool] = None
    failure_label: Optional[str] = None
    rationale: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    applicable: bool = True
    error: Optional[str] = None

    @classmethod
    def not_applicable(cls, name: str, kind: MetricKind, reason: str) -> "EvalResult":
        return cls(name=name, kind=kind, applicable=False, rationale=reason)

    @classmethod
    def errored(cls, name: str, kind: MetricKind, message: str) -> "EvalResult":
        return cls(name=name, kind=kind, applicable=False, error=message,
                   rationale=f"evaluator error: {message}")


class EvalReport(BaseModel):
    """All metric results for a single trace, plus aggregates."""

    trace_id: str
    results: list[EvalResult] = Field(default_factory=list)

    @property
    def scored(self) -> list[EvalResult]:
        return [r for r in self.results if r.applicable and r.score is not None]

    @property
    def overall_score(self) -> Optional[float]:
        """Unweighted mean of applicable metric scores (``None`` if none ran)."""
        scored = self.scored
        if not scored:
            return None
        return sum(r.score for r in scored) / len(scored)  # type: ignore[misc]

    @property
    def passed(self) -> Optional[bool]:
        """True iff every metric that rendered a verdict passed."""
        verdicts = [r.passed for r in self.results if r.passed is not None]
        if not verdicts:
            return None
        return all(verdicts)

    @property
    def failure_labels(self) -> list[str]:
        """Distinct taxonomy labels raised by failing metrics, order-preserved."""
        seen: list[str] = []
        for r in self.results:
            if r.failure_label and r.failure_label not in seen:
                seen.append(r.failure_label)
        return seen

    def get(self, name: str) -> Optional[EvalResult]:
        return next((r for r in self.results if r.name == name), None)

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "failure_labels": self.failure_labels,
            "metrics": len(self.results),
        }

    def render(self) -> str:
        """A compact, human-readable report (handy for demos and the dashboard)."""
        lines = [f"Eval report for trace {self.trace_id}"]
        overall = self.overall_score
        overall_str = f"{overall:.2f}" if overall is not None else "n/a"
        lines.append(f"  overall: {overall_str}   passed: {self.passed}")
        if self.failure_labels:
            lines.append(f"  failures: {', '.join(self.failure_labels)}")
        lines.append("  metrics:")
        for r in self.results:
            if not r.applicable:
                verdict = "skip" if r.error is None else "ERR "
                score_str = "  -  "
            else:
                verdict = {True: "pass", False: "FAIL", None: " -- "}[r.passed]
                score_str = f"{r.score:.2f}" if r.score is not None else "  -  "
            label = f"  [{r.failure_label}]" if r.failure_label else ""
            lines.append(f"    [{verdict}] {score_str}  {r.name:24} {r.rationale}{label}")
        return "\n".join(lines)


class Expectations(BaseModel):
    """Optional reference data for a trace (reference-based evaluation).

    Everything is optional. Provide only what you have; metrics that need a
    field they don't get report themselves as *not applicable* rather than
    guessing. Phase 3's Dataset cases will produce these.
    """

    goal: Optional[str] = None
    """What the agent was supposed to accomplish. Falls back to ``trace.input``."""

    reference_output: Optional[str] = None
    """A gold/expected answer, if one exists (enables reference-based judging)."""

    expected_tools: Optional[list[str]] = None
    """Tools that *should* have been used (by span name)."""

    allowed_tools: Optional[list[str]] = None
    """The full tool registry — calls to anything outside this set are wrong-tool."""

    tool_schemas: Optional[dict[str, dict]] = None
    """JSON-schema per tool name, for argument-validity checking."""

    max_steps: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
