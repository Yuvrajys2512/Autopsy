"""Failure classifier — the Phase 2 differentiator.

Maps a low-scoring trace + its EvalReport to a root cause diagnosis with evidence.

A hybrid approach: the LLM synthesizes the failure signals (from the report's
failure_labels and details) with the trace itself, constrained to the FailureLabel
taxonomy. Rule-based signals back the LLM's reasoning and let us fall back when
no judge is available.

Gate: a failing run is returned with a labeled cause, rationale, and culprit span.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..schema import Trace
from ..taxonomy import FailureLabel
from .judge import Judge
from .models import EvalReport


class FailureClassification(BaseModel):
    """The output of a failure classifier — a diagnosis with supporting evidence.

    Attributes:
        root_cause: The primary FailureLabel that best explains the low score.
        rationale: Why this label is the root cause (what led to this diagnosis).
        culprit_span_id: The span_id of the span most responsible for the failure.
        confidence: Optional confidence score in [0,1] (from the classifier).
        evidence_labels: The failure_labels from the report that informed this diagnosis.
    """

    root_cause: str  # FailureLabel.value
    rationale: str
    culprit_span_id: Optional[str] = None
    confidence: Optional[float] = None
    evidence_labels: list[str] = Field(default_factory=list)


class FailureClassifier(ABC):
    """Abstract classifier: EvalReport + Trace -> FailureClassification."""

    @abstractmethod
    def classify(self, report: EvalReport, trace: Trace) -> FailureClassification:
        """Synthesize the report and trace into a root-cause diagnosis.

        Args:
            report: The EvalReport from Phase 1 (failure_labels, details, scores).
            trace: The underlying Trace (spans, timeline, the full trajectory).

        Returns:
            A FailureClassification with root cause, rationale, and culprit span.
        """
        ...


class RuleBasedFailureClassifier(FailureClassifier):
    """Simple heuristic classifier — fallback when no judge is available.

    Strategy:
    - If only one failure label, return it (high confidence).
    - If multiple, apply heuristics to identify root cause:
      - WRONG_TOOL_SELECTED is usually root (tool choice is fundamental).
      - HALLUCINATED_TOOL_ARGS often causes NO_RETRY_ON_ERROR (symptom chain).
      - REDUNDANT_CALLS / INFINITE_LOOP are often a symptom of tool-selection confusion.
      - BUDGET_EXCEEDED is rarely the root (usually a symptom of something else going wrong).

    This is intentionally simple; the LLM classifier below handles the nuanced cases.
    """

    def classify(self, report: EvalReport, trace: Trace) -> FailureClassification:
        labels = report.failure_labels
        if not labels:
            # No failures reported — should not happen if overall_score is low,
            # but handle gracefully.
            return FailureClassification(
                root_cause=FailureLabel.GOAL_NOT_COMPLETED.value,
                rationale="No specific failure detected; goal may not have been completed.",
                evidence_labels=[],
                confidence=0.5,
            )

        if len(labels) == 1:
            # Single failure signal — it's the root cause.
            label = labels[0]
            # Find a result with this label to extract span evidence.
            culprit = None
            for result in report.results:
                if result.failure_label == label and result.details.get("span_ids"):
                    culprit = result.details["span_ids"][0]
                    break
            return FailureClassification(
                root_cause=label,
                rationale=f"Single failure signal: {label}.",
                culprit_span_id=culprit,
                evidence_labels=labels,
                confidence=0.8,
            )

        # Multiple labels: apply heuristics.
        # Priority order: wrong tool > hallucinated args > no recovery > others
        priority = [
            FailureLabel.WRONG_TOOL_SELECTED.value,
            FailureLabel.HALLUCINATED_TOOL_ARGS.value,
            FailureLabel.NO_RETRY_ON_ERROR.value,
            FailureLabel.GAVE_UP_EARLY.value,
            FailureLabel.INFINITE_LOOP.value,
            FailureLabel.REDUNDANT_CALLS.value,
            FailureLabel.BUDGET_EXCEEDED.value,
            FailureLabel.CONTEXT_OVERFLOW.value,
        ]
        for label_val in priority:
            if label_val in labels:
                return FailureClassification(
                    root_cause=label_val,
                    rationale=f"Multiple failure signals detected; {label_val} is likely root cause.",
                    evidence_labels=labels,
                    confidence=0.6,
                )

        # Fallback: return the first label.
        return FailureClassification(
            root_cause=labels[0],
            rationale="Multiple failure signals; picking first.",
            evidence_labels=labels,
            confidence=0.4,
        )


class LLMFailureClassifier(FailureClassifier):
    """LLM-driven classifier with rule-signal backing.

    Uses a Judge to synthesize:
    1. The trace trajectory (spans, tool calls, errors).
    2. The failure signals from the report (which metrics failed, their labels).
    3. The failure taxonomy (constrained set of possible roots causes).

    The judge's output is constrained to FailureLabel via structured outputs.
    When a judge is not available, falls back to rule-based heuristics.
    """

    def __init__(self, judge: Optional[Judge] = None) -> None:
        self.judge = judge
        self._rule_fallback = RuleBasedFailureClassifier()

    def classify(self, report: EvalReport, trace: Trace) -> FailureClassification:
        # Fallback if no judge available.
        if self.judge is None:
            return self._rule_fallback.classify(report, trace)

        # Build the context: what failed and why.
        failure_details = self._build_failure_context(report, trace)
        if not failure_details["signals"]:
            # No failures — shouldn't happen, but use rule fallback.
            return self._rule_fallback.classify(report, trace)

        # Prepare the judge call.
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(failure_details, trace)
        schema = self._output_schema()

        try:
            result = self.judge.score(system=system_prompt, prompt=user_prompt, schema=schema)
            return self._parse_classification(result, report, trace)
        except Exception as exc:
            # Judge failed; fall back to rules.
            # (In production, log this; for now, silent fallback is ok.)
            return self._rule_fallback.classify(report, trace)

    def _build_failure_context(self, report: EvalReport, trace: Trace) -> dict[str, Any]:
        """Extract the failure signals and details from the report."""
        signals = []
        span_evidence = {}

        for result in report.results:
            if result.failure_label:
                signals.append(
                    {
                        "label": result.failure_label,
                        "metric": result.name,
                        "rationale": result.rationale,
                    }
                )
                # Collect span-level evidence.
                if result.details.get("span_ids"):
                    span_ids = result.details["span_ids"]
                    if not isinstance(span_ids, list):
                        span_ids = [span_ids]
                    for sid in span_ids:
                        if sid not in span_evidence:
                            span_evidence[sid] = []
                        span_evidence[sid].append(result.failure_label)

        return {
            "signals": signals,
            "labels": report.failure_labels,
            "span_evidence": span_evidence,
            "overall_score": report.overall_score,
        }

    def _system_prompt(self) -> str:
        """The system prompt for the classifier."""
        taxonomy_values = ", ".join(f.value for f in FailureLabel)
        return f"""You are a failure-analysis expert for LLM-powered agent systems.

Given an agent's execution trace and the evaluation metrics that failed, diagnose
the ROOT CAUSE of the failure. Distinguish between:
- Root cause: the fundamental decision or behavior that led to failure.
- Symptoms: downstream consequences of the root cause.

For example:
- If the agent called the wrong tool AND then had no recovery, the root cause is
  the wrong tool selection (the recovery failure is a symptom).
- If there are redundant calls AND budget exceeded, redundant calls is likely root
  (budget exceeded is a symptom).

Return your diagnosis as a JSON object constrained to the failure taxonomy.
Valid root causes: {taxonomy_values}"""

    def _user_prompt(self, failure_details: dict[str, Any], trace: Trace) -> str:
        """Build the user prompt with failure context and trace summary."""
        # Summarize the failure signals.
        signals_text = "\n".join(
            f"  - {s['label']} (from {s['metric']}): {s['rationale']}"
            for s in failure_details["signals"]
        )

        # Summarize the trace: tool calls, errors, timeline.
        trace_summary = self._summarize_trace(trace, failure_details.get("span_evidence", {}))

        prompt = f"""Agent Execution Analysis

FAILURE SIGNALS (from evaluation):
{signals_text}

EXECUTION TRACE SUMMARY:
{trace_summary}

TASK:
1. Identify the ROOT CAUSE among the failure signals above.
2. Explain why it is the root cause (not a symptom).
3. Point to the specific span/step in the trace that exemplifies it.
4. Provide your confidence in this diagnosis.

Output your diagnosis as JSON."""
        return prompt

    def _summarize_trace(self, trace: Trace, span_evidence: dict[str, list[str]]) -> str:
        """Create a concise summary of the trace for the judge."""
        if not trace.spans:
            return "  (empty trace — no spans recorded)"

        lines = []
        for i, span in enumerate(trace.spans):
            # Mark spans with evidence.
            evidence_marker = ""
            if span.span_id in span_evidence:
                labels = ", ".join(span_evidence[span.span_id])
                evidence_marker = f" ⚠ [{labels}]"

            # Brief summary of the span.
            status_marker = "✓" if span.status.value == "ok" else "✗"
            kind_short = span.kind.value[:4] if span.kind else "?????"
            inputs_str = json.dumps(span.inputs)[:30] if span.inputs else ""
            lines.append(
                f"  Step {i}: {status_marker} {kind_short:5} {span.name:20} {inputs_str:30}{evidence_marker}"
            )

        return "\n".join(lines)

    def _output_schema(self) -> dict[str, Any]:
        """JSON schema for the classifier's output."""
        return {
            "type": "object",
            "properties": {
                "root_cause": {
                    "type": "string",
                    "enum": [f.value for f in FailureLabel],
                    "description": "The root cause of the failure (a FailureLabel).",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this is the root cause (1-2 sentences).",
                },
                "culprit_span_index": {
                    "type": "integer",
                    "description": "The zero-based index of the span most responsible (or -1 if not applicable).",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in this diagnosis, [0,1].",
                },
            },
            "required": ["root_cause", "rationale", "culprit_span_index", "confidence"],
            "additionalProperties": False,
        }

    def _parse_classification(
        self, judge_output: dict[str, Any], report: EvalReport, trace: Trace
    ) -> FailureClassification:
        """Parse the judge's output into a FailureClassification."""
        # Map span index back to span_id if available.
        culprit_id = None
        span_idx = judge_output.get("culprit_span_index", -1)
        if span_idx >= 0 and span_idx < len(trace.spans):
            culprit_id = trace.spans[span_idx].span_id

        return FailureClassification(
            root_cause=judge_output.get("root_cause"),
            rationale=judge_output.get("rationale", ""),
            culprit_span_id=culprit_id,
            confidence=judge_output.get("confidence"),
            evidence_labels=report.failure_labels,
        )
