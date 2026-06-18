"""Autopsy evaluation layer (Phase 1).

Rule-based and LLM-as-judge metrics over a captured trace, aggregated into a
structured report by `evaluate`.
"""

from __future__ import annotations

from .base import Evaluator
from .judge import AnthropicJudge, Judge, judgement_schema
from .llm_metrics import Faithfulness, GoalCompletion, LLMJudgeEvaluator
from .models import EvalReport, EvalResult, Expectations
from .rules import (
    RULE_EVALUATORS,
    ErrorRecovery,
    ResourceBudget,
    ToolArgumentValidity,
    ToolSelectionAccuracy,
    TrajectoryEfficiency,
)
from .runner import default_evaluators, evaluate

__all__ = [
    "evaluate",
    "default_evaluators",
    "EvalReport",
    "EvalResult",
    "Expectations",
    "Evaluator",
    # rule metrics
    "TrajectoryEfficiency",
    "ErrorRecovery",
    "ToolArgumentValidity",
    "ToolSelectionAccuracy",
    "ResourceBudget",
    "RULE_EVALUATORS",
    # judge
    "Judge",
    "AnthropicJudge",
    "judgement_schema",
    "LLMJudgeEvaluator",
    "GoalCompletion",
    "Faithfulness",
]
