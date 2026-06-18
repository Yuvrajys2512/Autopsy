"""Autopsy evaluation layer (Phase 1, 2 & 3).

Rule-based and LLM-as-judge metrics over a captured trace, aggregated into a
structured report by `evaluate` (Phase 1). Failure classification synthesizes
the report into a root-cause diagnosis with evidence (Phase 2). Datasets and
batch runners orchestrate large-scale evaluation (Phase 3).
"""

from __future__ import annotations

from .base import Evaluator
from .batch import BatchResult, BatchRunner, batch_evaluate
from .classifier import (
    FailureClassification,
    FailureClassifier,
    LLMFailureClassifier,
    RuleBasedFailureClassifier,
)
from .dataset import Dataset, TestCase
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
    # classifier (Phase 2)
    "FailureClassification",
    "FailureClassifier",
    "LLMFailureClassifier",
    "RuleBasedFailureClassifier",
    # datasets & batch (Phase 3)
    "Dataset",
    "TestCase",
    "BatchRunner",
    "BatchResult",
    "batch_evaluate",
]
