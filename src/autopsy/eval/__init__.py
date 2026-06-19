"""Autopsy evaluation layer (Phase 1–4).

Rule-based and LLM-as-judge metrics over a captured trace, aggregated into a
structured report by `evaluate` (Phase 1). Failure classification synthesizes
the report into a root-cause diagnosis with evidence (Phase 2). Datasets and
batch runners orchestrate large-scale evaluation (Phase 3). Comparison and
history track batch results across runs to flag regressions and trends
(Phase 4).
"""

from __future__ import annotations

from .aggregation import (
    AggregateMetrics,
    BatchCaseResult,
    BatchResultSet,
    aggregate_diagnoses,
)
from .base import Evaluator
from .batch import BatchResult, BatchRunner, batch_evaluate
from .classifier import (
    FailureClassification,
    FailureClassifier,
    LLMFailureClassifier,
    RuleBasedFailureClassifier,
)
from .comparison import CaseDiff, ComparisonMetrics, compare
from .dataset import Dataset, TestCase
from .history import BatchRunMetadata, HistoryEntry, RunHistory
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
    # aggregation (Phase 3)
    "AggregateMetrics",
    "BatchCaseResult",
    "BatchResultSet",
    "aggregate_diagnoses",
    # comparison & history (Phase 4)
    "compare",
    "ComparisonMetrics",
    "CaseDiff",
    "RunHistory",
    "BatchRunMetadata",
    "HistoryEntry",
]
