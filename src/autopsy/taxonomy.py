"""The failure taxonomy — shared vocabulary for *why* a run is bad.

Phase 1 evaluators emit these labels when a metric fails; Phase 2's classifier
will reason over the same set. Keeping the names in one place means a metric and
the classifier can never drift on spelling. This is the "starter" taxonomy from
the project brief; it's an open set we'll extend as real failures show up.
"""

from __future__ import annotations

from enum import Enum


class FailureLabel(str, Enum):
    # Tool selection / arguments
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    HALLUCINATED_TOOL_ARGS = "hallucinated_tool_args"

    # Error handling
    NO_RETRY_ON_ERROR = "no_retry_on_error"
    GAVE_UP_EARLY = "gave_up_early"

    # Trajectory shape
    REDUNDANT_CALLS = "redundant_calls"
    INFINITE_LOOP = "infinite_loop"
    BUDGET_EXCEEDED = "budget_exceeded"
    CONTEXT_OVERFLOW = "context_overflow"

    # Retrieval / synthesis
    RETRIEVAL_MISS = "retrieval_miss"
    CORRECT_RETRIEVAL_WRONG_ANSWER = "correct_retrieval_wrong_answer"

    # Outcome
    GOAL_NOT_COMPLETED = "goal_not_completed"
