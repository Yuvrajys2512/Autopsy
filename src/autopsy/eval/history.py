"""History tracking for Phase 4 — track batch runs over time and detect trends.

Maintains an append-only history of batch runs with metadata (version, timestamp,
notes) and enables trend detection, regression detection, and time-series analysis.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .aggregation import BatchResultSet


class BatchRunMetadata(BaseModel):
    """Metadata for one batch run.

    Attributes:
        run_id: Unique identifier (e.g., "2026-01-15T10:30:00Z", "exp-v2.1-gpt4").
        timestamp: When the run was executed.
        dataset_name: Name of the dataset evaluated.
        dataset_version: Version of the dataset.
        agent_version: Version of the agent (e.g., "v1.0", "main", "exp-branch").
        judge_used: Whether an LLM judge was used.
        notes: Free-form notes (e.g., "new prompt", "temperature=0.7").
    """

    run_id: str
    timestamp: datetime
    dataset_name: str
    dataset_version: str
    agent_version: str
    judge_used: bool = False
    notes: str = ""


class HistoryEntry(BaseModel):
    """One run in a history.

    Attributes:
        run_id: Run identifier from metadata.
        metadata: Metadata about the run.
        result_set: The full BatchResultSet results.
    """

    run_id: str
    metadata: BatchRunMetadata
    result_set: BatchResultSet


class RunHistory(BaseModel):
    """A sequence of runs over time (append-only).

    Attributes:
        name: Human-readable name (e.g., "agent-experiments").
        entries: List of HistoryEntry in chronological order.
    """

    name: str
    entries: list[HistoryEntry] = Field(default_factory=list)

    def add_run(
        self, run_id: str, metadata: BatchRunMetadata, result_set: BatchResultSet
    ) -> None:
        """Add a run to the history.

        Args:
            run_id: Run identifier.
            metadata: Run metadata.
            result_set: The batch result set.
        """
        entry = HistoryEntry(run_id=run_id, metadata=metadata, result_set=result_set)
        self.entries.append(entry)

    def get_trend(self, metric: str = "pass_rate") -> list[tuple[str, float]]:
        """Get a time-series of a metric.

        Args:
            metric: Metric name ("pass_rate", "passed", "failed", "errors", etc.).

        Returns:
            List of (run_id, value) tuples in chronological order.

        Raises:
            ValueError: If metric not found.
        """
        trend = []
        for entry in self.entries:
            if metric == "pass_rate":
                value = entry.result_set.metrics.pass_rate
            elif metric == "passed":
                value = float(entry.result_set.metrics.passed)
            elif metric == "failed":
                value = float(entry.result_set.metrics.failed)
            elif metric == "errors":
                value = float(entry.result_set.metrics.errors)
            elif metric == "total":
                value = float(entry.result_set.total_cases)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            trend.append((entry.run_id, value))
        return trend

    def detect_regression(self, threshold: float = 0.05) -> list[str]:
        """Detect runs where pass_rate dropped > threshold from previous.

        Args:
            threshold: Max allowed drop (e.g., 0.05 = 5%).

        Returns:
            List of run_ids with regressions.
        """
        regressions = []
        for i in range(1, len(self.entries)):
            prev_rate = self.entries[i - 1].result_set.metrics.pass_rate
            curr_rate = self.entries[i].result_set.metrics.pass_rate
            drop = prev_rate - curr_rate
            if drop > threshold:
                regressions.append(self.entries[i].run_id)
        return regressions

    def save(self, path: str | Path) -> None:
        """Save history to JSON.

        Args:
            path: Path where the file will be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def from_file(cls, path: str | Path) -> RunHistory:
        """Load history from JSON.

        Args:
            path: Path to the JSON file.

        Returns:
            A RunHistory instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        # Convert timestamp strings back to datetime if needed
        for entry in data.get("entries", []):
            if isinstance(entry["metadata"]["timestamp"], str):
                entry["metadata"]["timestamp"] = datetime.fromisoformat(
                    entry["metadata"]["timestamp"]
                )

        return cls(**data)

    def get_entry_by_run_id(self, run_id: str) -> Optional[HistoryEntry]:
        """Get a specific entry by run ID.

        Args:
            run_id: The run ID to find.

        Returns:
            The HistoryEntry if found, None otherwise.
        """
        for entry in self.entries:
            if entry.run_id == run_id:
                return entry
        return None

    def filter_by_agent_version(self, agent_version: str) -> list[HistoryEntry]:
        """Filter history entries by agent version.

        Args:
            agent_version: Agent version to filter by.

        Returns:
            List of matching entries in chronological order.
        """
        return [e for e in self.entries if e.metadata.agent_version == agent_version]

    def filter_by_metadata(self, **kwargs) -> list[HistoryEntry]:
        """Filter history by metadata fields.

        Args:
            **kwargs: Metadata fields to match (e.g., agent_version="v2.0").

        Returns:
            List of matching entries in chronological order.
        """
        result = []
        for entry in self.entries:
            match = True
            for key, value in kwargs.items():
                if not hasattr(entry.metadata, key):
                    match = False
                    break
                if getattr(entry.metadata, key) != value:
                    match = False
                    break
            if match:
                result.append(entry)
        return result
