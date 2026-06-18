"""Dataset abstraction for Phase 3 — labeled test cases for batch evaluation.

A TestCase encodes a single evaluation scenario: input, expected output,
which tools should be used, and resource budgets. A Dataset is a collection
of TestCases with metadata (name, version, source).

Serialization is JSON for simplicity and portability; load/save are local files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .models import Expectations


class TestCase(BaseModel):
    """One labeled test case for batch evaluation.

    Attributes:
        input: The agent's input (e.g., a question or command).
        expected_output: Gold answer, if available (optional).
        expected_tools: Tools that *should* have been used (optional).
        allowed_tools: The full tool registry (optional).
        tool_schemas: JSON schemas for arguments (optional).
        max_steps: Token/step budget (optional).
        max_cost_usd: Cost budget (optional).
        max_latency_ms: Latency budget (optional).
        metadata: Free-form dict for caller use (tags, source, etc).
    """

    input: str
    expected_output: Optional[str] = None
    expected_tools: Optional[list[str]] = None
    allowed_tools: Optional[list[str]] = None
    tool_schemas: Optional[dict[str, dict]] = None
    max_steps: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_expectations(self, goal: Optional[str] = None) -> Expectations:
        """Convert this TestCase into an Expectations object for evaluation."""
        return Expectations(
            goal=goal or self.input,
            reference_output=self.expected_output,
            expected_tools=self.expected_tools,
            allowed_tools=self.allowed_tools,
            tool_schemas=self.tool_schemas,
            max_steps=self.max_steps,
            max_cost_usd=self.max_cost_usd,
            max_latency_ms=self.max_latency_ms,
        )


class Dataset(BaseModel):
    """A collection of labeled test cases.

    Attributes:
        name: Dataset name (e.g., "Q&A Basic", "Tool Selection Hard").
        version: Semantic version string (e.g., "1.0.0").
        description: Human-readable description of the dataset.
        source: Where the dataset came from (e.g., "human-labeled", "synthetic").
        cases: List of TestCases.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    source: str = ""
    cases: list[TestCase] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> Dataset:
        """Load a dataset from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A Dataset instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            ValueError: If the JSON structure doesn't match the Dataset schema.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Save the dataset to a JSON file.

        Args:
            path: Path where the file will be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, index: int) -> TestCase:
        return self.cases[index]
