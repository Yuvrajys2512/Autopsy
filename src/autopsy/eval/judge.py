"""The LLM-as-judge transport layer.

A clean split: the **metric** owns the *rubric* (the question, the instructions,
the output schema); the **Judge** owns the *transport* (how that prompt becomes a
model call and a parsed dict). That separation is the whole lesson of
LLM-as-judge — a judge is useless without a rubric, and the rubric is the part
you iterate on and calibrate against human labels.

`Judge` is a Protocol, so you can plug in any backend (a rotating multi-provider
client, a local model, a stub for tests). `AnthropicJudge` is the batteries-
included default, using structured outputs so the response is guaranteed to
match the schema.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """Anything that can grade a prompt against a JSON schema.

    Implementations must return a ``dict`` conforming to ``schema``.
    """

    def score(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        ...


class AnthropicJudge:
    """Default judge backed by the Anthropic SDK and structured outputs.

    Uses ``output_config.format`` (json_schema) so the model's reply is
    constrained to the rubric's schema — no brittle parsing of free text.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
    ) -> None:
        if client is None:
            import anthropic  # imported lazily so the core has no hard dep

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def score(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            None,
        )
        if text is None:
            raise ValueError("judge returned no text content")
        return json.loads(text)


# Reusable JSON schema for a graded judgement. ``additionalProperties: false`` is
# required by the structured-outputs feature.
def judgement_schema(*, verdict_key: str = "passed") -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number", "description": "Quality in [0,1], higher is better"},
            verdict_key: {"type": "boolean"},
            "rationale": {"type": "string", "description": "Brief justification for the score"},
        },
        "required": ["score", verdict_key, "rationale"],
        "additionalProperties": False,
    }
