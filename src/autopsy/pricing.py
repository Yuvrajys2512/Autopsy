"""Token pricing and cost estimation.

A small, explicit table — not a live feed. Prices are USD per 1M tokens as
``(input, output)``. Cache reads/writes are folded into the input rate as a
rough approximation (Phase 0); the failure metrics that care about exact cost
come later. Unknown models return ``None`` rather than guessing.
"""

from __future__ import annotations

from typing import Optional

from .schema import TokenUsage

# USD per 1,000,000 tokens: (input, output)
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # OpenAI (common models; extend as needed)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
}


def get_pricing(model: Optional[str]) -> Optional[tuple[float, float]]:
    """Look up ``(input, output)`` rates for a model id.

    Tries an exact match first, then the longest registered prefix — so a
    dated variant like ``claude-haiku-4-5-20251001`` still resolves.
    """
    if not model:
        return None
    if model in PRICING:
        return PRICING[model]
    matches = [key for key in PRICING if model.startswith(key)]
    if matches:
        return PRICING[max(matches, key=len)]
    return None


def compute_cost(model: Optional[str], usage: Optional[TokenUsage]) -> Optional[float]:
    """Estimate the USD cost of a call. ``None`` if the model is unknown."""
    if usage is None:
        return None
    rates = get_pricing(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    billable_input = (
        usage.input_tokens
        + usage.cache_read_input_tokens
        + usage.cache_creation_input_tokens
    )
    return (billable_input * input_rate + usage.output_tokens * output_rate) / 1_000_000.0
