"""Token usage and estimated cost of every Anthropic API call, taken from `response.usage`."""

import logging
from dataclasses import dataclass

from anthropic.types import Message, Usage

logger = logging.getLogger("telemetry")

CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute cache writes cost 1.25x the base input price
CACHE_READ_MULTIPLIER = 0.1  # default when a model has no explicit cache-read price


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens (Anthropic first-party API rates)."""

    input: float
    output: float
    cache_read: float | None = None

    def cost_usd(self, usage: Usage) -> float:
        cache_write = usage.cache_creation_input_tokens or 0
        cache_read = usage.cache_read_input_tokens or 0
        cache_read_price = self.cache_read if self.cache_read is not None else self.input * CACHE_READ_MULTIPLIER
        total = (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + cache_write * self.input * CACHE_WRITE_MULTIPLIER
            + cache_read * cache_read_price
        )
        return total / 1_000_000


# Matched by longest prefix, so dated snapshot ids resolve to their family.
MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(10.0, 50.0, cache_read=0.25),
    "claude-fable-5": ModelPrice(10.0, 50.0),
    "claude-opus-5-5": ModelPrice(4.0, 20.0, cache_read=0.20),
    "claude-opus-5": ModelPrice(5.0, 25.0),
    "claude-opus-4": ModelPrice(5.0, 25.0),
    "claude-sonnet-5-5": ModelPrice(2.0, 10.0, cache_read=0.20),
    "claude-sonnet-5": ModelPrice(2.0, 10.0),
    "claude-sonnet-4": ModelPrice(3.0, 15.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
}


def price_for(model: str) -> ModelPrice | None:
    matches = [prefix for prefix in MODEL_PRICES if model.startswith(prefix)]
    return MODEL_PRICES[max(matches, key=len)] if matches else None


def estimate_cost_usd(model: str, usage: Usage) -> float | None:
    price = price_for(model)
    return price.cost_usd(usage) if price is not None else None


def log_usage(operation: str, response: Message) -> None:
    """Log one structured line per LLM call; cost is omitted for models without a known price."""
    usage = response.usage
    cost = estimate_cost_usd(response.model, usage)
    logger.info(
        "llm_call operation=%s model=%s input_tokens=%d output_tokens=%d "
        "cache_write_tokens=%d cache_read_tokens=%d cost_usd=%s",
        operation,
        response.model,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens or 0,
        usage.cache_read_input_tokens or 0,
        f"{cost:.6f}" if cost is not None else "unknown",
    )
