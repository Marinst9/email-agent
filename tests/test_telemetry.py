"""Token usage and cost logging from `response.usage`."""

import logging

import pytest
from anthropic.types import Message, TextBlock, Usage

from app.core.telemetry import estimate_cost_usd, log_usage, price_for


def test_cost_uses_the_model_price_including_cache_tokens() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_creation_input_tokens=100_000,
        cache_read_input_tokens=1_000_000,
    )
    # Sonnet 4.6: $3 in, $15 out, cache write 1.25x, cache read 0.1x.
    assert estimate_cost_usd("claude-sonnet-4-6", usage) == pytest.approx(3 + 1.5 + 0.375 + 0.3)


def test_longest_prefix_wins_and_unknown_models_have_no_price() -> None:
    assert price_for("claude-opus-5-5") == price_for("claude-opus-5-5-20260901")
    assert price_for("claude-opus-5-5") != price_for("claude-opus-5")
    assert estimate_cost_usd("some-other-model", Usage(input_tokens=1, output_tokens=1)) is None


def test_log_usage_writes_one_structured_line(caplog: pytest.LogCaptureFixture) -> None:
    response = Message.model_construct(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[TextBlock(type="text", text="ok")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=120, output_tokens=30),
    )
    with caplog.at_level(logging.INFO, logger="telemetry"):
        log_usage("classify", response)
    (record,) = caplog.records
    message = record.getMessage()
    assert "operation=classify" in message and "input_tokens=120" in message and "output_tokens=30" in message
    assert "cost_usd=0.000810" in message
