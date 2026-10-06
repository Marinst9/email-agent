"""Self-correction loop for schema-validated Claude output (fake client, no network)."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import Usage

from app.agents import StructuredOutputError, create_validated, parse_json_output, schema_instructions
from app.schemas.agent import DraftAction, OrchestrationResult

VALID = {
    "action": "ОДГОВОР",
    "classification": {"category": "INQUIRY", "priority": "LOW", "language": "mk", "sentiment": "neutral"},
    "retrieved_docs": [],
    "draft": {
        "raw": "",
        "action": "ОДГОВОР",
        "response_text": "Здраво!",
        "confidence": 0.9,
        "docs_used": [],
        "reasoning": "Прашање",
        "forward_to": None,
    },
    "review": {"needs_review": False, "reason": "", "auto_send": True},
    "confidence": 0.9,
    "reasoning": "Реална личност бара одговор",
}


class FakeMessages:
    def __init__(self, outputs: list[tuple[str, str]]) -> None:
        self._outputs = outputs
        self.calls: list[list[dict[str, Any]]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(list(kwargs["messages"]))
        text, stop_reason = self._outputs[len(self.calls) - 1]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=stop_reason,
            model=kwargs["model"],
            usage=Usage(input_tokens=10, output_tokens=5),
        )


def _client(*outputs: tuple[str, str]) -> tuple[AsyncAnthropic, FakeMessages]:
    messages = FakeMessages(list(outputs))
    return cast(AsyncAnthropic, SimpleNamespace(messages=messages)), messages


async def _decide(client: AsyncAnthropic, max_attempts: int = 3) -> OrchestrationResult:
    return await create_validated(
        client,
        model="m",
        system="s",
        messages=[{"role": "user", "content": "email"}],
        output_model=OrchestrationResult,
        max_attempts=max_attempts,
    )


async def test_valid_first_answer_needs_one_call() -> None:
    client, fake = _client((json.dumps(VALID), "end_turn"))
    result = await _decide(client)
    assert result.action is DraftAction.REPLY and result.draft and result.draft.response_text == "Здраво!"
    assert len(fake.calls) == 1


async def test_invalid_json_then_schema_error_are_fed_back_until_valid() -> None:
    missing_review = {k: v for k, v in VALID.items() if k != "review"}
    client, fake = _client(
        ("Sure! Here is the JSON:", "end_turn"),
        (json.dumps(missing_review), "end_turn"),
        (f"```json\n{json.dumps(VALID)}\n```", "end_turn"),
    )
    result = await _decide(client)
    assert result.review.auto_send is True
    assert len(fake.calls) == 3
    # The third request carries both earlier failures and their error messages.
    third = fake.calls[2]
    assert [m["role"] for m in third] == ["user", "assistant", "user", "assistant", "user"]
    assert "review" in third[-1]["content"]


async def test_gives_up_after_max_attempts() -> None:
    client, fake = _client(("nope", "end_turn"), ("still nope", "max_tokens"))
    with pytest.raises(StructuredOutputError):
        await _decide(client, max_attempts=2)
    assert len(fake.calls) == 2


async def test_refusal_is_not_retried() -> None:
    client, fake = _client(("", "refusal"))
    with pytest.raises(StructuredOutputError):
        await _decide(client)
    assert len(fake.calls) == 1


def test_schema_instructions_embed_the_pydantic_schema() -> None:
    text = schema_instructions(OrchestrationResult)
    assert "ONLY a valid JSON object" in text
    assert "forward_to" in text and "ИГНОРИРАЈ" in text


def test_parse_json_output_accepts_fenced_json() -> None:
    assert parse_json_output(f"```json\n{json.dumps(VALID)}\n```", OrchestrationResult).confidence == 0.9
