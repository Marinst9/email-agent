"""Schema-validated Claude calls with a self-correction loop, and the single-call EmailAgentAnalyzer.

`create_validated` asks Claude for JSON matching a Pydantic model. If the reply is not valid JSON
or fails validation, the error is sent back to Claude so it can correct its output, up to
`max_attempts` times.
"""

import json
import logging
import re
from typing import TypeVar

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam
from pydantic import BaseModel, ValidationError

from app.core.telemetry import log_usage
from app.schemas.agent import OrchestrationResult, RetrievedDoc

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
LOW_CONFIDENCE_THRESHOLD = 0.85

ModelT = TypeVar("ModelT", bound=BaseModel)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class StructuredOutputError(Exception):
    """Claude did not produce schema-valid output (after all correction attempts, or it declined)."""


def schema_instructions(output_model: type[BaseModel]) -> str:
    schema = json.dumps(output_model.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "Return ONLY a valid JSON object - no prose and no Markdown code fences - "
        f"that conforms to this JSON schema:\n{schema}"
    )


def parse_json_output(raw: str, output_model: type[ModelT]) -> ModelT:
    """json.loads + Pydantic validation. Tolerates a surrounding ```json fence."""
    return output_model.model_validate(json.loads(_CODE_FENCE.sub("", raw.strip())))


def _response_text(response: Message) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


async def create_validated(
    client: AsyncAnthropic,
    *,
    model: str,
    system: str,
    messages: list[MessageParam],
    output_model: type[ModelT],
    max_attempts: int = 3,
    max_tokens: int = 4096,
) -> ModelT:
    conversation: list[MessageParam] = list(messages)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        response = await client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=conversation
        )
        log_usage(f"structured_output:{output_model.__name__}:attempt{attempt}", response)
        if response.stop_reason == "refusal":
            raise StructuredOutputError("Claude declined to answer this request")

        raw = _response_text(response)
        try:
            return parse_json_output(raw, output_model)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("Attempt %d/%d returned invalid output: %s", attempt, max_attempts, exc)
            truncated = (
                "\nYour output was cut off by the token limit - keep it shorter."
                if response.stop_reason == "max_tokens"
                else ""
            )
            conversation += [
                {"role": "assistant", "content": raw or "(empty response)"},
                {
                    "role": "user",
                    "content": (
                        "Your previous output was not valid JSON for the required schema. "
                        f"Error:\n{exc}{truncated}\n\n"
                        "Return ONLY the corrected JSON object."
                    ),
                },
            ]

    raise StructuredOutputError(
        f"No schema-valid output after {max_attempts} attempts: {last_error}"
    ) from last_error


class EmailAgentAnalyzer:
    def __init__(self, ai_client: AsyncAnthropic, model: str = DEFAULT_MODEL) -> None:
        self.ai_client = ai_client
        self.model = model

    async def analyze_and_orchestrate(
        self,
        email_body: str,
        sender: str,
        retrieved_docs: list[RetrievedDoc],
        max_retries: int = 3,
    ) -> OrchestrationResult:
        """Run the whole pipeline in one call, with schema enforcement and the self-correction loop."""
        docs_context = "\n".join(
            f"[{doc.index}] (Similarity: {doc.similarity:.2f}): {doc.content}" for doc in retrieved_docs
        )
        system_prompt = (
            "You are an enterprise email processing agent. "
            "Analyze the inbound email and context.\n" + schema_instructions(OrchestrationResult)
        )
        user_message = (
            f"Sender: {sender}\n"
            f"Email Body:\n{email_body}\n\n"
            f"Retrieved Knowledge Documents:\n{docs_context or 'No relevant context found.'}"
        )

        result = await create_validated(
            self.ai_client,
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            output_model=OrchestrationResult,
            max_attempts=max_retries,
        )

        # Safeguard: low-confidence results always go to a human.
        if result.confidence < LOW_CONFIDENCE_THRESHOLD and not result.review.needs_review:
            result.review.needs_review = True
            result.review.auto_send = False
            result.review.reason = f"{result.review.reason} [Auto-flagged: Low confidence score]".strip()

        return result
