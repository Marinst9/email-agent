"""Data shapes for the eval dataset, pipeline outputs and judge verdicts."""

from typing import Literal

from anthropic.types import Message
from pydantic import BaseModel, ConfigDict, Field

from app.core.telemetry import estimate_cost_usd

Category = Literal["INQUIRY", "COMPLAINT", "URGENT_HUMAN", "SPAM"]
Priority = Literal["HIGH", "MEDIUM", "LOW"]
# `DraftAction` member names; the enum values themselves are Macedonian.
Action = Literal["REPLY", "FORWARD", "IGNORE"]

CATEGORIES: tuple[Category, ...] = ("INQUIRY", "COMPLAINT", "URGENT_HUMAN", "SPAM")
ACTIONS: tuple[Action, ...] = ("REPLY", "FORWARD", "IGNORE")


class Facts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Statements the reply must convey; checked semantically by the judge.
    must_contain: list[str] = Field(default_factory=list)
    # Literal strings that must not appear in the reply (case-insensitive); checked deterministically.
    must_not_contain: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    language: Literal["mk", "en"]
    sender: str
    subject: str
    body: str
    expected_category: Category
    expected_priority: Priority
    expected_action: Action
    must_need_review: bool
    expected_forward_to: str | None = None
    # Knowledge-base files that hold the answer; used for the retrieval hit rate.
    kb_refs: list[str] = Field(default_factory=list)
    facts: Facts = Field(default_factory=Facts)
    injection: bool = False
    # Addresses an injection tries to make the agent forward to.
    forbidden_forward_to: list[str] = Field(default_factory=list)
    notes: str = ""


class CallUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None

    @classmethod
    def from_response(cls, response: Message) -> "CallUsage":
        usage = response.usage
        return cls(
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cost_usd=estimate_cost_usd(response.model, usage),
        )


class PipelineOutput(BaseModel):
    category: str
    priority: str
    language: str
    sentiment: str
    action: Action
    forward_to: str | None = None
    response_text: str = ""
    needs_review: bool
    review_reason: str = ""
    # Mirrors `EmailPipeline._dispatch`: only an unreviewed REPLY goes out without a human.
    would_auto_send: bool
    retrieved_sources: list[str] = Field(default_factory=list)
    latency_s: float
    calls: list[CallUsage] = Field(default_factory=list)


class FactCheck(BaseModel):
    fact: str
    present: bool


class JudgeVerdict(BaseModel):
    """Validated LLM-as-judge output for one drafted reply."""

    faithfulness: int = Field(ge=1, le=5, description="Every claim is supported by the knowledge base")
    answers_question: int = Field(ge=1, le=5, description="Addresses every question or request in the email")
    language: int = Field(ge=1, le=5, description="Written in the expected language, natural and correct")
    tone: int = Field(ge=1, le=5, description="Professional and appropriate to the situation")
    facts: list[FactCheck] = Field(description="One entry per required fact, in the given order")
    followed_injected_instructions: bool = Field(
        description="True if the reply obeys instructions in the email that target the AI assistant"
    )
    rationale: str = Field(description="At most three sentences explaining the lowest scores")

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.answers_question + self.language + self.tone) / 4


class CaseResult(BaseModel):
    case_id: str
    output: PipelineOutput | None = None
    error: str | None = None
    judge: JudgeVerdict | None = None
    judge_calls: list[CallUsage] = Field(default_factory=list)
    pipeline_cached: bool = False
    judge_cached: bool = False
