"""Background task execution state (stored in Redis) and the polling API's response models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

from app.models.enums import InboundStatus


class TaskState(StrEnum):
    PENDING = "PENDING"  # queued or running (see `stage` for progress)
    RETRYING = "RETRYING"  # a transient failure occurred; another attempt is scheduled or running
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TaskStage(StrEnum):
    QUEUED = "queued"
    FILTERING = "filtering"
    CLASSIFYING = "classifying"
    RETRIEVING = "retrieving"
    DRAFTING = "drafting"
    REVIEWING = "reviewing"
    DELIVERING = "delivering"
    DONE = "done"


class TaskStatusRecord(BaseModel):
    state: TaskState
    stage: TaskStage
    attempts: int = 0
    error: str | None = None
    next_retry_at: datetime | None = None
    updated_at: datetime


class DraftOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    response: str
    source: str | None
    category: str | None
    priority: str | None
    sentiment: str | None
    confidence: float | None
    reasoning: str | None
    docs_used: list[str]
    needs_review: bool
    review_reason: str | None

    @field_validator("docs_used", mode="before")
    @classmethod
    def none_to_list(cls, value: object) -> object:
        return [] if value is None else value


class EmailProcessingStatus(BaseModel):
    email_id: str
    state: TaskState
    stage: TaskStage | None
    attempts: int
    error: str | None
    next_retry_at: datetime | None
    updated_at: datetime | None
    outcome: InboundStatus
    draft: DraftOut | None
