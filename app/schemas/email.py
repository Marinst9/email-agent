from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import EmailStatus, FeedbackValue
from app.schemas.common import DisplayStr


class GmailMessage(BaseModel):
    id: str
    thread_id: str
    sender: str
    subject: str
    body: str


class EmailLogCreate(BaseModel):
    user_email: str
    sender: str
    subject: str
    response: str
    source: str
    status: EmailStatus
    category: str
    confidence: float | None = None
    reasoning: str | None = None
    docs_used: list[str] = Field(default_factory=list)
    priority: str | None = None
    sentiment: str | None = None


class EmailLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sender: DisplayStr
    subject: DisplayStr
    response: DisplayStr
    source: DisplayStr
    status: DisplayStr
    category: DisplayStr
    confidence: float | None
    priority: str | None
    sentiment: str | None
    timestamp: datetime | None


class ThreadTurn(BaseModel):
    """A previous exchange in the same Gmail thread, fed to the drafting agent as context."""

    model_config = ConfigDict(from_attributes=True)

    body: DisplayStr
    response: DisplayStr


class FeedbackForm(BaseModel):
    feedback: FeedbackValue
    corrected_response: str = ""


class FeedbackAck(BaseModel):
    status: Literal["ok"] = "ok"
