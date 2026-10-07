from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import DraftAction as DraftAction
from app.models.enums import EmailStatus, InboundStatus
from app.schemas.common import DisplayStr, normalize_email_address

# --- Forms -------------------------------------------------------------------


class StartAgentForm(BaseModel):
    mode: Literal["auto", "manual"] = "manual"


class ApproveForm(BaseModel):
    custom_response: str | None = None
    # Only used for FORWARD drafts: lets the reviewer set or correct the recipient.
    forward_to: str | None = None

    @field_validator("forward_to", mode="before")
    @classmethod
    def valid_address(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        address = normalize_email_address(str(value))
        if address is None:
            raise ValueError("forward_to must be an email address")
        return address


# --- Multi-agent pipeline ----------------------------------------------------


class Classification(BaseModel):
    category: str = "INQUIRY"  # INQUIRY / COMPLAINT / URGENT_HUMAN / SPAM
    priority: str = "MEDIUM"  # HIGH / MEDIUM / LOW
    language: str = "mk"
    sentiment: str = "neutral"


class RetrievedDoc(BaseModel):
    """A knowledge-base chunk returned by hybrid search, with the scores that ranked it."""

    chunk_id: int
    content: str
    filename: str | None = None
    chunk_index: int | None = None
    page: int | None = None
    # Reciprocal Rank Fusion score of the vector and full-text rankings.
    score: float
    # Cosine similarity to the query embedding; None when the chunk came only from full-text search.
    vector_similarity: float | None = None
    # Postgres ts_rank_cd; None when the chunk came only from vector search.
    text_rank: float | None = None
    # 0-3 relevance from the optional reranker.
    rerank_score: float | None = None


SNIPPET_CHARS = 300


class Citation(BaseModel):
    """A chunk the drafting agent says it used, as shown to the reviewer."""

    chunk_id: int
    filename: str | None = None
    page: int | None = None
    snippet: str

    @classmethod
    def from_doc(cls, doc: RetrievedDoc) -> "Citation":
        return cls(chunk_id=doc.chunk_id, filename=doc.filename, page=doc.page, snippet=doc.content[:SNIPPET_CHARS])

    def label(self) -> str:
        return f"{self.filename or '?'}" + (f", стр. {self.page}" if self.page is not None else "")


class DraftResult(BaseModel):
    raw: str
    action: DraftAction
    response_text: str
    # Retrieved chunks the draft cites (only ids that were actually retrieved are kept).
    citations: list[Citation] = Field(default_factory=list)
    reasoning: str
    # A user-written reply template, not an AI draft: it needs no knowledge-base grounding.
    from_template: bool = False
    # Recipient when action is FORWARD.
    forward_to: str | None = None


class ReviewDecision(BaseModel):
    needs_review: bool
    reason: str = ""
    auto_send: bool


class OrchestrationResult(BaseModel):
    action: DraftAction
    classification: Classification
    retrieved_docs: list[RetrievedDoc]
    draft: DraftResult | None
    review: ReviewDecision
    # Only set by `EmailAgentAnalyzer`, where Claude reports its own confidence; the multi-agent pipeline has none.
    confidence: float | None = None
    reasoning: str


# --- Dashboard state ---------------------------------------------------------

BODY_PREVIEW_CHARS = 300

_STATUS_LABELS: dict[str, EmailStatus] = {
    InboundStatus.SENT.value: EmailStatus.SENT,
    InboundStatus.REJECTED.value: EmailStatus.REJECTED,
}


class PendingEmail(BaseModel):
    """Dashboard view of an `InboundEmail` row that has a proposed reply."""

    model_config = ConfigDict(from_attributes=True)

    id: str  # InboundEmail.id (not the Gmail message id)
    sender: str
    subject: str
    body: str
    response: DisplayStr
    source: DisplayStr
    category: DisplayStr
    confidence: float | None
    reasoning: DisplayStr
    docs_used: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    priority: DisplayStr
    sentiment: DisplayStr
    needs_review: bool = False
    review_reason: DisplayStr = ""
    action: DraftAction = DraftAction.REPLY
    forward_to: str | None = None

    @field_validator("body")
    @classmethod
    def preview(cls, value: str) -> str:
        return value[:BODY_PREVIEW_CHARS]

    @field_validator("docs_used", "citations", mode="before")
    @classmethod
    def none_to_list(cls, value: Any) -> Any:
        return [] if value is None else value


class ProcessedEmail(PendingEmail):
    status: EmailStatus

    @field_validator("status", mode="before")
    @classmethod
    def to_label(cls, value: Any) -> Any:
        # Templates compare against the Macedonian labels used in the history log.
        return _STATUS_LABELS.get(str(value), value)


class AgentStateView(BaseModel):
    running: bool
    auto_mode: bool
    pending: list[PendingEmail]
    processed: list[ProcessedEmail]
