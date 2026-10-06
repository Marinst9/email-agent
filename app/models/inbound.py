from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, Float, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow
from app.models.enums import DraftAction, InboundStatus


def _new_id() -> str:
    return uuid4().hex


class InboundEmail(Base):
    """An email picked up from Gmail, plus the draft produced for it by the worker.

    Replaces the in-memory pending list: the review queue is now durable and shared by all processes.
    """

    __tablename__ = "inbound_email"
    __table_args__ = (
        # Idempotent ingestion: the same Gmail message is never queued twice for a user.
        UniqueConstraint("user_email", "gmail_message_id", name="uq_inbound_email_user_message"),
        Index("ix_inbound_email_user_status", "user_email", "status"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_email: Mapped[str] = mapped_column(String(100))
    gmail_message_id: Mapped[str] = mapped_column(String(100))
    thread_id: Mapped[str] = mapped_column(String(200), default="")
    sender: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    # Snapshot of the agent mode at ingestion time.
    auto_send: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(String(20), default=InboundStatus.QUEUED.value)

    # Draft (filled in by the worker)
    response: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(150))
    category: Mapped[str | None] = mapped_column(String(50))
    priority: Mapped[str | None] = mapped_column(String(20))
    sentiment: Mapped[str | None] = mapped_column(String(20))
    confidence: Mapped[float | None] = mapped_column(Float)
    reasoning: Mapped[str | None] = mapped_column(Text)
    docs_used: Mapped[list[str] | None] = mapped_column(JSON)
    needs_review: Mapped[bool] = mapped_column(default=False)
    review_reason: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(20), default=DraftAction.REPLY.value)
    # Recipient of a FORWARD draft (proposed by the AI, possibly corrected by the reviewer).
    forward_to: Mapped[str | None] = mapped_column(String(320))

    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
