from datetime import datetime

from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class KnowledgeDocument(Base):
    """One chunk of an uploaded document."""

    __tablename__ = "knowledge_document"
    __table_args__ = (Index("ix_knowledge_document_user_file", "user_email", "filename"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_email: Mapped[str | None] = mapped_column(String(100))
    filename: Mapped[str | None] = mapped_column(String(200))
    chunk_index: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str | None] = mapped_column(Text)
    timestamp: Mapped[datetime | None] = mapped_column(default=utcnow)
