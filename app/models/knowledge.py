from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow
from app.models.enums import EmbeddingStatus
from app.services.embeddings import EMBEDDING_DIMENSION

# Full-text configuration: 'simple' (lower-casing only), because Postgres ships no Macedonian stemmer.
FTS_CONFIG = "simple"


class KnowledgeDocument(Base):
    """One chunk of an uploaded document."""

    __tablename__ = "knowledge_document"
    __table_args__ = (
        Index("ix_knowledge_document_user_file", "user_email", "filename"),
        Index(
            "ix_knowledge_document_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_knowledge_document_search_vector", "search_vector", postgresql_using="gin"),
        Index("ix_knowledge_document_embedding_status", "embedding_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_email: Mapped[str | None] = mapped_column(String(100))
    filename: Mapped[str | None] = mapped_column(String(200))
    chunk_index: Mapped[int | None] = mapped_column(Integer)
    # 1-based PDF page; None for plain-text documents.
    page: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str | None] = mapped_column(Text)
    timestamp: Mapped[datetime | None] = mapped_column(default=utcnow)

    # Filled in by the Celery worker (`knowledge.embed_pending`), never in the upload request.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSION))
    embedding_status: Mapped[str] = mapped_column(String(20), default=EmbeddingStatus.PENDING.value)
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(f"to_tsvector('{FTS_CONFIG}', coalesce(content, ''))", persisted=True)
    )
