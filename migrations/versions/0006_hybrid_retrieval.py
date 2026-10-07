"""Hybrid retrieval: pgvector embeddings, generated full-text column, chunk metadata, draft citations.

Requires the pgvector extension to be installed on the server (pgvector/pgvector images ship it).
Existing chunks are kept and marked "pending"; the worker embeds them (`knowledge.embed_pending`).

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-06
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")

EMBEDDING_DIMENSION = 1024  # app.services.embeddings.EMBEDDING_DIMENSION at the time of this migration


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column("knowledge_document", sa.Column("page", sa.Integer, nullable=True))
    op.add_column("knowledge_document", sa.Column("token_count", sa.Integer, nullable=True))
    op.add_column("knowledge_document", sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True))
    op.add_column(
        "knowledge_document",
        sa.Column("embedding_status", sa.String(20), nullable=False, server_default="pending"),
    )
    op.add_column("knowledge_document", sa.Column("embedding_model", sa.String(100), nullable=True))
    op.add_column(
        "knowledge_document",
        sa.Column(
            "search_vector",
            TSVECTOR,
            sa.Computed("to_tsvector('simple', coalesce(content, ''))", persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_knowledge_document_embedding_hnsw",
        "knowledge_document",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_knowledge_document_search_vector", "knowledge_document", ["search_vector"], postgresql_using="gin"
    )
    op.create_index("ix_knowledge_document_embedding_status", "knowledge_document", ["embedding_status"])

    op.add_column("inbound_email", sa.Column("citations", sa.JSON, nullable=True))

    # to_tsvector lower-cases with the database's character locale. Under the libc "C" locale Cyrillic is
    # left as is ("Рекламации" never matches the lower-cased query term), so capitalized Macedonian words
    # would only be found by vector search. UTF-8 locales (en_US.utf8, C.UTF-8 builtin, ICU) are fine.
    if context.is_offline_mode():  # --sql: there is no database to ask
        return
    folds = op.get_bind().scalar(sa.text("SELECT to_tsvector('simple', 'Рекламација') @@ 'рекламација'::tsquery"))
    if not folds:
        logger.warning(
            "This database's locale does not lower-case Cyrillic for full-text search; capitalized Macedonian "
            "words will be found by vector search only. Use a UTF-8 locale (e.g. en_US.utf8)."
        )


def downgrade() -> None:
    op.drop_column("inbound_email", "citations")
    op.drop_index("ix_knowledge_document_embedding_status", table_name="knowledge_document")
    op.drop_index("ix_knowledge_document_search_vector", table_name="knowledge_document")
    op.drop_index("ix_knowledge_document_embedding_hnsw", table_name="knowledge_document")
    for column in ("search_vector", "embedding_model", "embedding_status", "embedding", "token_count", "page"):
        op.drop_column("knowledge_document", column)
    # The vector extension is left installed: other database objects may use it.
