"""Store profile picture and Google OAuth token on users; index per-user lookups.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEXES: tuple[tuple[str, str, list[str]], ...] = (
    ("ix_email_log_user_email", "email_log", ["user_email"]),
    ("ix_ai_feedback_user_email", "ai_feedback", ["user_email"]),
    ("ix_user_template_user_email", "user_template", ["user_email"]),
    ("ix_blocked_sender_user_email", "blocked_sender", ["user_email"]),
    ("ix_thread_memory_user_thread", "thread_memory", ["user_email", "thread_id"]),
    ("ix_knowledge_document_user_file", "knowledge_document", ["user_email", "filename"]),
)


def upgrade() -> None:
    op.add_column("user", sa.Column("picture", sa.String(500), nullable=True))
    op.add_column("user", sa.Column("google_token", sa.JSON, nullable=True))

    # The legacy app always set these; make the schema say so.
    op.execute(sa.text("""UPDATE "user" SET role = 'employee' WHERE role IS NULL"""))
    op.execute(sa.text("""UPDATE "user" SET active = TRUE WHERE active IS NULL"""))
    op.alter_column("user", "email", existing_type=sa.String(100), nullable=False)
    op.alter_column("user", "role", existing_type=sa.String(20), nullable=False)
    op.alter_column("user", "active", existing_type=sa.Boolean, nullable=False)

    for name, table, columns in _INDEXES:
        op.create_index(name, table, columns)


def downgrade() -> None:
    for name, table, _ in reversed(_INDEXES):
        op.drop_index(name, table_name=table)
    op.alter_column("user", "active", existing_type=sa.Boolean, nullable=True)
    op.alter_column("user", "role", existing_type=sa.String(20), nullable=True)
    op.alter_column("user", "email", existing_type=sa.String(100), nullable=True)
    op.drop_column("user", "google_token")
    op.drop_column("user", "picture")
