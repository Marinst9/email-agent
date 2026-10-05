"""Durable inbound email queue processed by the Celery worker.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "inbound_email",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_email", sa.String(100), nullable=False),
        sa.Column("gmail_message_id", sa.String(100), nullable=False),
        sa.Column("thread_id", sa.String(200), nullable=False),
        sa.Column("sender", sa.Text, nullable=False),
        sa.Column("subject", sa.Text, nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("auto_send", sa.Boolean, nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("response", sa.Text),
        sa.Column("source", sa.String(150)),
        sa.Column("category", sa.String(50)),
        sa.Column("priority", sa.String(20)),
        sa.Column("sentiment", sa.String(20)),
        sa.Column("confidence", sa.Float),
        sa.Column("reasoning", sa.Text),
        sa.Column("docs_used", sa.JSON),
        sa.Column("needs_review", sa.Boolean, nullable=False),
        sa.Column("review_reason", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("user_email", "gmail_message_id", name="uq_inbound_email_user_message"),
    )
    op.create_index("ix_inbound_email_user_status", "inbound_email", ["user_email", "status"])


def downgrade() -> None:
    op.drop_index("ix_inbound_email_user_status", table_name="inbound_email")
    op.drop_table("inbound_email")
