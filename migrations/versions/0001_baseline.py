"""Baseline: the schema created by the legacy Flask-SQLAlchemy app.

Idempotent — tables that already exist (an existing production database) are left untouched,
so `alembic upgrade head` works for both fresh and pre-existing databases.

Revision ID: 0001
Revises:
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _tables() -> list[tuple[str, list[sa.SchemaItem]]]:
    return [
        (
            "email_log",
            [
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("user_email", sa.String(100)),
                sa.Column("sender", sa.String(200)),
                sa.Column("subject", sa.String(300)),
                sa.Column("response", sa.Text),
                sa.Column("source", sa.String(50)),
                sa.Column("status", sa.String(20)),
                sa.Column("category", sa.String(50)),
                sa.Column("confidence", sa.Float),
                sa.Column("reasoning", sa.Text),
                sa.Column("docs_used", sa.Text),
                sa.Column("priority", sa.String(20)),
                sa.Column("sentiment", sa.String(20)),
                sa.Column("timestamp", sa.DateTime),
            ],
        ),
        (
            "knowledge_document",
            [
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("user_email", sa.String(100)),
                sa.Column("filename", sa.String(200)),
                sa.Column("chunk_index", sa.Integer),
                sa.Column("content", sa.Text),
                sa.Column("timestamp", sa.DateTime),
            ],
        ),
        (
            "user",
            [
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("email", sa.String(100)),
                sa.Column("name", sa.String(100)),
                sa.Column("role", sa.String(20)),
                sa.Column("active", sa.Boolean),
                sa.Column("created_at", sa.DateTime),
                sa.Column("last_login", sa.DateTime),
                sa.UniqueConstraint("email"),
            ],
        ),
        (
            "thread_memory",
            [
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("user_email", sa.String(100)),
                sa.Column("thread_id", sa.String(200)),
                sa.Column("sender", sa.String(200)),
                sa.Column("subject", sa.String(300)),
                sa.Column("body", sa.Text),
                sa.Column("response", sa.Text),
                sa.Column("timestamp", sa.DateTime),
            ],
        ),
        (
            "user_template",
            [
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("user_email", sa.String(100)),
                sa.Column("name", sa.String(100)),
                sa.Column("keywords", sa.String(500)),
                sa.Column("response", sa.Text),
            ],
        ),
        (
            "blocked_sender",
            [
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("user_email", sa.String(100)),
                sa.Column("word", sa.String(100)),
            ],
        ),
        (
            "ai_feedback",
            [
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("user_email", sa.String(100)),
                sa.Column("email_log_id", sa.Integer),
                sa.Column("feedback", sa.String(10)),
                sa.Column("original_response", sa.Text),
                sa.Column("corrected_response", sa.Text),
                sa.Column("timestamp", sa.DateTime),
            ],
        ),
    ]


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    for name, columns in _tables():
        if name not in existing:
            op.create_table(name, *columns)


def downgrade() -> None:
    for name, _ in reversed(_tables()):
        op.drop_table(name)
