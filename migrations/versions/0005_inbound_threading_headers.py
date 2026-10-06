"""Store the original Message-ID and References headers so replies stay in the thread.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("inbound_email", sa.Column("message_id_header", sa.Text, nullable=False, server_default=""))
    op.add_column("inbound_email", sa.Column("references", sa.Text, nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("inbound_email", "references")
    op.drop_column("inbound_email", "message_id_header")
