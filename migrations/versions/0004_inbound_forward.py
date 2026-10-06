"""Store the drafted action and the forward recipient on inbound emails.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Every draft stored before this migration was handled as a reply.
    op.add_column("inbound_email", sa.Column("action", sa.String(20), nullable=False, server_default="ОДГОВОР"))
    op.add_column("inbound_email", sa.Column("forward_to", sa.String(320)))


def downgrade() -> None:
    op.drop_column("inbound_email", "forward_to")
    op.drop_column("inbound_email", "action")
