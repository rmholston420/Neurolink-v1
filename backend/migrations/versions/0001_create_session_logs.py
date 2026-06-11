"""create session_logs table

Revision ID: 0001
Revises:
Create Date: 2026-06-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "session_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_model", sa.String(), nullable=False),
        sa.Column("adapter_type", sa.String(), nullable=False),
        sa.Column("address", sa.String(), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=False, default=0),
        sa.Column("final_region", sa.String(), nullable=True),
        sa.Column("final_stage", sa.String(), nullable=True),
        sa.Column("final_ea1_eligible", sa.Boolean(), nullable=False, default=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("session_logs")
