"""create session_log table

Revision ID: 0001
Revises:
Create Date: 2026-06-11

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "session_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("device_model", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("adapter_type", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("address", sa.String(length=128), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("final_region", sa.String(length=8), nullable=True),
        sa.Column("final_stage", sa.String(length=64), nullable=True),
        sa.Column("final_ea1_eligible", sa.Boolean(), nullable=True),
        sa.Column("baseline_alpha", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("session_log")
