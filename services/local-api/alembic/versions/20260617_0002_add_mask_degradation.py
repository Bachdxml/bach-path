"""Add mask degradation columns to inference_runs.

Revision ID: 20260617_0002
Revises: 20260526_0001
Create Date: 2026-06-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260617_0002"
down_revision = "20260526_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("inference_runs"):
        return
    columns = {column["name"] for column in inspector.get_columns("inference_runs")}
    if "mask_degradation_status" not in columns:
        op.add_column(
            "inference_runs",
            sa.Column(
                "mask_degradation_status",
                sa.String(length=32),
                server_default="full",
                nullable=False,
            ),
        )
    if "mask_downsample_factor" not in columns:
        op.add_column(
            "inference_runs",
            sa.Column("mask_downsample_factor", sa.Integer(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("inference_runs", "mask_downsample_factor")
    op.drop_column("inference_runs", "mask_degradation_status")
