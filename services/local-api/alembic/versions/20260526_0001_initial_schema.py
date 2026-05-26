"""Initial schema.

Revision ID: 20260526_0001
Revises:
Create Date: 2026-05-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260526_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing_tables = set(inspector.get_table_names())
    if existing_tables.intersection(
        {
            "audit_logs",
            "import_collections",
            "inference_run_events",
            "inference_runs",
            "regions",
            "slides",
            "users",
        }
    ):
        return

    op.create_table(
        "import_collections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_import_collections")),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "role IN ('admin','pathologist','technician','viewer')",
            name=op.f("ck_users_ck_users_role"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)
    op.create_table(
        "slides",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("original_path", sa.String(length=1024), nullable=True),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_path", sa.String(length=1024), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("import_collection_id", sa.Integer(), nullable=True),
        sa.Column("review_status", sa.String(length=32), server_default="unreviewed", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["import_collection_id"],
            ["import_collections.id"],
            name=op.f("fk_slides_import_collection_id_import_collections"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_slides")),
        sa.UniqueConstraint("stored_filename", name=op.f("uq_slides_stored_filename")),
    )
    op.create_index(op.f("ix_slides_import_collection_id"), "slides", ["import_collection_id"], unique=False)
    op.create_index(op.f("ix_slides_sha256"), "slides", ["sha256"], unique=False)
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=True),
        sa.Column("details_json", sa.Text(), nullable=True),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["users.id"],
            name=op.f("fk_audit_logs_actor_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_logs")),
    )
    op.create_index(op.f("ix_audit_logs_action"), "audit_logs", ["action"], unique=False)
    op.create_index(op.f("ix_audit_logs_actor_user_id"), "audit_logs", ["actor_user_id"], unique=False)
    op.create_index(op.f("ix_audit_logs_entity_type"), "audit_logs", ["entity_type"], unique=False)
    op.create_table(
        "inference_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slide_id", sa.Integer(), nullable=False),
        sa.Column("requested_by_user_id", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("output_json_path", sa.String(length=1024), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued','running','succeeded','failed','canceled')",
            name=op.f("ck_inference_runs_ck_inference_runs_status"),
        ),
        sa.ForeignKeyConstraint(
            ["requested_by_user_id"],
            ["users.id"],
            name=op.f("fk_inference_runs_requested_by_user_id_users"),
        ),
        sa.ForeignKeyConstraint(
            ["slide_id"],
            ["slides.id"],
            name=op.f("fk_inference_runs_slide_id_slides"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_runs")),
    )
    op.create_index(op.f("ix_inference_runs_slide_id"), "inference_runs", ["slide_id"], unique=False)
    op.create_table(
        "inference_run_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("from_status", sa.String(length=32), nullable=True),
        sa.Column("to_status", sa.String(length=32), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["inference_runs.id"],
            name=op.f("fk_inference_run_events_run_id_inference_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_run_events")),
    )
    op.create_index(op.f("ix_inference_run_events_changed_at"), "inference_run_events", ["changed_at"], unique=False)
    op.create_index(op.f("ix_inference_run_events_run_id"), "inference_run_events", ["run_id"], unique=False)
    op.create_index(op.f("ix_inference_run_events_to_status"), "inference_run_events", ["to_status"], unique=False)
    op.create_table(
        "regions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("inference_run_id", sa.Integer(), nullable=False),
        sa.Column("x", sa.Integer(), nullable=False),
        sa.Column("y", sa.Integer(), nullable=False),
        sa.Column("w", sa.Integer(), nullable=False),
        sa.Column("h", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["inference_run_id"],
            ["inference_runs.id"],
            name=op.f("fk_regions_inference_run_id_inference_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_regions")),
    )
    op.create_index(op.f("ix_regions_inference_run_id"), "regions", ["inference_run_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_regions_inference_run_id"), table_name="regions")
    op.drop_table("regions")
    op.drop_index(op.f("ix_inference_run_events_to_status"), table_name="inference_run_events")
    op.drop_index(op.f("ix_inference_run_events_run_id"), table_name="inference_run_events")
    op.drop_index(op.f("ix_inference_run_events_changed_at"), table_name="inference_run_events")
    op.drop_table("inference_run_events")
    op.drop_index(op.f("ix_inference_runs_slide_id"), table_name="inference_runs")
    op.drop_table("inference_runs")
    op.drop_index(op.f("ix_audit_logs_entity_type"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_actor_user_id"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_action"), table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index(op.f("ix_slides_sha256"), table_name="slides")
    op.drop_index(op.f("ix_slides_import_collection_id"), table_name="slides")
    op.drop_table("slides")
    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_table("users")
    op.drop_table("import_collections")
