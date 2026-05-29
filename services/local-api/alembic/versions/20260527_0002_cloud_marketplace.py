"""Cloud marketplace schema.

Revision ID: 20260527_0002
Revises: 20260526_0001
Create Date: 2026-05-27
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260527_0002"
down_revision = "20260526_0001"
branch_labels = None
depends_on = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return False
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    if "users" in tables and not _has_column("users", "cognito_sub"):
        op.add_column("users", sa.Column("cognito_sub", sa.String(length=255), nullable=True))
        op.create_index(op.f("ix_users_cognito_sub"), "users", ["cognito_sub"], unique=True)
    if "users" in tables and not _has_column("users", "email"):
        op.add_column("users", sa.Column("email", sa.String(length=255), nullable=True))
        op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)

    if "accounts" not in tables:
        op.create_table(
            "accounts",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("account_type", sa.String(length=32), nullable=False),
            sa.Column("marketplace_role", sa.String(length=32), nullable=False),
            sa.Column("is_approved", sa.Boolean(), server_default=sa.text("false"), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint("account_type IN ('individual','organization','internal')", name=op.f("ck_accounts_account_type")),
            sa.CheckConstraint("marketplace_role IN ('submitter','buyer','internal')", name=op.f("ck_accounts_marketplace_role")),
            sa.PrimaryKeyConstraint("id", name=op.f("pk_accounts")),
        )

    if "account_memberships" not in tables:
        op.create_table(
            "account_memberships",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("account_id", sa.Integer(), nullable=False),
            sa.Column("role", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint("role IN ('owner','submitter','curator','buyer','admin')", name=op.f("ck_account_memberships_role")),
            sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], name=op.f("fk_account_memberships_account_id_accounts"), ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_account_memberships_user_id_users"), ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id", name=op.f("pk_account_memberships")),
            sa.UniqueConstraint("user_id", "account_id", name=op.f("uq_account_memberships_user_account")),
        )
        op.create_index(op.f("ix_account_memberships_account_id"), "account_memberships", ["account_id"], unique=False)
        op.create_index(op.f("ix_account_memberships_user_id"), "account_memberships", ["user_id"], unique=False)

    if "submissions" not in tables:
        op.create_table(
            "submissions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("account_id", sa.Integer(), nullable=False),
            sa.Column("created_by_user_id", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("status", sa.String(length=32), server_default="draft", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint("status IN ('draft','submitted','reviewed','rejected')", name=op.f("ck_submissions_status")),
            sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], name=op.f("fk_submissions_created_by_user_id_users")),
            sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], name=op.f("fk_submissions_account_id_accounts")),
            sa.PrimaryKeyConstraint("id", name=op.f("pk_submissions")),
        )
        op.create_index(op.f("ix_submissions_created_by_user_id"), "submissions", ["created_by_user_id"], unique=False)
        op.create_index(op.f("ix_submissions_account_id"), "submissions", ["account_id"], unique=False)

    if "slide_assets" not in tables:
        op.create_table(
            "slide_assets",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("submission_id", sa.Integer(), nullable=False),
            sa.Column("account_id", sa.Integer(), nullable=False),
            sa.Column("created_by_user_id", sa.Integer(), nullable=False),
            sa.Column("filename", sa.String(length=255), nullable=False),
            sa.Column("s3_key", sa.String(length=1024), nullable=False),
            sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
            sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
            sa.Column("file_type", sa.String(length=32), nullable=False),
            sa.Column("deidentified", sa.Boolean(), server_default=sa.text("false"), nullable=False),
            sa.Column("review_status", sa.String(length=32), server_default="submitted", nullable=False),
            sa.Column("metadata_json", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint("review_status IN ('submitted','deidentified_approved','rejected')", name=op.f("ck_slide_assets_review_status")),
            sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], name=op.f("fk_slide_assets_created_by_user_id_users")),
            sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], name=op.f("fk_slide_assets_account_id_accounts")),
            sa.ForeignKeyConstraint(["submission_id"], ["submissions.id"], name=op.f("fk_slide_assets_submission_id_submissions")),
            sa.PrimaryKeyConstraint("id", name=op.f("pk_slide_assets")),
            sa.UniqueConstraint("s3_key", name=op.f("uq_slide_assets_s3_key")),
        )
        op.create_index(op.f("ix_slide_assets_created_by_user_id"), "slide_assets", ["created_by_user_id"], unique=False)
        op.create_index(op.f("ix_slide_assets_account_id"), "slide_assets", ["account_id"], unique=False)
        op.create_index(op.f("ix_slide_assets_submission_id"), "slide_assets", ["submission_id"], unique=False)

    if "datasets" not in tables:
        op.create_table(
            "datasets",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("price_cents", sa.Integer(), server_default="0", nullable=False),
            sa.Column("currency", sa.String(length=3), server_default="usd", nullable=False),
            sa.Column("license_terms", sa.Text(), nullable=False),
            sa.Column("allowed_use", sa.Text(), nullable=False),
            sa.Column("status", sa.String(length=32), server_default="draft", nullable=False),
            sa.Column("created_by_user_id", sa.Integer(), nullable=False),
            sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint("status IN ('draft','published','archived')", name=op.f("ck_datasets_status")),
            sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], name=op.f("fk_datasets_created_by_user_id_users")),
            sa.PrimaryKeyConstraint("id", name=op.f("pk_datasets")),
        )
        op.create_index(op.f("ix_datasets_created_by_user_id"), "datasets", ["created_by_user_id"], unique=False)

    if "orders" not in tables:
        op.create_table(
            "orders",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("dataset_id", sa.Integer(), nullable=False),
            sa.Column("buyer_account_id", sa.Integer(), nullable=False),
            sa.Column("requested_by_user_id", sa.Integer(), nullable=False),
            sa.Column("amount_cents", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(length=3), server_default="usd", nullable=False),
            sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
            sa.Column("stripe_checkout_session_id", sa.String(length=255), nullable=True),
            sa.Column("stripe_payment_intent_id", sa.String(length=255), nullable=True),
            sa.Column("checkout_url", sa.String(length=2048), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint("status IN ('pending','paid','failed','canceled')", name=op.f("ck_orders_status")),
            sa.ForeignKeyConstraint(["buyer_account_id"], ["accounts.id"], name=op.f("fk_orders_buyer_account_id_accounts")),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name=op.f("fk_orders_dataset_id_datasets")),
            sa.ForeignKeyConstraint(["requested_by_user_id"], ["users.id"], name=op.f("fk_orders_requested_by_user_id_users")),
            sa.PrimaryKeyConstraint("id", name=op.f("pk_orders")),
            sa.UniqueConstraint("stripe_checkout_session_id", name=op.f("uq_orders_stripe_checkout_session_id")),
        )
        op.create_index(op.f("ix_orders_buyer_account_id"), "orders", ["buyer_account_id"], unique=False)
        op.create_index(op.f("ix_orders_dataset_id"), "orders", ["dataset_id"], unique=False)
        op.create_index(op.f("ix_orders_requested_by_user_id"), "orders", ["requested_by_user_id"], unique=False)

    if "dataset_slides" not in tables:
        op.create_table(
            "dataset_slides",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("dataset_id", sa.Integer(), nullable=False),
            sa.Column("slide_asset_id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name=op.f("fk_dataset_slides_dataset_id_datasets"), ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["slide_asset_id"], ["slide_assets.id"], name=op.f("fk_dataset_slides_slide_asset_id_slide_assets")),
            sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_slides")),
            sa.UniqueConstraint("dataset_id", "slide_asset_id", name=op.f("uq_dataset_slides_dataset_slide")),
        )
        op.create_index(op.f("ix_dataset_slides_dataset_id"), "dataset_slides", ["dataset_id"], unique=False)
        op.create_index(op.f("ix_dataset_slides_slide_asset_id"), "dataset_slides", ["slide_asset_id"], unique=False)

    if "licenses" not in tables:
        op.create_table(
            "licenses",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("dataset_id", sa.Integer(), nullable=False),
            sa.Column("buyer_account_id", sa.Integer(), nullable=False),
            sa.Column("order_id", sa.Integer(), nullable=True),
            sa.Column("terms", sa.Text(), nullable=False),
            sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint("status IN ('active','expired','revoked')", name=op.f("ck_licenses_status")),
            sa.ForeignKeyConstraint(["buyer_account_id"], ["accounts.id"], name=op.f("fk_licenses_buyer_account_id_accounts")),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name=op.f("fk_licenses_dataset_id_datasets")),
            sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name=op.f("fk_licenses_order_id_orders")),
            sa.PrimaryKeyConstraint("id", name=op.f("pk_licenses")),
            sa.UniqueConstraint("dataset_id", "buyer_account_id", name=op.f("uq_licenses_dataset_buyer")),
        )
        op.create_index(op.f("ix_licenses_buyer_account_id"), "licenses", ["buyer_account_id"], unique=False)
        op.create_index(op.f("ix_licenses_dataset_id"), "licenses", ["dataset_id"], unique=False)
        op.create_index(op.f("ix_licenses_order_id"), "licenses", ["order_id"], unique=False)


def downgrade() -> None:
    for table_name in (
        "licenses",
        "dataset_slides",
        "orders",
        "datasets",
        "slide_assets",
        "submissions",
        "account_memberships",
        "accounts",
    ):
        op.drop_table(table_name)
    if _has_column("users", "email"):
        op.drop_index(op.f("ix_users_email"), table_name="users")
        op.drop_column("users", "email")
    if _has_column("users", "cognito_sub"):
        op.drop_index(op.f("ix_users_cognito_sub"), table_name="users")
        op.drop_column("users", "cognito_sub")
