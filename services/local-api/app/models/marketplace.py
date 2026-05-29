from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = (
        CheckConstraint("account_type IN ('individual','organization','internal')", name="ck_accounts_account_type"),
        CheckConstraint("marketplace_role IN ('submitter','buyer','internal')", name="ck_accounts_marketplace_role"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    account_type: Mapped[str] = mapped_column(String(32), nullable=False)
    marketplace_role: Mapped[str] = mapped_column(String(32), nullable=False)
    is_approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    memberships = relationship("AccountMembership", back_populates="account", cascade="all, delete-orphan")


class AccountMembership(Base):
    __tablename__ = "account_memberships"
    __table_args__ = (
        CheckConstraint("role IN ('owner','submitter','curator','buyer','admin')", name="ck_account_memberships_role"),
        UniqueConstraint("user_id", "account_id", name="uq_account_memberships_user_account"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    account = relationship("Account", back_populates="memberships")
    user = relationship("User")


class Submission(Base):
    __tablename__ = "submissions"
    __table_args__ = (
        CheckConstraint("status IN ('draft','submitted','reviewed','rejected')", name="ck_submissions_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False, index=True)
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft", server_default="draft")
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class SlideAsset(Base):
    __tablename__ = "slide_assets"
    __table_args__ = (
        CheckConstraint(
            "review_status IN ('submitted','deidentified_approved','rejected')",
            name="ck_slide_assets_review_status",
        ),
        UniqueConstraint("s3_key", name="uq_slide_assets_s3_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), nullable=False, index=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False, index=True)
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    deidentified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, default="submitted", server_default="submitted")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        CheckConstraint("status IN ('draft','published','archived')", name="ck_datasets_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="usd", server_default="usd")
    license_terms: Mapped[str] = mapped_column(Text, nullable=False)
    allowed_use: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft", server_default="draft")
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    published_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DatasetSlide(Base):
    __tablename__ = "dataset_slides"
    __table_args__ = (
        UniqueConstraint("dataset_id", "slide_asset_id", name="uq_dataset_slides_dataset_slide"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    slide_asset_id: Mapped[int] = mapped_column(ForeignKey("slide_assets.id"), nullable=False, index=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class License(Base):
    __tablename__ = "licenses"
    __table_args__ = (
        CheckConstraint("status IN ('active','expired','revoked')", name="ck_licenses_status"),
        UniqueConstraint("dataset_id", "buyer_account_id", name="uq_licenses_dataset_buyer"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    buyer_account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False, index=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True, index=True)
    terms: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", server_default="active")
    expires_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint("status IN ('pending','paid','failed','canceled')", name="ck_orders_status"),
        UniqueConstraint("stripe_checkout_session_id", name="uq_orders_stripe_checkout_session_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    buyer_account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False, index=True)
    requested_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="usd", server_default="usd")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    stripe_checkout_session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_payment_intent_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    checkout_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
