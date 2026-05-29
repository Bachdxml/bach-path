from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


AccountRole = Literal["owner", "submitter", "curator", "buyer", "admin"]
AccountType = Literal["individual", "organization", "internal"]
MarketplaceRole = Literal["submitter", "buyer", "internal"]
SlideReviewStatus = Literal["submitted", "deidentified_approved", "rejected"]
DatasetStatus = Literal["draft", "published", "archived"]
LicenseStatus = Literal["active", "expired", "revoked"]


class AccountResponse(BaseModel):
    id: int
    name: str
    account_type: AccountType
    marketplace_role: MarketplaceRole
    is_approved: bool


class AccountCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    account_type: AccountType = "organization"
    marketplace_role: MarketplaceRole
    is_approved: bool = False


class AccountApprovalRequest(BaseModel):
    is_approved: bool


class SubmissionCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)

    @field_validator("title")
    @classmethod
    def trim_title(cls, value: str) -> str:
        return value.strip()


class SubmissionResponse(BaseModel):
    id: int
    title: str
    status: Literal["draft", "submitted", "reviewed", "rejected"]
    account_id: int
    created_at: datetime


class UploadUrlRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    file_type: str = Field(..., min_length=1, max_length=32)
    file_size_bytes: int = Field(..., gt=0)
    checksum_sha256: str = Field(..., min_length=64, max_length=64)

    @field_validator("filename", "file_type", "checksum_sha256")
    @classmethod
    def trim_strings(cls, value: str) -> str:
        return value.strip()


class UploadUrlResponse(BaseModel):
    upload_url: str
    s3_key: str
    expires_in_seconds: int


class SlideFinalizeRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    s3_key: str = Field(..., min_length=1, max_length=1024)
    checksum_sha256: str = Field(..., min_length=64, max_length=64)
    file_size_bytes: int = Field(..., gt=0)
    file_type: str = Field(..., min_length=1, max_length=32)
    metadata: dict[str, object] = Field(default_factory=dict)


class SlideAssetResponse(BaseModel):
    id: int
    submission_id: int
    filename: str
    s3_key: str
    checksum_sha256: str
    file_size_bytes: int
    file_type: str
    deidentified: bool
    review_status: SlideReviewStatus


class SlideReviewRequest(BaseModel):
    review_status: Literal["deidentified_approved", "rejected"]


class DatasetCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    price_cents: int = Field(default=0, ge=0)
    currency: str = Field(default="usd", min_length=3, max_length=3)
    license_terms: str = Field(..., min_length=1)
    allowed_use: str = Field(..., min_length=1)


class DatasetUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    price_cents: int | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    license_terms: str | None = Field(default=None, min_length=1)
    allowed_use: str | None = Field(default=None, min_length=1)


class DatasetSlideAddRequest(BaseModel):
    slide_asset_id: int


class DatasetResponse(BaseModel):
    id: int
    title: str
    description: str | None
    price_cents: int
    currency: str
    license_terms: str
    allowed_use: str
    status: DatasetStatus
    created_at: datetime
    published_at: datetime | None = None


class CheckoutResponse(BaseModel):
    order_id: int
    checkout_url: str


class LicenseResponse(BaseModel):
    id: int
    dataset_id: int
    buyer_account_id: int
    status: LicenseStatus
    expires_at: datetime | None = None


class DownloadUrlResponse(BaseModel):
    urls: list[str]
    expires_in_seconds: int
