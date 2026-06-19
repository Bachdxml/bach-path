from __future__ import annotations
from typing import Literal
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

class SlideImportResponse(BaseModel):
    slide_id: int
    stored_path: str = Field(..., description="Managed slide identifier, not an absolute filesystem path")

class ImportCollectionResponse(BaseModel):
    id: int
    title: str | None = None
    source_type: str
    created_at: datetime

class ImportCollectionCreateRequest(BaseModel):
    title: str | None = Field(default=None, description="Optional collection title")
    source_type: str | None = Field(default=None, description="Optional collection source label")

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            return None
        return trimmed[:255]

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            return None
        return trimmed[:64]

class ImportCollectionRenameRequest(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("title must be a non-empty string")
        return trimmed[:255]

class SlideReviewRequest(BaseModel):
    review_status: Literal["unreviewed", "positive", "negative", "indeterminate"]

class SlideReviewResponse(BaseModel):
    id: int
    review_status: Literal["unreviewed", "positive", "negative", "indeterminate"]

class SlideListItem(BaseModel):
    id: int
    original_path: str | None = Field(default=None, description="Original slide filename only")
    created_at: datetime
    inference_result: Literal["positive", "negative", "needs_review", "unchecked"] = "unchecked"
    review_status: Literal["unreviewed", "positive", "negative", "indeterminate"] = "unreviewed"
    folder_label: str = "Uncategorized"
    folder_key: str = "uncategorized"
    collection_id: int | None = None
    collection_title: str | None = None
    collection_created_at: datetime | None = None

class SlideListResponse(BaseModel):
    slides: list[SlideListItem]

class SlideMetadataResponse(BaseModel):
    slide_id: int
    vendor: str | None = None
    level_count: int
    dimensions: tuple[int, int]
    level_dimensions: list[tuple[int, int]]
    mpp_x: float | None = None
    mpp_y: float | None = None
    properties: dict[str, str]
