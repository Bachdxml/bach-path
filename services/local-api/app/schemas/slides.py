from __future__ import annotations
from pathlib import Path
from typing import Literal
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

class SlideImportRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to an SVS file accessible to the local machine")
    compute_sha256: bool = False
    collection_id: int | None = Field(default=None, description="Existing import collection to attach the slide to")
    allow_tile_like_import: bool = Field(
        default=False,
        description="Explicit override for importing generated tile-like raster images as slides",
    )

    @field_validator("file_path")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("file_path must be an absolute path")
        return value

class SlideImportResponse(BaseModel):
    slide_id: int
    stored_path: str = Field(..., description="Managed slide identifier, not an absolute filesystem path")

class ImportCollectionResponse(BaseModel):
    id: int
    title: str | None = None
    source_type: str
    created_at: datetime

class SlideImportCollectionRequest(BaseModel):
    file_paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Absolute paths to slide files to import together",
    )
    title: str | None = Field(default=None, description="Optional collection title")
    source_type: str | None = Field(default=None, description="Optional collection source label")
    allow_tile_like_import: bool = Field(
        default=False,
        description="Explicit override for importing generated tile-like raster images as slides",
    )

    @field_validator("file_paths")
    @classmethod
    def validate_absolute_paths(cls, values: list[str]) -> list[str]:
        for value in values:
            if not Path(value).is_absolute():
                raise ValueError("file_paths entries must be absolute paths")
        return values

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

class SlideImportCollectionResponse(BaseModel):
    collection: ImportCollectionResponse
    imported_count: int
    imported_slide_ids: list[int]

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
    inference_result: Literal["positive", "negative", "indecisive", "unchecked"] = "unchecked"
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
