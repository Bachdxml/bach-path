from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class SlideImportRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to an SVS file accessible to the local machine")
    compute_sha256: bool = False

    @field_validator("file_path")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("file_path must be an absolute path")
        return value

class SlideImportResponse(BaseModel):
    slide_id: int
    stored_path: str

class SlideListItem(BaseModel):
    id: int
    original_path: str | None
    created_at: str
    inference_result: Literal["positive", "negative", "unchecked"] = "unchecked"
    folder_label: str = "Uncategorized"
    folder_key: str = "uncategorized"

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
