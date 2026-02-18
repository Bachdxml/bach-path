from __future__ import annotations
from pydantic import BaseModel, Field

class SlideImportRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to an SVS file accessible to the local machine")
    compute_sha256: bool = False

class SlideImportResponse(BaseModel):
    slide_id: int
    stored_path: str

class SlideMetadataResponse(BaseModel):
    slide_id: int
    vendor: str | None = None
    level_count: int
    dimensions: tuple[int, int]
    level_dimensions: list[tuple[int, int]]
    mpp_x: float | None = None
    mpp_y: float | None = None
    properties: dict[str, str]
