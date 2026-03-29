from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime


class InferenceRunCreate(BaseModel):
    model_name: str = "ResidualAttentionUNet"
    model_version: str = "1.0"
    model_file: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class InferenceBatchRunCreate(InferenceRunCreate):
    slide_ids: list[int] = Field(default_factory=list, min_length=1)


class InferenceFolderRunCreate(InferenceRunCreate):
    folder_key: str = Field(min_length=1)


class InferenceRunResponse(BaseModel):
    id: int
    slide_id: int
    model_name: str
    model_version: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    summary: dict | None = None
    error_message: str | None = None

    class Config:
        from_attributes = True


class RegionResponse(BaseModel):
    id: int
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str | None

    class Config:
        from_attributes = True


class InferenceModelInfo(BaseModel):
    id: str
    label: str
    path: str
    size_bytes: int
    modified_at: str | None = None


class InferenceModelListResponse(BaseModel):
    models: list[InferenceModelInfo]
    default_model_id: str | None = None


class InferenceBatchRunResponse(BaseModel):
    run_ids: list[int]
    slide_ids: list[int]
