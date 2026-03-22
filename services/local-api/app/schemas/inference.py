from __future__ import annotations
from pydantic import BaseModel
from datetime import datetime


class InferenceRunCreate(BaseModel):
    model_name: str = "ResidualAttentionUNet"
    model_version: str = "1.0"
    model_file: str | None = None


class InferenceRunResponse(BaseModel):
    id: int
    slide_id: int
    model_name: str
    model_version: str
    status: str
    started_at: str | None
    finished_at: str | None
    created_at: str
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
