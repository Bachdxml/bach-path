from __future__ import annotations
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from pathlib import Path
import os

from app.api.deps import get_db
from app.schemas.slides import SlideImportRequest, SlideImportResponse, SlideMetadataResponse
from app.models.slide import Slide
from app.slides.storage import copy_into_managed_storage
from app.slides.metadata import read_openslide_metadata

from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/slides", tags=["slides"])

@router.post("/import", response_model=SlideImportResponse)
def import_slide(payload: SlideImportRequest, request: Request, db: Session = Depends(get_db)):
    settings = request.app.state.settings

    src = Path(payload.file_path)
    if not src.exists():
        raise AppError(ErrorCode.SLIDE_NOT_FOUND, f"File not found: {src}")
    if not src.is_file():
        raise AppError(ErrorCode.SLIDE_INVALID, f"Not a file: {src}")

    # Basic extension gate (don’t rely solely on it)
    if src.suffix.lower() not in {".svs", ".tif", ".tiff"}:
        raise AppError(ErrorCode.SLIDE_INVALID, f"Unsupported extension: {src.suffix}")

    # Create DB row first to reserve an ID (stable filename)
    size = src.stat().st_size
    slide = Slide(
        original_path=str(src),
        stored_filename="pending",
        stored_path="pending",
        file_size_bytes=size,
        sha256=None,
    )
    db.add(slide)
    db.flush()  # assigns slide.id without committing yet

    dest_filename = f"{slide.id}{src.suffix.lower()}"
    try:
        dest_path = copy_into_managed_storage(src, settings.slides_dir, dest_filename)
    except PermissionError:
        raise AppError(ErrorCode.SLIDE_PERMISSION, "Access denied when copying slide")
    except FileExistsError as e:
        raise AppError(ErrorCode.CONFLICT, str(e))
    except OSError as e:
        raise AppError(ErrorCode.IO_ERROR, f"Failed to copy slide: {e}")

    slide.stored_filename = dest_filename
    slide.stored_path = str(dest_path)

    return SlideImportResponse(slide_id=slide.id, stored_path=slide.stored_path)

@router.get("/{slide_id}/metadata", response_model=SlideMetadataResponse)
def slide_metadata(slide_id: int, db: Session = Depends(get_db)):
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    p = Path(slide.stored_path)
    if not p.exists():
        # Clinical-track: DB says it exists but file is missing => integrity issue
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")

    try:
        meta = read_openslide_metadata(p)
    except Exception as e:
        # OpenSlide errors vary; treat as “unprocessable slide”
        raise AppError(ErrorCode.SLIDE_UNREADABLE, f"OpenSlide failed: {e}")

    return SlideMetadataResponse(slide_id=slide_id, **meta)
