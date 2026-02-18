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

from fastapi import Response
import io

import openslide
from PIL import Image

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

@router.get("/{slide_id}/tiles/{level}/{x}/{y}.jpg")
def slide_tile(
    slide_id: int,
    level: int,
    x: int,
    y: int,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings

    # 1) Load slide from DB
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    slide_path = Path(slide.stored_path)
    if not slide_path.exists():
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")

    # Tile settings
    TILE_SIZE = 256  # common default; adjust if your frontend expects something else

    # 7) Cache tile to disk
    cache_dir = Path(settings.tiles_cache_dir) if hasattr(settings, "tiles_cache_dir") else (settings.app_data_dir / "tiles_cache")
    tile_path = cache_dir / str(slide_id) / str(level) / f"{x}_{y}.jpg"
    tile_path.parent.mkdir(parents=True, exist_ok=True)

    if tile_path.exists():
        return Response(content=tile_path.read_bytes(), media_type="image/jpeg")

    # 2) Validate level exists + 3) compute tile region
    try:
        osr = openslide.OpenSlide(str(slide_path))
    except Exception as e:
        raise AppError(ErrorCode.SLIDE_UNREADABLE, f"OpenSlide failed: {e}")

    try:
        if level < 0 or level >= osr.level_count:
            raise AppError(ErrorCode.SLIDE_INVALID, f"Invalid level {level}; slide has {osr.level_count} levels")

        level_w, level_h = osr.level_dimensions[level]

        # Tile location in *level coordinates*
        px_level = x * TILE_SIZE
        py_level = y * TILE_SIZE

        # Reject completely out-of-bounds tiles (avoids doing work for nonsense coords)
        if px_level >= level_w or py_level >= level_h or x < 0 or y < 0:
            raise AppError(ErrorCode.NOT_FOUND, f"Tile out of bounds: level={level} x={x} y={y}")

        # OpenSlide read_region location is in level-0 coordinates
        downsample = float(osr.level_downsamples[level])
        px0 = int(px_level * downsample)
        py0 = int(py_level * downsample)

        # Clip tile size at edges (right/bottom border tiles can be smaller)
        w = min(TILE_SIZE, level_w - px_level)
        h = min(TILE_SIZE, level_h - py_level)

        # 4) read_region (returns RGBA)
        img = osr.read_region((px0, py0), level, (w, h))

        # 5) Convert to RGB (JPEG needs no alpha)
        img = img.convert("RGB")

        # If you want every tile to be exactly 256x256 for the client,
        # pad edge tiles up to TILE_SIZE.
        if w != TILE_SIZE or h != TILE_SIZE:
            padded = Image.new("RGB", (TILE_SIZE, TILE_SIZE))
            padded.paste(img, (0, 0))
            img = padded

        # 6) Return JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        jpg_bytes = buf.getvalue()

        # write-through cache
        tile_path.write_bytes(jpg_bytes)

        return Response(content=jpg_bytes, media_type="image/jpeg")
    finally:
        try:
            osr.close()
        except Exception:
            pass
