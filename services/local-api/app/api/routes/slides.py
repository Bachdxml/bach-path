from __future__ import annotations
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from pathlib import Path
import logging
import shutil

from app.api.deps import get_db
from app.schemas.slides import SlideImportRequest, SlideImportResponse, SlideListResponse, SlideListItem, SlideMetadataResponse
from app.models.slide import Slide
from app.models.inference_run import InferenceRun
from app.models.region import Region
from app.models.enums import InferenceStatus
from app.slides.storage import copy_into_managed_storage
from app.slides.metadata import read_openslide_metadata, read_raster_metadata, RASTER_EXTENSIONS
from app.slides.deepzoom import deepzoom_paths, ensure_deepzoom, has_deepzoom

from fastapi import Response
from fastapi.responses import FileResponse
import io

import openslide
from PIL import Image

from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/slides", tags=["slides"])
logger = logging.getLogger(__name__)

WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".png"}

TILE_SIZE_DEFAULT = 256


def _raster_thumbnail_jpeg(slide_path: Path, size: int) -> bytes:
    with Image.open(slide_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if w > size or h > size:
            ratio = min(size / w, size / h)
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _raster_tile_jpeg(slide_path: Path, level: int, x: int, y: int, tile_size: int) -> bytes:
    if level != 0:
        raise AppError(ErrorCode.SLIDE_INVALID, f"Raster images have only level 0; got {level}")
    with Image.open(slide_path) as img:
        img = img.convert("RGB")
        iw, ih = img.size
        px = x * tile_size
        py = y * tile_size
        if px >= iw or py >= ih or x < 0 or y < 0:
            raise AppError(ErrorCode.NOT_FOUND, f"Tile out of bounds: level={level} x={x} y={y}")
        w = min(tile_size, iw - px)
        h = min(tile_size, ih - py)
        crop = img.crop((px, py, px + w, py + h))
        if w != tile_size or h != tile_size:
            padded = Image.new("RGB", (tile_size, tile_size))
            padded.paste(crop, (0, 0))
            crop = padded
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


@router.delete("/{slide_id}")
def delete_slide(slide_id: int, request: Request, db: Session = Depends(get_db)):
    """Remove slide from storage, caches, inference artifacts, and database."""
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    settings = request.app.state.settings

    runs = db.query(InferenceRun).filter(InferenceRun.slide_id == slide_id).all()
    run_json_paths = [settings.inference_runs_dir / f"{run.id}.json" for run in runs]
    slide_path = Path(slide.stored_path)
    cache_slide = settings.tiles_cache_dir / str(slide_id)

    db.delete(slide)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    for json_path in run_json_paths:
        if json_path.is_file():
            try:
                json_path.unlink()
            except OSError as e:
                logger.warning("Could not delete inference output %s: %s", json_path, e)

    if slide_path.is_file():
        try:
            slide_path.unlink()
        except OSError as e:
            logger.warning("Could not delete slide file %s: %s", slide_path, e)

    if cache_slide.exists():
        shutil.rmtree(cache_slide, ignore_errors=True)

    return {"ok": True, "id": slide_id}


# List all slides in db for gallery
@router.get("", response_model=SlideListResponse)
def list_slides(db: Session = Depends(get_db)):
    slides = db.query(Slide).order_by(Slide.created_at.desc()).all()
    slide_ids = [s.id for s in slides]
    latest_run_by_slide: dict[int, InferenceRun] = {}
    positive_count_by_run: dict[int, int] = {}

    if slide_ids:
        runs = (
            db.query(InferenceRun)
            .filter(
                InferenceRun.slide_id.in_(slide_ids),
                InferenceRun.status == InferenceStatus.succeeded.value,
            )
            .order_by(InferenceRun.slide_id.asc(), InferenceRun.created_at.desc())
            .all()
        )
        for run in runs:
            latest_run_by_slide.setdefault(run.slide_id, run)

        run_ids = [r.id for r in latest_run_by_slide.values()]
        if run_ids:
            rows = (
                db.query(Region.inference_run_id, func.count(Region.id))
                .filter(
                    Region.inference_run_id.in_(run_ids),
                    Region.label == "fungus_positive",
                )
                .group_by(Region.inference_run_id)
                .all()
            )
            positive_count_by_run = {run_id: count for run_id, count in rows}

    def _inference_result(slide_id: int) -> str:
        run = latest_run_by_slide.get(slide_id)
        if not run:
            return "unchecked"
        return "positive" if positive_count_by_run.get(run.id, 0) > 0 else "negative"

    def _folder_info(original_path: str | None) -> tuple[str, str]:
        if not original_path:
            return "Uncategorized", "uncategorized"
        p = Path(original_path)
        parent = p.parent
        label = parent.name or "(root)"
        key = str(parent)
        return label, key

    items: list[SlideListItem] = []
    for s in slides:
        folder_label, folder_key = _folder_info(s.original_path)
        items.append(
            SlideListItem(
                id=s.id,
                original_path=s.original_path,
                created_at=s.created_at.isoformat() if s.created_at else "",
                inference_result=_inference_result(s.id),
                folder_label=folder_label,
                folder_key=folder_key,
            )
        )
    return SlideListResponse(slides=items)


@router.post("/import", response_model=SlideImportResponse)
def import_slide(payload: SlideImportRequest, request: Request, db: Session = Depends(get_db)):
    settings = request.app.state.settings

    src = Path(payload.file_path)
    if not src.exists():
        raise AppError(ErrorCode.SLIDE_NOT_FOUND, f"File not found: {src}")
    if not src.is_file():
        raise AppError(ErrorCode.SLIDE_INVALID, f"Not a file: {src}")

    # Basic extension gate (don’t rely solely on it)
    if src.suffix.lower() not in WSI_EXTENSIONS:
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

    try:
        ensure_deepzoom(dest_path, settings.tiles_cache_dir, slide.id)
    except Exception as e:
        # Keep import successful even if pre-generation fails; viewer can still use dynamic tiles.
        logger.warning("DeepZoom generation failed for slide %s: %s", slide.id, e)

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

    if p.suffix.lower() in RASTER_EXTENSIONS:
        try:
            meta = read_raster_metadata(p)
        except Exception as e:
            raise AppError(ErrorCode.SLIDE_UNREADABLE, f"Could not read image: {e}")
    else:
        try:
            meta = read_openslide_metadata(p)
        except Exception as e:
            # OpenSlide errors vary; treat as “unprocessable slide”
            raise AppError(ErrorCode.SLIDE_UNREADABLE, f"OpenSlide failed: {e}")

    return SlideMetadataResponse(slide_id=slide_id, **meta)

# Small preview image for each slide in gallery
@router.get("/{slide_id}/thumbnail")
def slide_thumbnail(
    slide_id: int,
    size: int = 256,
    db: Session = Depends(get_db),
):
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    slide_path = Path(slide.stored_path)
    if not slide_path.exists():
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")

    if slide_path.suffix.lower() in RASTER_EXTENSIONS:
        try:
            jpg = _raster_thumbnail_jpeg(slide_path, size)
            return Response(content=jpg, media_type="image/jpeg")
        except AppError:
            raise
        except Exception as e:
            raise AppError(ErrorCode.SLIDE_UNREADABLE, f"Could not read image: {e}")

    try:
        osr = openslide.OpenSlide(str(slide_path))
    except Exception as e:
        raise AppError(ErrorCode.SLIDE_UNREADABLE, f"OpenSlide failed: {e}")

    try:
        # Use lowest-resolution level (highest level index)
        level = osr.level_count - 1
        level_w, level_h = osr.level_dimensions[level]
        img = osr.read_region((0, 0), level, (level_w, level_h))
        img = img.convert("RGB")

        # Scale to fit within size (max dimension)
        w, h = img.size
        if w > size or h > size:
            ratio = min(size / w, size / h)
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    finally:
        try:
            osr.close()
        except Exception:
            pass


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
    TILE_SIZE = TILE_SIZE_DEFAULT
    overlap = 0  # set to e.g. 10 for 10px overlap between tiles (helps with some viewers)

    # 7) Cache tile to disk
    tile_path = (
        settings.app_data_dir
        / "tiles_cache"
        / str(slide_id)
        / f"{TILE_SIZE}_{overlap}"
        / str(level)
        / f"{x}_{y}.jpg"
    )
    tile_path.parent.mkdir(parents=True, exist_ok=True)

    if tile_path.exists():
        return Response(content=tile_path.read_bytes(), media_type="image/jpeg")

    if slide_path.suffix.lower() in RASTER_EXTENSIONS:
        try:
            jpg_bytes = _raster_tile_jpeg(slide_path, level, x, y, TILE_SIZE)
            tile_path.write_bytes(jpg_bytes)
            return Response(content=jpg_bytes, media_type="image/jpeg")
        except AppError:
            raise
        except Exception as e:
            raise AppError(ErrorCode.SLIDE_UNREADABLE, f"Could not read image: {e}")

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


@router.get("/{slide_id}/deepzoom.dzi")
def slide_deepzoom_descriptor(
    slide_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    slide_path = Path(slide.stored_path)
    if not slide_path.exists():
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")

    dz_paths = deepzoom_paths(settings.tiles_cache_dir, slide_id)
    if not has_deepzoom(dz_paths):
        raise AppError(ErrorCode.NOT_FOUND, "DeepZoom tiles not pre-generated for this slide")
    return FileResponse(path=dz_paths.descriptor, media_type="application/xml")


@router.get("/{slide_id}/slide_files/{level}/{x}_{y}.jpg")
def slide_deepzoom_tile(
    slide_id: int,
    level: int,
    x: int,
    y: int,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = request.app.state.settings
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    slide_path = Path(slide.stored_path)
    if not slide_path.exists():
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")

    dz_paths = deepzoom_paths(settings.tiles_cache_dir, slide_id)
    if not has_deepzoom(dz_paths):
        raise AppError(ErrorCode.NOT_FOUND, "DeepZoom tiles not pre-generated for this slide")
    tile_path = dz_paths.tiles_dir / str(level) / f"{x}_{y}.jpg"
    if not tile_path.exists():
        raise AppError(ErrorCode.NOT_FOUND, f"Tile out of bounds: level={level} x={x} y={y}")
    return FileResponse(path=tile_path, media_type="image/jpeg")
