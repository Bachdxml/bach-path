from __future__ import annotations
import io
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_db
from app.models.enums import InferenceStatus
from app.models.inference_run import InferenceRun
from app.models.import_collection import ImportCollection
from app.models.region import Region
from app.models.slide import Slide
from app.schemas.slides import (
    ImportCollectionRenameRequest,
    ImportCollectionResponse,
    SlideImportCollectionRequest,
    SlideImportCollectionResponse,
    SlideImportRequest,
    SlideImportResponse,
    SlideListItem,
    SlideListResponse,
    SlideMetadataResponse,
    SlideReviewRequest,
    SlideReviewResponse,
)
from app.slides.deepzoom import deepzoom_paths, has_deepzoom, ensure_deepzoom
from app.slides.metadata import RASTER_EXTENSIONS, read_openslide_metadata, read_raster_metadata
from app.slides.paths import folder_key_for_original_path
from app.slides.storage import copy_into_managed_storage
from app.util.exceptions import AppError, ErrorCode
from app.util.openslide_runtime import configure_openslide_runtime

router = APIRouter(prefix="/slides", tags=["slides"])
collections_router = APIRouter(tags=["slides"])
logger = logging.getLogger(__name__)

WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".png"}

TILE_SIZE_DEFAULT = 256
THUMBNAIL_SIZE_MIN = 32
THUMBNAIL_SIZE_MAX = 2048
MAX_IMPORT_COLLECTION_ITEMS = 256
TILE_FILENAME_RE = re.compile(r"(?:^|[_-])tile[_-]?x?\d+[_-]y?\d+(?:[_-]|$)", re.IGNORECASE)
GENERATED_TILE_PATH_PARTS = {"mastertile", "tiles", "patches"}
GENERATED_TILE_CLASS_PARTS = {"high", "medium", "low", "negative", "unclassified", "images", "masks"}


def _display_original_path(original_path: str | None) -> str | None:
    if not original_path:
        return None
    return Path(original_path).name or original_path


def _folder_info(original_path: str | None) -> tuple[str, str]:
    if not original_path:
        return "Uncategorized", "uncategorized"
    parent = Path(original_path).parent
    label = parent.name or "(root)"
    return label, folder_key_for_original_path(original_path)


def _managed_slide_identifier(stored_filename: str) -> str:
    return f"slides/{stored_filename}"


def _collection_title_from_timestamp(timestamp: datetime | None) -> str:
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat(timespec="seconds")


def _serialize_collection(collection: ImportCollection) -> ImportCollectionResponse:
    return ImportCollectionResponse(
        id=collection.id,
        title=collection.title,
        source_type=collection.source_type,
        created_at=collection.created_at,
    )


def _create_import_collection(
    db: Session,
    *,
    title: str | None,
    source_type: str | None,
    created_at: datetime | None = None,
) -> ImportCollection:
    collection = ImportCollection(
        title=title if title is not None else _collection_title_from_timestamp(created_at),
        source_type=(source_type or "import"),
    )
    if created_at is not None:
        collection.created_at = created_at
    db.add(collection)
    db.flush()
    db.commit()
    db.refresh(collection)
    return collection


def _ensure_import_allowed(src: Path, allowed_roots: tuple[Path, ...]) -> None:
    if not allowed_roots:
        return

    resolved_src = src.resolve()
    for root in allowed_roots:
        try:
            resolved_src.relative_to(root)
            return
        except ValueError:
            continue
    raise AppError(
        ErrorCode.SLIDE_PERMISSION,
        "Slide import path is not allowed",
        http_status=403,
    )


def _looks_like_generated_training_tile(src: Path) -> bool:
    suffix = src.suffix.lower()
    if suffix not in RASTER_EXTENSIONS:
        return False
    parts = {part.lower() for part in src.parts}
    stem = src.stem.lower()
    if TILE_FILENAME_RE.search(stem):
        return True
    if parts.intersection(GENERATED_TILE_PATH_PARTS) and parts.intersection(GENERATED_TILE_CLASS_PARTS):
        return True
    return False


def _validate_import_source(
    src: Path,
    allowed_roots: tuple[Path, ...],
    *,
    allow_tile_like_import: bool = False,
) -> None:
    if not src.exists():
        raise AppError(ErrorCode.SLIDE_NOT_FOUND, "File not found")
    if not src.is_file():
        raise AppError(ErrorCode.SLIDE_INVALID, "Import path must be a file")
    _ensure_import_allowed(src, allowed_roots)
    if src.suffix.lower() not in WSI_EXTENSIONS:
        raise AppError(ErrorCode.SLIDE_INVALID, "Unsupported slide file extension")
    if not allow_tile_like_import and _looks_like_generated_training_tile(src):
        raise AppError(
            ErrorCode.SLIDE_INVALID,
            "This looks like a generated training tile, not a whole slide. Import the source .svs instead.",
        )


def _resolve_managed_slide_path(slide: Slide, settings, *, must_exist: bool = True) -> Path:
    path = Path(slide.stored_path).resolve(strict=False)
    slides_root = settings.slides_dir.resolve(strict=False)
    try:
        path.relative_to(slides_root)
    except ValueError as e:
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide path is outside managed storage") from e
    if must_exist and (not path.exists() or not path.is_file()):
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")
    return path


def _import_slide_file(
    *,
    db: Session,
    src: Path,
    settings,
    collection_id: int | None,
) -> Slide:
    slide = Slide(
        original_path=str(src.resolve()),
        stored_filename="pending",
        stored_path="pending",
        file_size_bytes=src.stat().st_size,
        sha256=None,
        import_collection_id=collection_id,
    )
    db.add(slide)
    db.flush()
    db.commit()
    db.refresh(slide)

    dest_filename = f"{slide.id}{src.suffix.lower()}"
    dest_path = None
    for attempt in range(4):
        try:
            dest_path = copy_into_managed_storage(src, settings.slides_dir, dest_filename)
            break
        except FileExistsError:
            if attempt == 0:
                logger.warning(
                    "Managed storage collision for slide id=%s filename=%s; retrying with unique suffix",
                    slide.id,
                    dest_filename,
                )
            dest_filename = f"{slide.id}-{uuid.uuid4().hex[:8]}{src.suffix.lower()}"
        except PermissionError:
            _delete_pending_slide_row(db, slide.id)
            raise AppError(ErrorCode.SLIDE_PERMISSION, "Access denied when copying slide")
        except ValueError:
            _delete_pending_slide_row(db, slide.id)
            raise AppError(ErrorCode.SLIDE_INVALID, "Invalid managed storage destination")
        except OSError:
            logger.exception("Failed to copy slide from %s", src)
            _delete_pending_slide_row(db, slide.id)
            raise AppError(ErrorCode.IO_ERROR, "Failed to copy slide")
    if dest_path is None:
        _delete_pending_slide_row(db, slide.id)
        raise AppError(ErrorCode.CONFLICT, "Managed slide already exists")

    slide.stored_filename = dest_filename
    slide.stored_path = str(dest_path)
    db.add(slide)
    db.commit()
    db.refresh(slide)
    try:
        logger.warning("ATTEMPTING DEEPZOOM for slide %s at %s", slide.id, dest_path)
        ensure_deepzoom(dest_path, settings.tiles_cache_dir, slide.id)
        logger.warning("DEEPZOOM COMPLETE for slide %s", slide.id)
    except Exception as e:
        logger.warning("DeepZoom pre-generation failed for slide %s: %s", slide.id, e, exc_info=True)
    return slide

def _delete_pending_slide_row(db: Session, slide_id: int) -> None:
    slide = db.get(Slide, slide_id)
    if not slide:
        return
    db.delete(slide)
    db.commit()


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


def _thumbnail_cache_path(settings, slide_id: int, size: int) -> Path:
    return settings.tiles_cache_dir / str(slide_id) / "thumbnails" / f"{size}.jpg"


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
    slide_path = _resolve_managed_slide_path(slide, settings, must_exist=False)
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
    slides = (
        db.query(Slide)
        .options(selectinload(Slide.import_collection))
        .order_by(Slide.created_at.desc())
        .all()
    )
    slide_ids = [s.id for s in slides]
    latest_run_by_slide: dict[int, InferenceRun] = {}
    positive_count_by_run: dict[int, int] = {}
    negative_count_by_run: dict[int, int] = {}

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
                db.query(Region.inference_run_id, Region.label, func.count(Region.id))
                .filter(
                    Region.inference_run_id.in_(run_ids),
                    Region.label.in_(["fungus_positive", "fungus_negative"]),
                )
                .group_by(Region.inference_run_id, Region.label)
                .all()
            )
            for run_id, label, count in rows:
                if label == "fungus_positive":
                    positive_count_by_run[run_id] = count
                elif label == "fungus_negative":
                    negative_count_by_run[run_id] = count

    def _inference_result(slide_id: int) -> str:
        run = latest_run_by_slide.get(slide_id)
        if not run:
            return "unchecked"
        if positive_count_by_run.get(run.id, 0) > 0:
            return "positive"
        if negative_count_by_run.get(run.id, 0) > 0:
            return "negative"
        return "needs_review"

    items: list[SlideListItem] = []
    for s in slides:
        folder_label, folder_key = _folder_info(s.original_path)
        items.append(
            SlideListItem(
                id=s.id,
                original_path=_display_original_path(s.original_path),
                created_at=s.created_at,
                inference_result=_inference_result(s.id),
                review_status=s.review_status or "unreviewed",
                folder_label=folder_label,
                folder_key=folder_key,
                collection_id=s.import_collection.id if s.import_collection else s.import_collection_id,
                collection_title=s.import_collection.title if s.import_collection else None,
                collection_created_at=s.import_collection.created_at if s.import_collection else None,
            )
        )
    return SlideListResponse(slides=items)


@router.patch("/{slide_id}/review", response_model=SlideReviewResponse)
def update_slide_review(
    slide_id: int,
    payload: SlideReviewRequest,
    db: Session = Depends(get_db),
):
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    slide.review_status = payload.review_status
    db.add(slide)
    db.commit()
    db.refresh(slide)
    return SlideReviewResponse(id=slide.id, review_status=slide.review_status)


@router.post("/import", response_model=SlideImportResponse)
def import_slide(payload: SlideImportRequest, request: Request, db: Session = Depends(get_db)):
    settings = request.app.state.settings

    src = Path(payload.file_path)
    _validate_import_source(
        src,
        settings.import_allowed_roots,
        allow_tile_like_import=payload.allow_tile_like_import,
    )

    collection: ImportCollection | None = None
    if payload.collection_id is not None:
        collection = db.get(ImportCollection, payload.collection_id)
        if not collection:
            raise AppError(ErrorCode.NOT_FOUND, f"Import collection {payload.collection_id} not found")
    else:
        collection = _create_import_collection(
            db,
            title=None,
            source_type="import",
        )

    try:
        slide = _import_slide_file(db=db, src=src, settings=settings, collection_id=collection.id if collection else None)
    except Exception:
        if collection is not None and payload.collection_id is None:
            existing_collection = db.get(ImportCollection, collection.id)
            if existing_collection is not None:
                db.delete(existing_collection)
                db.commit()
        raise
    if slide is None:
        raise AppError(ErrorCode.IO_ERROR, "Slide import did not complete")
    return SlideImportResponse(slide_id=slide.id, stored_path=_managed_slide_identifier(slide.stored_filename))


@router.post("/import-collection", response_model=SlideImportCollectionResponse)
@router.post("/import-collections", response_model=SlideImportCollectionResponse)
def import_slide_collection(payload: SlideImportCollectionRequest, request: Request, db: Session = Depends(get_db)):
    settings = request.app.state.settings

    # Keep batch imports bounded to avoid long-running single requests.
    if len(payload.file_paths) > MAX_IMPORT_COLLECTION_ITEMS:
        raise AppError(
            ErrorCode.SLIDE_INVALID,
            f"Too many files in one import collection (max {MAX_IMPORT_COLLECTION_ITEMS})",
        )

    deduped_file_paths = list(dict.fromkeys(payload.file_paths))
    sources = [Path(file_path) for file_path in deduped_file_paths]
    for src in sources:
        _validate_import_source(
            src,
            settings.import_allowed_roots,
            allow_tile_like_import=payload.allow_tile_like_import,
        )

    collection = _create_import_collection(
        db,
        title=payload.title,
        source_type=payload.source_type,
    )

    imported_slide_ids: list[int] = []
    copied_paths: list[Path] = []
    try:
        for src in sources:
            slide = _import_slide_file(db=db, src=src, settings=settings, collection_id=collection.id)
            imported_slide_ids.append(slide.id)
            copied_paths.append(Path(slide.stored_path))
    except Exception:
        for slide_id in imported_slide_ids:
            shutil.rmtree(settings.tiles_cache_dir / str(slide_id), ignore_errors=True)
            imported_slide = db.get(Slide, slide_id)
            if imported_slide:
                db.delete(imported_slide)
        for copied_path in copied_paths:
            if copied_path.is_file():
                try:
                    copied_path.unlink()
                except OSError:
                    logger.warning("Could not delete copied slide file %s after failed collection import", copied_path)
        db.delete(collection)
        db.commit()
        raise

    return SlideImportCollectionResponse(
        collection=_serialize_collection(collection),
        imported_count=len(imported_slide_ids),
        imported_slide_ids=imported_slide_ids,
    )


@collections_router.get("/import-collections/{collection_id}", response_model=ImportCollectionResponse)
def get_import_collection(
    collection_id: int,
    db: Session = Depends(get_db),
):
    collection = db.get(ImportCollection, collection_id)
    if not collection:
        raise AppError(ErrorCode.NOT_FOUND, f"Import collection {collection_id} not found")
    return _serialize_collection(collection)


@collections_router.patch("/import-collections/{collection_id}", response_model=ImportCollectionResponse)
@collections_router.post("/import-collections/{collection_id}/rename", response_model=ImportCollectionResponse)
def rename_import_collection(
    collection_id: int,
    payload: ImportCollectionRenameRequest,
    db: Session = Depends(get_db),
):
    collection = db.get(ImportCollection, collection_id)
    if not collection:
        raise AppError(ErrorCode.NOT_FOUND, f"Import collection {collection_id} not found")

    collection.title = payload.title
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return _serialize_collection(collection)


@router.get("/{slide_id}/metadata", response_model=SlideMetadataResponse)
def slide_metadata(slide_id: int, request: Request, db: Session = Depends(get_db)):
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    settings = request.app.state.settings
    p = _resolve_managed_slide_path(slide, settings)

    if p.suffix.lower() in RASTER_EXTENSIONS:
        try:
            meta = read_raster_metadata(p)
        except Exception:
            logger.exception("Could not read raster metadata for slide %s", slide_id)
            raise AppError(ErrorCode.SLIDE_UNREADABLE, "Could not read slide metadata")
    else:
        try:
            meta = read_openslide_metadata(p)
        except Exception:
            # OpenSlide errors vary; treat as “unprocessable slide”
            logger.exception("OpenSlide metadata read failed for slide %s", slide_id)
            raise AppError(ErrorCode.SLIDE_UNREADABLE, "Could not read slide metadata")

    return SlideMetadataResponse(slide_id=slide_id, **meta)

# Small preview image for each slide in gallery
@router.get("/{slide_id}/thumbnail")
def slide_thumbnail(
    slide_id: int,
    request: Request,
    size: int = Query(default=256, ge=THUMBNAIL_SIZE_MIN, le=THUMBNAIL_SIZE_MAX),
    db: Session = Depends(get_db),
):
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    settings = request.app.state.settings
    slide_path = _resolve_managed_slide_path(slide, settings)
    cache_path = _thumbnail_cache_path(settings, slide_id, size)
    if cache_path.is_file():
        return FileResponse(path=cache_path, media_type="image/jpeg")

    if slide_path.suffix.lower() in RASTER_EXTENSIONS:
        try:
            jpg = _raster_thumbnail_jpeg(slide_path, size)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(jpg)
            return Response(content=jpg, media_type="image/jpeg")
        except AppError:
            raise
        except Exception:
            logger.exception("Could not generate raster thumbnail for slide %s", slide_id)
            raise AppError(ErrorCode.SLIDE_UNREADABLE, "Could not read slide image")

    try:
        configure_openslide_runtime()
        import openslide
        osr = openslide.OpenSlide(str(slide_path))
    except Exception:
        logger.exception("OpenSlide thumbnail open failed for slide %s", slide_id)
        raise AppError(ErrorCode.SLIDE_UNREADABLE, "Could not read slide image")

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
        jpg = buf.getvalue()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(jpg)
        return Response(content=jpg, media_type="image/jpeg")
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

    slide_path = _resolve_managed_slide_path(slide, settings)

    tile_size = TILE_SIZE_DEFAULT
    tile_path = (
        settings.tiles_cache_dir
        / str(slide_id)
        / str(tile_size)
        / str(level)
        / f"{x}_{y}.jpg"
    )

    if slide_path.suffix.lower() in RASTER_EXTENSIONS:
        try:
            if tile_path.exists():
                with Image.open(slide_path) as img:
                    iw, ih = img.size
                if x < 0 or y < 0 or x * tile_size >= iw or y * tile_size >= ih or level != 0:
                    raise AppError(ErrorCode.NOT_FOUND, f"Tile out of bounds: level={level} x={x} y={y}")
                return Response(content=tile_path.read_bytes(), media_type="image/jpeg")
            jpg_bytes = _raster_tile_jpeg(slide_path, level, x, y, tile_size)
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tile_path.write_bytes(jpg_bytes)
            return Response(content=jpg_bytes, media_type="image/jpeg")
        except AppError:
            raise
        except Exception:
            logger.exception("Could not generate raster tile for slide %s", slide_id)
            raise AppError(ErrorCode.SLIDE_UNREADABLE, "Could not read slide image")

    # 2) Validate level exists + 3) compute tile region
    try:
        configure_openslide_runtime()
        import openslide
        osr = openslide.OpenSlide(str(slide_path))
    except Exception:
        logger.exception("OpenSlide tile open failed for slide %s", slide_id)
        raise AppError(ErrorCode.SLIDE_UNREADABLE, "Could not read slide image")

    try:
        if level < 0 or level >= osr.level_count:
            raise AppError(ErrorCode.SLIDE_INVALID, f"Invalid level {level}; slide has {osr.level_count} levels")

        level_w, level_h = osr.level_dimensions[level]

        px_level = x * tile_size
        py_level = y * tile_size

        # Reject completely out-of-bounds tiles (avoids doing work for nonsense coords)
        if px_level >= level_w or py_level >= level_h or x < 0 or y < 0:
            raise AppError(ErrorCode.NOT_FOUND, f"Tile out of bounds: level={level} x={x} y={y}")

        if tile_path.exists():
            return Response(content=tile_path.read_bytes(), media_type="image/jpeg")

        # OpenSlide read_region location is in level-0 coordinates
        downsample = float(osr.level_downsamples[level])
        px0 = int(px_level * downsample)
        py0 = int(py_level * downsample)

        w = min(tile_size, level_w - px_level)
        h = min(tile_size, level_h - py_level)

        # 4) read_region (returns RGBA)
        img = osr.read_region((px0, py0), level, (w, h))

        # 5) Convert to RGB (JPEG needs no alpha)
        img = img.convert("RGB")

        if w != tile_size or h != tile_size:
            padded = Image.new("RGB", (tile_size, tile_size))
            padded.paste(img, (0, 0))
            img = padded

        # 6) Return JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        jpg_bytes = buf.getvalue()

        # write-through cache
        tile_path.parent.mkdir(parents=True, exist_ok=True)
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

    _resolve_managed_slide_path(slide, settings)

    dz_paths = deepzoom_paths(settings.tiles_cache_dir, slide_id)
    if not has_deepzoom(dz_paths):
        raise AppError(
            ErrorCode.NOT_FOUND,
            "DeepZoom tiles not pre-generated for this slide",
            http_status=404,
        )
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

    _resolve_managed_slide_path(slide, settings)

    dz_paths = deepzoom_paths(settings.tiles_cache_dir, slide_id)
    if not has_deepzoom(dz_paths):
        raise AppError(
            ErrorCode.NOT_FOUND,
            "DeepZoom tiles not pre-generated for this slide",
            http_status=404,
        )
    if level < 0 or x < 0 or y < 0:
        raise AppError(ErrorCode.NOT_FOUND, f"Tile out of bounds: level={level} x={x} y={y}")
    tile_path = dz_paths.tiles_dir / str(level) / f"{x}_{y}.jpg"
    if not tile_path.exists():
        raise AppError(ErrorCode.NOT_FOUND, f"Tile out of bounds: level={level} x={x} y={y}")
    return FileResponse(path=tile_path, media_type="image/jpeg")
