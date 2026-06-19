from __future__ import annotations
import io
import logging
import re
import shutil
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_db
from app.inference.results import classify_inference_result
from app.models.enums import InferenceStatus
from app.models.inference_run import InferenceRun
from app.models.import_collection import ImportCollection
from app.models.region import Region
from app.models.slide import Slide
from app.schemas.slides import (
    ImportCollectionCreateRequest,
    ImportCollectionRenameRequest,
    ImportCollectionResponse,
    SlideImportResponse,
    SlideListItem,
    SlideListResponse,
    SlideMetadataResponse,
    SlideReviewRequest,
    SlideReviewResponse,
)
from app.slides.access import resolve_managed_slide_path as _resolve_managed_slide_path
from app.slides.deepzoom import deepzoom_paths, has_deepzoom, ensure_deepzoom
from app.slides.metadata import RASTER_EXTENSIONS, read_openslide_metadata, read_raster_metadata
from app.slides.paths import folder_key_for_original_path
from app.slides.storage import NotEnoughDiskSpaceError, stream_into_managed_storage
from app.util.exceptions import AppError, ErrorCode
from app.util.openslide_runtime import configure_openslide_runtime

router = APIRouter(prefix="/slides", tags=["slides"])
collections_router = APIRouter(tags=["slides"])
logger = logging.getLogger(__name__)

WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".png"}

TILE_SIZE_DEFAULT = 256
THUMBNAIL_SIZE_MIN = 32
THUMBNAIL_SIZE_MAX = 2048
TILE_FILENAME_RE = re.compile(r"(?:^|[_-])tile[_-]?x?\d+[_-]y?\d+(?:[_-]|$)", re.IGNORECASE)

# --- OpenSlide handle cache (Finding #11) -----------------------------------
# Opening a WSI parses pyramid metadata and header offsets, which is expensive.
# The viewer fires many tile requests per second during pan/zoom, so we share a
# bounded set of OpenSlide handles across requests keyed on the resolved path.
#
# Concurrency design: the lock is held only to look up or insert/evict a handle
# in the OrderedDict. The actual read_region happens outside the lock once the
# caller holds a handle reference. On eviction we simply drop the dict entry and
# never call .close() on it -- another in-flight request may still be reading
# through that same handle. The handle is closed by OpenSlide's own __del__ once
# the last reference is garbage-collected, so we never close an in-use handle.
_OPENSLIDE_CACHE_MAXSIZE = 8
_openslide_cache: "OrderedDict[str, object]" = OrderedDict()
_openslide_cache_lock = threading.Lock()


def _get_cached_openslide(slide_path: Path):
    """Return a shared OpenSlide handle for ``slide_path``, opening it on miss.

    The handle is owned by the cache; callers must not close it. See the module
    comment above for the concurrency rationale.
    """
    key = str(slide_path)
    with _openslide_cache_lock:
        handle = _openslide_cache.get(key)
        if handle is not None:
            _openslide_cache.move_to_end(key)
            return handle

    # Open outside the lock so a slow open() does not block other slides.
    configure_openslide_runtime()
    import openslide

    handle = openslide.OpenSlide(str(slide_path))

    with _openslide_cache_lock:
        existing = _openslide_cache.get(key)
        if existing is not None:
            # Another thread opened it first; reuse theirs and drop ours
            # (ours is closed by GC once this function returns).
            _openslide_cache.move_to_end(key)
            return existing
        _openslide_cache[key] = handle
        _openslide_cache.move_to_end(key)
        while len(_openslide_cache) > _OPENSLIDE_CACHE_MAXSIZE:
            # Drop the reference only; never .close() a possibly in-use handle.
            _openslide_cache.popitem(last=False)
    return handle


def _evict_openslide_handle(slide_path: Path) -> None:
    """Drop and close any cached handle for ``slide_path``.

    Called before deleting a slide file so a cached handle does not keep the
    file open (notably on Windows, where an open handle blocks unlink). Unlike
    capacity eviction, this path explicitly closes because the file is going
    away; any concurrent read will fail, which is acceptable for a deletion.
    """
    key = str(slide_path)
    with _openslide_cache_lock:
        handle = _openslide_cache.pop(key, None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


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


def _filename_looks_like_generated_tile(filename: str) -> bool:
    """Detect a generated training-tile filename (e.g. ``tile_x0_y0.png``).

    Filename-only by design: the upload endpoint never receives a source folder,
    so the previous folder-structure heuristic no longer applies (Requirement 7).
    """
    name = Path(filename).name
    if Path(name).suffix.lower() not in RASTER_EXTENSIONS:
        return False
    stem = Path(name).stem.lower()
    return bool(TILE_FILENAME_RE.search(stem))


def _validate_upload_filename(filename: str, *, allow_tile_like_import: bool = False) -> str:
    """Validate an uploaded slide's filename and return its lowercased suffix."""
    name = Path(filename).name
    if not name.strip():
        raise AppError(ErrorCode.SLIDE_INVALID, "Uploaded file is missing a filename")
    suffix = Path(name).suffix.lower()
    if suffix not in WSI_EXTENSIONS:
        raise AppError(ErrorCode.SLIDE_INVALID, "Unsupported slide file extension")
    if not allow_tile_like_import and _filename_looks_like_generated_tile(name):
        raise AppError(
            ErrorCode.SLIDE_INVALID,
            "This looks like a generated training tile, not a whole slide. Import the source .svs instead.",
        )
    return suffix


def _import_uploaded_slide(
    *,
    db: Session,
    upload_file: UploadFile,
    settings,
    collection_id: int | None,
    allow_tile_like_import: bool = False,
) -> Slide:
    """Stream an uploaded slide to managed storage and run the import pipeline.

    Mirrors the path-import pipeline (managed-storage placement, DB row, DeepZoom
    pre-gen) so an uploaded slide is indistinguishable from one imported before,
    but the bytes arrive over multipart instead of a shared filesystem path.
    """
    filename = upload_file.filename or ""
    suffix = _validate_upload_filename(filename, allow_tile_like_import=allow_tile_like_import)
    # original_path stores only the original filename for display; host paths are
    # never meaningful here (Constraints / Open Question on original_path).
    original_name = Path(filename).name

    slide = Slide(
        original_path=original_name,
        stored_filename="pending",
        stored_path="pending",
        file_size_bytes=0,
        sha256=None,
        import_collection_id=collection_id,
    )
    db.add(slide)
    db.flush()
    db.commit()
    db.refresh(slide)

    src = upload_file.file
    try:
        src.seek(0)
    except (OSError, ValueError):
        pass
    expected_size = getattr(upload_file, "size", None)

    dest_filename = f"{slide.id}{suffix}"
    dest_path = None
    bytes_written = 0
    for attempt in range(4):
        try:
            dest_path, bytes_written = stream_into_managed_storage(
                src,
                settings.slides_dir,
                dest_filename,
                expected_size=expected_size,
            )
            break
        except FileExistsError:
            if attempt == 0:
                logger.warning(
                    "Managed storage collision for slide id=%s filename=%s; retrying with unique suffix",
                    slide.id,
                    dest_filename,
                )
            dest_filename = f"{slide.id}-{uuid.uuid4().hex[:8]}{suffix}"
            try:
                src.seek(0)
            except (OSError, ValueError):
                pass
        except NotEnoughDiskSpaceError:
            _delete_pending_slide_row(db, slide.id)
            raise AppError(
                ErrorCode.IO_ERROR,
                "Not enough disk space to import this slide. Free up space and try again.",
                http_status=507,
            )
        except ValueError:
            _delete_pending_slide_row(db, slide.id)
            raise AppError(ErrorCode.SLIDE_INVALID, "Invalid managed storage destination")
        except OSError:
            logger.exception("Failed to write uploaded slide %s", original_name)
            _delete_pending_slide_row(db, slide.id)
            raise AppError(ErrorCode.IO_ERROR, "Failed to save uploaded slide")
    if dest_path is None:
        _delete_pending_slide_row(db, slide.id)
        raise AppError(ErrorCode.CONFLICT, "Managed slide already exists")

    # A zero-byte (empty or truncated) upload is an unreadable slide, not a
    # successful import; clean up the partial file and the pending row.
    if bytes_written == 0:
        try:
            dest_path.unlink()
        except OSError:
            pass
        _delete_pending_slide_row(db, slide.id)
        raise AppError(ErrorCode.SLIDE_UNREADABLE, "Uploaded file was empty or unreadable")

    slide.stored_filename = dest_filename
    slide.stored_path = str(dest_path)
    slide.file_size_bytes = bytes_written
    db.add(slide)
    db.commit()
    db.refresh(slide)
    try:
        ensure_deepzoom(dest_path, settings.tiles_cache_dir, slide.id)
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
        if x < 0 or y < 0:
            raise AppError(ErrorCode.SLIDE_INVALID, f"Invalid tile coordinates: level={level} x={x} y={y}")
        px = x * tile_size
        py = y * tile_size
        if px >= iw or py >= ih:
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

    _evict_openslide_handle(slide_path)

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
        # "Succeeded" here means a latest succeeded run exists for the slide.
        run = latest_run_by_slide.get(slide_id)
        if not run:
            return classify_inference_result(0, 0, succeeded=False)
        return classify_inference_result(
            positive_count_by_run.get(run.id, 0),
            negative_count_by_run.get(run.id, 0),
            succeeded=True,
        )

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


@router.post("/upload", response_model=SlideImportResponse)
def upload_slide(
    request: Request,
    file: UploadFile = File(...),
    collection_id: int | None = Form(default=None),
    allow_tile_like_import: bool = Form(default=False),
    db: Session = Depends(get_db),
):
    """Import a single slide from its uploaded bytes (multipart), not a path.

    The client and backend no longer share a filesystem; the raw file is streamed
    to managed storage and run through the existing import pipeline. A batch
    import pre-creates a collection and calls this once per file so per-file
    progress and skip-and-continue work (Requirements 1-5).
    """
    settings = request.app.state.settings

    collection: ImportCollection | None = None
    created_collection = False
    if collection_id is not None:
        collection = db.get(ImportCollection, collection_id)
        if not collection:
            raise AppError(ErrorCode.NOT_FOUND, f"Import collection {collection_id} not found")
    else:
        collection = _create_import_collection(db, title=None, source_type="upload")
        created_collection = True

    try:
        slide = _import_uploaded_slide(
            db=db,
            upload_file=file,
            settings=settings,
            collection_id=collection.id,
            allow_tile_like_import=allow_tile_like_import,
        )
    except Exception:
        # Only tear down a collection this request created; never an existing one
        # the client is uploading more files into.
        if created_collection:
            existing_collection = db.get(ImportCollection, collection.id)
            if existing_collection is not None:
                db.delete(existing_collection)
                db.commit()
        raise
    return SlideImportResponse(slide_id=slide.id, stored_path=_managed_slide_identifier(slide.stored_filename))


@collections_router.post("/import-collections", response_model=ImportCollectionResponse)
def create_import_collection(payload: ImportCollectionCreateRequest, db: Session = Depends(get_db)):
    """Create an empty import collection that uploaded slides are grouped into."""
    collection = _create_import_collection(
        db,
        title=payload.title,
        source_type=payload.source_type or "upload",
    )
    return _serialize_collection(collection)


@collections_router.delete("/import-collections/{collection_id}")
def delete_import_collection(collection_id: int, db: Session = Depends(get_db)):
    """Delete an import collection only when it holds no slides.

    Used to clean up after a batch where every file failed, so an empty
    collection is never left behind (Requirement 4).
    """
    collection = db.get(ImportCollection, collection_id)
    if not collection:
        raise AppError(ErrorCode.NOT_FOUND, f"Import collection {collection_id} not found")
    slide_count = db.query(Slide).filter(Slide.import_collection_id == collection_id).count()
    if slide_count > 0:
        raise AppError(
            ErrorCode.CONFLICT,
            "Cannot delete an import collection that still contains slides",
        )
    db.delete(collection)
    db.commit()
    return {"ok": True, "id": collection_id}


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
        osr = _get_cached_openslide(slide_path)
    except Exception:
        logger.exception("OpenSlide thumbnail open failed for slide %s", slide_id)
        raise AppError(ErrorCode.SLIDE_UNREADABLE, "Could not read slide image")

    # Cached handle is owned by the cache; do not close it here.
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
        # Cheap validation before any cache lookup or image open, so cache-hit
        # and cache-miss paths return identical status codes.
        if level != 0:
            raise AppError(ErrorCode.SLIDE_INVALID, f"Raster images have only level 0; got {level}")
        if x < 0 or y < 0:
            raise AppError(ErrorCode.SLIDE_INVALID, f"Invalid tile coordinates: level={level} x={x} y={y}")
        try:
            if tile_path.exists():
                # Cached tiles were only written after passing bounds checks in
                # _raster_tile_jpeg, so no need to re-open the source image.
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
        osr = _get_cached_openslide(slide_path)
    except Exception:
        logger.exception("OpenSlide tile open failed for slide %s", slide_id)
        raise AppError(ErrorCode.SLIDE_UNREADABLE, "Could not read slide image")

    # Cached handle is owned by the cache; do not close it here. The read below
    # runs outside the cache lock since the handle reference is now held.
    if level < 0 or level >= osr.level_count:
        raise AppError(ErrorCode.SLIDE_INVALID, f"Invalid level {level}; slide has {osr.level_count} levels")

    level_w, level_h = osr.level_dimensions[level]

    px_level = x * tile_size
    py_level = y * tile_size

    # Negative coordinates are invalid client input, not a missing resource.
    if x < 0 or y < 0:
        raise AppError(ErrorCode.SLIDE_INVALID, f"Invalid tile coordinates: level={level} x={x} y={y}")

    # Reject completely out-of-bounds tiles (avoids doing work for nonsense coords)
    if px_level >= level_w or py_level >= level_h:
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
        raise AppError(ErrorCode.SLIDE_INVALID, f"Invalid tile coordinates: level={level} x={x} y={y}")
    tile_path = dz_paths.tiles_dir / str(level) / f"{x}_{y}.jpg"
    if not tile_path.exists():
        raise AppError(ErrorCode.NOT_FOUND, f"Tile out of bounds: level={level} x={x} y={y}")
    return FileResponse(path=tile_path, media_type="image/jpeg")
