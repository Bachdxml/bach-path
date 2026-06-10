from __future__ import annotations
from pathlib import Path

from app.models.slide import Slide
from app.util.exceptions import AppError, ErrorCode


def resolve_managed_slide_path(slide: Slide, settings, *, must_exist: bool = True) -> Path:
    """Resolve a slide's on-disk path, asserting it lives inside managed storage.

    Canonical implementation shared by the slides and inference routers. The
    ``must_exist`` flag preserves both historical callers' behavior: most call
    sites require the file to exist on disk (``must_exist=True``), while the
    slide-deletion path resolves the location without requiring presence
    (``must_exist=False``).
    """
    path = Path(slide.stored_path).resolve(strict=False)
    slides_root = settings.slides_dir.resolve(strict=False)
    try:
        path.relative_to(slides_root)
    except ValueError as e:
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide path is outside managed storage") from e
    if must_exist and (not path.exists() or not path.is_file()):
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")
    return path
