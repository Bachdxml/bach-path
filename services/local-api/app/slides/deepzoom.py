from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import threading

logger = logging.getLogger(__name__)

# pyvips is optional at runtime; native libvips must be installed (e.g. macOS: brew install vips).
_pyvips = None
_pyvips_probe_done = False
_missing_libvips_warned = False


@dataclass(frozen=True)
class DeepZoomPaths:
    prefix: Path
    descriptor: Path
    tiles_dir: Path
    complete_marker: Path


_deepzoom_locks: dict[int, threading.Lock] = {}
_deepzoom_locks_guard = threading.Lock()


def _lock_for_slide(slide_id: int) -> threading.Lock:
    with _deepzoom_locks_guard:
        lock = _deepzoom_locks.get(slide_id)
        if lock is None:
            lock = threading.Lock()
            _deepzoom_locks[slide_id] = lock
        return lock


def deepzoom_paths(cache_root: Path, slide_id: int) -> DeepZoomPaths:
    slide_dir = cache_root / str(slide_id) / "deepzoom"
    prefix = slide_dir / "slide"
    return DeepZoomPaths(
        prefix=prefix,
        descriptor=slide_dir / "slide.dzi",
        tiles_dir=slide_dir / "slide_files",
        complete_marker=slide_dir / ".complete",
    )


def has_deepzoom(paths: DeepZoomPaths) -> bool:
    return (
        paths.descriptor.exists()
        and paths.tiles_dir.exists()
        and paths.complete_marker.exists()
    )


def _get_pyvips():
    """Return pyvips module if import + libvips load succeed; else None."""
    global _pyvips, _pyvips_probe_done, _missing_libvips_warned
    if _pyvips_probe_done:
        return _pyvips
    _pyvips_probe_done = True
    try:
        import pyvips

        _pyvips = pyvips
    except (ImportError, OSError, EnvironmentError) as e:
        _pyvips = None
        if not _missing_libvips_warned:
            _missing_libvips_warned = True
            logger.warning(
                "pyvips/libvips is not usable (%s). Deep Zoom pre-caching is disabled; "
                "the viewer will use on-demand tiles instead. On macOS install native libvips: "
                "`brew install vips`, then restart the API.",
                type(e).__name__,
            )
    return _pyvips


def ensure_deepzoom(slide_path: Path, cache_root: Path, slide_id: int, tile_size: int = 256) -> DeepZoomPaths:
    paths = deepzoom_paths(cache_root, slide_id)
    with _lock_for_slide(slide_id):
        if has_deepzoom(paths):
            return paths

        # Regenerate from scratch when output is partial/corrupt.
        if paths.prefix.parent.exists():
            shutil.rmtree(paths.prefix.parent, ignore_errors=True)
        paths.prefix.parent.mkdir(parents=True, exist_ok=True)

        pyvips = _get_pyvips()
        if pyvips is None:
            return paths

        image = pyvips.Image.new_from_file(str(slide_path), access="sequential")
        image.dzsave(
            str(paths.prefix),
            tile_size=tile_size,
            overlap=0,
            suffix=".jpg[Q=85]",
        )
        paths.complete_marker.write_text("ok", encoding="utf-8")
    return paths
