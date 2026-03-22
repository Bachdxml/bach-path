from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class DeepZoomPaths:
    prefix: Path
    descriptor: Path
    tiles_dir: Path
    complete_marker: Path


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


def ensure_deepzoom(slide_path: Path, cache_root: Path, slide_id: int, tile_size: int = 256) -> DeepZoomPaths:
    paths = deepzoom_paths(cache_root, slide_id)
    if has_deepzoom(paths):
        return paths

    # Regenerate from scratch when output is partial/corrupt.
    if paths.prefix.parent.exists():
        shutil.rmtree(paths.prefix.parent, ignore_errors=True)
    paths.prefix.parent.mkdir(parents=True, exist_ok=True)

    import pyvips  # lazy import: keeps API boot fast if unavailable

    image = pyvips.Image.new_from_file(str(slide_path), access="sequential")
    image.dzsave(
        str(paths.prefix),
        tile_size=tile_size,
        overlap=0,
        suffix=".jpg[Q=85]",
    )
    paths.complete_marker.write_text("ok", encoding="utf-8")
    return paths
