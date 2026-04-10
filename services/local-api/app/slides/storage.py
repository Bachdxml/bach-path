from __future__ import annotations
from pathlib import Path
import shutil


def _validate_dest_filename(dest_filename: str) -> str:
    cleaned = dest_filename.strip()
    if not cleaned:
        raise ValueError("Destination filename is required")
    if cleaned in {".", ".."}:
        raise ValueError("Destination filename is invalid")
    if Path(cleaned).name != cleaned:
        raise ValueError("Destination filename must not include path separators")
    return cleaned


def copy_into_managed_storage(src: Path, dest_dir: Path, dest_filename: str) -> Path:
    safe_name = _validate_dest_filename(dest_filename)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = (dest_dir / safe_name).resolve()
    resolved_dest_dir = dest_dir.resolve()
    try:
        dest_path.relative_to(resolved_dest_dir)
    except ValueError as exc:
        raise ValueError("Destination path escapes managed storage") from exc

    # Ensure overwrite is explicit (not accidental)
    if dest_path.exists():
        raise FileExistsError(f"Destination already exists: {dest_path}")

    shutil.copy2(src, dest_path)
    return dest_path
