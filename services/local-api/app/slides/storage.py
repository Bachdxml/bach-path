from __future__ import annotations
from pathlib import Path
import errno
import os
import shutil
from typing import BinaryIO

# Bytes pulled from the upload stream per write. Bounds peak additional memory
# for an upload to one chunk regardless of slide size (spec Requirement 2).
UPLOAD_CHUNK_SIZE = 1024 * 1024


class NotEnoughDiskSpaceError(Exception):
    """Raised when the destination disk cannot hold an uploaded slide."""


def _validate_dest_filename(dest_filename: str) -> str:
    cleaned = dest_filename.strip()
    if not cleaned:
        raise ValueError("Destination filename is required")
    if cleaned in {".", ".."}:
        raise ValueError("Destination filename is invalid")
    if Path(cleaned).name != cleaned:
        raise ValueError("Destination filename must not include path separators")
    return cleaned


def _resolve_managed_dest(dest_dir: Path, dest_filename: str) -> Path:
    safe_name = _validate_dest_filename(dest_filename)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = (dest_dir / safe_name).resolve()
    resolved_dest_dir = dest_dir.resolve()
    try:
        dest_path.relative_to(resolved_dest_dir)
    except ValueError as exc:
        raise ValueError("Destination path escapes managed storage") from exc
    if dest_path.exists():
        raise FileExistsError(f"Destination already exists: {dest_path}")
    return dest_path


def stream_into_managed_storage(
    src_fileobj: BinaryIO,
    dest_dir: Path,
    dest_filename: str,
    *,
    expected_size: int | None = None,
    chunk_size: int = UPLOAD_CHUNK_SIZE,
) -> tuple[Path, int]:
    """Stream ``src_fileobj`` into managed storage in bounded chunks.

    The bytes are written to a sibling ``.part`` file and atomically renamed into
    place only after the whole stream is consumed, so a failed or cancelled upload
    never leaves a usable-looking slide behind. Peak additional memory is one
    ``chunk_size`` buffer, never the file size (Requirement 2).

    Raises ``NotEnoughDiskSpaceError`` when the disk genuinely cannot hold the
    upload (either a pre-check against ``expected_size`` or an ENOSPC mid-write),
    after deleting the partial file. ``FileExistsError`` is raised if the
    destination already exists so the caller can retry with a unique name.
    """
    dest_path = _resolve_managed_dest(dest_dir, dest_filename)

    if expected_size is not None:
        try:
            free = shutil.disk_usage(dest_path.parent).free
        except OSError:
            free = None
        if free is not None and free < expected_size:
            raise NotEnoughDiskSpaceError(
                "Not enough disk space to store the uploaded slide"
            )

    tmp_path = dest_path.parent / (dest_path.name + ".part")
    bytes_written = 0
    try:
        with open(tmp_path, "wb") as out:
            while True:
                chunk = src_fileobj.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                bytes_written += len(chunk)
        os.replace(tmp_path, dest_path)
    except OSError as exc:
        _silent_unlink(tmp_path)
        if exc.errno == errno.ENOSPC:
            raise NotEnoughDiskSpaceError(
                "Ran out of disk space while writing the uploaded slide"
            ) from exc
        raise
    except BaseException:
        # Cancellation or any other failure: never leave an orphaned partial.
        _silent_unlink(tmp_path)
        raise
    return dest_path, bytes_written


def _silent_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
