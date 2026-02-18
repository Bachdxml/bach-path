from __future__ import annotations
from pathlib import Path
import shutil

def copy_into_managed_storage(src: Path, dest_dir: Path, dest_filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = (dest_dir / dest_filename).resolve()

    # Ensure overwrite is explicit (not accidental)
    if dest_path.exists():
        raise FileExistsError(f"Destination already exists: {dest_path}")

    shutil.copy2(src, dest_path)
    return dest_path
