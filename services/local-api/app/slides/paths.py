from __future__ import annotations

import hashlib
from pathlib import Path


def folder_key_for_original_path(original_path: str | None) -> str:
    if not original_path:
        return "uncategorized"
    parent = Path(original_path).parent
    normalized = parent.resolve(strict=False).as_posix()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"folder_{digest}"
