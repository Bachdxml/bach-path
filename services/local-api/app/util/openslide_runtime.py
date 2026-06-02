from __future__ import annotations

import os
from pathlib import Path

_dll_directory_added = False


def configure_openslide_runtime() -> None:
    """Make openslide-bin DLLs discoverable on Windows before importing openslide."""
    global _dll_directory_added
    if _dll_directory_added or os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    try:
        import openslide_bin
    except ImportError:
        _dll_directory_added = True
        return

    dll_dir = Path(openslide_bin.__file__).resolve().parent
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))
    _dll_directory_added = True
