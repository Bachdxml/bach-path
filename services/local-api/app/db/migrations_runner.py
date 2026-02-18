from __future__ import annotations
from pathlib import Path
import sys
from alembic import command
from alembic.config import Config

def _resource_path(rel: str) -> str:
    """
    Works in dev and in PyInstaller one-dir/one-file.
    If you bundle the alembic folder, point to it here.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parents[2]  # backend/app/db -> backend
    return str((base / rel).resolve())

def run_migrations(sqlite_path: Path) -> None:
    alembic_ini = _resource_path("alembic.ini")         # bundle this
    script_location = _resource_path("alembic")         # bundle this folder

    cfg = Config(alembic_ini)
    cfg.set_main_option("script_location", script_location)
    cfg.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{sqlite_path.as_posix()}")

    # Upgrade to head at startup
    command.upgrade(cfg, "head")
