from __future__ import annotations
from pathlib import Path
import sys

def _base_path() -> Path:
    """
    Works in dev and in PyInstaller one-dir/one-file.
    If you bundle the alembic folder, point to it here.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]  # services/local-api

def _resource_path(rel: str) -> Path:
    return (_base_path() / rel).resolve()

def migrations_available() -> bool:
    alembic_ini = _resource_path("alembic.ini")
    script_location = _resource_path("alembic")
    return alembic_ini.is_file() and script_location.is_dir()

def run_migrations(sqlite_path: Path) -> None:
    from alembic import command
    from alembic.config import Config

    alembic_ini = _resource_path("alembic.ini")
    script_location = _resource_path("alembic")
    if not alembic_ini.is_file():
        raise FileNotFoundError(f"Alembic config not found: {alembic_ini}")
    if not script_location.is_dir():
        raise FileNotFoundError(f"Alembic script directory not found: {script_location}")

    cfg = Config(str(alembic_ini))
    cfg.set_main_option("script_location", str(script_location))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{sqlite_path.as_posix()}")

    # Upgrade to head at startup
    command.upgrade(cfg, "head")
