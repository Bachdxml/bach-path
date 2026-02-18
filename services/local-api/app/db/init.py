from __future__ import annotations
from pathlib import Path
from app.settings import Settings
from app.db.session import make_engine
from app.db.base import Base

# Explicitly import all model modules so tables are registered
from app.models import (
    audit_log,
    enums,
    inference_run,
    region,
    slide,
    user,
)


REQUIRED_DIRS = ("slides", "inference_runs", "tiles_cache", "logs")

def ensure_dirs(settings: Settings) -> None:
    settings.app_data_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        (settings.app_data_dir / name).mkdir(parents=True, exist_ok=True)

def init_database(settings: Settings) -> None:
    ensure_dirs(settings)
    engine = make_engine(settings.sqlite_path)
    Base.metadata.create_all(bind=engine)
