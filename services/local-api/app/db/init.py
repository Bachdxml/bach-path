from __future__ import annotations
from pathlib import Path
import logging
from app.settings import Settings
from app.db.session import make_engine
from app.db.base import Base
from app.db.migrations_runner import migrations_available, run_migrations

# Explicitly import all model modules so tables are registered
from app.models import (
    audit_log,
    enums,
    inference_run,
    region,
    slide,
    user,
)


REQUIRED_DIRS = ("slides", "inference_runs", "training_runs", "tiles_cache", "logs")
logger = logging.getLogger(__name__)

def ensure_dirs(settings: Settings) -> None:
    settings.app_data_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        (settings.app_data_dir / name).mkdir(parents=True, exist_ok=True)

def init_database(settings: Settings) -> None:
    ensure_dirs(settings)
    if migrations_available():
        run_migrations(settings.sqlite_path)
        return
    logger.warning("Alembic assets not found; falling back to metadata.create_all().")
    engine = make_engine(settings.sqlite_path)
    Base.metadata.create_all(bind=engine)
