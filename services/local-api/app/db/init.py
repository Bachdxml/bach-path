from __future__ import annotations
from pathlib import Path
import logging
from datetime import datetime, timezone

from sqlalchemy import inspect

from app.settings import Settings
from app.db.session import make_engine, make_session_factory
from app.db.base import Base
from app.db.migrations_runner import migrations_available, run_migrations

# Explicitly import all model modules so tables are registered
from app.models import (
    audit_log,
    enums,
    inference_run,
    import_collection,
    region,
    slide,
    user,
)
from app.models.import_collection import ImportCollection
from app.models.slide import Slide


REQUIRED_DIRS = ("slides", "inference_runs", "training_runs", "tiles_cache", "logs")
logger = logging.getLogger(__name__)

def ensure_dirs(settings: Settings) -> None:
    settings.app_data_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        (settings.app_data_dir / name).mkdir(parents=True, exist_ok=True)

def _legacy_collection_title(timestamp: datetime | None) -> str:
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat(timespec="seconds")

def _upgrade_sqlite_schema(sqlite_path: Path) -> None:
    engine = make_engine(sqlite_path)
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        inspector = inspect(conn)
        if inspector.has_table("slides"):
            slide_columns = {column["name"] for column in inspector.get_columns("slides")}
            if "import_collection_id" not in slide_columns:
                conn.exec_driver_sql(
                    "ALTER TABLE slides ADD COLUMN import_collection_id INTEGER "
                    "REFERENCES import_collections(id) ON DELETE SET NULL"
                )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_slides_import_collection_id "
                "ON slides (import_collection_id)"
            )

    session_factory = make_session_factory(engine)
    with session_factory() as db:
        legacy_slides = (
            db.query(Slide)
            .filter(Slide.import_collection_id.is_(None))
            .order_by(Slide.id.asc())
            .all()
        )
        if not legacy_slides:
            return

        for slide in legacy_slides:
            collection = ImportCollection(
                title=_legacy_collection_title(slide.created_at),
                source_type="legacy_import",
            )
            if slide.created_at is not None:
                collection.created_at = slide.created_at
            db.add(collection)
            db.flush()
            slide.import_collection_id = collection.id

        db.commit()

def init_database(settings: Settings) -> None:
    ensure_dirs(settings)
    if migrations_available():
        run_migrations(settings.sqlite_path)
    else:
        logger.warning("Alembic assets not found; falling back to metadata.create_all().")
    _upgrade_sqlite_schema(settings.sqlite_path)
