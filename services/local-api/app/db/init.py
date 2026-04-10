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
    inference_run_event,
    import_collection,
    region,
    slide,
    user,
)
from app.models.enums import InferenceStatus
from app.models.inference_run import InferenceRun
from app.models.inference_run_event import InferenceRunEvent
from app.models.import_collection import ImportCollection
from app.models.slide import Slide


REQUIRED_DIRS = ("slides", "inference_runs", "training_runs", "tiles_cache", "logs")
logger = logging.getLogger(__name__)
_STARTUP_FAILURE_CODE = "startup_recovery"
_STARTUP_FAILURE_MESSAGE = (
    "Inference run was interrupted by an unclean API shutdown and was marked failed during startup reconciliation."
)

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

def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()

def _reconcile_orphaned_inference_runs(sqlite_path: Path) -> int:
    engine = make_engine(sqlite_path)
    session_factory = make_session_factory(engine)
    now = datetime.now(timezone.utc)
    repaired_count = 0

    with session_factory() as db:
        orphaned_runs = (
            db.query(InferenceRun)
            .filter(InferenceRun.status.in_([InferenceStatus.queued.value, InferenceStatus.running.value]))
            .order_by(InferenceRun.id.asc())
            .all()
        )
        if not orphaned_runs:
            return 0

        for run in orphaned_runs:
            previous_status = run.status
            run.status = InferenceStatus.failed.value
            run.finished_at = now
            if _is_blank(run.error_code):
                run.error_code = _STARTUP_FAILURE_CODE
            if _is_blank(run.error_message):
                run.error_message = _STARTUP_FAILURE_MESSAGE
            db.add(
                InferenceRunEvent(
                    run_id=run.id,
                    from_status=previous_status,
                    to_status=InferenceStatus.failed.value,
                    changed_at=now,
                    detail="Startup reconciliation recovered an interrupted inference run.",
                    error=run.error_code,
                )
            )
            repaired_count += 1

        db.commit()

    return repaired_count

def init_database(settings: Settings) -> int:
    ensure_dirs(settings)
    if migrations_available():
        run_migrations(settings.sqlite_path)
    else:
        logger.warning("Alembic assets not found; falling back to metadata.create_all().")
    _upgrade_sqlite_schema(settings.sqlite_path)
    repaired_count = _reconcile_orphaned_inference_runs(settings.sqlite_path)
    if repaired_count:
        logger.info(
            "Startup reconciliation marked %d orphaned inference run(s) as failed.",
            repaired_count,
        )
    else:
        logger.info("Startup reconciliation found no orphaned inference runs.")
    return repaired_count
