from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from app import models  # noqa: F401
from app.db.base import Base
from app.db.migrations_runner import migrations_available, run_migrations


def test_migrations_are_available() -> None:
    assert migrations_available() is True


def test_migrations_create_fresh_schema_and_version(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "fresh.db"

    run_migrations(sqlite_path)

    engine = create_engine(f"sqlite+pysqlite:///{sqlite_path.as_posix()}", future=True)
    with engine.connect() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        assert {
            "alembic_version",
            "audit_logs",
            "import_collections",
            "inference_run_events",
            "inference_runs",
            "regions",
            "slides",
            "users",
        }.issubset(tables)
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "20260526_0001"


def test_migrations_adopt_existing_metadata_schema(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "existing.db"
    engine = create_engine(f"sqlite+pysqlite:///{sqlite_path.as_posix()}", future=True)
    Base.metadata.create_all(bind=engine)

    run_migrations(sqlite_path)

    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "20260526_0001"
