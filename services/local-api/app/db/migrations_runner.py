from __future__ import annotations
from pathlib import Path
import sys

from sqlalchemy import create_engine, inspect, text


REQUIRED_SCHEMA: dict[str, set[str]] = {
    "audit_logs": {
        "id",
        "actor_user_id",
        "action",
        "entity_type",
        "entity_id",
        "details_json",
        "ip",
        "created_at",
    },
    "import_collections": {"id", "title", "source_type", "created_at"},
    "inference_run_events": {
        "id",
        "run_id",
        "from_status",
        "to_status",
        "changed_at",
        "detail",
        "error",
    },
    "inference_runs": {
        "id",
        "slide_id",
        "requested_by_user_id",
        "model_name",
        "model_version",
        "status",
        "started_at",
        "finished_at",
        "output_json_path",
        "error_code",
        "error_message",
        "created_at",
    },
    "regions": {"id", "inference_run_id", "x", "y", "w", "h", "score", "label", "payload_json"},
    "slides": {
        "id",
        "original_path",
        "stored_filename",
        "stored_path",
        "file_size_bytes",
        "sha256",
        "import_collection_id",
        "review_status",
        "created_at",
    },
    "users": {"id", "username", "password_hash", "role", "is_active", "created_at"},
}

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


def _database_revision(sqlite_path: Path) -> str | None:
    if not sqlite_path.exists():
        return None

    engine = create_engine(f"sqlite+pysqlite:///{sqlite_path.as_posix()}", future=True)
    with engine.connect() as connection:
        inspector = inspect(connection)
        if not inspector.has_table("alembic_version"):
            return None
        return connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).scalar_one_or_none()


def _schema_satisfies_current_app(sqlite_path: Path) -> bool:
    if not sqlite_path.exists():
        return False

    engine = create_engine(f"sqlite+pysqlite:///{sqlite_path.as_posix()}", future=True)
    with engine.connect() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        if not set(REQUIRED_SCHEMA).issubset(tables):
            return False

        for table_name, required_columns in REQUIRED_SCHEMA.items():
            columns = {column["name"] for column in inspector.get_columns(table_name)}
            if not required_columns.issubset(columns):
                return False

    return True


def _stamp_database_revision(sqlite_path: Path, revision: str) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{sqlite_path.as_posix()}", future=True)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM alembic_version"))
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": revision},
        )


def run_migrations(sqlite_path: Path) -> None:
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    alembic_ini = _resource_path("alembic.ini")
    script_location = _resource_path("alembic")
    if not alembic_ini.is_file():
        raise FileNotFoundError(f"Alembic config not found: {alembic_ini}")
    if not script_location.is_dir():
        raise FileNotFoundError(f"Alembic script directory not found: {script_location}")

    cfg = Config(str(alembic_ini))
    cfg.set_main_option("script_location", str(script_location))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{sqlite_path.as_posix()}")

    script = ScriptDirectory.from_config(cfg)
    known_revisions = {revision.revision for revision in script.walk_revisions()}
    current_revision = _database_revision(sqlite_path)
    if current_revision and current_revision not in known_revisions and _schema_satisfies_current_app(sqlite_path):
        _stamp_database_revision(sqlite_path, script.get_current_head())
        return

    # Upgrade to head at startup
    command.upgrade(cfg, "head")
