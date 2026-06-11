from __future__ import annotations
import threading
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path

def make_engine(sqlite_path: Path) -> Engine:
    url = f"sqlite+pysqlite:///{sqlite_path.as_posix()}"
    engine = create_engine(
        url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},  # required for FastAPI concurrency
    )

    # SQLite pragmas that matter in production:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")      # better crash resilience + concurrency
        cursor.execute("PRAGMA synchronous=NORMAL;")    # balanced durability/perf
        cursor.execute("PRAGMA busy_timeout=60000;")    # wait through long WSI import/cache bursts
        cursor.close()

    return engine

def make_session_factory(engine: Engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True, class_=Session)

# Single lock guarding lazy init of the shared engine on app.state. All code
# paths (request deps, background inference worker) must go through
# get_app_session_factory so double-checked locking stays safe.
_app_engine_lock = threading.Lock()

def get_app_session_factory(app):
    """Lazily create and cache the shared engine/session factory on app.state."""
    state = app.state
    if not hasattr(state, "engine"):
        with _app_engine_lock:
            if not hasattr(state, "engine"):
                state.engine = make_engine(state.settings.sqlite_path)
                state.SessionLocal = make_session_factory(state.engine)
    return state.SessionLocal
