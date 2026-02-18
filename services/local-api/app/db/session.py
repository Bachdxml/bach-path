from __future__ import annotations
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
        cursor.execute("PRAGMA busy_timeout=5000;")     # avoid “database is locked” spikes
        cursor.close()

    return engine

def make_session_factory(engine: Engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True, class_=Session)
