from __future__ import annotations
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path

def _is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite:")


def make_engine_from_url(url: str) -> Engine:
    connect_args = {"check_same_thread": False} if _is_sqlite_url(url) else {}
    engine = create_engine(
        url,
        echo=False,
        future=True,
        connect_args=connect_args,
    )

    if _is_sqlite_url(url):
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


def make_engine(sqlite_path: Path) -> Engine:
    return make_engine_from_url(f"sqlite+pysqlite:///{sqlite_path.as_posix()}")


def make_session_factory(engine: Engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True, class_=Session)
