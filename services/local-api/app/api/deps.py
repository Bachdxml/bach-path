from __future__ import annotations
from fastapi import Request
from sqlalchemy.orm import Session
from typing import Generator
import threading

from app.db.session import make_engine, make_session_factory

_db_init_lock = threading.Lock()

def get_db(request: Request) -> Generator[Session, None, None]:
    settings = request.app.state.settings
    # Cache engine/session factory on app.state for performance
    if not hasattr(request.app.state, "engine"):
        with _db_init_lock:
            if not hasattr(request.app.state, "engine"):
                request.app.state.engine = make_engine(settings.sqlite_path)
                request.app.state.SessionLocal = make_session_factory(request.app.state.engine)

    db: Session = request.app.state.SessionLocal()
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()
