from __future__ import annotations
import secrets
import threading
from typing import Generator

from fastapi import Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.settings import DeploymentMode
from app.db.session import make_engine, make_engine_from_url, make_session_factory

_db_init_lock = threading.Lock()

def get_db(request: Request) -> Generator[Session, None, None]:
    settings = request.app.state.settings
    # Cache engine/session factory on app.state for performance
    if not hasattr(request.app.state, "engine"):
        with _db_init_lock:
            if not hasattr(request.app.state, "engine"):
                if settings.deployment_mode is DeploymentMode.LOCAL or not settings.database_url:
                    request.app.state.engine = make_engine(settings.sqlite_path)
                else:
                    request.app.state.engine = make_engine_from_url(settings.database_url)
                request.app.state.SessionLocal = make_session_factory(request.app.state.engine)

    db: Session = request.app.state.SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> None:
    settings = getattr(request.app.state, "settings", None)
    expected = getattr(settings, "api_key", None)
    if not expected:
        return

    provided_api_key = x_api_key
    allow_query_api_key = bool(getattr(settings, "allow_query_api_key", False))
    if provided_api_key is None and allow_query_api_key:
        provided_api_key = request.query_params.get("api_key")

    if not provided_api_key or not secrets.compare_digest(provided_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
