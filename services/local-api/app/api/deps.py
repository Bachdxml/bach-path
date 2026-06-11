from __future__ import annotations
import secrets
from typing import Generator

from fastapi import Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db.session import get_app_session_factory

def get_db(request: Request) -> Generator[Session, None, None]:
    # Engine/session factory is lazily created and cached on app.state by the
    # shared helper, so all code paths use the same lock and engine.
    db: Session = get_app_session_factory(request.app)()
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
