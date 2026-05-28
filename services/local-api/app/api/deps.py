from __future__ import annotations
import secrets
import threading
from typing import Generator

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth.tokens import decode_access_token
from app.db.session import make_engine, make_session_factory
from app.models.user import User
from app.util.exceptions import AppError, ErrorCode

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


def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AppError(ErrorCode.UNAUTHORIZED, "Sign in required", http_status=401)
    token = authorization.split(" ", 1)[1].strip()
    claims = decode_access_token(token, request.app.state.settings)
    try:
        user_id = int(claims.get("sub"))
    except (TypeError, ValueError) as exc:
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid session", http_status=401) from exc
    user = db.get(User, user_id)
    if not user or not user.is_active:
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid session", http_status=401)
    return user
