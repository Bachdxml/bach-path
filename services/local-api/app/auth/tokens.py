from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from app.settings import Settings
from app.util.exceptions import AppError, ErrorCode

TOKEN_ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 12


def _token_secret(settings: Settings) -> str:
    if settings.api_key:
        return settings.api_key
    raise AppError(ErrorCode.INTERNAL, "Token signing is not configured", http_status=500)


def create_access_token(*, user_id: int, username: str, role: str, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=TOKEN_TTL_HOURS)).timestamp()),
    }
    return jwt.encode(payload, _token_secret(settings), algorithm=TOKEN_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    try:
        return jwt.decode(token, _token_secret(settings), algorithms=[TOKEN_ALGORITHM])
    except JWTError as exc:
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid session", http_status=401) from exc
