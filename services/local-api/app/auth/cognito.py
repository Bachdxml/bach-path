from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.request import urlopen
import json

from jose import jwt

from app.settings import Settings
from app.util.exceptions import AppError, ErrorCode


@dataclass(frozen=True)
class CognitoClaims:
    subject: str
    email: str | None
    raw: dict[str, Any]


@lru_cache(maxsize=8)
def _jwks(issuer: str) -> dict[str, Any]:
    with urlopen(f"{issuer.rstrip('/')}/.well-known/jwks.json", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_cognito_token(token: str, settings: Settings) -> CognitoClaims:
    if not settings.cognito_issuer or not settings.cognito_audience:
        raise AppError(ErrorCode.INTERNAL, "Cognito auth is not configured", http_status=500)
    try:
        claims = jwt.decode(
            token,
            _jwks(settings.cognito_issuer),
            algorithms=["RS256"],
            audience=settings.cognito_audience,
            issuer=settings.cognito_issuer.rstrip("/"),
        )
    except Exception as exc:
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid Cognito token", http_status=401) from exc

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid Cognito token subject", http_status=401)
    email = claims.get("email")
    return CognitoClaims(subject=subject, email=email if isinstance(email, str) else None, raw=claims)
