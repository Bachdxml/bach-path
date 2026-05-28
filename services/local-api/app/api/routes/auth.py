from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.auth.passwords import verify_password
from app.auth.tokens import create_access_token
from app.models.user import User
from app.schemas.auth import AuthUserResponse, LoginRequest, LoginResponse
from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_response(user: User) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        username=user.username,
        email=getattr(user, "email", None),
        role=user.role,
    )


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    username = payload.username.strip()
    user = db.query(User).filter(User.username == username).one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid username or password", http_status=401)
    if not user.is_active:
        raise AppError(ErrorCode.FORBIDDEN, "Account is inactive", http_status=403)

    token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
        settings=request.app.state.settings,
    )
    return LoginResponse(access_token=token, user=_user_response(user))
