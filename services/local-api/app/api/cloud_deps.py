from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.auth.cognito import verify_cognito_token
from app.models.marketplace import AccountMembership, Account
from app.models.user import User
from app.settings import DeploymentMode
from app.util.exceptions import AppError, ErrorCode


@dataclass(frozen=True)
class CloudPrincipal:
    user: User
    account: Account
    membership: AccountMembership

    @property
    def role(self) -> str:
        return self.membership.role


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AppError(ErrorCode.UNAUTHORIZED, "Missing bearer token", http_status=401)
    return authorization.split(" ", 1)[1].strip()


def get_cloud_principal(
    request: Request,
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> CloudPrincipal:
    settings = request.app.state.settings
    if settings.deployment_mode is not DeploymentMode.CLOUD:
        raise AppError(ErrorCode.NOT_FOUND, "Cloud marketplace routes are only available in cloud mode", http_status=404)

    claims = verify_cognito_token(_bearer_token(authorization), settings)
    user = db.query(User).filter(User.cognito_sub == claims.subject).one_or_none()
    if user is None:
        username = claims.subject[:64]
        user = User(
            username=username,
            password_hash="cognito",
            cognito_sub=claims.subject,
            email=claims.email,
            role="viewer",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    membership = (
        db.query(AccountMembership)
        .join(Account, Account.id == AccountMembership.account_id)
        .filter(AccountMembership.user_id == user.id)
        .order_by(AccountMembership.id.asc())
        .first()
    )
    if membership is None:
        raise AppError(ErrorCode.FORBIDDEN, "User is not a member of an account", http_status=403)
    account = db.get(Account, membership.account_id)
    if account is None:
        raise AppError(ErrorCode.FORBIDDEN, "Account membership is invalid", http_status=403)
    return CloudPrincipal(user=user, account=account, membership=membership)


def require_role(principal: CloudPrincipal, *roles: str) -> None:
    allowed = set(roles)
    if "submitter" in allowed or "buyer" in allowed:
        allowed.add("owner")
    if principal.role not in allowed:
        raise AppError(ErrorCode.FORBIDDEN, "Insufficient role for this action", http_status=403)


def require_approved_account(principal: CloudPrincipal) -> None:
    if not principal.account.is_approved:
        raise AppError(ErrorCode.FORBIDDEN, "Account is not approved", http_status=403)
