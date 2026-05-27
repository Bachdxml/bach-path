from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.auth.cognito import verify_cognito_token
from app.models.marketplace import Membership, Organization
from app.models.user import User
from app.settings import DeploymentMode
from app.util.exceptions import AppError, ErrorCode


@dataclass(frozen=True)
class CloudPrincipal:
    user: User
    organization: Organization
    membership: Membership

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
        db.query(Membership)
        .join(Organization, Organization.id == Membership.organization_id)
        .filter(Membership.user_id == user.id)
        .order_by(Membership.id.asc())
        .first()
    )
    if membership is None:
        raise AppError(ErrorCode.FORBIDDEN, "User is not a member of an organization", http_status=403)
    organization = db.get(Organization, membership.organization_id)
    if organization is None:
        raise AppError(ErrorCode.FORBIDDEN, "Organization membership is invalid", http_status=403)
    return CloudPrincipal(user=user, organization=organization, membership=membership)


def require_role(principal: CloudPrincipal, *roles: str) -> None:
    if principal.role not in set(roles):
        raise AppError(ErrorCode.FORBIDDEN, "Insufficient role for this action", http_status=403)


def require_approved_org(principal: CloudPrincipal) -> None:
    if not principal.organization.is_approved:
        raise AppError(ErrorCode.FORBIDDEN, "Organization is not approved", http_status=403)
