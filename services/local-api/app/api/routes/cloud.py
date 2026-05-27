from __future__ import annotations

from datetime import datetime, timezone
import json

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.cloud_deps import CloudPrincipal, get_cloud_principal, require_approved_org, require_role
from app.api.deps import get_db
from app.models.audit_log import AuditLog
from app.models.marketplace import (
    Dataset,
    DatasetSlide,
    License,
    Order,
    Organization,
    SlideAsset,
    Submission,
)
from app.payments.stripe import create_checkout_session, verify_webhook
from app.schemas.marketplace import (
    CheckoutResponse,
    DatasetCreateRequest,
    DatasetResponse,
    DatasetSlideAddRequest,
    DatasetUpdateRequest,
    DownloadUrlResponse,
    LicenseResponse,
    OrganizationApprovalRequest,
    OrganizationCreateRequest,
    OrganizationResponse,
    SlideAssetResponse,
    SlideFinalizeRequest,
    SlideReviewRequest,
    SubmissionCreateRequest,
    SubmissionResponse,
    UploadUrlRequest,
    UploadUrlResponse,
)
from app.storage.cloud import (
    DOWNLOAD_EXPIRES_SECONDS,
    create_download_url,
    create_upload_url,
    slide_object_key,
)
from app.util.exceptions import AppError, ErrorCode


router = APIRouter(prefix="/cloud", tags=["cloud-marketplace"])
webhooks_router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _audit(
    db: Session,
    *,
    principal: CloudPrincipal | None,
    action: str,
    entity_type: str,
    entity_id: int | str | None,
    details: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            actor_user_id=principal.user.id if principal else None,
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id) if entity_id is not None else None,
            details_json=json.dumps(details or {}, sort_keys=True),
        )
    )


def _submission_response(submission: Submission) -> SubmissionResponse:
    return SubmissionResponse(
        id=submission.id,
        title=submission.title,
        status=submission.status,
        organization_id=submission.organization_id,
        created_at=submission.created_at,
    )


def _slide_response(slide: SlideAsset) -> SlideAssetResponse:
    return SlideAssetResponse(
        id=slide.id,
        submission_id=slide.submission_id,
        filename=slide.filename,
        s3_key=slide.s3_key,
        checksum_sha256=slide.checksum_sha256,
        file_size_bytes=slide.file_size_bytes,
        file_type=slide.file_type,
        deidentified=slide.deidentified,
        review_status=slide.review_status,
    )


def _dataset_response(dataset: Dataset) -> DatasetResponse:
    return DatasetResponse(
        id=dataset.id,
        title=dataset.title,
        description=dataset.description,
        price_cents=dataset.price_cents,
        currency=dataset.currency,
        license_terms=dataset.license_terms,
        allowed_use=dataset.allowed_use,
        status=dataset.status,
        created_at=dataset.created_at,
        published_at=dataset.published_at,
    )


def _license_response(license_: License) -> LicenseResponse:
    return LicenseResponse(
        id=license_.id,
        dataset_id=license_.dataset_id,
        buyer_organization_id=license_.buyer_organization_id,
        status=license_.status,
        expires_at=license_.expires_at,
    )


def _is_expired(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at < datetime.now(timezone.utc)


def _organization_response(organization: Organization) -> OrganizationResponse:
    return OrganizationResponse(
        id=organization.id,
        name=organization.name,
        org_type=organization.org_type,
        is_approved=organization.is_approved,
    )


@router.post("/admin/organizations", response_model=OrganizationResponse)
def admin_create_organization(
    payload: OrganizationCreateRequest,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "admin")
    organization = Organization(
        name=payload.name.strip(),
        org_type=payload.org_type,
        is_approved=payload.is_approved,
    )
    db.add(organization)
    db.flush()
    _audit(db, principal=principal, action="organization.create", entity_type="organization", entity_id=organization.id)
    db.commit()
    db.refresh(organization)
    return _organization_response(organization)


@router.patch("/admin/organizations/{organization_id}/approval", response_model=OrganizationResponse)
def admin_update_organization_approval(
    organization_id: int,
    payload: OrganizationApprovalRequest,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "admin")
    organization = db.get(Organization, organization_id)
    if not organization:
        raise AppError(ErrorCode.NOT_FOUND, "Organization not found", http_status=404)
    organization.is_approved = payload.is_approved
    db.add(organization)
    _audit(
        db,
        principal=principal,
        action="organization.approval",
        entity_type="organization",
        entity_id=organization.id,
        details={"is_approved": payload.is_approved},
    )
    db.commit()
    db.refresh(organization)
    return _organization_response(organization)


@router.post("/submissions", response_model=SubmissionResponse)
def create_submission(
    payload: SubmissionCreateRequest,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "submitter", "admin")
    require_approved_org(principal)
    submission = Submission(
        organization_id=principal.organization.id,
        created_by_user_id=principal.user.id,
        title=payload.title,
        status="draft",
    )
    db.add(submission)
    db.flush()
    _audit(db, principal=principal, action="submission.create", entity_type="submission", entity_id=submission.id)
    db.commit()
    db.refresh(submission)
    return _submission_response(submission)


@router.get("/submissions", response_model=list[SubmissionResponse])
def list_submissions(
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "submitter", "admin")
    submissions = (
        db.query(Submission)
        .filter(Submission.organization_id == principal.organization.id)
        .order_by(Submission.created_at.desc())
        .all()
    )
    return [_submission_response(submission) for submission in submissions]


@router.get("/submissions/{submission_id}", response_model=SubmissionResponse)
def get_submission(
    submission_id: int,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    submission = db.get(Submission, submission_id)
    if not submission or submission.organization_id != principal.organization.id:
        raise AppError(ErrorCode.NOT_FOUND, "Submission not found", http_status=404)
    return _submission_response(submission)


@router.post("/submissions/{submission_id}/upload-url", response_model=UploadUrlResponse)
def create_submission_upload_url(
    submission_id: int,
    payload: UploadUrlRequest,
    request: Request,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "submitter", "admin")
    require_approved_org(principal)
    submission = db.get(Submission, submission_id)
    if not submission or submission.organization_id != principal.organization.id:
        raise AppError(ErrorCode.NOT_FOUND, "Submission not found", http_status=404)
    key = slide_object_key(
        organization_id=principal.organization.id,
        submission_id=submission.id,
        filename=payload.filename,
    )
    signed = create_upload_url(request.app.state.settings, key=key, content_type=payload.file_type)
    _audit(
        db,
        principal=principal,
        action="submission.upload_url",
        entity_type="submission",
        entity_id=submission.id,
        details={"s3_key": key},
    )
    db.commit()
    return UploadUrlResponse(upload_url=signed.url, s3_key=signed.key, expires_in_seconds=signed.expires_in_seconds)


@router.post("/submissions/{submission_id}/slides/finalize", response_model=SlideAssetResponse)
def finalize_submission_slide(
    submission_id: int,
    payload: SlideFinalizeRequest,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "submitter", "admin")
    require_approved_org(principal)
    submission = db.get(Submission, submission_id)
    if not submission or submission.organization_id != principal.organization.id:
        raise AppError(ErrorCode.NOT_FOUND, "Submission not found", http_status=404)
    prefix = f"slides/org-{principal.organization.id}/submission-{submission.id}/"
    if not payload.s3_key.startswith(prefix):
        raise AppError(ErrorCode.FORBIDDEN, "Upload key is not authorized for this submission", http_status=403)
    slide = SlideAsset(
        submission_id=submission.id,
        organization_id=principal.organization.id,
        created_by_user_id=principal.user.id,
        filename=payload.filename,
        s3_key=payload.s3_key,
        checksum_sha256=payload.checksum_sha256,
        file_size_bytes=payload.file_size_bytes,
        file_type=payload.file_type,
        deidentified=False,
        review_status="submitted",
        metadata_json=json.dumps(payload.metadata, sort_keys=True),
    )
    submission.status = "submitted"
    db.add(slide)
    db.add(submission)
    db.flush()
    _audit(db, principal=principal, action="slide.finalize", entity_type="slide_asset", entity_id=slide.id)
    db.commit()
    db.refresh(slide)
    return _slide_response(slide)


@router.get("/admin/submissions", response_model=list[SubmissionResponse])
def admin_list_submissions(
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "curator", "admin")
    return [_submission_response(s) for s in db.query(Submission).order_by(Submission.created_at.desc()).all()]


@router.patch("/admin/slides/{slide_asset_id}/review", response_model=SlideAssetResponse)
def review_slide_asset(
    slide_asset_id: int,
    payload: SlideReviewRequest,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "curator", "admin")
    slide = db.get(SlideAsset, slide_asset_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, "Slide asset not found", http_status=404)
    slide.review_status = payload.review_status
    slide.deidentified = payload.review_status == "deidentified_approved"
    db.add(slide)
    _audit(
        db,
        principal=principal,
        action="slide.review",
        entity_type="slide_asset",
        entity_id=slide.id,
        details={"review_status": payload.review_status},
    )
    db.commit()
    db.refresh(slide)
    return _slide_response(slide)


@router.post("/datasets", response_model=DatasetResponse)
def create_dataset(
    payload: DatasetCreateRequest,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "curator", "admin")
    dataset = Dataset(
        title=payload.title.strip(),
        description=payload.description,
        price_cents=payload.price_cents,
        currency=payload.currency.lower(),
        license_terms=payload.license_terms,
        allowed_use=payload.allowed_use,
        status="draft",
        created_by_user_id=principal.user.id,
    )
    db.add(dataset)
    db.flush()
    _audit(db, principal=principal, action="dataset.create", entity_type="dataset", entity_id=dataset.id)
    db.commit()
    db.refresh(dataset)
    return _dataset_response(dataset)


@router.patch("/datasets/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    dataset_id: int,
    payload: DatasetUpdateRequest,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "curator", "admin")
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise AppError(ErrorCode.NOT_FOUND, "Dataset not found", http_status=404)
    if dataset.status != "draft":
        raise AppError(ErrorCode.CONFLICT, "Only draft datasets can be edited", http_status=409)
    for field in ("title", "description", "price_cents", "currency", "license_terms", "allowed_use"):
        value = getattr(payload, field)
        if value is not None:
            setattr(dataset, field, value.lower() if field == "currency" else value)
    db.add(dataset)
    _audit(db, principal=principal, action="dataset.update", entity_type="dataset", entity_id=dataset.id)
    db.commit()
    db.refresh(dataset)
    return _dataset_response(dataset)


@router.post("/datasets/{dataset_id}/slides")
def add_slide_to_dataset(
    dataset_id: int,
    payload: DatasetSlideAddRequest,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "curator", "admin")
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise AppError(ErrorCode.NOT_FOUND, "Dataset not found", http_status=404)
    if dataset.status != "draft":
        raise AppError(ErrorCode.CONFLICT, "Only draft datasets can be modified", http_status=409)
    slide = db.get(SlideAsset, payload.slide_asset_id)
    if not slide or slide.review_status != "deidentified_approved":
        raise AppError(ErrorCode.FORBIDDEN, "Only de-identified approved slides can be added", http_status=403)
    row = DatasetSlide(dataset_id=dataset.id, slide_asset_id=slide.id)
    db.add(row)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise AppError(ErrorCode.CONFLICT, "Slide is already in dataset", http_status=409)
    _audit(db, principal=principal, action="dataset.slide_add", entity_type="dataset", entity_id=dataset.id)
    db.commit()
    return {"ok": True, "dataset_id": dataset.id, "slide_asset_id": slide.id}


@router.post("/datasets/{dataset_id}/publish", response_model=DatasetResponse)
def publish_dataset(
    dataset_id: int,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "curator", "admin")
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise AppError(ErrorCode.NOT_FOUND, "Dataset not found", http_status=404)
    slide_count = db.query(DatasetSlide).filter(DatasetSlide.dataset_id == dataset.id).count()
    if slide_count < 1:
        raise AppError(ErrorCode.CONFLICT, "Dataset must contain at least one approved slide", http_status=409)
    dataset.status = "published"
    dataset.published_at = datetime.now(timezone.utc)
    db.add(dataset)
    _audit(db, principal=principal, action="dataset.publish", entity_type="dataset", entity_id=dataset.id)
    db.commit()
    db.refresh(dataset)
    return _dataset_response(dataset)


@router.get("/datasets", response_model=list[DatasetResponse])
def list_published_datasets(
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "buyer", "admin", "curator")
    return [
        _dataset_response(dataset)
        for dataset in db.query(Dataset).filter(Dataset.status == "published").order_by(Dataset.created_at.desc()).all()
    ]


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_published_dataset(
    dataset_id: int,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "buyer", "admin", "curator")
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.status != "published":
        raise AppError(ErrorCode.NOT_FOUND, "Dataset not found", http_status=404)
    return _dataset_response(dataset)


@router.post("/datasets/{dataset_id}/checkout", response_model=CheckoutResponse)
def create_dataset_checkout(
    dataset_id: int,
    request: Request,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "buyer", "admin")
    require_approved_org(principal)
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.status != "published":
        raise AppError(ErrorCode.NOT_FOUND, "Dataset not found", http_status=404)
    order = Order(
        dataset_id=dataset.id,
        buyer_organization_id=principal.organization.id,
        requested_by_user_id=principal.user.id,
        amount_cents=dataset.price_cents,
        currency=dataset.currency,
        status="pending",
    )
    db.add(order)
    db.flush()
    session = create_checkout_session(
        request.app.state.settings,
        dataset_id=dataset.id,
        title=dataset.title,
        amount_cents=dataset.price_cents,
        currency=dataset.currency,
        success_url=f"{request.app.state.settings.remote_api_base_url}/cloud/checkout/success",
        cancel_url=f"{request.app.state.settings.remote_api_base_url}/cloud/checkout/cancel",
    )
    order.stripe_checkout_session_id = session.id
    order.checkout_url = session.url
    db.add(order)
    _audit(db, principal=principal, action="order.checkout", entity_type="order", entity_id=order.id)
    db.commit()
    return CheckoutResponse(order_id=order.id, checkout_url=session.url)


@router.get("/licenses", response_model=list[LicenseResponse])
def list_licenses(
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "buyer", "admin")
    return [
        _license_response(license_)
        for license_ in db.query(License)
        .filter(License.buyer_organization_id == principal.organization.id)
        .order_by(License.created_at.desc())
        .all()
    ]


@router.post("/licenses/{license_id}/download-url", response_model=DownloadUrlResponse)
def create_license_download_urls(
    license_id: int,
    request: Request,
    principal: CloudPrincipal = Depends(get_cloud_principal),
    db: Session = Depends(get_db),
):
    require_role(principal, "buyer", "admin")
    license_ = db.get(License, license_id)
    if not license_ or license_.buyer_organization_id != principal.organization.id:
        raise AppError(ErrorCode.NOT_FOUND, "License not found", http_status=404)
    if license_.status != "active":
        raise AppError(ErrorCode.FORBIDDEN, "License is not active", http_status=403)
    if _is_expired(license_.expires_at):
        raise AppError(ErrorCode.FORBIDDEN, "License has expired", http_status=403)
    rows = (
        db.query(SlideAsset)
        .join(DatasetSlide, DatasetSlide.slide_asset_id == SlideAsset.id)
        .filter(DatasetSlide.dataset_id == license_.dataset_id)
        .all()
    )
    urls = [create_download_url(request.app.state.settings, key=row.s3_key) for row in rows]
    _audit(db, principal=principal, action="license.download_url", entity_type="license", entity_id=license_.id)
    db.commit()
    return DownloadUrlResponse(urls=urls, expires_in_seconds=DOWNLOAD_EXPIRES_SECONDS)


@webhooks_router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
    db: Session = Depends(get_db),
):
    event = verify_webhook(await request.body(), stripe_signature, request.app.state.settings.stripe_webhook_secret)
    if event.get("type") != "checkout.session.completed":
        return {"ok": True, "ignored": True}
    session = event.get("data", {}).get("object", {})
    session_id = session.get("id")
    if not session_id:
        raise AppError(ErrorCode.SLIDE_INVALID, "Stripe event missing session id", http_status=400)
    order = db.query(Order).filter(Order.stripe_checkout_session_id == session_id).one_or_none()
    if not order:
        raise AppError(ErrorCode.NOT_FOUND, "Order not found", http_status=404)
    if order.status == "paid":
        return {"ok": True, "order_id": order.id, "idempotent": True}
    order.status = "paid"
    payment_intent = session.get("payment_intent")
    if isinstance(payment_intent, str):
        order.stripe_payment_intent_id = payment_intent
    dataset = db.get(Dataset, order.dataset_id)
    if not dataset:
        raise AppError(ErrorCode.NOT_FOUND, "Dataset not found", http_status=404)
    license_ = (
        db.query(License)
        .filter(
            License.dataset_id == order.dataset_id,
            License.buyer_organization_id == order.buyer_organization_id,
        )
        .one_or_none()
    )
    if license_ is None:
        license_ = License(
            dataset_id=order.dataset_id,
            buyer_organization_id=order.buyer_organization_id,
            order_id=order.id,
            terms=dataset.license_terms,
            status="active",
        )
        db.add(license_)
    else:
        license_.status = "active"
        license_.order_id = order.id
        db.add(license_)
    db.add(order)
    _audit(db, principal=None, action="stripe.checkout_completed", entity_type="order", entity_id=order.id)
    db.commit()
    return {"ok": True, "order_id": order.id}
