from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.settings import Settings
from app.util.exceptions import AppError, ErrorCode


UPLOAD_EXPIRES_SECONDS = 900
DOWNLOAD_EXPIRES_SECONDS = 900


@dataclass(frozen=True)
class PresignedUrl:
    url: str
    key: str
    expires_in_seconds: int


def _client(settings: Settings):
    try:
        import boto3
    except ImportError as exc:
        raise AppError(ErrorCode.INTERNAL, "boto3 is required for S3 cloud storage", http_status=500) from exc
    return boto3.client("s3", region_name=settings.s3_region)


def slide_object_key(*, organization_id: int, submission_id: int, filename: str) -> str:
    safe_name = filename.replace("/", "_").replace("\\", "_")
    return f"slides/org-{organization_id}/submission-{submission_id}/{uuid4().hex}-{safe_name}"


def create_upload_url(settings: Settings, *, key: str, content_type: str) -> PresignedUrl:
    if not settings.s3_bucket:
        raise AppError(ErrorCode.INTERNAL, "S3 bucket is not configured", http_status=500)
    url = _client(settings).generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.s3_bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=UPLOAD_EXPIRES_SECONDS,
    )
    return PresignedUrl(url=url, key=key, expires_in_seconds=UPLOAD_EXPIRES_SECONDS)


def create_download_url(settings: Settings, *, key: str) -> str:
    if not settings.s3_bucket:
        raise AppError(ErrorCode.INTERNAL, "S3 bucket is not configured", http_status=500)
    return _client(settings).generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=DOWNLOAD_EXPIRES_SECONDS,
    )
