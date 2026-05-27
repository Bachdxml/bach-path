from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth.cognito import CognitoClaims
from app.main import create_app
from app.models.marketplace import Membership, Organization
from app.models.user import User
from app.db.session import make_engine_from_url, make_session_factory
from app.storage.cloud import PresignedUrl


def _cloud_env(monkeypatch, tmp_path: Path) -> None:
    app_data_dir = tmp_path / "app-data"
    app_data_dir.mkdir()
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "cloud")
    monkeypatch.setenv("APP_DATA_DIR", str(app_data_dir))
    monkeypatch.setenv("APP_DATABASE_URL", f"sqlite+pysqlite:///{(tmp_path / 'cloud.db').as_posix()}")
    monkeypatch.setenv("APP_API_KEY", "service-key")
    monkeypatch.setenv("APP_REMOTE_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("APP_REMOTE_AUTH_PROVIDER_URL", "https://auth.example.test")
    monkeypatch.setenv("APP_REMOTE_STORAGE_URL", "s3://bach-path-test")
    monkeypatch.setenv("APP_COGNITO_ISSUER", "https://cognito-idp.us-east-1.amazonaws.com/pool")
    monkeypatch.setenv("APP_COGNITO_AUDIENCE", "client-id")
    monkeypatch.setenv("APP_S3_BUCKET", "bach-path-test")
    monkeypatch.setenv("APP_S3_REGION", "us-east-1")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_key")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")


def _seed_member(app, *, sub: str, email: str, org_name: str, org_type: str, role: str, approved: bool = True):
    session_factory = getattr(app.state, "SessionLocal", None)
    if session_factory is None:
        engine = make_engine_from_url(app.state.settings.database_url)
        session_factory = make_session_factory(engine)
    db = session_factory()
    try:
        user = User(username=sub[:64], password_hash="cognito", cognito_sub=sub, email=email, role="viewer")
        org = Organization(name=org_name, org_type=org_type, is_approved=approved)
        db.add_all([user, org])
        db.flush()
        db.add(Membership(user_id=user.id, organization_id=org.id, role=role))
        db.commit()
        return user.id, org.id
    finally:
        db.close()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _patch_cognito(monkeypatch):
    def fake_verify(token, settings):
        return CognitoClaims(subject=token, email=f"{token}@example.test", raw={"sub": token})

    monkeypatch.setattr("app.api.cloud_deps.verify_cognito_token", fake_verify)


def test_approved_submitter_can_create_submission_and_finalize_slide(monkeypatch, tmp_path):
    _cloud_env(monkeypatch, tmp_path)
    _patch_cognito(monkeypatch)
    monkeypatch.setattr(
        "app.api.routes.cloud.create_upload_url",
        lambda settings, *, key, content_type: PresignedUrl(url=f"https://upload.test/{key}", key=key, expires_in_seconds=900),
    )

    app = create_app()
    with TestClient(app) as client:
        _seed_member(app, sub="submitter", email="submitter@example.test", org_name="Lab", org_type="submitter", role="submitter")

        create_response = client.post("/cloud/submissions", json={"title": "Batch A"}, headers=_auth("submitter"))
        assert create_response.status_code == 200, create_response.text
        submission_id = create_response.json()["id"]

        upload_response = client.post(
            f"/cloud/submissions/{submission_id}/upload-url",
            json={
                "filename": "case-001.svs",
                "file_type": "svs",
                "file_size_bytes": 123,
                "checksum_sha256": "a" * 64,
            },
            headers=_auth("submitter"),
        )
        assert upload_response.status_code == 200, upload_response.text
        s3_key = upload_response.json()["s3_key"]
        assert s3_key.startswith("slides/org-")

        finalize_response = client.post(
            f"/cloud/submissions/{submission_id}/slides/finalize",
            json={
                "filename": "case-001.svs",
                "s3_key": s3_key,
                "checksum_sha256": "a" * 64,
                "file_size_bytes": 123,
                "file_type": "svs",
                "metadata": {"organ": "lung"},
            },
            headers=_auth("submitter"),
        )
        assert finalize_response.status_code == 200, finalize_response.text
        assert finalize_response.json()["review_status"] == "submitted"


def test_unapproved_submitter_cannot_create_submission(monkeypatch, tmp_path):
    _cloud_env(monkeypatch, tmp_path)
    _patch_cognito(monkeypatch)

    app = create_app()
    with TestClient(app) as client:
        _seed_member(app, sub="blocked", email="blocked@example.test", org_name="Blocked", org_type="submitter", role="submitter", approved=False)
        response = client.post("/cloud/submissions", json={"title": "Batch"}, headers=_auth("blocked"))

    assert response.status_code == 403


def test_admin_can_create_and_approve_organization(monkeypatch, tmp_path):
    _cloud_env(monkeypatch, tmp_path)
    _patch_cognito(monkeypatch)

    app = create_app()
    with TestClient(app) as client:
        _seed_member(app, sub="admin", email="admin@example.test", org_name="Internal", org_type="internal", role="admin")

        create_response = client.post(
            "/cloud/admin/organizations",
            json={"name": "New Lab", "org_type": "submitter"},
            headers=_auth("admin"),
        )
        assert create_response.status_code == 200, create_response.text
        organization_id = create_response.json()["id"]
        assert create_response.json()["is_approved"] is False

        approval_response = client.patch(
            f"/cloud/admin/organizations/{organization_id}/approval",
            json={"is_approved": True},
            headers=_auth("admin"),
        )
        assert approval_response.status_code == 200, approval_response.text
        assert approval_response.json()["is_approved"] is True


def test_dataset_requires_deidentified_approved_slide(monkeypatch, tmp_path):
    _cloud_env(monkeypatch, tmp_path)
    _patch_cognito(monkeypatch)

    app = create_app()
    with TestClient(app) as client:
        _seed_member(app, sub="submitter", email="submitter@example.test", org_name="Lab", org_type="submitter", role="submitter")
        _seed_member(app, sub="curator", email="curator@example.test", org_name="Internal", org_type="internal", role="curator")

        submission_id = client.post("/cloud/submissions", json={"title": "Batch"}, headers=_auth("submitter")).json()["id"]
        slide_response = client.post(
            f"/cloud/submissions/{submission_id}/slides/finalize",
            json={
                "filename": "case.svs",
                "s3_key": f"slides/org-1/submission-{submission_id}/case.svs",
                "checksum_sha256": "b" * 64,
                "file_size_bytes": 100,
                "file_type": "svs",
            },
            headers=_auth("submitter"),
        )
        slide_id = slide_response.json()["id"]
        dataset_id = client.post(
            "/cloud/datasets",
            json={
                "title": "Fungal cohort",
                "price_cents": 1000,
                "license_terms": "research use",
                "allowed_use": "model training",
            },
            headers=_auth("curator"),
        ).json()["id"]

        denied = client.post(
            f"/cloud/datasets/{dataset_id}/slides",
            json={"slide_asset_id": slide_id},
            headers=_auth("curator"),
        )
        assert denied.status_code == 403

        review = client.patch(
            f"/cloud/admin/slides/{slide_id}/review",
            json={"review_status": "deidentified_approved"},
            headers=_auth("curator"),
        )
        assert review.status_code == 200, review.text

        added = client.post(
            f"/cloud/datasets/{dataset_id}/slides",
            json={"slide_asset_id": slide_id},
            headers=_auth("curator"),
        )
        assert added.status_code == 200, added.text


def _stripe_signature(payload: bytes, secret: str) -> str:
    timestamp = str(int(time.time()))
    digest = hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode("utf-8") + payload, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def test_stripe_checkout_activates_license_and_download_requires_license(monkeypatch, tmp_path):
    _cloud_env(monkeypatch, tmp_path)
    _patch_cognito(monkeypatch)
    monkeypatch.setattr(
        "app.api.routes.cloud.create_checkout_session",
        lambda settings, **kwargs: type("Session", (), {"id": "cs_test_123", "url": "https://checkout.test/session"})(),
    )
    monkeypatch.setattr("app.api.routes.cloud.create_download_url", lambda settings, *, key: f"https://download.test/{key}")

    app = create_app()
    with TestClient(app) as client:
        _seed_member(app, sub="submitter", email="submitter@example.test", org_name="Lab", org_type="submitter", role="submitter")
        _seed_member(app, sub="curator", email="curator@example.test", org_name="Internal", org_type="internal", role="curator")
        _seed_member(app, sub="buyer", email="buyer@example.test", org_name="Buyer", org_type="buyer", role="buyer")

        submission_id = client.post("/cloud/submissions", json={"title": "Batch"}, headers=_auth("submitter")).json()["id"]
        slide_id = client.post(
            f"/cloud/submissions/{submission_id}/slides/finalize",
            json={
                "filename": "case.svs",
                "s3_key": f"slides/org-1/submission-{submission_id}/case.svs",
                "checksum_sha256": "c" * 64,
                "file_size_bytes": 100,
                "file_type": "svs",
            },
            headers=_auth("submitter"),
        ).json()["id"]
        client.patch(f"/cloud/admin/slides/{slide_id}/review", json={"review_status": "deidentified_approved"}, headers=_auth("curator"))
        dataset_id = client.post(
            "/cloud/datasets",
            json={
                "title": "Dataset",
                "price_cents": 1000,
                "license_terms": "research use",
                "allowed_use": "model training",
            },
            headers=_auth("curator"),
        ).json()["id"]
        client.post(f"/cloud/datasets/{dataset_id}/slides", json={"slide_asset_id": slide_id}, headers=_auth("curator"))
        client.post(f"/cloud/datasets/{dataset_id}/publish", headers=_auth("curator"))

        checkout = client.post(f"/cloud/datasets/{dataset_id}/checkout", headers=_auth("buyer"))
        assert checkout.status_code == 200, checkout.text

        assert client.get("/cloud/licenses", headers=_auth("buyer")).json() == []

        event = {
            "type": "checkout.session.completed",
            "data": {"object": {"id": "cs_test_123", "payment_intent": "pi_test"}},
        }
        payload = json.dumps(event).encode("utf-8")
        webhook = client.post(
            "/webhooks/stripe",
            content=payload,
            headers={"Stripe-Signature": _stripe_signature(payload, "whsec_test")},
        )
        assert webhook.status_code == 200, webhook.text

        licenses = client.get("/cloud/licenses", headers=_auth("buyer")).json()
        assert len(licenses) == 1

        download = client.post(f"/cloud/licenses/{licenses[0]['id']}/download-url", headers=_auth("buyer"))
        assert download.status_code == 200, download.text
        assert download.json()["urls"] == ["https://download.test/slides/org-1/submission-1/case.svs"]

        replay = client.post(
            "/webhooks/stripe",
            content=payload,
            headers={"Stripe-Signature": _stripe_signature(payload, "whsec_test")},
        )
        assert replay.status_code == 200
        assert replay.json()["idempotent"] is True
