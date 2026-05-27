from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from app.settings import Settings
from app.util.exceptions import AppError, ErrorCode


@dataclass(frozen=True)
class CheckoutSession:
    id: str
    url: str


def create_checkout_session(
    settings: Settings,
    *,
    dataset_id: int,
    title: str,
    amount_cents: int,
    currency: str,
    success_url: str,
    cancel_url: str,
) -> CheckoutSession:
    try:
        import stripe
    except ImportError as exc:
        raise AppError(ErrorCode.INTERNAL, "stripe package is required for checkout", http_status=500) from exc
    stripe.api_key = settings.stripe_secret_key
    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[
            {
                "quantity": 1,
                "price_data": {
                    "currency": currency,
                    "unit_amount": amount_cents,
                    "product_data": {"name": title},
                },
            }
        ],
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"dataset_id": str(dataset_id)},
    )
    return CheckoutSession(id=session["id"], url=session["url"])


def verify_webhook(payload: bytes, signature_header: str | None, secret: str | None) -> dict[str, Any]:
    if not secret:
        raise AppError(ErrorCode.INTERNAL, "Stripe webhook secret is not configured", http_status=500)
    if not signature_header:
        raise AppError(ErrorCode.UNAUTHORIZED, "Missing Stripe signature", http_status=401)

    parts = dict(part.split("=", 1) for part in signature_header.split(",") if "=" in part)
    timestamp = parts.get("t")
    signature = parts.get("v1")
    if not timestamp or not signature:
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid Stripe signature", http_status=401)
    try:
        ts = int(timestamp)
    except ValueError as exc:
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid Stripe signature timestamp", http_status=401) from exc
    if abs(time.time() - ts) > 300:
        raise AppError(ErrorCode.UNAUTHORIZED, "Expired Stripe signature", http_status=401)

    signed_payload = f"{timestamp}.".encode("utf-8") + payload
    expected = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise AppError(ErrorCode.UNAUTHORIZED, "Invalid Stripe signature", http_status=401)
    event = json.loads(payload.decode("utf-8"))
    if not isinstance(event, dict):
        raise AppError(ErrorCode.SLIDE_INVALID, "Invalid Stripe event", http_status=400)
    return event
