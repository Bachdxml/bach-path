from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient

from app.main import create_app


def _assert_validation_error(payload: dict, *, request_id: str | None = None) -> None:
    assert set(payload) == {"error"}
    error = payload["error"]
    assert error["code"] == "validation_error"
    assert error["message"] == "Request validation failed. Check the submitted values and try again."
    assert isinstance(error["details"], list)
    assert 1 <= len(error["details"]) <= 5
    for detail in error["details"]:
        assert set(detail) == {"loc", "msg", "type"}
        assert isinstance(detail["loc"], list)
        assert isinstance(detail["msg"], str)
        assert isinstance(detail["type"], str)

    if request_id is None:
        UUID(error["request_id"])
    else:
        assert error["request_id"] == request_id


def test_body_validation_error_uses_normalized_shape(app_paths):
    app = create_app()
    with TestClient(app) as client:
        # The upload endpoint requires a multipart `file` field; omitting it
        # exercises body validation on the current (path-free) import API.
        response = client.post(
            "/slides/upload",
            data={"allow_tile_like_import": "false"},
            headers={"x-request-id": "body-validation-rid"},
        )

    assert response.status_code == 422
    payload = response.json()
    _assert_validation_error(payload, request_id="body-validation-rid")
    assert payload["error"]["details"][0]["loc"] == ["body", "file"]


def test_query_validation_error_uses_normalized_shape(app_paths):
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/slides/1/thumbnail", params={"size": 1})

    assert response.status_code == 422
    payload = response.json()
    _assert_validation_error(payload)
    assert payload["error"]["details"][0]["loc"] == ["query", "size"]


def test_path_validation_error_uses_normalized_shape(app_paths):
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/slides/not-an-int/metadata")

    assert response.status_code == 422
    payload = response.json()
    _assert_validation_error(payload)
    assert payload["error"]["details"][0]["loc"] == ["path", "slide_id"]
