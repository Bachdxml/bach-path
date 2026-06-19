from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def upload_slide(
    client: TestClient,
    file_path: Path | None = None,
    *,
    filename: str | None = None,
    collection_id: int | None = None,
    allow_tile_like_import: bool = False,
    content: bytes | None = None,
):
    """POST a slide's bytes to the multipart upload endpoint.

    Mirrors how the desktop app imports: the raw file bytes plus the original
    filename are sent as multipart form data; no client filesystem path is used.
    """
    if content is None:
        if file_path is None:
            raise ValueError("Provide either file_path or content")
        content = Path(file_path).read_bytes()
    name = filename or (Path(file_path).name if file_path is not None else "slide.png")
    form: dict[str, str] = {}
    if collection_id is not None:
        form["collection_id"] = str(collection_id)
    if allow_tile_like_import:
        form["allow_tile_like_import"] = "true"
    return client.post(
        "/slides/upload",
        files={"file": (name, content, "application/octet-stream")},
        data=form,
    )


def create_collection(client: TestClient, *, title: str | None = None, source_type: str | None = None):
    """Create an empty import collection that uploads are grouped into."""
    payload: dict[str, str] = {}
    if title is not None:
        payload["title"] = title
    if source_type is not None:
        payload["source_type"] = source_type
    return client.post("/import-collections", json=payload)
