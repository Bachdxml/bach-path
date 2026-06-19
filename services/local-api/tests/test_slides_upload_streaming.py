from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.slides.storage import (
    NotEnoughDiskSpaceError,
    UPLOAD_CHUNK_SIZE,
    stream_into_managed_storage,
)
from tests.upload_helpers import upload_slide


class _ChunkSpyFile:
    """File-like wrapper that records the largest single read it is asked for."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)
        self.max_read = 0
        self.read_calls = 0

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        if size is not None and size >= 0:
            self.max_read = max(self.max_read, size)
        return self._buf.read(size)

    def seek(self, *args, **kwargs):
        return self._buf.seek(*args, **kwargs)


def test_stream_into_managed_storage_reads_in_bounded_chunks(tmp_path):
    # ~5 chunks worth of data: peak memory is one chunk, not the whole payload.
    payload = b"x" * (UPLOAD_CHUNK_SIZE * 5 + 123)
    spy = _ChunkSpyFile(payload)
    dest_dir = tmp_path / "slides"

    dest_path, written = stream_into_managed_storage(spy, dest_dir, "42.svs")

    assert written == len(payload)
    assert dest_path.read_bytes() == payload
    # No single read ever asked for more than one chunk, so peak additional
    # memory does not scale with file size (Requirement 2 / DoD #2).
    assert spy.max_read <= UPLOAD_CHUNK_SIZE
    assert spy.read_calls >= 5


def test_stream_into_managed_storage_precheck_rejects_when_disk_too_small(tmp_path, monkeypatch):
    dest_dir = tmp_path / "slides"
    dest_dir.mkdir()

    class _Usage:
        free = 10

    monkeypatch.setattr("app.slides.storage.shutil.disk_usage", lambda _p: _Usage())

    with pytest.raises(NotEnoughDiskSpaceError):
        stream_into_managed_storage(io.BytesIO(b"a" * 1000), dest_dir, "1.svs", expected_size=1000)

    # The pre-check fires before any bytes are written, so nothing is left behind.
    assert list(dest_dir.glob("*")) == []


def test_stream_enospc_midwrite_cleans_partial_and_raises(tmp_path, monkeypatch):
    dest_dir = tmp_path / "slides"

    class _NoSpaceFile:
        def __init__(self) -> None:
            self._chunks = iter([b"a" * 1024, b"b" * 1024])

        def read(self, _size: int = -1) -> bytes:
            return next(self._chunks, b"")

        def seek(self, *args, **kwargs):
            return 0

    import errno

    real_open = open

    def fail_write_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if "w" in mode:
            original_write = handle.write

            def boom(_data):
                raise OSError(errno.ENOSPC, "No space left on device")

            handle.write = boom  # type: ignore[assignment]
        return handle

    monkeypatch.setattr("builtins.open", fail_write_open)

    with pytest.raises(NotEnoughDiskSpaceError):
        stream_into_managed_storage(_NoSpaceFile(), dest_dir, "9.svs")

    # The partial .part file is removed; nothing is left in managed storage.
    leftover = list(dest_dir.glob("*")) if dest_dir.exists() else []
    assert leftover == []


def test_large_upload_imports_and_is_viewable(app_paths):
    # A multi-MB payload exercises the streaming path end to end without OOM.
    big_image = Image.new("RGB", (1600, 1200), color=(50, 90, 160))
    buf = io.BytesIO()
    big_image.save(buf, format="PNG")
    big_bytes = buf.getvalue()

    app = create_app()
    with TestClient(app) as client:
        response = upload_slide(client, filename="large-slide.png", content=big_bytes)
        assert response.status_code == 200, response.text
        slide_id = response.json()["slide_id"]

        stored_filename = response.json()["stored_path"].split("/", 1)[1]
        stored_path = app_paths["app_data_dir"] / "slides" / stored_filename
        assert stored_path.exists()
        assert stored_path.stat().st_size == len(big_bytes)

        thumb = client.get(f"/slides/{slide_id}/thumbnail?size=128")
        assert thumb.status_code == 200, thumb.text


def test_zero_byte_upload_is_rejected_and_cleaned_up(app_paths):
    app = create_app()
    with TestClient(app) as client:
        response = upload_slide(client, filename="empty.png", content=b"")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "slide_unreadable"

        gallery = client.get("/slides")
        assert gallery.json()["slides"] == []

    slides_dir = app_paths["app_data_dir"] / "slides"
    leftover = list(slides_dir.glob("*")) if slides_dir.exists() else []
    assert leftover == []
