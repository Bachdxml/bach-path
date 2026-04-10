from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.api.routes import inference as inference_routes


def _create_sample_slide(path: Path, color: tuple[int, int, int] = (120, 80, 200)) -> None:
    Image.new("RGB", (48, 36), color=color).save(path)


def test_batch_inference_rejects_too_many_slide_ids(app_paths):
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/inference/slides/batch-run",
            json={"slide_ids": list(range(1, 258))},
        )

    assert response.status_code == 422


def test_folder_inference_rejects_oversized_folder(app_paths, monkeypatch):
    monkeypatch.setattr(inference_routes, "MAX_FOLDER_INFERENCE_SLIDES", 2)

    source_dir = app_paths["source_dir"]
    first_slide = source_dir / "folder-a.png"
    second_slide = source_dir / "folder-b.png"
    third_slide = source_dir / "folder-c.png"
    _create_sample_slide(first_slide, color=(200, 80, 80))
    _create_sample_slide(second_slide, color=(80, 200, 80))
    _create_sample_slide(third_slide, color=(80, 80, 200))

    app = create_app()
    with TestClient(app) as client:
        for slide_path in (first_slide, second_slide, third_slide):
            import_response = client.post("/slides/import", json={"file_path": str(slide_path)})
            assert import_response.status_code == 200, import_response.text

        list_response = client.get("/slides")
        assert list_response.status_code == 200, list_response.text
        first_item = list_response.json()["slides"][0]
        folder_key = first_item["folder_key"]

        response = client.post(
            "/inference/folders/run",
            json={"folder_key": folder_key},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "slide_invalid"

