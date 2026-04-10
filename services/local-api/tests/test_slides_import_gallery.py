from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


def _create_sample_slide(path: Path) -> None:
    Image.new("RGB", (32, 24), color=(200, 80, 40)).save(path)


def test_imported_slide_is_visible_in_gallery(app_paths):
    slide_path = app_paths["source_dir"] / "gallery-import.png"
    _create_sample_slide(slide_path)

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post(
            "/slides/import",
            json={"file_path": str(slide_path)},
        )
        assert import_response.status_code == 200, import_response.text

        import_payload = import_response.json()
        slide_id = import_payload["slide_id"]
        assert import_payload["stored_path"].startswith("slides/")

        gallery_response = client.get("/slides")
        assert gallery_response.status_code == 200, gallery_response.text

        slides = gallery_response.json()["slides"]
        imported_slide = next((item for item in slides if item["id"] == slide_id), None)

        assert imported_slide is not None
        assert imported_slide["original_path"] == "gallery-import.png"
        assert imported_slide["inference_result"] == "unchecked"
        assert imported_slide["collection_id"] is not None
