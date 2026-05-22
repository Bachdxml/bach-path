from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.models.inference_run import InferenceRun
from app.models.region import Region


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
        assert imported_slide["review_status"] == "unreviewed"
        assert imported_slide["collection_id"] is not None


def test_slide_review_status_can_be_updated(app_paths):
    slide_path = app_paths["source_dir"] / "reviewable.png"
    _create_sample_slide(slide_path)

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post("/slides/import", json={"file_path": str(slide_path)})
        assert import_response.status_code == 200, import_response.text
        slide_id = import_response.json()["slide_id"]

        update_response = client.patch(
            f"/slides/{slide_id}/review",
            json={"review_status": "positive"},
        )
        assert update_response.status_code == 200, update_response.text
        assert update_response.json() == {"id": slide_id, "review_status": "positive"}

        gallery_response = client.get("/slides")
        assert gallery_response.status_code == 200, gallery_response.text
        reviewed_slide = next(item for item in gallery_response.json()["slides"] if item["id"] == slide_id)

    assert reviewed_slide["review_status"] == "positive"


def test_slide_review_status_rejects_invalid_values(app_paths):
    slide_path = app_paths["source_dir"] / "review-invalid.png"
    _create_sample_slide(slide_path)

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post("/slides/import", json={"file_path": str(slide_path)})
        assert import_response.status_code == 200, import_response.text
        slide_id = import_response.json()["slide_id"]

        update_response = client.patch(
            f"/slides/{slide_id}/review",
            json={"review_status": "malignant-ish"},
        )

    assert update_response.status_code == 422


def test_import_rejects_generated_training_tile_without_creating_slide(app_paths):
    tile_dir = app_paths["source_dir"] / "MASTERTILE" / "example.svs[--series, 0]" / "negative" / "images"
    tile_dir.mkdir(parents=True)
    tile_path = tile_dir / "tile_x18432_y112128.png"
    _create_sample_slide(tile_path)

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post("/slides/import", json={"file_path": str(tile_path)})
        gallery_response = client.get("/slides")

    assert import_response.status_code == 400
    payload = import_response.json()
    assert payload["error"]["code"] == "slide_invalid"
    assert "generated training tile" in payload["error"]["message"]
    assert gallery_response.status_code == 200
    assert gallery_response.json()["slides"] == []


def test_import_allows_generated_training_tile_with_explicit_override(app_paths):
    tile_dir = app_paths["source_dir"] / "MASTERTILE" / "example.svs[--series, 0]" / "negative" / "images"
    tile_dir.mkdir(parents=True)
    tile_path = tile_dir / "tile_x1_y2.png"
    _create_sample_slide(tile_path)

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post(
            "/slides/import",
            json={"file_path": str(tile_path), "allow_tile_like_import": True},
        )

    assert import_response.status_code == 200, import_response.text


def test_collection_import_rejects_generated_tile_atomically(app_paths):
    valid_slide = app_paths["source_dir"] / "valid-slide.png"
    _create_sample_slide(valid_slide)
    tile_dir = app_paths["source_dir"] / "MASTERTILE" / "example.svs[--series, 0]" / "negative" / "images"
    tile_dir.mkdir(parents=True)
    tile_path = tile_dir / "tile_x3_y4.png"
    _create_sample_slide(tile_path)

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post(
            "/slides/import-collection",
            json={"file_paths": [str(valid_slide), str(tile_path)], "title": "Mixed batch"},
        )
        gallery_response = client.get("/slides")

    assert import_response.status_code == 400
    assert gallery_response.status_code == 200
    assert gallery_response.json()["slides"] == []


def test_gallery_distinguishes_negative_and_needs_review_slide_inference(app_paths):
    negative_slide_path = app_paths["source_dir"] / "ai-negative.png"
    needs_review_slide_path = app_paths["source_dir"] / "ai-needs-review.png"
    _create_sample_slide(negative_slide_path)
    _create_sample_slide(needs_review_slide_path)

    app = create_app()
    with TestClient(app) as client:
        negative_import = client.post("/slides/import", json={"file_path": str(negative_slide_path)})
        assert negative_import.status_code == 200, negative_import.text
        negative_slide_id = negative_import.json()["slide_id"]

        needs_review_import = client.post("/slides/import", json={"file_path": str(needs_review_slide_path)})
        assert needs_review_import.status_code == 200, needs_review_import.text
        needs_review_slide_id = needs_review_import.json()["slide_id"]

        db = app.state.SessionLocal()
        try:
            negative_run = InferenceRun(
                slide_id=negative_slide_id,
                model_name="ResidualAttentionUNet",
                model_version="test",
                status="succeeded",
            )
            db.add(negative_run)
            db.flush()
            db.add(
                Region(
                    inference_run_id=negative_run.id,
                    x=0,
                    y=0,
                    w=512,
                    h=512,
                    score=0.01,
                    label="fungus_negative",
                )
            )
            db.add(
                InferenceRun(
                    slide_id=needs_review_slide_id,
                    model_name="ResidualAttentionUNet",
                    model_version="test",
                    status="succeeded",
                )
            )
            db.commit()
        finally:
            db.close()

        gallery_response = client.get("/slides")
        assert gallery_response.status_code == 200, gallery_response.text
        gallery_by_id = {item["id"]: item for item in gallery_response.json()["slides"]}

    assert gallery_by_id[negative_slide_id]["inference_result"] == "negative"
    assert gallery_by_id[needs_review_slide_id]["inference_result"] == "needs_review"
