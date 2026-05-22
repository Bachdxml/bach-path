from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.models.inference_run import InferenceRun
from app.models.region import Region
from app.models.slide import Slide


def _create_sample_slide(path: Path, color: tuple[int, int, int] = (80, 140, 210)) -> None:
    Image.new("RGB", (48, 36), color=color).save(path)


def test_slides_endpoint_requires_api_key_when_configured(app_paths, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "test-key")

    app = create_app()
    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200

        missing_key_response = client.get("/slides")
        assert missing_key_response.status_code == 401
        assert missing_key_response.json()["error"]["code"] == "http_error"

        query_key_response = client.get("/slides", params={"api_key": "test-key"})
        assert query_key_response.status_code == 401

        valid_key_response = client.get("/slides", headers={"x-api-key": "test-key"})
        assert valid_key_response.status_code == 200


def test_slides_endpoint_allows_query_key_when_explicitly_enabled(app_paths, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "test-key")
    monkeypatch.setenv("APP_ALLOW_QUERY_API_KEY", "1")

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/slides", params={"api_key": "test-key"})
        assert response.status_code == 200


def test_import_rejects_disallowed_slide_path(app_paths, tmp_path):
    disallowed_dir = tmp_path / "outside-source"
    disallowed_dir.mkdir()
    slide_path = disallowed_dir / "outside.png"
    _create_sample_slide(slide_path)

    app = create_app()
    with TestClient(app) as client:
        response = client.post("/slides/import", json={"file_path": str(slide_path)})

    assert response.status_code == 403
    payload = response.json()
    assert payload["error"]["code"] == "slide_permission"


def test_import_collection_deduplicates_and_supports_rename(app_paths):
    first_slide = app_paths["source_dir"] / "dup-a.png"
    second_slide = app_paths["source_dir"] / "dup-b.png"
    _create_sample_slide(first_slide, color=(220, 80, 70))
    _create_sample_slide(second_slide, color=(70, 180, 90))

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post(
            "/slides/import-collection",
            json={
                "file_paths": [str(first_slide), str(first_slide), str(second_slide)],
                "title": "  Initial Batch  ",
                "source_type": "manual",
            },
        )
        assert import_response.status_code == 200, import_response.text

        import_payload = import_response.json()
        assert import_payload["imported_count"] == 2
        assert len(import_payload["imported_slide_ids"]) == 2

        collection_id = import_payload["collection"]["id"]

        rename_response = client.patch(
            f"/import-collections/{collection_id}",
            json={"title": "  Renamed Batch  "},
        )
        assert rename_response.status_code == 200, rename_response.text
        assert rename_response.json()["title"] == "Renamed Batch"

        gallery_response = client.get("/slides")
        assert gallery_response.status_code == 200, gallery_response.text
        gallery_by_id = {item["id"]: item for item in gallery_response.json()["slides"]}

    imported_ids = set(import_payload["imported_slide_ids"])
    assert imported_ids.issubset(gallery_by_id.keys())

    for slide_id in imported_ids:
        slide = gallery_by_id[slide_id]
        assert slide["collection_id"] == collection_id
        assert slide["collection_title"] == "Renamed Batch"


def test_delete_slide_removes_file_and_tile_cache(app_paths):
    slide_path = app_paths["source_dir"] / "deletable.png"
    _create_sample_slide(slide_path)

    app_data_dir = app_paths["app_data_dir"]

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post("/slides/import", json={"file_path": str(slide_path)})
        assert import_response.status_code == 200, import_response.text

        import_payload = import_response.json()
        slide_id = import_payload["slide_id"]
        stored_filename = import_payload["stored_path"].split("/", 1)[1]
        stored_slide_path = app_data_dir / "slides" / stored_filename

        assert stored_slide_path.exists()

        tile_response = client.get(f"/slides/{slide_id}/tiles/0/0/0.jpg")
        assert tile_response.status_code == 200, tile_response.text
        assert tile_response.headers["content-type"].startswith("image/jpeg")

        cached_tile_path = app_data_dir / "tiles_cache" / str(slide_id) / "256_0" / "0" / "0_0.jpg"
        assert cached_tile_path.exists()

        delete_response = client.delete(f"/slides/{slide_id}")
        assert delete_response.status_code == 200, delete_response.text
        assert delete_response.json()["ok"] is True

        gallery_response = client.get("/slides")
        assert gallery_response.status_code == 200
        gallery_ids = {item["id"] for item in gallery_response.json()["slides"]}

        metadata_response = client.get(f"/slides/{slide_id}/metadata")

    assert slide_id not in gallery_ids
    assert not stored_slide_path.exists()
    assert not cached_tile_path.exists()
    assert metadata_response.status_code == 400
    assert metadata_response.json()["error"]["code"] == "not_found"


def test_cached_raster_tile_still_rejects_invalid_coordinates(app_paths):
    slide_path = app_paths["source_dir"] / "cache-guard.png"
    _create_sample_slide(slide_path)

    app_data_dir = app_paths["app_data_dir"]

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post("/slides/import", json={"file_path": str(slide_path)})
        assert import_response.status_code == 200, import_response.text
        slide_id = import_response.json()["slide_id"]

        poisoned_cache = app_data_dir / "tiles_cache" / str(slide_id) / "256_0" / "0" / "-1_0.jpg"
        poisoned_cache.parent.mkdir(parents=True, exist_ok=True)
        poisoned_cache.write_bytes(b"not really a jpg")

        tile_response = client.get(f"/slides/{slide_id}/tiles/0/-1/0.jpg")

    assert tile_response.status_code == 400
    assert tile_response.json()["error"]["code"] == "not_found"


def test_metadata_rejects_stored_path_outside_managed_storage(app_paths):
    outside_file = app_paths["app_data_dir"].parent / "outside-managed.png"
    _create_sample_slide(outside_file)

    app = create_app()
    with TestClient(app) as client:
        # Ensure app state/session factory is initialized.
        response = client.get("/slides")
        assert response.status_code == 200

        db = app.state.SessionLocal()
        try:
            rogue = Slide(
                original_path=str(outside_file),
                stored_filename="rogue.png",
                stored_path=str(outside_file),
                file_size_bytes=outside_file.stat().st_size,
                sha256=None,
                import_collection_id=None,
            )
            db.add(rogue)
            db.commit()
            db.refresh(rogue)
            rogue_id = rogue.id
        finally:
            db.close()

        metadata_response = client.get(f"/slides/{rogue_id}/metadata")
        assert metadata_response.status_code == 400
        payload = metadata_response.json()
        assert payload["error"]["code"] == "storage_inconsistent"


def test_metadata_rejects_directory_stored_path(app_paths):
    managed_slides_dir = app_paths["app_data_dir"] / "slides"
    rogue_dir = managed_slides_dir / "not-a-file"
    rogue_dir.mkdir(parents=True, exist_ok=True)

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/slides")
        assert response.status_code == 200

        db = app.state.SessionLocal()
        try:
            rogue = Slide(
                original_path=str(rogue_dir),
                stored_filename="not-a-file",
                stored_path=str(rogue_dir),
                file_size_bytes=0,
                sha256=None,
                import_collection_id=None,
            )
            db.add(rogue)
            db.commit()
            db.refresh(rogue)
            rogue_id = rogue.id
        finally:
            db.close()

        metadata_response = client.get(f"/slides/{rogue_id}/metadata")
        assert metadata_response.status_code == 400
        payload = metadata_response.json()
        assert payload["error"]["code"] == "storage_inconsistent"


def test_batch_inference_rejects_oversized_payload(app_paths):
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/inference/slides/batch-run",
            json={"slide_ids": list(range(1, 66))},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "slide_invalid"


def test_inference_regions_include_hotspot_payload(app_paths):
    slide_path = app_paths["source_dir"] / "region-payload.png"
    _create_sample_slide(slide_path, color=(180, 90, 120))

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post("/slides/import", json={"file_path": str(slide_path)})
        assert import_response.status_code == 200, import_response.text
        slide_id = import_response.json()["slide_id"]

        db = app.state.SessionLocal()
        try:
            run = InferenceRun(
                slide_id=slide_id,
                model_name="ResidualAttentionUNet",
                model_version="test",
                status="succeeded",
            )
            db.add(run)
            db.flush()

            region = Region(
                inference_run_id=run.id,
                x=100,
                y=200,
                w=512,
                h=512,
                score=0.82,
                label="fungus_positive",
                payload_json=(
                    '{"hotspot":{"cx":164.5,"cy":248.25,"x":140,"y":226,"w":72,"h":60,'
                    '"coverage":0.118,"source":"segmentation_centroid"}}'
                ),
            )
            db.add(region)
            db.commit()
            run_id = run.id
        finally:
            db.close()

        regions_response = client.get(f"/inference/runs/{run_id}/regions")

    assert regions_response.status_code == 200, regions_response.text
    payload = regions_response.json()
    assert len(payload["regions"]) == 1
    hotspot = payload["regions"][0]["payload"]["hotspot"]
    assert hotspot["cx"] == 164.5
    assert hotspot["cy"] == 248.25
    assert hotspot["w"] == 72
    assert hotspot["coverage"] == 0.118
