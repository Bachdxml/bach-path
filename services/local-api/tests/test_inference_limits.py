from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import create_app
from app.api.routes import inference as inference_routes
from app.util.exceptions import AppError


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


def test_env_inference_checkpoint_requires_existing_file(monkeypatch):
    monkeypatch.setenv("INFERENCE_CHECKPOINT", "/tmp/definitely-missing-checkpoint-file.pth.gz")

    with pytest.raises(AppError, match="INFERENCE_CHECKPOINT must point to an existing file"):
        inference_routes._resolve_model_checkpoint(None)


def test_inference_python_prefers_local_api_venv(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    local_python = project_root / "services" / "local-api" / ".venv" / "bin" / "python"
    local_python.parent.mkdir(parents=True)
    local_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    monkeypatch.delenv("INFERENCE_PYTHON", raising=False)
    monkeypatch.setattr(inference_routes, "_get_project_root", lambda: project_root)

    assert inference_routes._get_inference_python() == str(local_python)


def test_inference_python_env_override_wins(monkeypatch, tmp_path):
    custom_python = tmp_path / "custom-python"

    monkeypatch.setenv("INFERENCE_PYTHON", str(custom_python))

    assert inference_routes._get_inference_python() == str(custom_python)
