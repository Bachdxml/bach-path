from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.db.session import make_engine, make_session_factory
from app.main import create_app
from app.models.inference_run import InferenceRun
from app.models.slide import Slide
from app.models.enums import InferenceStatus


def _db_for(app_paths):
    factory = make_session_factory(make_engine(app_paths["app_data_dir"] / "app.db"))
    return factory()


def _create_run(db, output_json_path: Path | None) -> int:
    slide = Slide(
        original_path="C:/slides/sample.svs",
        stored_filename="sample.svs",
        stored_path=str(Path("sample.svs")),
        file_size_bytes=1,
        sha256="0" * 64,
    )
    db.add(slide)
    db.flush()
    run = InferenceRun(
        slide_id=slide.id,
        model_name="fungus",
        model_version="v1",
        status=InferenceStatus.succeeded.value,
        output_json_path=str(output_json_path) if output_json_path else None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run.id


def _write_sidecar(output_json_path: Path, arrays: dict, threshold: float | None) -> None:
    json_path = Path(output_json_path)
    sidecar = json_path.with_name(f"{json_path.stem}_masks.npz")
    payload = dict(arrays)
    if threshold is not None:
        payload["__threshold__"] = np.asarray(threshold, dtype=np.float64)
    np.savez_compressed(sidecar, **payload)


def test_tile_mask_roundtrips(app_paths, tmp_path):
    app = create_app()
    output_json = tmp_path / "runs" / "7.json"
    output_json.parent.mkdir(parents=True)

    mask = np.array(
        [
            [True, False, True, False, True],
            [False, False, True, True, False],
            [True, True, False, False, True],
        ],
        dtype=bool,
    )

    with TestClient(app) as client:
        db = _db_for(app_paths)
        try:
            run_id = _create_run(db, output_json)
        finally:
            db.close()
        _write_sidecar(output_json, {"10_20": mask}, threshold=0.5)

        response = client.get(f"/inference/runs/{run_id}/masks/10/20")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["encoding"] == "bitpack"
    assert payload["width"] == 5
    assert payload["height"] == 3
    assert payload["threshold"] == 0.5

    raw = base64.b64decode(payload["data"])
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    decoded = bits[: mask.size].reshape(mask.shape).astype(bool)
    assert np.array_equal(decoded, mask)


def test_tile_mask_missing_key_returns_404(app_paths, tmp_path):
    app = create_app()
    output_json = tmp_path / "runs" / "8.json"
    output_json.parent.mkdir(parents=True)
    mask = np.ones((2, 2), dtype=bool)

    with TestClient(app) as client:
        db = _db_for(app_paths)
        try:
            run_id = _create_run(db, output_json)
        finally:
            db.close()
        _write_sidecar(output_json, {"0_0": mask}, threshold=None)

        response = client.get(f"/inference/runs/{run_id}/masks/99/99")

    assert response.status_code == 404, response.text


def test_tile_mask_missing_run_returns_404(app_paths):
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/inference/runs/424242/masks/0/0")
    assert response.status_code == 404, response.text


def test_tile_mask_missing_sidecar_returns_404(app_paths, tmp_path):
    app = create_app()
    output_json = tmp_path / "runs" / "9.json"
    output_json.parent.mkdir(parents=True)

    with TestClient(app) as client:
        db = _db_for(app_paths)
        try:
            run_id = _create_run(db, output_json)
        finally:
            db.close()
        # Note: no sidecar written.
        response = client.get(f"/inference/runs/{run_id}/masks/0/0")

    assert response.status_code == 404, response.text


def test_tile_mask_threshold_absent_returns_null(app_paths, tmp_path):
    app = create_app()
    output_json = tmp_path / "runs" / "11.json"
    output_json.parent.mkdir(parents=True)
    mask = np.array([[True, False]], dtype=bool)

    with TestClient(app) as client:
        db = _db_for(app_paths)
        try:
            run_id = _create_run(db, output_json)
        finally:
            db.close()
        _write_sidecar(output_json, {"3_4": mask}, threshold=None)

        response = client.get(f"/inference/runs/{run_id}/masks/3/4")

    assert response.status_code == 200, response.text
    assert response.json()["threshold"] is None
