from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import inference as inference_routes
from app.db.session import make_engine, make_session_factory
from app.main import create_app
from app.models.inference_run import InferenceRun


def _create_sample_slide(path: Path, color: tuple[int, int, int] = (120, 80, 200)) -> None:
    Image.new("RGB", (48, 36), color=color).save(path)


def _import_slide(client: TestClient, slide_path: Path) -> int:
    response = client.post("/slides/import", json={"file_path": str(slide_path)})
    assert response.status_code == 200, response.text
    return response.json()["slide_id"]


def _install_synchronous_inference(
    app_paths,
    monkeypatch,
    *,
    output_payload: dict[str, object] | None,
    returncode: int = 0,
    recorded_commands: list[list[str]] | None = None,
):
    checkpoint_path = app_paths["app_data_dir"] / "test-checkpoint.pth.gz"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    def fake_resolve_model_checkpoint(model_file: str | None):
        return checkpoint_path, model_file or "models/test-checkpoint.pth.gz"

    def fake_submit(fn, *args, **kwargs):
        fn(*args, **kwargs)
        return SimpleNamespace(result=lambda: None)

    def fake_run(cmd, **kwargs):
        if recorded_commands is not None:
            recorded_commands.append(list(cmd))
        if output_payload is not None:
            output_path = Path(cmd[cmd.index("--output-json") + 1])
            output_path.write_text(json.dumps(output_payload), encoding="utf-8")
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr(inference_routes, "_resolve_model_checkpoint", fake_resolve_model_checkpoint)
    monkeypatch.setattr(inference_routes._inference_executor, "submit", fake_submit)
    monkeypatch.setattr(inference_routes.subprocess, "run", fake_run)


def test_inference_run_persists_success_and_is_queryable_after_restart(app_paths, monkeypatch):
    slide_path = app_paths["source_dir"] / "success-slide.png"
    _create_sample_slide(slide_path, color=(90, 180, 120))
    expected_hotspot = {
        "cx": 164.5,
        "cy": 248.25,
        "x": 140,
        "y": 226,
        "w": 72,
        "h": 60,
        "coverage": 0.118,
        "source": "segmentation_centroid",
    }

    script_path = app_paths["app_data_dir"] / "run_inference_api.py"
    script_path.write_text("# test stub\n", encoding="utf-8")
    _install_synchronous_inference(
        app_paths,
        monkeypatch,
        output_payload={
            "regions": [
                {
                    "x": 1,
                    "y": 2,
                    "w": 3,
                    "h": 4,
                    "score": 0.91,
                    "label": "fungus_positive",
                    "hotspot": expected_hotspot,
                },
                {"x": 5, "y": 6, "w": 7, "h": 8, "score": 0.27, "label": "fungus_negative"},
            ]
        },
    )
    monkeypatch.setattr(inference_routes, "_get_script_path", lambda: script_path)

    app = create_app()
    with TestClient(app) as client:
        slide_id = _import_slide(client, slide_path)
        create_response = client.post(f"/inference/slides/{slide_id}/run", json={"model_name": "fungus"})
        assert create_response.status_code == 200, create_response.text
        run_id = create_response.json()["id"]

    app = create_app()
    with TestClient(app) as client:
        session_factory = make_session_factory(make_engine(app_paths["app_data_dir"] / "app.db"))
        db = session_factory()
        try:
            run = db.get(InferenceRun, run_id)
            assert run is not None
            assert run.output_json_path is not None
            persisted_output = json.loads(Path(run.output_json_path).read_text(encoding="utf-8"))
        finally:
            db.close()

        run_response = client.get(f"/inference/runs/{run_id}")
        runs_response = client.get(f"/inference/slides/{slide_id}/runs")
        regions_response = client.get(f"/inference/runs/{run_id}/regions")
        lifecycle_response = client.get(f"/inference/runs/{run_id}/lifecycle-events")

    assert run_response.status_code == 200, run_response.text
    run_payload = run_response.json()
    assert run_payload["id"] == run_id
    assert run_payload["slide_id"] == slide_id
    assert run_payload["status"] == "succeeded"
    assert run_payload["started_at"] is not None
    assert run_payload["finished_at"] is not None
    assert run_payload["summary"] == {"total": 2, "fungus_positive": 1, "fungus_negative": 1}
    assert run_payload["error_message"] is None

    assert runs_response.status_code == 200, runs_response.text
    runs_payload = runs_response.json()["runs"]
    assert [item["id"] for item in runs_payload] == [run_id]
    assert runs_payload[0]["status"] == "succeeded"
    assert runs_payload[0]["summary"] == {"total": 2, "fungus_positive": 1, "fungus_negative": 1}

    assert persisted_output["regions"][0]["hotspot"] == expected_hotspot
    assert regions_response.status_code == 200, regions_response.text
    regions_payload = regions_response.json()["regions"]
    assert len(regions_payload) == 2
    assert {region["label"] for region in regions_payload} == {"fungus_positive", "fungus_negative"}
    assert regions_payload[0]["payload"] == {"hotspot": expected_hotspot}
    assert lifecycle_response.status_code == 200, lifecycle_response.text
    lifecycle = lifecycle_response.json()["events"]
    assert [event["to_status"] for event in lifecycle] == ["queued", "running", "succeeded"]


def test_inference_run_persists_failure_and_is_queryable_after_restart(app_paths, monkeypatch):
    slide_path = app_paths["source_dir"] / "failure-slide.png"
    _create_sample_slide(slide_path, color=(180, 90, 90))

    script_path = app_paths["app_data_dir"] / "run_inference_api.py"
    script_path.write_text("# test stub\n", encoding="utf-8")
    _install_synchronous_inference(app_paths, monkeypatch, output_payload=None, returncode=1)
    monkeypatch.setattr(inference_routes, "_get_script_path", lambda: script_path)

    app = create_app()
    with TestClient(app) as client:
        slide_id = _import_slide(client, slide_path)
        create_response = client.post(f"/inference/slides/{slide_id}/run", json={"model_name": "fungus"})
        assert create_response.status_code == 200, create_response.text
        run_id = create_response.json()["id"]

    app = create_app()
    with TestClient(app) as client:
        run_response = client.get(f"/inference/runs/{run_id}")
        runs_response = client.get(f"/inference/slides/{slide_id}/runs")
        regions_response = client.get(f"/inference/runs/{run_id}/regions")
        lifecycle_response = client.get(f"/inference/runs/{run_id}/lifecycle-events")

    assert run_response.status_code == 200, run_response.text
    run_payload = run_response.json()
    assert run_payload["status"] == "failed"
    assert run_payload["started_at"] is not None
    assert run_payload["finished_at"] is not None
    assert run_payload["error_message"] == "Inference failed. Check server logs for details."
    assert run_payload["summary"] is None

    assert runs_response.status_code == 200, runs_response.text
    runs_payload = runs_response.json()["runs"]
    assert [item["id"] for item in runs_payload] == [run_id]
    assert runs_payload[0]["status"] == "failed"
    assert runs_payload[0]["error_message"] == "Inference failed. Check server logs for details."

    assert regions_response.status_code == 200, regions_response.text
    assert regions_response.json()["regions"] == []
    assert lifecycle_response.status_code == 200, lifecycle_response.text
    lifecycle = lifecycle_response.json()["events"]
    assert [event["to_status"] for event in lifecycle] == ["queued", "running", "failed"]


def test_inference_subprocess_invocation_includes_model_metadata_and_optional_threshold(
    app_paths,
    monkeypatch,
):
    slide_path = app_paths["source_dir"] / "invocation-slide.png"
    _create_sample_slide(slide_path, color=(110, 70, 180))

    script_path = app_paths["app_data_dir"] / "run_inference_api.py"
    script_path.write_text("# test stub\n", encoding="utf-8")
    recorded_commands: list[list[str]] = []
    _install_synchronous_inference(
        app_paths,
        monkeypatch,
        output_payload={"regions": []},
        recorded_commands=recorded_commands,
    )
    monkeypatch.setattr(inference_routes, "_get_script_path", lambda: script_path)

    app = create_app()
    with TestClient(app) as client:
        slide_id = _import_slide(client, slide_path)

        with_threshold = client.post(
            f"/inference/slides/{slide_id}/run",
            json={
                "model_name": "fungus-detector",
                "model_file": "models/custom-checkpoint.pth.gz",
                "threshold": 0.35,
            },
        )
        without_threshold = client.post(
            f"/inference/slides/{slide_id}/run",
            json={
                "model_name": "fungus-detector",
                "model_file": "models/custom-checkpoint.pth.gz",
            },
        )

    assert with_threshold.status_code == 200, with_threshold.text
    assert without_threshold.status_code == 200, without_threshold.text
    assert len(recorded_commands) == 2

    with_threshold_payload = with_threshold.json()
    without_threshold_payload = without_threshold.json()
    first_command, second_command = recorded_commands

    assert "--checkpoint" in first_command
    assert first_command[first_command.index("--checkpoint") + 1] == str(
        app_paths["app_data_dir"] / "test-checkpoint.pth.gz"
    )
    assert first_command[first_command.index("--model-name") + 1] == with_threshold_payload["model_name"]
    assert first_command[first_command.index("--model-version") + 1] == with_threshold_payload["model_version"]
    assert first_command[first_command.index("--threshold") + 1] == "0.35"

    assert "--checkpoint" in second_command
    assert second_command[second_command.index("--checkpoint") + 1] == str(
        app_paths["app_data_dir"] / "test-checkpoint.pth.gz"
    )
    assert second_command[second_command.index("--model-name") + 1] == without_threshold_payload["model_name"]
    assert second_command[second_command.index("--model-version") + 1] == without_threshold_payload["model_version"]
    assert "--threshold" not in second_command
