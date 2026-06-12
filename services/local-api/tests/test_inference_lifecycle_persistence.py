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
    stdout: str = "",
    stderr: str = "",
    recorded_commands: list[list[str]] | None = None,
    recorded_run_kwargs: list[dict[str, object]] | None = None,
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
        if recorded_run_kwargs is not None:
            recorded_run_kwargs.append(dict(kwargs))
        if output_payload is not None:
            output_path = Path(cmd[cmd.index("--output-json") + 1])
            output_path.write_text(json.dumps(output_payload), encoding="utf-8")
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(inference_routes, "_resolve_model_checkpoint", fake_resolve_model_checkpoint)
    monkeypatch.setattr(inference_routes._inference_executor, "submit", fake_submit)
    monkeypatch.setattr(inference_routes.subprocess, "run", fake_run)


def test_inference_run_persists_success_and_is_queryable_after_restart(app_paths, monkeypatch):
    slide_path = app_paths["source_dir"] / "success-slide.png"
    _create_sample_slide(slide_path, color=(90, 180, 120))
    expected_segmentation = {
        "kind": "segmentation",
        "source": "prediction_tile",
        "coverage": 0.25,
        "prediction_mask": {
            "encoding": "bitpack",
            "width": 4,
            "height": 4,
            "threshold": 0.5,
            "data": "BkY=",
        },
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
                    **expected_segmentation,
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
    assert run_payload["inference_result"] == "positive"
    assert run_payload["started_at"] is not None
    assert run_payload["finished_at"] is not None
    assert run_payload["summary"] == {"total": 2, "fungus_positive": 1, "fungus_negative": 1}
    assert run_payload["error_message"] is None

    assert runs_response.status_code == 200, runs_response.text
    runs_payload = runs_response.json()["runs"]
    assert [item["id"] for item in runs_payload] == [run_id]
    assert runs_payload[0]["status"] == "succeeded"
    assert runs_payload[0]["inference_result"] == "positive"
    assert runs_payload[0]["summary"] == {"total": 2, "fungus_positive": 1, "fungus_negative": 1}

    for key, value in expected_segmentation.items():
        assert persisted_output["regions"][0][key] == value
    assert regions_response.status_code == 200, regions_response.text
    regions_payload = regions_response.json()["regions"]
    assert len(regions_payload) == 2
    assert {region["label"] for region in regions_payload} == {"fungus_positive", "fungus_negative"}
    assert regions_payload[0]["payload"] == expected_segmentation
    assert lifecycle_response.status_code == 200, lifecycle_response.text
    lifecycle = lifecycle_response.json()["events"]
    assert [event["to_status"] for event in lifecycle] == ["queued", "running", "succeeded"]


def test_inference_run_persists_failure_and_is_queryable_after_restart(app_paths, monkeypatch):
    slide_path = app_paths["source_dir"] / "failure-slide.png"
    _create_sample_slide(slide_path, color=(180, 90, 90))

    script_path = app_paths["app_data_dir"] / "run_inference_api.py"
    script_path.write_text("# test stub\n", encoding="utf-8")
    _install_synchronous_inference(
        app_paths,
        monkeypatch,
        output_payload=None,
        returncode=1,
        stderr="ModuleNotFoundError: No module named 'torchvision'\napi_key=super-secret",
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
        run_response = client.get(f"/inference/runs/{run_id}")
        runs_response = client.get(f"/inference/slides/{slide_id}/runs")
        regions_response = client.get(f"/inference/runs/{run_id}/regions")
        lifecycle_response = client.get(f"/inference/runs/{run_id}/lifecycle-events")

    assert run_response.status_code == 200, run_response.text
    run_payload = run_response.json()
    assert run_payload["status"] == "failed"
    assert run_payload["inference_result"] == "unchecked"
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
    assert "ModuleNotFoundError" in lifecycle[-1]["detail"]
    assert "torchvision" in lifecycle[-1]["detail"]
    assert "super-secret" not in lifecycle[-1]["detail"]
    assert "[redacted]" in lifecycle[-1]["detail"]


def test_inference_run_response_classifies_negative_and_needs_review(app_paths, monkeypatch):
    negative_slide_path = app_paths["source_dir"] / "run-negative.png"
    needs_review_slide_path = app_paths["source_dir"] / "run-needs-review.png"
    _create_sample_slide(negative_slide_path, color=(60, 70, 80))
    _create_sample_slide(needs_review_slide_path, color=(80, 70, 60))

    script_path = app_paths["app_data_dir"] / "run_inference_api.py"
    script_path.write_text("# test stub\n", encoding="utf-8")

    outputs = [
        {"regions": [{"x": 1, "y": 2, "w": 3, "h": 4, "score": 0.01, "label": "fungus_negative"}]},
        {"regions": []},
    ]

    def fake_run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("--output-json") + 1])
        output_path.write_text(json.dumps(outputs.pop(0)), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    _install_synchronous_inference(app_paths, monkeypatch, output_payload=None)
    monkeypatch.setattr(inference_routes, "_get_script_path", lambda: script_path)
    monkeypatch.setattr(inference_routes.subprocess, "run", fake_run)

    app = create_app()
    with TestClient(app) as client:
        negative_slide_id = _import_slide(client, negative_slide_path)
        needs_review_slide_id = _import_slide(client, needs_review_slide_path)

        negative_create = client.post(f"/inference/slides/{negative_slide_id}/run", json={"model_name": "fungus"})
        needs_review_create = client.post(
            f"/inference/slides/{needs_review_slide_id}/run",
            json={"model_name": "fungus"},
        )
        assert negative_create.status_code == 200, negative_create.text
        assert needs_review_create.status_code == 200, needs_review_create.text

        negative_response = client.get(f"/inference/runs/{negative_create.json()['id']}")
        needs_review_response = client.get(f"/inference/runs/{needs_review_create.json()['id']}")

    assert negative_response.status_code == 200, negative_response.text
    assert negative_response.json()["inference_result"] == "negative"
    assert negative_response.json()["summary"] == {"total": 1, "fungus_positive": 0, "fungus_negative": 1}

    assert needs_review_response.status_code == 200, needs_review_response.text
    assert needs_review_response.json()["inference_result"] == "needs_review"
    assert needs_review_response.json()["summary"] == {"total": 0, "fungus_positive": 0, "fungus_negative": 0}


def test_inference_rejects_malformed_region_output(app_paths, monkeypatch):
    slide_path = app_paths["source_dir"] / "malformed-output-slide.png"
    _create_sample_slide(slide_path, color=(100, 110, 120))
    _install_synchronous_inference(
        app_paths,
        monkeypatch,
        output_payload={
            "regions": [
                {"x": 0, "y": 0, "w": -1, "h": 8, "score": 0.5, "label": "fungus_positive"}
            ]
        },
    )

    app = create_app()
    with TestClient(app) as client:
        slide_id = _import_slide(client, slide_path)
        create_response = client.post(f"/inference/slides/{slide_id}/run", json={"model_name": "fungus"})
        assert create_response.status_code == 200, create_response.text
        run_id = create_response.json()["id"]

        run_response = client.get(f"/inference/runs/{run_id}")
        regions_response = client.get(f"/inference/runs/{run_id}/regions")
        lifecycle_response = client.get(f"/inference/runs/{run_id}/lifecycle-events")

    run_payload = run_response.json()
    assert run_payload["status"] == "failed"
    assert run_payload["error_message"] == "Inference output was malformed."
    assert regions_response.json()["regions"] == []
    lifecycle = lifecycle_response.json()["events"]
    assert lifecycle[-1]["error"] == "malformed_output"


def test_inference_subprocess_invocation_includes_model_metadata_and_optional_threshold(
    app_paths,
    monkeypatch,
):
    monkeypatch.setenv("APP_INFERENCE_TARGET_TILES", "1500")
    slide_path = app_paths["source_dir"] / "invocation-slide.png"
    _create_sample_slide(slide_path, color=(110, 70, 180))

    script_path = app_paths["app_data_dir"] / "run_inference_api.py"
    script_path.write_text("# test stub\n", encoding="utf-8")
    recorded_commands: list[list[str]] = []
    recorded_run_kwargs: list[dict[str, object]] = []
    _install_synchronous_inference(
        app_paths,
        monkeypatch,
        output_payload={"regions": []},
        recorded_commands=recorded_commands,
        recorded_run_kwargs=recorded_run_kwargs,
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
    assert len(recorded_run_kwargs) == 2

    with_threshold_payload = with_threshold.json()
    without_threshold_payload = without_threshold.json()
    first_command, second_command = recorded_commands
    first_kwargs = recorded_run_kwargs[0]

    assert "--checkpoint" in first_command
    assert first_command[first_command.index("--checkpoint") + 1] == str(
        app_paths["app_data_dir"] / "test-checkpoint.pth.gz"
    )
    assert first_command[first_command.index("--model-name") + 1] == with_threshold_payload["model_name"]
    assert first_command[first_command.index("--model-version") + 1] == with_threshold_payload["model_version"]
    assert first_command[first_command.index("--batch-size") + 1] == "4"
    assert first_command[first_command.index("--device") + 1] == "auto"
    assert first_command[first_command.index("--level") + 1] == "auto"
    assert first_command[first_command.index("--target-tiles") + 1] == "1500"
    assert first_command[first_command.index("--threshold") + 1] == "0.35"
    assert first_kwargs["cwd"] == str(script_path.parent.parent)
    assert first_kwargs["capture_output"] is True
    assert first_kwargs["text"] is True
    assert isinstance(first_kwargs["env"], dict)
    assert str(Path(__file__).resolve().parents[3] / "wsi-fungal-segmentation") in first_kwargs["env"]["PYTHONPATH"]

    assert "--checkpoint" in second_command
    assert second_command[second_command.index("--checkpoint") + 1] == str(
        app_paths["app_data_dir"] / "test-checkpoint.pth.gz"
    )
    assert second_command[second_command.index("--model-name") + 1] == without_threshold_payload["model_name"]
    assert second_command[second_command.index("--model-version") + 1] == without_threshold_payload["model_version"]
    assert second_command[second_command.index("--batch-size") + 1] == "4"
    assert second_command[second_command.index("--device") + 1] == "auto"
    assert second_command[second_command.index("--level") + 1] == "auto"
    assert second_command[second_command.index("--target-tiles") + 1] == "1500"
    assert "--threshold" not in second_command


def test_inference_subprocess_invocation_uses_configured_tile_batch_size(
    app_paths,
    monkeypatch,
):
    monkeypatch.setenv("APP_INFERENCE_TILE_BATCH_SIZE", "3")
    slide_path = app_paths["source_dir"] / "configured-batch-slide.png"
    _create_sample_slide(slide_path, color=(100, 80, 60))

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
        response = client.post(
            f"/inference/slides/{slide_id}/run",
            json={"model_name": "fungus-detector", "model_file": "models/custom-checkpoint.pth.gz"},
        )

    assert response.status_code == 200, response.text
    command = recorded_commands[0]
    assert command[command.index("--batch-size") + 1] == "3"


def test_inference_python_prefers_wsi_runtime_before_api_runtime(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    wsi_python = project_root / "wsi-fungal-segmentation" / ".venv" / "Scripts" / "python.exe"
    api_python = project_root / "services" / "local-api" / ".venv" / "Scripts" / "python.exe"
    wsi_python.parent.mkdir(parents=True)
    api_python.parent.mkdir(parents=True)
    wsi_python.write_text("", encoding="utf-8")
    api_python.write_text("", encoding="utf-8")

    monkeypatch.delenv("INFERENCE_PYTHON", raising=False)
    monkeypatch.setattr(inference_routes, "_get_project_root", lambda: project_root)

    assert inference_routes._get_inference_python() == str(wsi_python)


def test_inference_reports_memory_pressure_when_subprocess_is_sigkilled(
    app_paths,
    monkeypatch,
):
    slide_path = app_paths["source_dir"] / "sigkill-slide.png"
    _create_sample_slide(slide_path, color=(20, 120, 160))
    _install_synchronous_inference(
        app_paths,
        monkeypatch,
        output_payload={"regions": []},
        returncode=-9,
        stderr="",
        stdout="",
    )

    app = create_app()
    with TestClient(app) as client:
        slide_id = _import_slide(client, slide_path)
        create_response = client.post(
            f"/inference/slides/{slide_id}/run",
            json={"model_name": "fungus-detector", "model_file": "models/custom-checkpoint.pth.gz"},
        )
        assert create_response.status_code == 200, create_response.text
        run_id = create_response.json()["id"]

    app = create_app()
    with TestClient(app) as client:
        run_response = client.get(f"/inference/runs/{run_id}")
        lifecycle_response = client.get(f"/inference/runs/{run_id}/lifecycle-events")

    assert run_response.status_code == 200, run_response.text
    run_payload = run_response.json()
    assert run_payload["status"] == "failed"
    assert "memory pressure" in run_payload["error_message"]
    assert lifecycle_response.status_code == 200, lifecycle_response.text
    lifecycle = lifecycle_response.json()["events"]
    assert lifecycle[-1]["to_status"] == "failed"
    assert "non-zero exit code -9" in lifecycle[-1]["detail"]
