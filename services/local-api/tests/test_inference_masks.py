from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import inference as inference_routes
from app.main import create_app


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
    recorded_commands: list[list[str]] | None = None,
):
    checkpoint_path = app_paths["app_data_dir"] / "test-checkpoint.pth.gz"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    script_path = app_paths["app_data_dir"] / "run_inference_api.py"
    script_path.write_text("# test stub\n", encoding="utf-8")

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
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(inference_routes, "_resolve_model_checkpoint", fake_resolve_model_checkpoint)
    monkeypatch.setattr(inference_routes, "_get_script_path", lambda: script_path)
    monkeypatch.setattr(inference_routes._inference_executor, "submit", fake_submit)
    monkeypatch.setattr(inference_routes.subprocess, "run", fake_run)


def _run_single_slide(app_paths, monkeypatch, *, output_payload, color, recorded_commands=None):
    slide_path = app_paths["source_dir"] / f"mask-{color[0]}-{color[1]}-{color[2]}.png"
    _create_sample_slide(slide_path, color=color)
    _install_synchronous_inference(
        app_paths,
        monkeypatch,
        output_payload=output_payload,
        recorded_commands=recorded_commands,
    )

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
    assert run_response.status_code == 200, run_response.text
    assert runs_response.status_code == 200, runs_response.text
    return run_response.json(), runs_response.json()["runs"][0]


def test_downsampled_status_is_persisted_and_exposed(app_paths, monkeypatch):
    payload = {
        "mask_degradation": {"status": "downsampled", "factor": 4},
        "regions": [{"x": 1, "y": 2, "w": 3, "h": 4, "score": 0.8, "label": "fungus_positive"}],
    }
    run_payload, list_payload = _run_single_slide(
        app_paths, monkeypatch, output_payload=payload, color=(90, 180, 120)
    )
    assert run_payload["status"] == "succeeded"
    assert run_payload["mask_degradation_status"] == "downsampled"
    assert run_payload["mask_downsample_factor"] == 4
    assert list_payload["mask_degradation_status"] == "downsampled"
    assert list_payload["mask_downsample_factor"] == 4


def test_dropped_status_is_persisted_and_run_succeeds(app_paths, monkeypatch):
    payload = {
        "mask_degradation": {"status": "dropped", "factor": None},
        "regions": [{"x": 1, "y": 2, "w": 3, "h": 4, "score": 0.8, "label": "fungus_positive"}],
    }
    run_payload, _ = _run_single_slide(
        app_paths, monkeypatch, output_payload=payload, color=(180, 90, 90)
    )
    assert run_payload["status"] == "succeeded"
    assert run_payload["mask_degradation_status"] == "dropped"
    assert run_payload["mask_downsample_factor"] is None


def test_full_status_default_for_output_without_field(app_paths, monkeypatch):
    payload = {
        "regions": [{"x": 1, "y": 2, "w": 3, "h": 4, "score": 0.8, "label": "fungus_positive"}],
    }
    run_payload, _ = _run_single_slide(
        app_paths, monkeypatch, output_payload=payload, color=(70, 120, 200)
    )
    assert run_payload["status"] == "succeeded"
    assert run_payload["mask_degradation_status"] == "full"
    assert run_payload["mask_downsample_factor"] is None


def test_worker_passes_output_budget_at_ninety_percent_of_limit(app_paths, monkeypatch):
    recorded_commands: list[list[str]] = []
    _run_single_slide(
        app_paths,
        monkeypatch,
        output_payload={"regions": []},
        color=(110, 70, 180),
        recorded_commands=recorded_commands,
    )
    command = recorded_commands[0]
    assert "--max-output-bytes" in command
    budget = int(command[command.index("--max-output-bytes") + 1])
    # Default limit is 10,000,000; budget targets ~90% as a safety margin.
    assert budget == int(10_000_000 * 0.9)


def test_worker_budget_tracks_env_override(app_paths, monkeypatch):
    monkeypatch.setenv("APP_MAX_INFERENCE_OUTPUT_BYTES", "4000000")
    recorded_commands: list[list[str]] = []
    _run_single_slide(
        app_paths,
        monkeypatch,
        output_payload={"regions": []},
        color=(60, 160, 160),
        recorded_commands=recorded_commands,
    )
    command = recorded_commands[0]
    budget = int(command[command.index("--max-output-bytes") + 1])
    assert budget == int(4_000_000 * 0.9)
