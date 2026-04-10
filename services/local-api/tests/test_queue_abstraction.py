from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.api.routes import inference as inference_routes
from app.queue import InMemoryQueue


def _create_sample_slide(path: Path, color: tuple[int, int, int] = (120, 80, 200)) -> None:
    Image.new("RGB", (48, 36), color=color).save(path)


def test_in_memory_queue_lifecycle() -> None:
    queue = InMemoryQueue()

    first = queue.enqueue({"slide_id": 1})
    second = queue.enqueue({"slide_id": 2})

    assert first.status.value == "queued"
    assert second.status.value == "queued"

    claimed_first = queue.claim()
    assert claimed_first is not None
    assert claimed_first.id == first.id
    assert claimed_first.status.value == "running"
    assert claimed_first.attempts == 1

    acked = queue.ack(claimed_first.id)
    assert acked is not None
    assert acked.status.value == "succeeded"

    claimed_second = queue.claim()
    assert claimed_second is not None
    assert claimed_second.id == second.id
    failed = queue.fail(claimed_second.id, "boom")
    assert failed is not None
    assert failed.status.value == "failed"
    assert failed.error == "boom"

    assert queue.cancel(claimed_second.id) is None


def test_inference_run_enqueues_job(app_paths, monkeypatch):
    slide_path = app_paths["source_dir"] / "queued-slide.png"
    _create_sample_slide(slide_path)

    script_path = app_paths["app_data_dir"] / "run_inference_api.py"
    script_path.write_text("# test stub\n", encoding="utf-8")
    checkpoint_path = app_paths["app_data_dir"] / "checkpoint.pt.gz"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    enqueue_payloads: list[dict[str, object]] = []

    class SpyQueue(InMemoryQueue):
        def enqueue(self, payload: dict[str, object]):
            enqueue_payloads.append(dict(payload))
            return super().enqueue(payload)

    monkeypatch.setattr(inference_routes, "_inference_queue", SpyQueue())
    monkeypatch.setattr(inference_routes, "_get_script_path", lambda: script_path)
    monkeypatch.setattr(
        inference_routes,
        "_resolve_model_checkpoint",
        lambda model_file: (checkpoint_path, model_file or "models/checkpoint.pt.gz"),
    )

    def fake_submit(*args, **kwargs):
        class _DummyFuture:
            def result(self) -> None:
                return None

        return _DummyFuture()

    monkeypatch.setattr(inference_routes._inference_executor, "submit", fake_submit)

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post("/slides/import", json={"file_path": str(slide_path)})
        assert import_response.status_code == 200, import_response.text
        slide_id = import_response.json()["slide_id"]

        response = client.post(f"/inference/slides/{slide_id}/run", json={"model_name": "fungus"})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "queued"
    assert enqueue_payloads
    first_payload = enqueue_payloads[0]
    assert first_payload["run_id"] == payload["id"]
    assert "slide_path" in first_payload

