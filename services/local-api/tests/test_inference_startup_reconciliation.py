from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app.db.base import Base
from app.db.init import _STARTUP_FAILURE_CODE, _STARTUP_FAILURE_MESSAGE
from app.db.session import make_engine, make_session_factory
from app.main import create_app
from app.models.enums import InferenceStatus
from app.models.inference_run import InferenceRun
from app.models.inference_run_event import InferenceRunEvent
from app.models.slide import Slide


def _seed_orphaned_inference_runs(app_data_dir: Path) -> tuple[int, int, int]:
    sqlite_path = app_data_dir / "app.db"
    engine = make_engine(sqlite_path)
    Base.metadata.create_all(bind=engine)
    session_factory = make_session_factory(engine)

    with session_factory() as db:
        slide_path = app_data_dir / "slides" / "restart-simulation-slide.svs"
        slide_path.parent.mkdir(parents=True, exist_ok=True)
        slide_path.write_text("slide", encoding="utf-8")

        slide = Slide(
            original_path=str(slide_path),
            stored_filename="restart-simulation-slide.svs",
            stored_path=str(slide_path),
            file_size_bytes=5,
            sha256="restart-simulation-slide",
        )
        db.add(slide)
        db.flush()

        queued_run = InferenceRun(
            slide_id=slide.id,
            model_name="fungus",
            model_version="1.0",
            status=InferenceStatus.queued.value,
        )
        running_run = InferenceRun(
            slide_id=slide.id,
            model_name="fungus",
            model_version="1.0",
            status=InferenceStatus.running.value,
            started_at=datetime(2026, 4, 10, 12, 0, 0, tzinfo=UTC),
        )
        succeeded_run = InferenceRun(
            slide_id=slide.id,
            model_name="fungus",
            model_version="1.0",
            status=InferenceStatus.succeeded.value,
            started_at=datetime(2026, 4, 10, 11, 30, 0, tzinfo=UTC),
            finished_at=datetime(2026, 4, 10, 11, 31, 0, tzinfo=UTC),
            error_code=None,
            error_message=None,
        )
        db.add_all([queued_run, running_run, succeeded_run])
        db.flush()

        db.add_all(
            [
                InferenceRunEvent(
                    run_id=queued_run.id,
                    from_status=None,
                    to_status=InferenceStatus.queued.value,
                    changed_at=datetime(2026, 4, 10, 12, 0, 1, tzinfo=UTC),
                ),
                InferenceRunEvent(
                    run_id=running_run.id,
                    from_status=None,
                    to_status=InferenceStatus.running.value,
                    changed_at=datetime(2026, 4, 10, 12, 0, 2, tzinfo=UTC),
                ),
                InferenceRunEvent(
                    run_id=succeeded_run.id,
                    from_status=None,
                    to_status=InferenceStatus.succeeded.value,
                    changed_at=datetime(2026, 4, 10, 11, 31, 0, tzinfo=UTC),
                ),
            ]
        )
        db.commit()

        return queued_run.id, running_run.id, succeeded_run.id


def test_restart_reconciles_orphaned_inference_runs_and_persists_lifecycle_events(app_paths):
    queued_run_id, running_run_id, succeeded_run_id = _seed_orphaned_inference_runs(app_paths["app_data_dir"])

    app = create_app()
    with TestClient(app) as client:
        queued_response = client.get(f"/inference/runs/{queued_run_id}")
        running_response = client.get(f"/inference/runs/{running_run_id}")
        succeeded_response = client.get(f"/inference/runs/{succeeded_run_id}")

        queued_events_response = client.get(f"/inference/runs/{queued_run_id}/lifecycle-events")
        running_events_response = client.get(f"/inference/runs/{running_run_id}/lifecycle-events")
        succeeded_events_response = client.get(f"/inference/runs/{succeeded_run_id}/lifecycle-events")

    assert queued_response.status_code == 200, queued_response.text
    assert running_response.status_code == 200, running_response.text
    assert succeeded_response.status_code == 200, succeeded_response.text

    queued_payload = queued_response.json()
    running_payload = running_response.json()
    succeeded_payload = succeeded_response.json()

    assert queued_payload["status"] == "failed"
    assert running_payload["status"] == "failed"
    assert succeeded_payload["status"] == "succeeded"
    assert succeeded_payload["error_message"] is None

    session_factory = make_session_factory(make_engine(app_paths["app_data_dir"] / "app.db"))
    with session_factory() as db:
        queued_run = db.get(InferenceRun, queued_run_id)
        running_run = db.get(InferenceRun, running_run_id)
        succeeded_run = db.get(InferenceRun, succeeded_run_id)

        assert queued_run is not None
        assert running_run is not None
        assert succeeded_run is not None

        assert queued_run.status == InferenceStatus.failed.value
        assert queued_run.error_code == _STARTUP_FAILURE_CODE
        assert queued_run.error_message == _STARTUP_FAILURE_MESSAGE
        assert queued_run.finished_at is not None

        assert running_run.status == InferenceStatus.failed.value
        assert running_run.error_code == _STARTUP_FAILURE_CODE
        assert running_run.error_message == _STARTUP_FAILURE_MESSAGE
        assert running_run.finished_at is not None

        assert succeeded_run.status == InferenceStatus.succeeded.value
        assert succeeded_run.error_code is None
        assert succeeded_run.error_message is None
        assert succeeded_run.finished_at is not None

        queued_events = (
            db.query(InferenceRunEvent)
            .filter(InferenceRunEvent.run_id == queued_run_id)
            .order_by(InferenceRunEvent.changed_at.asc(), InferenceRunEvent.id.asc())
            .all()
        )
        running_events = (
            db.query(InferenceRunEvent)
            .filter(InferenceRunEvent.run_id == running_run_id)
            .order_by(InferenceRunEvent.changed_at.asc(), InferenceRunEvent.id.asc())
            .all()
        )
        succeeded_events = (
            db.query(InferenceRunEvent)
            .filter(InferenceRunEvent.run_id == succeeded_run_id)
            .order_by(InferenceRunEvent.changed_at.asc(), InferenceRunEvent.id.asc())
            .all()
        )

    assert queued_events_response.status_code == 200, queued_events_response.text
    assert running_events_response.status_code == 200, running_events_response.text
    assert succeeded_events_response.status_code == 200, succeeded_events_response.text

    assert [event["to_status"] for event in queued_events_response.json()["events"]] == [
        InferenceStatus.queued.value,
        InferenceStatus.failed.value,
    ]
    assert [event["to_status"] for event in running_events_response.json()["events"]] == [
        InferenceStatus.running.value,
        InferenceStatus.failed.value,
    ]
    assert [event["to_status"] for event in succeeded_events_response.json()["events"]] == [
        InferenceStatus.succeeded.value,
    ]

    assert [event.to_status for event in queued_events] == [
        InferenceStatus.queued.value,
        InferenceStatus.failed.value,
    ]
    assert [event.to_status for event in running_events] == [
        InferenceStatus.running.value,
        InferenceStatus.failed.value,
    ]
    assert [event.to_status for event in succeeded_events] == [
        InferenceStatus.succeeded.value,
    ]

    assert succeeded_payload["status"] == "succeeded"
    assert succeeded_events_response.json()["events"][0]["to_status"] == "succeeded"
