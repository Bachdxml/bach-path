from __future__ import annotations
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.slide import Slide
from app.models.inference_run import InferenceRun
from app.models.region import Region
from app.models.enums import InferenceStatus
from app.schemas.inference import InferenceRunCreate, InferenceRunResponse, RegionResponse
from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/inference", tags=["inference"])
_inference_executor = ThreadPoolExecutor(max_workers=2)


def _get_script_path() -> Path:
    """Path to run_inference_api.py"""
    base = Path(__file__).resolve().parent.parent.parent.parent.parent
    script = base / "wsi-fungal-segmentation" / "scripts" / "run_inference_api.py"
    return script


def _get_checkpoint_path() -> Path:
    """Default checkpoint path; override with INFERENCE_CHECKPOINT env"""
    import os
    env_path = os.environ.get("INFERENCE_CHECKPOINT")
    if env_path:
        return Path(env_path).resolve()
    base = Path(__file__).resolve().parent.parent.parent.parent.parent
    return base / "wsi-fungal-segmentation" / "checkpoints" / "best_model.pth"


def _get_inference_python() -> str:
    """Python executable for inference (needs torch, openslide, wsi-fungal-segmentation)"""
    import os
    env_py = os.environ.get("INFERENCE_PYTHON")
    if env_py:
        return env_py
    base = Path(__file__).resolve().parent.parent.parent.parent.parent
    for venv in [
        base / "wsi-fungal-segmentation" / ".venv" / "bin" / "python",
        base / "wsi-fungal-segmentation" / ".venv" / "Scripts" / "python.exe",
        base / ".venv" / "bin" / "python",
        base / ".venv" / "Scripts" / "python.exe",
    ]:
        if venv.exists():
            return str(venv)
    return "python"


def _run_inference_task(run_id: int, slide_path: str, output_path: Path, checkpoint: Path):
    """Background task: run script, then load JSON and update DB."""
    from app.db.session import make_engine, make_session_factory
    from app.settings import load_settings

    settings = load_settings()
    engine = make_engine(settings.sqlite_path)
    SessionLocal = make_session_factory(engine)
    db = SessionLocal()

    try:
        run = db.get(InferenceRun, run_id)
        if not run:
            return
        run.status = InferenceStatus.running.value
        run.started_at = datetime.now(timezone.utc)
        db.commit()

        script_path = _get_script_path()
        python_path = _get_inference_python()
        cmd = [
            python_path,
            str(script_path),
            "--slide-path", slide_path,
            "--output-json", str(output_path),
            "--checkpoint", str(checkpoint),
            "--model-name", run.model_name,
            "--model-version", run.model_version,
        ]

        result = subprocess.run(
            cmd,
            cwd=str(script_path.parent.parent),
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            run.status = InferenceStatus.failed.value
            run.finished_at = datetime.now(timezone.utc)
            run.error_code = "inference_failed"
            run.error_message = (result.stderr or result.stdout or f"Exit code {result.returncode}")[:2000]
            db.commit()
            return

        run.output_json_path = str(output_path)
        run.status = InferenceStatus.succeeded.value
        run.finished_at = datetime.now(timezone.utc)

        with open(output_path) as f:
            data = json.load(f)

        for r in data.get("regions", []):
            region = Region(
                inference_run_id=run_id,
                x=r["x"],
                y=r["y"],
                w=r["w"],
                h=r["h"],
                score=r["score"],
                label=r.get("label"),
            )
            db.add(region)

        db.commit()
    except subprocess.TimeoutExpired:
        run.status = InferenceStatus.failed.value
        run.finished_at = datetime.now(timezone.utc)
        run.error_code = "timeout"
        run.error_message = "Inference timed out after 1 hour"
        db.commit()
    except Exception as e:
        run = db.get(InferenceRun, run_id)
        if run:
            run.status = InferenceStatus.failed.value
            run.finished_at = datetime.now(timezone.utc)
            run.error_code = "internal"
            run.error_message = str(e)[:2000]
            db.commit()
    finally:
        db.close()


@router.post("/slides/{slide_id}/run", response_model=InferenceRunResponse)
def run_inference(
    slide_id: int,
    request: Request,
    payload: InferenceRunCreate | None = None,
    db: Session = Depends(get_db),
):
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")

    slide_path = Path(slide.stored_path)
    if not slide_path.exists():
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing")

    script_path = _get_script_path()
    if not script_path.exists():
        raise AppError(ErrorCode.IO_ERROR, f"Inference script not found: {script_path}")

    checkpoint = _get_checkpoint_path()
    if not checkpoint.exists():
        raise AppError(ErrorCode.IO_ERROR, f"Checkpoint not found: {checkpoint}. Set INFERENCE_CHECKPOINT.")

    settings = request.app.state.settings
    settings.inference_runs_dir.mkdir(parents=True, exist_ok=True)

    payload = payload or InferenceRunCreate()
    run = InferenceRun(
        slide_id=slide_id,
        model_name=payload.model_name,
        model_version=payload.model_version,
        status=InferenceStatus.queued.value,
        requested_by_user_id=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    output_path = settings.inference_runs_dir / f"{run.id}.json"

    _inference_executor.submit(
        _run_inference_task,
        run.id,
        str(slide_path),
        output_path,
        checkpoint,
    )

    return _run_to_response(run)


@router.get("/runs/{run_id}", response_model=InferenceRunResponse)
def get_inference_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(InferenceRun, run_id)
    if not run:
        raise AppError(ErrorCode.NOT_FOUND, f"Inference run {run_id} not found")
    return _run_to_response(run, db)


@router.get("/runs/{run_id}/regions")
def get_inference_regions(run_id: int, db: Session = Depends(get_db)):
    run = db.get(InferenceRun, run_id)
    if not run:
        raise AppError(ErrorCode.NOT_FOUND, f"Inference run {run_id} not found")
    regions = db.query(Region).filter(Region.inference_run_id == run_id).all()
    return {"regions": [RegionResponse.model_validate(r) for r in regions]}


@router.get("/slides/{slide_id}/runs")
def list_slide_inference_runs(slide_id: int, db: Session = Depends(get_db)):
    runs = db.query(InferenceRun).filter(InferenceRun.slide_id == slide_id).order_by(InferenceRun.created_at.desc()).all()
    return {"runs": [_run_to_response(r, db) for r in runs]}


def _run_to_response(run: InferenceRun, db: Session | None = None) -> InferenceRunResponse:
    summary = None
    if db and run.status == InferenceStatus.succeeded.value:
        regions = db.query(Region).filter(Region.inference_run_id == run.id).all()
        n_pos = sum(1 for r in regions if r.label == "fungus_positive")
        n_neg = sum(1 for r in regions if r.label == "fungus_negative")
        summary = {"total": len(regions), "fungus_positive": n_pos, "fungus_negative": n_neg}
    return InferenceRunResponse(
        id=run.id,
        slide_id=run.slide_id,
        model_name=run.model_name,
        model_version=run.model_version,
        status=run.status,
        started_at=run.started_at.isoformat() if run.started_at else None,
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        created_at=run.created_at.isoformat() if run.created_at else "",
        summary=summary,
    )
