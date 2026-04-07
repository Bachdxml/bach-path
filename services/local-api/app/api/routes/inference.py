from __future__ import annotations
import hashlib
import json
import logging
import subprocess
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
from app.schemas.inference import (
    InferenceRunCreate,
    InferenceBatchRunCreate,
    InferenceFolderRunCreate,
    InferenceBatchRunResponse,
    InferenceRunResponse,
    RegionResponse,
    InferenceModelInfo,
    InferenceModelListResponse,
)
from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/inference", tags=["inference"])
_inference_executor = ThreadPoolExecutor(max_workers=2)
logger = logging.getLogger(__name__)


def _is_deployable_weight(p: Path) -> bool:
    if not p.is_file():
        return False
    n = p.name.lower()
    return n.endswith((".pth.gz", ".pt.gz"))


def _folder_key_for_slide(original_path: str | None) -> str:
    if not original_path:
        return "uncategorized"
    parent = Path(original_path).parent
    normalized = parent.resolve(strict=False).as_posix()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"folder_{digest}"


def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent.parent.parent


def _get_script_path() -> Path:
    """Path to run_inference_api.py"""
    return _get_project_root() / "wsi-fungal-segmentation" / "scripts" / "run_inference_api.py"


def _get_models_dir() -> Path:
    models_dir = _get_project_root() / "wsi-fungal-segmentation" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _get_model_roots() -> list[tuple[str, Path]]:
    return [
        ("models", _get_models_dir().resolve()),
    ]


def _model_infos() -> list[InferenceModelInfo]:
    roots = _get_model_roots()
    files = []
    for prefix, root in roots:
        for p in root.rglob("*"):
            if not _is_deployable_weight(p):
                continue
            rel = p.relative_to(root).as_posix()
            stat = p.stat()
            model_id = f"{prefix}/{rel}"
            files.append(
                InferenceModelInfo(
                    id=model_id,
                    label=model_id,
                    path=model_id,
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                )
            )
    files.sort(key=lambda m: m.modified_at or "", reverse=True)
    return files


def _resolve_model_checkpoint(model_file: str | None) -> tuple[Path, str]:
    """
    Resolve requested checkpoint from model id relative to models/.
    Returns (path, model_id_for_display).
    """
    import os

    models_dir = _get_models_dir().resolve()
    model_roots = _get_model_roots()
    model_infos = _model_infos()

    if model_file:
        requested = model_file.strip()
        root_prefix = None
        rel = None
        for prefix, root in model_roots:
            prefix_token = f"{prefix}/"
            if requested.startswith(prefix_token):
                root_prefix = (prefix, root)
                rel = requested[len(prefix_token):]
                break
        if root_prefix is None:
            # backward-compatible: plain id resolves under models/
            root_prefix = ("models", models_dir)
            rel = requested
        _, root = root_prefix
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as e:
            raise AppError(ErrorCode.IO_ERROR, "Invalid model path") from e
        if not candidate.exists() or not candidate.is_file():
            raise AppError(ErrorCode.IO_ERROR, "Requested model not found")
        return candidate, requested

    # 1) Explicit env override keeps highest priority for deployments.
    env_path = os.environ.get("INFERENCE_CHECKPOINT")
    if env_path:
        return Path(env_path).resolve(), "env:INFERENCE_CHECKPOINT"

    # 2) Prefer default deploy artifact if present.
    default_model = models_dir / "deploy.pth.gz"
    if default_model.exists():
        return default_model, "models/deploy.pth.gz"

    # 3) First discovered model file.
    if model_infos:
        first = model_infos[0]
        return (_get_models_dir() / first.id.removeprefix("models/")).resolve(), first.id

    raise AppError(
        ErrorCode.IO_ERROR,
        "No deploy model found. Add .pth.gz or .pt.gz weights to the models directory or set INFERENCE_CHECKPOINT.",
    )


def _get_inference_python() -> str:
    """Python executable for inference (needs torch, openslide, wsi-fungal-segmentation)"""
    import os
    env_py = os.environ.get("INFERENCE_PYTHON")
    if env_py:
        return env_py
    base = _get_project_root()
    for venv in [
        base / "wsi-fungal-segmentation" / ".venv" / "bin" / "python",
        base / "wsi-fungal-segmentation" / ".venv" / "Scripts" / "python.exe",
        base / ".venv" / "bin" / "python",
        base / ".venv" / "Scripts" / "python.exe",
    ]:
        if venv.exists():
            return str(venv)
    return "python"


def _run_inference_task(
    run_id: int,
    slide_path: str,
    output_path: Path,
    checkpoint: Path,
    threshold: float | None = None,
):
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
        if threshold is not None:
            cmd.extend(["--threshold", str(float(threshold))])

        result = subprocess.run(
            cmd,
            cwd=str(script_path.parent.parent),
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            logger.warning("Inference subprocess failed for run %s with exit code %s", run_id, result.returncode)
            run.status = InferenceStatus.failed.value
            run.finished_at = datetime.now(timezone.utc)
            run.error_code = "inference_failed"
            run.error_message = "Inference failed. Check server logs for details."
            db.commit()
            return

        run.output_json_path = str(output_path)
        run.status = InferenceStatus.succeeded.value
        run.finished_at = datetime.now(timezone.utc)

        with open(output_path) as f:
            data = json.load(f)

        parsed_regions = []
        for r in data.get("regions", []):
            parsed_regions.append(
                {
                    "x": int(r["x"]),
                    "y": int(r["y"]),
                    "w": int(r["w"]),
                    "h": int(r["h"]),
                    "score": float(r["score"]),
                    "label": r.get("label"),
                }
            )

        for r in parsed_regions:
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
        logger.warning("Inference timed out for run %s", run_id)
        run.status = InferenceStatus.failed.value
        run.finished_at = datetime.now(timezone.utc)
        run.error_code = "timeout"
        run.error_message = "Inference timed out after 1 hour"
        db.commit()
    except Exception:
        logger.exception("Unhandled inference task failure for run %s", run_id)
        run = db.get(InferenceRun, run_id)
        if run:
            run.status = InferenceStatus.failed.value
            run.finished_at = datetime.now(timezone.utc)
            run.error_code = "internal"
            run.error_message = "Inference failed due to an internal error."
            db.commit()
    finally:
        db.close()


def _queue_inference_run_for_slide(
    *,
    slide: Slide,
    payload: InferenceRunCreate,
    settings,
    db: Session,
) -> InferenceRun:
    slide_path = Path(slide.stored_path)
    if not slide_path.exists():
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")

    checkpoint, model_id = _resolve_model_checkpoint(payload.model_file)
    run = InferenceRun(
        slide_id=slide.id,
        model_name=payload.model_name,
        model_version=model_id or payload.model_version,
        status=InferenceStatus.queued.value,
        requested_by_user_id=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    output_path = settings.inference_runs_dir / f"{run.id}.json"
    try:
        _inference_executor.submit(
            _run_inference_task,
            run.id,
            str(slide_path),
            output_path,
            checkpoint,
            payload.threshold,
        )
    except Exception:
        logger.exception("Failed to submit inference run %s", run.id)
        run.status = InferenceStatus.failed.value
        run.finished_at = datetime.now(timezone.utc)
        run.error_code = "executor_submit_failed"
        run.error_message = "Could not start inference task."
        db.commit()
        raise AppError(ErrorCode.IO_ERROR, "Could not start inference task")
    return run


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

    script_path = _get_script_path()
    if not script_path.exists():
        raise AppError(ErrorCode.IO_ERROR, "Inference script is not available on this server")

    settings = request.app.state.settings
    settings.inference_runs_dir.mkdir(parents=True, exist_ok=True)

    payload = payload or InferenceRunCreate()
    run = _queue_inference_run_for_slide(slide=slide, payload=payload, settings=settings, db=db)

    return _run_to_response(run)


@router.post("/slides/batch-run", response_model=InferenceBatchRunResponse)
def run_batch_inference(
    request: Request,
    payload: InferenceBatchRunCreate,
    db: Session = Depends(get_db),
):
    script_path = _get_script_path()
    if not script_path.exists():
        raise AppError(ErrorCode.IO_ERROR, "Inference script is not available on this server")

    settings = request.app.state.settings
    settings.inference_runs_dir.mkdir(parents=True, exist_ok=True)

    slide_ids = list(dict.fromkeys(payload.slide_ids))
    slides = db.query(Slide).filter(Slide.id.in_(slide_ids)).all()
    found_by_id = {s.id: s for s in slides}
    missing = [sid for sid in slide_ids if sid not in found_by_id]
    if missing:
        raise AppError(ErrorCode.NOT_FOUND, f"Slides not found: {missing}")

    base_payload = InferenceRunCreate(
        model_name=payload.model_name,
        model_version=payload.model_version,
        model_file=payload.model_file,
        threshold=payload.threshold,
    )
    runs = []
    for sid in slide_ids:
        run = _queue_inference_run_for_slide(
            slide=found_by_id[sid],
            payload=base_payload,
            settings=settings,
            db=db,
        )
        runs.append(run)
    return InferenceBatchRunResponse(run_ids=[r.id for r in runs], slide_ids=slide_ids)


@router.post("/folders/run", response_model=InferenceBatchRunResponse)
def run_folder_inference(
    request: Request,
    payload: InferenceFolderRunCreate,
    db: Session = Depends(get_db),
):
    script_path = _get_script_path()
    if not script_path.exists():
        raise AppError(ErrorCode.IO_ERROR, "Inference script is not available on this server")

    settings = request.app.state.settings
    settings.inference_runs_dir.mkdir(parents=True, exist_ok=True)

    requested_key = payload.folder_key.strip()
    if not requested_key:
        raise AppError(ErrorCode.SLIDE_INVALID, "folder_key is required")

    slides = db.query(Slide).order_by(Slide.created_at.desc()).all()
    target_slides = [s for s in slides if _folder_key_for_slide(s.original_path) == requested_key]
    if not target_slides:
        raise AppError(ErrorCode.NOT_FOUND, "No slides found in that folder")

    base_payload = InferenceRunCreate(
        model_name=payload.model_name,
        model_version=payload.model_version,
        model_file=payload.model_file,
        threshold=payload.threshold,
    )
    runs = []
    for slide in target_slides:
        run = _queue_inference_run_for_slide(
            slide=slide,
            payload=base_payload,
            settings=settings,
            db=db,
        )
        runs.append(run)
    return InferenceBatchRunResponse(
        run_ids=[r.id for r in runs],
        slide_ids=[s.id for s in target_slides],
    )


@router.get("/models", response_model=InferenceModelListResponse)
def list_inference_models():
    models = _model_infos()
    default_model_id = None
    try:
        _, default_model_id = _resolve_model_checkpoint(None)
    except AppError:
        default_model_id = None
    return InferenceModelListResponse(models=models, default_model_id=default_model_id)


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
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")
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
        started_at=run.started_at,
        finished_at=run.finished_at,
        created_at=run.created_at,
        summary=summary,
        error_message=_public_error_message(run.error_message),
    )


def _public_error_message(message: str | None) -> str | None:
    if not message:
        return None
    if "/" in message or "\\" in message:
        return "Inference failed. Check server logs for details."
    return message
