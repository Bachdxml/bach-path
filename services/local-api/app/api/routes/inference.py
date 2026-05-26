from __future__ import annotations
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.slide import Slide
from app.models.inference_run import InferenceRun
from app.models.inference_run_event import InferenceRunEvent
from app.models.region import Region
from app.models.enums import InferenceStatus
from app.queue import InMemoryQueue, JobRecord, QueueInterface
from app.schemas.inference import (
    InferenceRunCreate,
    InferenceBatchRunCreate,
    InferenceFolderRunCreate,
    InferenceBatchRunResponse,
    InferenceRunResponse,
    InferenceRunLifecycleEventListResponse,
    InferenceRunLifecycleEventResponse,
    RegionResponse,
    InferenceModelInfo,
    InferenceModelListResponse,
)
from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/inference", tags=["inference"])
# Whole-slide inference is memory-heavy; serializing jobs avoids competing
# OpenSlide/PyTorch subprocesses being killed by local OS memory pressure.
_inference_executor = ThreadPoolExecutor(max_workers=1)
_inference_queue: QueueInterface = InMemoryQueue()
logger = logging.getLogger(__name__)
MAX_FOLDER_INFERENCE_SLIDES = 512
MAX_REGIONS_PER_RUN = 100_000
MAX_SUBPROCESS_DIAGNOSTIC_CHARS = 2000
ALLOWED_REGION_LABELS = {"fungus_positive", "fungus_negative"}
SIGKILL_RETURN_CODE = -9
SIGKILL_ERROR_MESSAGE = (
    "Inference process was killed, likely due to memory pressure. "
    "Try closing other apps or reducing APP_INFERENCE_TILE_BATCH_SIZE."
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)([^\s,;]+)"),
)


def _region_payload_from_json(payload_json: str | None) -> dict | None:
    if not payload_json:
        return None
    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        logger.warning("Skipping invalid region payload JSON")
        return None
    return payload if isinstance(payload, dict) else None


def _region_to_response(region: Region) -> RegionResponse:
    return RegionResponse(
        id=region.id,
        x=region.x,
        y=region.y,
        w=region.w,
        h=region.h,
        score=region.score,
        label=region.label,
        payload=_region_payload_from_json(region.payload_json),
    )


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
        env_checkpoint = Path(env_path).resolve()
        if not env_checkpoint.exists() or not env_checkpoint.is_file():
            raise AppError(ErrorCode.IO_ERROR, "INFERENCE_CHECKPOINT must point to an existing file")
        return env_checkpoint, "env:INFERENCE_CHECKPOINT"

    # 2) Prefer default deploy artifact if present (canonical name, then legacy).
    for name, label in (
        ("deploy-fungus.pth.gz", "models/deploy-fungus.pth.gz"),
        ("deploy.pth.gz", "models/deploy.pth.gz"),
    ):
        default_model = models_dir / name
        if default_model.exists():
            return default_model, label

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
        base / "services" / "local-api" / ".venv" / "bin" / "python",
        base / "services" / "local-api" / ".venv" / "Scripts" / "python.exe",
        base / "wsi-fungal-segmentation" / ".venv" / "bin" / "python",
        base / "wsi-fungal-segmentation" / ".venv" / "Scripts" / "python.exe",
        base / ".venv" / "bin" / "python",
        base / ".venv" / "Scripts" / "python.exe",
    ]:
        if venv.exists():
            return str(venv)
    return sys.executable or "python"


def _subprocess_diagnostic(result: subprocess.CompletedProcess | object) -> str:
    parts = []
    for label, value in (("stderr", getattr(result, "stderr", None)), ("stdout", getattr(result, "stdout", None))):
        if not value:
            continue
        text = str(value).replace("\x00", "").strip()
        if text:
            parts.append(f"{label}: {text}")
    diagnostic = "\n".join(parts).strip()
    if not diagnostic:
        return "Inference subprocess exited without diagnostic output."
    for pattern in _SECRET_PATTERNS:
        diagnostic = pattern.sub(r"\1\2[redacted]", diagnostic)
    if len(diagnostic) > MAX_SUBPROCESS_DIAGNOSTIC_CHARS:
        return diagnostic[:MAX_SUBPROCESS_DIAGNOSTIC_CHARS].rstrip() + "\n...[truncated]"
    return diagnostic


def _inference_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    project_root = _get_project_root()
    python_entries = [
        str(project_root / "wsi-fungal-segmentation"),
        str(project_root / "services" / "local-api"),
        env.get("PYTHONPATH"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(entry for entry in python_entries if entry)
    return env


def _subprocess_failure_message(returncode: int | None) -> str:
    if returncode == SIGKILL_RETURN_CODE:
        return SIGKILL_ERROR_MESSAGE
    return "Inference failed. Check server logs for details."


def _parse_inference_region(raw_region: object) -> dict:
    if not isinstance(raw_region, dict):
        raise ValueError("Inference output region must be an object")
    x = int(raw_region["x"])
    y = int(raw_region["y"])
    w = int(raw_region["w"])
    h = int(raw_region["h"])
    score = float(raw_region["score"])
    label = raw_region.get("label")
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("Inference output region has invalid geometry")
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("Inference output region has invalid score")
    if label is not None and label not in ALLOWED_REGION_LABELS:
        raise ValueError("Inference output region has invalid label")
    payload = {
        k: v
        for k, v in raw_region.items()
        if k not in {"x", "y", "w", "h", "score", "label"}
    }
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "score": score,
        "label": label,
        "payload_json": json.dumps(payload, allow_nan=False) if payload else None,
    }


def _resolve_managed_slide_path(slide: Slide, settings) -> Path:
    path = Path(slide.stored_path).resolve(strict=False)
    slides_root = settings.slides_dir.resolve(strict=False)
    try:
        path.relative_to(slides_root)
    except ValueError as e:
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide path is outside managed storage") from e
    if not path.exists() or not path.is_file():
        raise AppError(ErrorCode.STORAGE_INCONSISTENT, "Slide file missing from managed storage")
    return path


def _record_inference_run_transition(
    db: Session,
    run: InferenceRun,
    status: InferenceStatus | str,
    *,
    changed_at: datetime | None = None,
    detail: str | None = None,
    error: str | None = None,
) -> datetime:
    transition_at = changed_at or datetime.now(timezone.utc)
    status_value = status.value if isinstance(status, InferenceStatus) else str(status)
    supported = {s.value for s in InferenceStatus}
    if status_value not in supported:
        raise ValueError(f"Unsupported inference status transition: {status_value}")
    previous_status = run.status

    run.status = status_value
    if status_value == InferenceStatus.running.value:
        run.started_at = transition_at
    elif status_value in {InferenceStatus.succeeded.value, InferenceStatus.failed.value, InferenceStatus.canceled.value}:
        run.finished_at = transition_at

    db.add(
        InferenceRunEvent(
            run_id=run.id,
            from_status=previous_status,
            to_status=status_value,
            changed_at=transition_at,
            detail=detail,
            error=error,
        )
    )
    return transition_at


def _run_inference_task(
    run_id: int,
    slide_path: str,
    output_path: Path,
    checkpoint: Path,
    threshold: float | None = None,
):
    """Background task: claim a queued inference job, run it, and finalize queue state."""
    claimed_job = _inference_queue.claim()
    if claimed_job is None:
        logger.warning("Inference worker started for run %s but no queued job was available", run_id)
        return

    try:
        _process_inference_job(
            claimed_job,
            fallback_run_id=run_id,
            fallback_slide_path=slide_path,
            fallback_output_path=output_path,
            fallback_checkpoint=checkpoint,
            fallback_threshold=threshold,
        )
    except Exception:
        logger.exception("Inference job %s could not be processed", claimed_job.id)
        _inference_queue.fail(claimed_job.id, "internal")


def _process_inference_job(
    job: JobRecord,
    *,
    fallback_run_id: int,
    fallback_slide_path: str,
    fallback_output_path: Path,
    fallback_checkpoint: Path,
    fallback_threshold: float | None,
):
    """Run the claimed job and update both the DB row and queue record."""
    from app.db.session import make_engine, make_session_factory
    from app.settings import load_settings

    settings = load_settings()
    engine = make_engine(settings.sqlite_path)
    SessionLocal = make_session_factory(engine)
    db = SessionLocal()

    payload = job.payload or {}
    run_id = fallback_run_id
    slide_path = fallback_slide_path
    output_path = fallback_output_path
    checkpoint = fallback_checkpoint
    threshold = fallback_threshold
    run = None
    try:
        if "run_id" in payload:
            run_id = int(payload["run_id"])
        if "slide_path" in payload:
            slide_path = str(payload["slide_path"])
        if "output_path" in payload:
            output_path = Path(payload["output_path"])
        if "checkpoint" in payload:
            checkpoint = Path(payload["checkpoint"])
        if "threshold" in payload:
            threshold = payload["threshold"]

        run = db.get(InferenceRun, run_id)
        if not run:
            _inference_queue.cancel(job.id)
            return
        _record_inference_run_transition(db, run, InferenceStatus.running)
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
            "--batch-size", str(settings.inference_tile_batch_size),
            "--device", settings.inference_device,
            "--level", settings.inference_level,
            "--target-tiles", str(settings.inference_target_tiles),
        ]
        if threshold is not None:
            cmd.extend(["--threshold", str(float(threshold))])

        result = subprocess.run(
            cmd,
            cwd=str(script_path.parent.parent),
            env=_inference_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            diagnostic = _subprocess_diagnostic(result)
            logger.warning(
                "Inference subprocess failed for run %s with exit code %s: %s",
                run_id,
                result.returncode,
                diagnostic,
            )
            run.error_code = "inference_failed"
            run.error_message = _subprocess_failure_message(result.returncode)
            _record_inference_run_transition(
                db,
                run,
                InferenceStatus.failed,
                detail=f"Inference subprocess returned non-zero exit code {result.returncode}.\n{diagnostic}",
                error="inference_failed",
            )
            db.commit()
            _inference_queue.fail(job.id, "inference_failed")
            return

        if not output_path.exists():
            run.error_code = "missing_output"
            run.error_message = "Inference did not produce output."
            _record_inference_run_transition(
                db,
                run,
                InferenceStatus.failed,
                detail="Inference subprocess did not create output JSON",
                error="missing_output",
            )
            db.commit()
            _inference_queue.fail(job.id, "missing_output")
            return
        if output_path.stat().st_size > settings.max_inference_output_bytes:
            run.error_code = "output_too_large"
            run.error_message = "Inference output exceeded server limits."
            _record_inference_run_transition(
                db,
                run,
                InferenceStatus.failed,
                detail="Output JSON exceeded configured size limit",
                error="output_too_large",
            )
            db.commit()
            _inference_queue.fail(job.id, "output_too_large")
            return

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        raw_regions = data.get("regions", [])
        if not isinstance(raw_regions, list):
            raise ValueError("Inference output regions must be a list")
        if len(raw_regions) > MAX_REGIONS_PER_RUN:
            raise ValueError(f"Inference output exceeded max regions ({MAX_REGIONS_PER_RUN})")

        try:
            parsed_regions = [_parse_inference_region(r) for r in raw_regions]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            run.error_code = "malformed_output"
            run.error_message = "Inference output was malformed."
            _record_inference_run_transition(
                db,
                run,
                InferenceStatus.failed,
                detail=f"Inference output validation failed: {exc}",
                error="malformed_output",
            )
            db.commit()
            _inference_queue.fail(job.id, "malformed_output")
            return

        db.add_all(
            Region(
                inference_run_id=run_id,
                x=r["x"],
                y=r["y"],
                w=r["w"],
                h=r["h"],
                score=r["score"],
                label=r.get("label"),
                payload_json=r.get("payload_json"),
            )
            for r in parsed_regions
        )

        run.output_json_path = str(output_path)
        _record_inference_run_transition(db, run, InferenceStatus.succeeded)
        db.commit()
        _inference_queue.ack(job.id)
    except subprocess.TimeoutExpired as exc:
        logger.warning("Inference timed out for run %s", run_id)
        run = run or db.get(InferenceRun, run_id)
        if run:
            diagnostic = _subprocess_diagnostic(exc)
            run.error_code = "timeout"
            run.error_message = "Inference timed out after 1 hour"
            _record_inference_run_transition(
                db,
                run,
                InferenceStatus.failed,
                detail=f"Inference subprocess timed out.\n{diagnostic}",
                error="timeout",
            )
            db.commit()
        _inference_queue.fail(job.id, "timeout")
    except Exception:
        logger.exception("Unhandled inference task failure for run %s", run_id)
        run = run or db.get(InferenceRun, run_id)
        if run:
            run.error_code = "internal"
            run.error_message = "Inference failed due to an internal error."
            _record_inference_run_transition(
                db,
                run,
                InferenceStatus.failed,
                detail="Unhandled exception while processing inference job",
                error="internal",
            )
            db.commit()
        _inference_queue.fail(job.id, "internal")
    finally:
        db.close()


def _queue_inference_run_for_slide(
    *,
    slide: Slide,
    payload: InferenceRunCreate,
    settings,
    db: Session,
) -> InferenceRun:
    slide_path = _resolve_managed_slide_path(slide, settings)

    checkpoint, model_id = _resolve_model_checkpoint(payload.model_file)
    run = InferenceRun(
        slide_id=slide.id,
        model_name=payload.model_name,
        model_version=model_id or payload.model_version,
        status=InferenceStatus.queued.value,
        requested_by_user_id=None,
    )
    db.add(run)
    db.flush()
    _record_inference_run_transition(db, run, InferenceStatus.queued)
    db.commit()
    db.refresh(run)

    output_path = settings.inference_runs_dir / f"{run.id}.json"
    job = None
    try:
        job = _inference_queue.enqueue(
            {
                "run_id": run.id,
                "slide_path": str(slide_path),
                "output_path": str(output_path),
                "checkpoint": str(checkpoint),
                "threshold": payload.threshold,
            }
        )
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
        if job is not None:
            _inference_queue.cancel(job.id)
        run.error_code = "executor_submit_failed"
        run.error_message = "Could not start inference task."
        _record_inference_run_transition(
            db,
            run,
            InferenceStatus.failed,
            detail="ThreadPool executor rejected inference submission",
            error="executor_submit_failed",
        )
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
    if len(slide_ids) > settings.max_batch_inference_items:
        raise AppError(
            ErrorCode.SLIDE_INVALID,
            f"Too many slides in one batch run (max {settings.max_batch_inference_items})",
        )
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
    if len(target_slides) > settings.max_batch_inference_items:
        raise AppError(
            ErrorCode.SLIDE_INVALID,
            f"Too many slides in one folder run (max {settings.max_batch_inference_items})",
        )
    if len(target_slides) > MAX_FOLDER_INFERENCE_SLIDES:
        raise AppError(
            ErrorCode.SLIDE_INVALID,
            f"Too many slides in folder for one run (max {MAX_FOLDER_INFERENCE_SLIDES})",
        )

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
    return {"regions": [_region_to_response(r) for r in regions]}


@router.get("/runs/{run_id}/lifecycle-events", response_model=InferenceRunLifecycleEventListResponse)
def get_inference_run_lifecycle_events(run_id: int, db: Session = Depends(get_db)):
    run = db.get(InferenceRun, run_id)
    if not run:
        raise AppError(ErrorCode.NOT_FOUND, f"Inference run {run_id} not found")

    events = (
        db.query(InferenceRunEvent)
        .filter(InferenceRunEvent.run_id == run_id)
        .order_by(InferenceRunEvent.changed_at.asc(), InferenceRunEvent.id.asc())
        .all()
    )
    return InferenceRunLifecycleEventListResponse(
        events=[
            InferenceRunLifecycleEventResponse(
                id=event.id,
                run_id=event.run_id,
                from_status=event.from_status,
                to_status=event.to_status,
                changed_at=event.changed_at,
                detail=event.detail,
                error=event.error,
            )
            for event in events
        ]
    )


@router.get("/slides/{slide_id}/runs")
def list_slide_inference_runs(slide_id: int, db: Session = Depends(get_db)):
    slide = db.get(Slide, slide_id)
    if not slide:
        raise AppError(ErrorCode.NOT_FOUND, f"Slide {slide_id} not found")
    runs = db.query(InferenceRun).filter(InferenceRun.slide_id == slide_id).order_by(InferenceRun.created_at.desc()).all()
    summaries = _run_summaries(db, runs)
    return {"runs": [_run_to_response(r, summary=summaries.get(r.id)) for r in runs]}


def _run_summaries(db: Session, runs: list[InferenceRun]) -> dict[int, dict]:
    succeeded_run_ids = [run.id for run in runs if run.status == InferenceStatus.succeeded.value]
    if not succeeded_run_ids:
        return {}

    rows = (
        db.query(Region.inference_run_id, Region.label, func.count(Region.id))
        .filter(Region.inference_run_id.in_(succeeded_run_ids))
        .group_by(Region.inference_run_id, Region.label)
        .all()
    )
    summaries = {
        run_id: {"total": 0, "fungus_positive": 0, "fungus_negative": 0}
        for run_id in succeeded_run_ids
    }
    for run_id, label, count in rows:
        summary = summaries[run_id]
        summary["total"] += int(count)
        if label == "fungus_positive":
            summary["fungus_positive"] = int(count)
        elif label == "fungus_negative":
            summary["fungus_negative"] = int(count)
    return summaries


def _run_to_response(
    run: InferenceRun,
    db: Session | None = None,
    *,
    summary: dict | None = None,
) -> InferenceRunResponse:
    if summary is None and db and run.status == InferenceStatus.succeeded.value:
        summary = _run_summaries(db, [run]).get(run.id)
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
