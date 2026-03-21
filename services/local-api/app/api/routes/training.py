from __future__ import annotations
import json
import subprocess
import threading
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/training", tags=["training"])

_training_process = None
_training_progress_path: Path | None = None


def _get_train_script_path() -> Path:
    # training.py -> routes -> api -> app -> local-api -> services -> project_root
    base = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    return base / "wsi-fungal-segmentation" / "scripts" / "train.py"


def _get_training_python() -> str:
    import os
    env_py = os.environ.get("INFERENCE_PYTHON") or os.environ.get("TRAINING_PYTHON")
    if env_py:
        return env_py
    base = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    for venv in [
        base / "wsi-fungal-segmentation" / ".venv" / "bin" / "python",
        base / "wsi-fungal-segmentation" / ".venv" / "Scripts" / "python.exe",
        base / ".venv" / "bin" / "python",
        base / ".venv" / "Scripts" / "python.exe",
    ]:
        if venv.exists():
            return str(venv)
    return "python"


class TrainingStartRequest(BaseModel):
    export_root: str


class TrainingStatusResponse(BaseModel):
    status: str  # idle | running | succeeded | failed
    epoch: int | None = None
    train_loss: float | None = None
    train_dice: float | None = None
    val_loss: float | None = None
    val_dice: float | None = None
    best_dice: float | None = None
    checkpoint_path: str | None = None
    error_message: str | None = None


def _run_training_task(export_root: str, progress_path: Path):
    global _training_process
    script_path = _get_train_script_path()
    python_path = _get_training_python()
    cfg_path = script_path.parent.parent / "configs" / "default.yaml"
    _training_progress_path = progress_path

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w") as f:
        json.dump({"status": "starting"}, f)

    cmd = [
        python_path,
        str(script_path),
        "--config", str(cfg_path),
        "--export-root", export_root,
        "--progress-file", str(progress_path),
    ]
    cwd = str(script_path.parent.parent)

    try:
        _training_process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        _, stderr = _training_process.communicate(timeout=86400)  # 24h max
        if _training_process.returncode != 0:
            with open(progress_path, "w") as f:
                json.dump({
                    "status": "failed",
                    "error_message": (stderr or f"Exit code {_training_process.returncode}")[:2000],
                }, f, indent=2)
    except subprocess.TimeoutExpired:
        _training_process.kill()
        with open(progress_path, "w") as f:
            json.dump({"status": "failed", "error_message": "Training timed out"}, f, indent=2)
    except Exception as e:
        with open(progress_path, "w") as f:
            json.dump({"status": "failed", "error_message": str(e)[:2000]}, f, indent=2)
    finally:
        _training_process = None


@router.post("/start", response_model=TrainingStatusResponse)
def start_training(
    request: Request,
    payload: TrainingStartRequest,
):
    global _training_process
    if _training_process is not None and _training_process.poll() is None:
        raise AppError(ErrorCode.CONFLICT, "Training already in progress")

    export_root = Path(payload.export_root)
    if not export_root.exists():
        raise AppError(ErrorCode.IO_ERROR, f"Export folder not found: {export_root}")
    if not export_root.is_dir():
        raise AppError(ErrorCode.IO_ERROR, "Export path must be a directory")

    script_path = _get_train_script_path()
    if not script_path.exists():
        raise AppError(ErrorCode.IO_ERROR, f"Training script not found: {script_path}")

    settings = request.app.state.settings
    settings.training_runs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = settings.training_runs_dir / "current.json"

    thread = threading.Thread(
        target=_run_training_task,
        args=(str(export_root), progress_path),
        daemon=True,
    )
    thread.start()

    return TrainingStatusResponse(status="running", epoch=0)


@router.get("/status", response_model=TrainingStatusResponse)
def get_training_status(request: Request):
    settings = request.app.state.settings
    progress_path = settings.training_runs_dir / "current.json"

    if not progress_path.exists():
        return TrainingStatusResponse(status="idle")

    try:
        with open(progress_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return TrainingStatusResponse(status="idle")

    return TrainingStatusResponse(
        status=data.get("status", "unknown"),
        epoch=data.get("epoch"),
        train_loss=data.get("train_loss"),
        train_dice=data.get("train_dice"),
        val_loss=data.get("val_loss"),
        val_dice=data.get("val_dice"),
        best_dice=data.get("best_dice"),
        checkpoint_path=data.get("checkpoint_path"),
        error_message=data.get("error_message"),
    )
