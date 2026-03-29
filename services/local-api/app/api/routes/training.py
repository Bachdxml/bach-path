from __future__ import annotations
import json
import subprocess
import threading
from pathlib import Path

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/training", tags=["training"])

_training_process = None
_training_progress_path: Path | None = None
_training_starting = False
_training_lock = threading.Lock()


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
    status: str  # idle | running | stopped | succeeded | failed
    epoch: int | None = None
    train_loss: float | None = None
    train_dice: float | None = None
    val_loss: float | None = None
    val_dice: float | None = None
    best_dice: float | None = None
    checkpoint_path: str | None = None
    error_message: str | None = None


def _run_training_task(export_root: str, progress_path: Path):
    global _training_process, _training_progress_path, _training_starting
    script_path = _get_train_script_path()
    python_path = _get_training_python()
    cfg_path = script_path.parent.parent / "configs" / "default.yaml"
    _training_progress_path = progress_path

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = progress_path.parent / "training_console.log"
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

    log_file = None
    proc = None
    try:
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _training_process = proc
        _training_starting = False
        try:
            proc.wait(timeout=86400)  # 24h max
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except OSError:
                    pass
                log_file = None

        if proc.returncode != 0:
            code = proc.returncode
            try:
                with open(progress_path) as f:
                    current = json.load(f)
                if current.get("status") in {"stopped", "succeeded"}:
                    return
            except (json.JSONDecodeError, OSError):
                pass
            log_tail = ""
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                pass
            signal_hint = ""
            if code == -9:
                signal_hint = (
                    "Process was killed (SIGKILL, exit -9). "
                    "This is often due to system memory pressure. "
                    "Try smaller training settings (e.g., batch size 1 or smaller image size), "
                    "close other heavy apps, then retry.\n\n"
                )
            elif code == -15:
                signal_hint = "Training stopped by user.\n\n"
            with open(progress_path, "w") as f:
                json.dump(
                    {
                        "status": "stopped" if code == -15 else "failed",
                        "error_message": (signal_hint + (log_tail or f"Exit code {code}"))[:2000],
                    },
                    f,
                    indent=2,
                )
    except subprocess.TimeoutExpired:
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                pass
        with open(progress_path, "w") as f:
            json.dump({"status": "failed", "error_message": "Training timed out"}, f, indent=2)
    except Exception as e:
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                pass
        with open(progress_path, "w") as f:
            json.dump({"status": "failed", "error_message": str(e)[:2000]}, f, indent=2)
    finally:
        _training_process = None
        _training_starting = False


@router.post("/start", response_model=TrainingStatusResponse)
def start_training(
    request: Request,
    payload: TrainingStartRequest,
):
    global _training_process, _training_starting
    with _training_lock:
        if _training_starting or (_training_process is not None and _training_process.poll() is None):
            raise AppError(ErrorCode.CONFLICT, "Training already in progress")
        _training_starting = True

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
    try:
        thread.start()
    except Exception:
        with _training_lock:
            _training_starting = False
        raise

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


@router.post("/stop", response_model=TrainingStatusResponse)
def stop_training(request: Request):
    global _training_process, _training_starting
    if _training_starting and (_training_process is None or _training_process.poll() is not None):
        raise AppError(ErrorCode.CONFLICT, "Training is still starting; try again in a moment")
    if _training_process is None or _training_process.poll() is not None:
        raise AppError(ErrorCode.CONFLICT, "No training process is currently running")

    _training_process.terminate()
    settings = request.app.state.settings
    progress_path = settings.training_runs_dir / "current.json"
    try:
        with open(progress_path, "w") as f:
            json.dump({"status": "running", "error_message": "Stop requested..."}, f, indent=2)
    except OSError:
        pass

    return TrainingStatusResponse(status="running", error_message="Stop requested...")


@router.get("/log")
def training_log_tail(request: Request, tail: int = Query(200, ge=1, le=5000)):
    """Last N lines of the training subprocess stdout/stderr log."""
    settings = request.app.state.settings
    log_path = settings.training_runs_dir / "training_console.log"
    if not log_path.is_file():
        return {"lines": []}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"lines": []}
    lines = text.splitlines()
    return {"lines": lines[-tail:]}
