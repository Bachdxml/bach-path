# app/api/routes/health.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

import os
import sys
import time
import platform
from pathlib import Path

from app.api.deps import get_db
from app.util.exceptions import AppError, ErrorCode

router = APIRouter(prefix="/health", tags=["health"])

# Track process uptime (monotonic; survives wall-clock changes)
_PROCESS_START = time.monotonic()


class HealthResponse(BaseModel):
    status: str  # "ok"
    uptime_seconds: float


class ReadyResponse(BaseModel):
    status: str  # "ready"
    db_ok: bool
    sqlite_path: str
    uptime_seconds: float


class InfoResponse(BaseModel):
    app_name: str
    version: str
    frozen: bool
    python: str
    platform: str
    exe_path: str | None = None
    app_data_dir: str
    sqlite_path: str
    uptime_seconds: float


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse:
    # Liveness: no dependencies, always quick
    return HealthResponse(
        status="ok",
        uptime_seconds=time.monotonic() - _PROCESS_START,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request, db: Session = Depends(get_db)) -> ReadyResponse:
    """
    Readiness: validates we can serve real traffic.
    - Settings are loaded (lifespan already did this)
    - DB connects and answers a trivial query
    - (Optional) verify Alembic version table exists (migrations ran)
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise AppError(ErrorCode.INTERNAL, "Settings not initialized", http_status=500)

    sqlite_path: Path = settings.sqlite_path

    # Basic filesystem sanity for frozen builds (helpful for support)
    if not sqlite_path.parent.exists():
        raise AppError(ErrorCode.IO_ERROR, f"Database directory missing: {sqlite_path.parent}", http_status=500)

    # DB connectivity check
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        raise AppError(ErrorCode.DB_ERROR, f"DB not reachable: {e}", http_status=500)

    # Verify migrations table exists (best-effort; helpful signal)
    # NOTE: Alembic creates alembic_version by default.
    try:
        db.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
    except Exception:
        # If you prefer to be strict, turn this into a 500.
        # For now, mark db_ok true (connectivity ok) but you could fail readiness.
        pass

    return ReadyResponse(
        status="ready",
        db_ok=True,
        sqlite_path=str(sqlite_path),
        uptime_seconds=time.monotonic() - _PROCESS_START,
    )


@router.get("/info", response_model=InfoResponse)
def info(request: Request) -> InfoResponse:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise AppError(ErrorCode.INTERNAL, "Settings not initialized", http_status=500)

    frozen = bool(getattr(sys, "frozen", False))
    exe_path = None
    if frozen:
        # In PyInstaller builds, sys.executable points to the bundled exe
        exe_path = sys.executable

    return InfoResponse(
        app_name="Pathology Local API",
        version=getattr(request.app, "version", "unknown"),
        frozen=frozen,
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        exe_path=exe_path,
        app_data_dir=str(settings.app_data_dir),
        sqlite_path=str(settings.sqlite_path),
        uptime_seconds=time.monotonic() - _PROCESS_START,
    )
