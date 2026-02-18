from __future__ import annotations
from pydantic import BaseModel
import os
from pathlib import Path

class Settings(BaseModel):
    app_data_dir: Path
    log_dir: Path
    slides_dir: Path
    inference_runs_dir: Path
    tiles_cache_dir: Path
    sqlite_path: Path

def load_settings() -> Settings:
    # Required for your CLI/launcher: set APP_DATA_DIR before starting local-api.exe
    data_dir = os.environ.get("APP_DATA_DIR")
    if not data_dir:
        raise RuntimeError("APP_DATA_DIR is not set")

    app_data_dir = Path(data_dir).resolve()

    log_dir = app_data_dir / "logs"
    slides_dir = app_data_dir / "slides"
    inference_runs_dir = app_data_dir / "inference_runs"
    tiles_cache_dir = app_data_dir / "tiles_cache"
    sqlite_path = app_data_dir / "app.db"

    return Settings(
        app_data_dir=app_data_dir,
        log_dir=log_dir,
        slides_dir=slides_dir,
        inference_runs_dir=inference_runs_dir,
        tiles_cache_dir=tiles_cache_dir,
        sqlite_path=sqlite_path,
    )
