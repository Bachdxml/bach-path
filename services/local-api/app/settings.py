from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    app_data_dir: Path
    log_dir: Path
    log_level: str
    slides_dir: Path
    inference_runs_dir: Path
    training_runs_dir: Path
    tiles_cache_dir: Path
    sqlite_path: Path
    api_key: str | None = None
    import_allowed_roots: tuple[Path, ...] = ()


def _parse_allowed_roots(raw: str | None) -> tuple[Path, ...]:
    if not raw:
        return ()

    roots: list[Path] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        root = Path(value)
        if not root.is_absolute():
            raise RuntimeError("APP_IMPORT_ALLOWED_ROOTS entries must be absolute paths")
        roots.append(root.resolve())
    return tuple(roots)

def load_settings() -> Settings:
    # Required for your CLI/launcher: set APP_DATA_DIR before starting local-api.exe
    data_dir = os.environ.get("APP_DATA_DIR")
    if not data_dir:
        raise RuntimeError("APP_DATA_DIR is not set")

    app_data_dir = Path(data_dir).resolve()

    log_dir_raw = os.environ.get("APP_LOG_DIR")
    log_dir = Path(log_dir_raw).resolve() if log_dir_raw else (app_data_dir / "logs")
    log_level = (os.environ.get("APP_LOG_LEVEL") or "INFO").upper()
    slides_dir = app_data_dir / "slides"
    inference_runs_dir = app_data_dir / "inference_runs"
    training_runs_dir = app_data_dir / "training_runs"
    tiles_cache_dir = app_data_dir / "tiles_cache"
    sqlite_path = app_data_dir / "app.db"
    api_key = (os.environ.get("APP_API_KEY") or "").strip() or None
    import_allowed_roots = _parse_allowed_roots(os.environ.get("APP_IMPORT_ALLOWED_ROOTS"))

    return Settings(
        app_data_dir=app_data_dir,
        log_dir=log_dir,
        log_level=log_level,
        slides_dir=slides_dir,
        inference_runs_dir=inference_runs_dir,
        training_runs_dir=training_runs_dir,
        tiles_cache_dir=tiles_cache_dir,
        sqlite_path=sqlite_path,
        api_key=api_key,
        import_allowed_roots=import_allowed_roots,
    )
