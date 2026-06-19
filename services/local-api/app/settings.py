from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from pydantic import BaseModel

class DeploymentMode(str, Enum):
    LOCAL = "local"
    HYBRID = "hybrid"
    CLOUD = "cloud"


class Settings(BaseModel):
    deployment_mode: DeploymentMode
    app_data_dir: Path
    log_dir: Path
    log_level: str
    slides_dir: Path
    inference_runs_dir: Path
    training_runs_dir: Path
    tiles_cache_dir: Path
    sqlite_path: Path
    api_key: str | None = None
    allow_query_api_key: bool = False
    remote_api_base_url: str | None = None
    remote_auth_provider_url: str | None = None
    remote_storage_url: str | None = None
    import_allowed_roots: tuple[Path, ...] = ()
    max_batch_inference_items: int = 64
    max_inference_output_bytes: int = 10_000_000
    inference_tile_batch_size: int = 4
    inference_device: str = "auto"
    inference_level: int | None = None
    inference_target_tiles: int = 30000
    project_root: str = ""


def _parse_deployment_mode(raw: str | None) -> DeploymentMode:
    value = (raw or "").strip().lower() or DeploymentMode.LOCAL.value
    for mode in DeploymentMode:
        if mode.value == value:
            return mode

    allowed = ", ".join(mode.value for mode in DeploymentMode)
    raise RuntimeError(
        f"Invalid APP_DEPLOYMENT_MODE={raw!r}. Expected one of: {allowed}."
    )


def _normalize_filesystem_path(raw: str, *, env_name: str) -> Path:
    value = raw.strip()
    if not value:
        raise RuntimeError(f"{env_name} is empty")
    if "://" in value:
        raise RuntimeError(f"{env_name} must be a filesystem path, not a URL")
    return Path(value).expanduser().resolve()


def _optional_env(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value or None


def _validate_remote_url(
    name: str,
    value: str | None,
    errors: list[str],
    *,
    allowed_schemes: tuple[str, ...] = ("https", "http"),
) -> None:
    if not value:
        return
    parsed = urlparse(value)
    if parsed.scheme not in set(allowed_schemes) or not parsed.netloc:
        scheme_label = "|".join(allowed_schemes)
        errors.append(f"{name} must be an absolute URL with scheme {scheme_label}")
        return
    if parsed.username or parsed.password:
        errors.append(f"{name} must not include embedded credentials")


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


def _parse_positive_int(raw: str | None, *, default: int) -> int:
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError("Expected positive integer value") from exc
    if parsed < 1:
        raise RuntimeError("Expected positive integer value")
    return parsed


def _validate_profile_settings(
    *,
    deployment_mode: DeploymentMode,
    app_data_dir: Path,
    log_dir: Path,
    api_key: str | None,
    allow_query_api_key: bool,
    remote_api_base_url: str | None,
    remote_auth_provider_url: str | None,
    remote_storage_url: str | None,
) -> None:
    errors: list[str] = []

    if deployment_mode is DeploymentMode.LOCAL:
        # Local mode keeps the current single-workstation storage model.
        if not app_data_dir.is_absolute():
            errors.append("APP_DATA_DIR must resolve to an absolute filesystem path")
        if not log_dir.is_absolute():
            errors.append("APP_LOG_DIR must resolve to an absolute filesystem path when set")
    elif deployment_mode is DeploymentMode.HYBRID:
        if not api_key:
            errors.append("APP_API_KEY is required for hybrid mode")
        if allow_query_api_key:
            errors.append("APP_ALLOW_QUERY_API_KEY must be false in hybrid mode")
        if not remote_api_base_url:
            errors.append("APP_REMOTE_API_BASE_URL is required for hybrid mode")
        if not remote_auth_provider_url:
            errors.append("APP_REMOTE_AUTH_PROVIDER_URL is required for hybrid mode")
        _validate_remote_url("APP_REMOTE_API_BASE_URL", remote_api_base_url, errors)
        _validate_remote_url("APP_REMOTE_AUTH_PROVIDER_URL", remote_auth_provider_url, errors)
    elif deployment_mode is DeploymentMode.CLOUD:
        if not api_key:
            errors.append("APP_API_KEY is required for cloud mode")
        if allow_query_api_key:
            errors.append("APP_ALLOW_QUERY_API_KEY must be false in cloud mode")
        if not remote_api_base_url:
            errors.append("APP_REMOTE_API_BASE_URL is required for cloud mode")
        if not remote_auth_provider_url:
            errors.append("APP_REMOTE_AUTH_PROVIDER_URL is required for cloud mode")
        if not remote_storage_url:
            errors.append("APP_REMOTE_STORAGE_URL is required for cloud mode")
        _validate_remote_url("APP_REMOTE_API_BASE_URL", remote_api_base_url, errors)
        _validate_remote_url("APP_REMOTE_AUTH_PROVIDER_URL", remote_auth_provider_url, errors)
        _validate_remote_url(
            "APP_REMOTE_STORAGE_URL",
            remote_storage_url,
            errors,
            allowed_schemes=("https", "http", "s3"),
        )

    if errors:
        details = "; ".join(errors)
        raise RuntimeError(f"Invalid {deployment_mode.value} deployment configuration: {details}.")


def load_settings() -> Settings:
    deployment_mode = _parse_deployment_mode(os.environ.get("APP_DEPLOYMENT_MODE"))

    app_data_dir_raw = os.environ.get("APP_DATA_DIR")
    if not app_data_dir_raw:
        raise RuntimeError(
            f"APP_DATA_DIR is required for {deployment_mode.value} deployment mode"
        )

    app_data_dir = _normalize_filesystem_path(app_data_dir_raw, env_name="APP_DATA_DIR")

    log_dir_raw = os.environ.get("APP_LOG_DIR")
    log_dir = (
        _normalize_filesystem_path(log_dir_raw, env_name="APP_LOG_DIR")
        if log_dir_raw
        else (app_data_dir / "logs")
    )
    log_level = (os.environ.get("APP_LOG_LEVEL") or "INFO").upper()
    slides_dir = app_data_dir / "slides"
    inference_runs_dir = app_data_dir / "inference_runs"
    training_runs_dir = app_data_dir / "training_runs"
    tiles_cache_dir = app_data_dir / "tiles_cache"
    sqlite_path = app_data_dir / "app.db"
    api_key = _optional_env("APP_API_KEY")
    allow_query_api_key = (os.environ.get("APP_ALLOW_QUERY_API_KEY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    remote_api_base_url = _optional_env("APP_REMOTE_API_BASE_URL")
    remote_auth_provider_url = _optional_env("APP_REMOTE_AUTH_PROVIDER_URL")
    remote_storage_url = _optional_env("APP_REMOTE_STORAGE_URL")
    import_allowed_roots = _parse_allowed_roots(os.environ.get("APP_IMPORT_ALLOWED_ROOTS"))
    max_batch_inference_items = _parse_positive_int(
        os.environ.get("APP_MAX_BATCH_INFERENCE_ITEMS"),
        default=64,
    )
    max_inference_output_bytes = _parse_positive_int(
        os.environ.get("APP_MAX_INFERENCE_OUTPUT_BYTES"),
        default=10_000_000,
    )
    inference_tile_batch_size = _parse_positive_int(
        os.environ.get("APP_INFERENCE_TILE_BATCH_SIZE"),
        default=4,
    )
    inference_device = (os.environ.get("APP_INFERENCE_DEVICE") or "auto").strip().lower()
    if inference_device not in {"auto", "cpu", "cuda", "mps"} and not inference_device.startswith("cuda:"):
        raise RuntimeError(
            "APP_INFERENCE_DEVICE must be one of auto, cpu, cuda, cuda:<index>, or mps"
        )
    inference_level_raw = (os.environ.get("APP_INFERENCE_LEVEL") or "").strip().lower()
    if not inference_level_raw or inference_level_raw == "auto":
        inference_level: int | None = None
    else:
        try:
            parsed_level = int(inference_level_raw)
        except ValueError as exc:
            raise RuntimeError("APP_INFERENCE_LEVEL must be 'auto' or a non-negative integer") from exc
        if parsed_level < 0:
            raise RuntimeError("APP_INFERENCE_LEVEL must be 'auto' or a non-negative integer")
        inference_level = parsed_level
    project_root = os.environ.get("APP_PROJECT_ROOT", "")
    inference_target_tiles = _parse_positive_int(
        os.environ.get("APP_INFERENCE_TARGET_TILES"),
        default=30000,
    )

    _validate_profile_settings(
        deployment_mode=deployment_mode,
        app_data_dir=app_data_dir,
        log_dir=log_dir,
        api_key=api_key,
        allow_query_api_key=allow_query_api_key,
        remote_api_base_url=remote_api_base_url,
        remote_auth_provider_url=remote_auth_provider_url,
        remote_storage_url=remote_storage_url,
    )

    return Settings(
        deployment_mode=deployment_mode,
        app_data_dir=app_data_dir,
        log_dir=log_dir,
        log_level=log_level,
        slides_dir=slides_dir,
        inference_runs_dir=inference_runs_dir,
        training_runs_dir=training_runs_dir,
        tiles_cache_dir=tiles_cache_dir,
        sqlite_path=sqlite_path,
        api_key=api_key,
        allow_query_api_key=allow_query_api_key,
        import_allowed_roots=import_allowed_roots,
        max_batch_inference_items=max_batch_inference_items,
        max_inference_output_bytes=max_inference_output_bytes,
        inference_tile_batch_size=inference_tile_batch_size,
        inference_device=inference_device,
        inference_level=inference_level,
        inference_target_tiles=inference_target_tiles,
        project_root=project_root,
        remote_api_base_url=remote_api_base_url,
        remote_auth_provider_url=remote_auth_provider_url,
        remote_storage_url=remote_storage_url,
    )
