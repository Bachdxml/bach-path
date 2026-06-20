from __future__ import annotations
import os

from app.util.openslide_runtime import configure_openslide_runtime

# Make the openslide-bin DLLs discoverable before openslide is imported. This is a
# no-op off Windows (e.g. the Docker image), where OpenSlide comes from the OS
# (libopenslide0) and is resolved by the dynamic linker.
configure_openslide_runtime()
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.settings import load_settings, Settings
from app.version import read_backend_version
from app.api.router import api_router
from app.api.errors import register_exception_handlers
from app.logging_config import configure_logging
from app.db.init import init_database
from fastapi.middleware.cors import CORSMiddleware


def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw:
        parsed = [v.strip() for v in raw.split(",") if v.strip()]
        if parsed:
            return parsed
    return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cors_origins() -> list[str]:
    origins = _csv_env("APP_CORS_ORIGINS", ["http://127.0.0.1", "http://localhost"])
    if _bool_env("APP_CORS_ALLOW_FILE_ORIGIN", False):
        for origin in _csv_env("APP_CORS_FILE_ORIGINS", ["null"]):
            if origin not in origins:
                origins.append(origin)
    return origins


def _cors_methods() -> list[str]:
    return _csv_env("APP_CORS_METHODS", ["GET", "POST", "PATCH", "DELETE"])


def _cors_headers() -> list[str]:
    return _csv_env(
        "APP_CORS_HEADERS",
        ["Content-Type", "X-API-Key", "X-Request-Id"],
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings: Settings = load_settings()
    except RuntimeError as exc:
        raise RuntimeError(f"API startup failed during settings validation: {exc}") from exc
    app.state.settings = settings

    configure_logging(settings.log_dir, settings.log_level)
    init_database(settings)

    yield

def create_app() -> FastAPI:
    app = FastAPI(
        title="Pathology Local API",
        # Sourced from the baked VERSION file so the reported version always
        # reflects the source this image was built from (spec M8/M10).
        version=read_backend_version(),
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=_cors_methods(),
        allow_headers=_cors_headers(),
    )
    return app

app = create_app()
