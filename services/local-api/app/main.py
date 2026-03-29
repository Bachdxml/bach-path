from __future__ import annotations
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.settings import load_settings, Settings
from app.api.router import api_router
from app.api.errors import register_exception_handlers
from app.logging_config import configure_logging
from app.db.init import init_database
from fastapi.middleware.cors import CORSMiddleware


def _cors_origins() -> list[str]:
    raw = os.environ.get("APP_CORS_ORIGINS")
    if raw:
        parsed = [v.strip() for v in raw.split(",") if v.strip()]
        if parsed:
            return parsed
    # Electron desktop (file:// / null origin) + localhost dev defaults.
    return ["null", "http://127.0.0.1", "http://localhost"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = load_settings()
    app.state.settings = settings

    configure_logging(settings.log_dir, settings.log_level)
    init_database(settings)

    yield

def create_app() -> FastAPI:
    app = FastAPI(
        title="Pathology Local API",
        version="0.1.0",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

app = create_app()
