from __future__ import annotations
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.settings import load_settings, Settings
from app.api.router import api_router
from app.api.errors import register_exception_handlers
from app.logging_config import configure_logging
from app.db.init import init_database

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = load_settings()
    app.state.settings = settings

    configure_logging(settings.log_dir)
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
    return app

app = create_app()
