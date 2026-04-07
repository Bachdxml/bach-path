from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import logging
import uuid

from app.util.exceptions import AppError, ErrorCode

logger = logging.getLogger("app")

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        # Put details in the message line (stderr formatters often ignore `extra`).
        # NOT_FOUND is common during WSI viewing (tile probes, race conditions) — avoid WARNING spam.
        line = f"{exc.code}: {exc.message} | {request.method} {request.url.path} | rid={rid}"
        if exc.code == ErrorCode.NOT_FOUND:
            logger.debug("AppError %s", line)
        elif exc.code in (
            ErrorCode.SLIDE_UNREADABLE,
            ErrorCode.STORAGE_INCONSISTENT,
            ErrorCode.DB_ERROR,
            ErrorCode.INTERNAL,
        ) or exc.http_status >= 500:
            logger.warning("AppError %s", line)
        else:
            logger.info("AppError %s", line)
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": rid,
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        logger.info(
            "HTTPException status=%s path=%s rid=%s",
            exc.status_code,
            request.url.path,
            rid,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "http_error",
                    "message": str(exc.detail),
                    "request_id": rid,
                }
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def db_error_handler(request: Request, exc: SQLAlchemyError):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        logger.exception("DBError rid=%s path=%s", rid, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": ErrorCode.DB_ERROR, "message": "Database operation failed", "request_id": rid}},
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        logger.exception("UnhandledError rid=%s path=%s", rid, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": ErrorCode.INTERNAL, "message": "Internal error", "request_id": rid}},
        )
