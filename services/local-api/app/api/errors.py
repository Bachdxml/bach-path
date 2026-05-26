from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import logging
import uuid

from app.util.exceptions import AppError, ErrorCode

logger = logging.getLogger("app")


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def _validation_details(exc: RequestValidationError) -> list[dict[str, object]]:
    details = []
    for error in exc.errors()[:5]:
        details.append(
            {
                "loc": list(error.get("loc", ())),
                "msg": error.get("msg", "Invalid value"),
                "type": error.get("type", "value_error"),
            }
        )
    return details


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        rid = _request_id(request)
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

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError):
        rid = _request_id(request)
        logger.info(
            "RequestValidationError path=%s rid=%s",
            request.url.path,
            rid,
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed. Check the submitted values and try again.",
                    "request_id": rid,
                    "details": _validation_details(exc),
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        rid = _request_id(request)
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
        rid = _request_id(request)
        logger.exception("DBError rid=%s path=%s", rid, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": ErrorCode.DB_ERROR, "message": "Database operation failed", "request_id": rid}},
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        rid = _request_id(request)
        logger.exception("UnhandledError rid=%s path=%s", rid, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": ErrorCode.INTERNAL, "message": "Internal error", "request_id": rid}},
        )
