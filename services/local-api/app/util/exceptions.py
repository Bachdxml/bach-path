from __future__ import annotations
from enum import Enum


class ErrorCode(str, Enum):
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    IO_ERROR = "io_error"
    SLIDE_NOT_FOUND = "slide_not_found"
    SLIDE_INVALID = "slide_invalid"
    SLIDE_PERMISSION = "slide_permission"
    SLIDE_UNREADABLE = "slide_unreadable"
    STORAGE_INCONSISTENT = "storage_inconsistent"
    DB_ERROR = "db_error"
    INTERNAL = "internal"


_CODE_STATUS: dict[str, int] = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.SLIDE_NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.SLIDE_PERMISSION: 403,
}


class AppError(Exception):
    def __init__(self, code: str, message: str, http_status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status if http_status is not None else _CODE_STATUS.get(code, 400)
