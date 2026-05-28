from __future__ import annotations
from dataclasses import dataclass

class ErrorCode:
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    IO_ERROR = "io_error"
    SLIDE_NOT_FOUND = "slide_not_found"
    SLIDE_INVALID = "slide_invalid"
    SLIDE_PERMISSION = "slide_permission"
    SLIDE_UNREADABLE = "slide_unreadable"
    STORAGE_INCONSISTENT = "storage_inconsistent"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    DB_ERROR = "db_error"
    INTERNAL = "internal"

@dataclass
class AppError(Exception):
    code: str
    message: str
    http_status: int = 400
