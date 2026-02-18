from __future__ import annotations
import enum

class UserRole(str, enum.Enum):
    admin = "admin"
    pathologist = "pathologist"
    technician = "technician"
    viewer = "viewer"

class InferenceStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"
