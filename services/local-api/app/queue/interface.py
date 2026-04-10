from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True, frozen=True)
class JobRecord:
    id: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.queued
    attempts: int = 0
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    claimed_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class QueueInterface(ABC):
    @abstractmethod
    def enqueue(self, payload: dict[str, Any]) -> JobRecord: ...

    @abstractmethod
    def claim(self) -> JobRecord | None: ...

    @abstractmethod
    def ack(self, job_id: str) -> JobRecord | None: ...

    @abstractmethod
    def fail(self, job_id: str, error: str | None = None) -> JobRecord | None: ...

    @abstractmethod
    def cancel(self, job_id: str) -> JobRecord | None: ...


def _update_job(job: JobRecord, *, updated_at: datetime | None = None, **changes: Any) -> JobRecord:
    return replace(job, updated_at=updated_at or _utc_now(), **changes)
