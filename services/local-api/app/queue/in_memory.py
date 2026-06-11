from __future__ import annotations

from collections import deque
from datetime import timedelta
from threading import RLock
from typing import Any
from uuid import uuid4

from .interface import JobRecord, JobStatus, QueueInterface, _update_job, _utc_now

_COMPLETED_JOB_TTL_SECS = 300
_TERMINAL_STATUSES = frozenset({JobStatus.succeeded, JobStatus.failed, JobStatus.canceled})


class InMemoryQueue(QueueInterface):
    def __init__(self) -> None:
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._pending: deque[str] = deque()

    def enqueue(self, payload: dict[str, Any]) -> JobRecord:
        job = JobRecord(id=uuid4().hex, payload=dict(payload))
        with self._lock:
            self._evict_expired()
            self._jobs[job.id] = job
            self._pending.append(job.id)
            return job

    def claim(self) -> JobRecord | None:
        with self._lock:
            self._evict_expired()
            while self._pending:
                job_id = self._pending.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status is not JobStatus.queued:
                    continue
                now = _utc_now()
                claimed = _update_job(
                    job,
                    status=JobStatus.running,
                    attempts=job.attempts + 1,
                    claimed_at=now,
                    updated_at=now,
                )
                self._jobs[job_id] = claimed
                return claimed
            return None

    def ack(self, job_id: str) -> JobRecord | None:
        return self._finish(job_id, JobStatus.succeeded)

    def fail(self, job_id: str, error: str | None = None) -> JobRecord | None:
        with self._lock:
            self._evict_expired()
            job = self._jobs.get(job_id)
            if job is None or job.status in {JobStatus.succeeded, JobStatus.canceled}:
                return None
            now = _utc_now()
            failed = _update_job(
                job,
                status=JobStatus.failed,
                finished_at=now,
                error=error,
                updated_at=now,
            )
            self._jobs[job_id] = failed
            return failed

    def cancel(self, job_id: str) -> JobRecord | None:
        return self._finish(job_id, JobStatus.canceled)

    def _finish(self, job_id: str, status: JobStatus) -> JobRecord | None:
        with self._lock:
            self._evict_expired()
            job = self._jobs.get(job_id)
            if job is None or job.status in {JobStatus.succeeded, JobStatus.failed, JobStatus.canceled}:
                return None
            now = _utc_now()
            finished = _update_job(job, status=status, finished_at=now, updated_at=now)
            self._jobs[job_id] = finished
            return finished

    def _evict_expired(self) -> None:
        """Lazily purge terminal jobs older than the TTL.

        Called from every public method while the lock is held, so no
        per-job timer threads are needed (see code-review finding #10).
        """
        cutoff = _utc_now() - timedelta(seconds=_COMPLETED_JOB_TTL_SECS)
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in _TERMINAL_STATUSES
            and job.finished_at is not None
            and job.finished_at <= cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]
