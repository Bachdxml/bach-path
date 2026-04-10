# ADR 0002: Queue Abstraction

## Status
Accepted

## Context
Bach Path currently executes long-running work synchronously in the local service path. That matches the current codebase reality and keeps the first implementation simple: the API can validate a request, perform the work inline, persist results, and return state without introducing queue infrastructure, background workers, or cross-process orchestration.

Roadmap item P1-B1 calls for a queue abstraction next. We need a design that preserves today's synchronous behavior while creating a stable seam for future asynchronous execution. The abstraction should let us move from inline execution to queued jobs without changing business logic, UI contracts, or persistence shapes more than necessary.

## Decision
We will introduce a queue abstraction layer now, but we will keep the default implementation synchronous.

The abstraction will expose a small job-oriented interface for enqueuing work, observing status, and reporting completion or failure. Under current conditions, the default adapter will execute work immediately in-process and persist the resulting state exactly as today. Future adapters may route the same job contract to a real queue and worker pool.

This ADR intentionally separates the execution contract from the execution mechanism. Callers should depend on the abstraction, not on direct inline execution details.

## Consequences
- Existing flows remain simple and deterministic during the transition.
- We avoid a premature dependency on queue infrastructure before there is a confirmed operational need.
- Business logic can be reused when the implementation switches to an async worker model.
- The synchronous adapter becomes the baseline behavior and the fallback path if queue infrastructure is unavailable.
- The abstraction adds a small amount of indirection and a modest upfront design cost.

## Migration Plan Cues
1. Define the queue/job interface around the operations we already need: submit, inspect status, mark success, mark failure, and retrieve results.
2. Route current synchronous execution through that interface so callers stop depending on inline execution details.
3. Keep persistence and UI state compatible with both immediate completion and later asynchronous completion.
4. Add a queue-backed adapter only when P1-B1 work requires it, leaving the synchronous adapter available for local mode and tests.
5. Expand observability and retries only after the abstraction is in place and proven by real jobs.

## Notes
The goal is not to add queue infrastructure today. The goal is to make queue support an implementation detail we can add next without reworking the application boundary.
