# Phase 1 Roadmap (0-30 Days)

This document tracks the current Phase 1 execution roadmap in a versioned docs location.

## Sprint Cadence

- Sprint length: 1 week
- Phase 1 target: 4 sprints
- Planning: Monday
- Demo/retro: Friday

## Board

| ID | Title | Stream | Priority | Estimate | Status | Owner | Depends On | Sprint | Done Date | PR/Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| P1-A1 | Define deployment topologies (`local`, `hybrid`, `cloud`) doc | Platform | P0 | S | done | codex | none | S1 | 2026-04-10 | pending-commit |
| P1-A2 | Add env profile loader and config split for `local`/`hybrid`/`cloud` | Platform | P0 | M | done | codex | P1-A1 | S1 | 2026-04-10 | pending-commit |
| P1-A3 | Add ADRs: DB, queue, object storage, auth provider | Platform | P1 | S | done | codex | P1-A1 | S1 | 2026-04-10 | pending-commit |
| P1-B1 | Add queue abstraction interface (`enqueue`, `claim`, `ack`, `fail`, `cancel`) | Platform | P0 | M | done | codex | P1-A2 | S1 | 2026-04-10 | pending-commit |
| P1-B2 | Persist job lifecycle states and timestamps in DB | Platform | P0 | M | todo | unassigned | P1-B1 | S1 |  |  |
| P1-B3 | Add crash recovery reconciliation on API startup | Platform | P0 | M | todo | unassigned | P1-B2 | S2 |  |  |
| P1-B4 | Add cancellation endpoint and cooperative cancellation checks | Platform | P1 | M | todo | unassigned | P1-B2 | S2 |  |  |
| P1-B5 | Add progress reporting contract (`stage`, `percent`, `eta`) | App/API | P1 | M | todo | unassigned | P1-B2 | S2 |  |  |
| P1-C1 | Add request/job correlation IDs in structured logs | Platform | P0 | S | todo | unassigned | P1-B2 | S2 |  |  |
| P1-C2 | Add metrics instrumentation (latency, queue depth, failure rate) | Platform | P0 | M | todo | unassigned | P1-B2 | S2 |  |  |
| P1-C3 | Add readiness endpoint with dependency checks | Platform | P1 | S | todo | unassigned | P1-C2 | S2 |  |  |
| P1-C4 | Add baseline alert definitions and runbook | Platform | P1 | S | todo | unassigned | P1-C2 | S3 |  |  |
| P1-D1 | Add API contract tests for runs/regions/models/training info | QA/Release | P0 | M | todo | unassigned | P1-B2 | S3 |  |  |
| P1-D2 | Add E2E flow test: import -> inference -> overlay render | QA/Release | P0 | L | todo | unassigned | P1-D1 | S3 |  |  |
| P1-D3 | Add gallery/viewer regression tests | QA/Release | P1 | M | todo | unassigned | P1-D2 | S4 |  |  |
| P1-D4 | Enforce CI gate for Phase 1 tests | QA/Release | P0 | M | todo | unassigned | P1-D1 | S4 |  |  |

## Acceptance Criteria

- P1-A1: `docs/architecture/deployment-topologies.md` exists and defines boundaries and supported modes.
- P1-A2: Config profile selection is environment-driven and validated at startup.
- P1-A3: At least 4 ADR files exist under `docs/adr/` and are linked from an index.
- P1-B1: Queue interface implemented and used by inference submit path.
- P1-B2: Job state transitions are persisted and queryable from API.
- P1-B3: Restart simulation test proves orphaned running jobs are reconciled.
- P1-B4: Cancel request moves eligible jobs to `cancelled`; running jobs honor cancellation checkpoints.
- P1-B5: API returns progress payload for running jobs with stable schema.
- P1-C1: Every request/job log entry includes correlation IDs.
- P1-C2: Metrics endpoint exposes counters/histograms for key Phase 1 signals.
- P1-C3: Readiness fails when DB/queue dependencies fail.
- P1-C4: Alert rules documented with owner and response steps.
- P1-D1: Contract tests pass for critical endpoints and failure modes.
- P1-D2: E2E test runs in CI and validates import-through-overlay.
- P1-D3: Regression suite covers viewer state transitions and known edge cases.
- P1-D4: CI blocks merges on failing Phase 1 tests.
