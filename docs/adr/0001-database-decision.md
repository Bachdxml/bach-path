# ADR 0001: Database Decision

## Status
Accepted

## Context
Bach Path is desktop-first and the current `services/local-api` implementation uses a local SQLite database stored under the application data directory. The active settings code resolves `sqlite_path` to `app_data_dir / "app.db"`, and the database layer creates the engine directly from that path with SQLite pragmas tuned for local concurrency and durability.

This matches the current product shape: a single-user desktop app launching a local FastAPI service on the same machine. It also aligns with the profile-aware configuration model already in place, where deployment mode can be `local`, `hybrid`, or `cloud`.

We need a decision that preserves the current local SQLite setup while leaving room for future modes that may need a different persistence backend or a remote database service.

## Decision
We will keep SQLite as the default and current database for `local` mode, with the database file rooted in the per-user application data directory.

The local API will continue to own schema initialization, migrations, and database access through its existing service layer. For `hybrid` and `cloud` profiles, the application will treat the database as a profile-specific dependency selected by configuration rather than hard-coding the local SQLite path into business logic.

In practice, that means:
- `local` mode uses `app_data_dir/app.db`.
- `hybrid` mode may continue to use local SQLite for workstation state if needed, but must be able to point at a profile-appropriate shared or remote database when the deployment requires it.
- `cloud` mode will not assume local filesystem-backed persistence and must use the backend selected by its configuration and infrastructure.

## Consequences
- The current local workflow stays simple, deterministic, and offline-friendly.
- Schema creation and migrations remain centralized in the local API, which keeps startup behavior predictable.
- SQLite pragmas such as WAL and a busy timeout continue to provide good single-workstation behavior.
- The path to shared or remote persistence is explicit, but it will require a later adapter or configuration split for non-local modes.
- We avoid prematurely introducing a heavier database stack before the local product boundary needs it.

## Alternatives
- Adopt Postgres now for every mode. This would simplify future shared deployments but add operational overhead immediately and make the local desktop experience heavier than necessary.
- Use in-memory storage for local mode. This would simplify tests but would not meet the product requirement for durable local state.
- Keep SQLite forever for all modes. This preserves simplicity but limits the cloud path and makes multi-user or shared deployment scenarios harder to support cleanly.
- Split each deployment mode into a separate database implementation now. This is flexible, but it adds complexity before the non-local modes are fully defined.

## Notes
This ADR covers the database decision only. Queue, object storage, and auth provider decisions are tracked in separate ADRs so each boundary can evolve independently.
