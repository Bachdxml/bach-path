# ADR 0003: Object Storage Strategy

Status: Accepted

## Context

[`docs/architecture/deployment-topologies.md`](/Users/ryanvu/Documents/bach-path/docs/architecture/deployment-topologies.md) defines three supported deployment modes: local, hybrid, and cloud. Those modes have different trust boundaries and operational expectations, but they all need a consistent way to store and retrieve large binary assets.

The application handles three classes of objects:

- Slide source files.
- Derived artifacts, including tiles, previews, thumbnails, overlays, and cacheable analysis outputs.
- Model artifacts, including deployable checkpoints, bundles, and other versioned inference assets.

The storage strategy needs to satisfy these constraints:

- Local mode must work without internet access and without mandatory external infrastructure.
- Hybrid mode may place durable storage in a trusted internal environment while the desktop app stays local.
- Cloud mode must support remote persistence with centralized identity, audit, and access control.
- The same application workflows should work across modes without coupling business logic to a specific backend vendor.
- Large files and generated artifacts need predictable lifecycle management, cleanup, and replay behavior.

## Decision

Use a storage abstraction with two concrete backend families:

1. **Filesystem-backed storage for local mode**
   - Slides, derived artifacts, and model artifacts are stored on the workstation or on mounted storage.
   - Paths are resolved through the API and constrained to approved roots or explicit user-selected locations.
   - This keeps local mode fully offline-capable and avoids introducing object storage infrastructure as a hard dependency.

2. **Object-storage-backed storage for hybrid and cloud modes**
   - Slides, derived artifacts, and model artifacts are stored in an S3-compatible object store or an equivalent managed object service.
   - The API addresses objects through logical keys, not physical filesystem paths.
   - Buckets or logical namespaces are separated by purpose so slides, derived artifacts, and model artifacts can use independent retention and access policies.

Across all modes, the application uses the same logical storage categories:

- `slides` for original uploads or imported source files.
- `derived` for generated previews, tiles, caches, and analysis outputs.
- `models` for deployable artifacts and versioned inference assets.

Operational rules:

- Source slides are treated as durable inputs and are never overwritten in place by derived processing.
- Derived artifacts are disposable and may be regenerated from source slides and model artifacts.
- Model artifacts are versioned and immutable once published for inference use.
- The API remains the sole authority for reads, writes, retention decisions, and cleanup.

## Consequences

### Positive

- Local mode stays simple, offline-friendly, and easy to run on a single workstation.
- Hybrid and cloud modes can share the same application behavior while swapping storage backends.
- Object lifecycle policies can be tuned independently for slides, derived artifacts, and model artifacts.
- The abstraction reduces vendor lock-in and makes it easier to support S3-compatible services, private object stores, or future backends.
- Derived artifact regeneration becomes explicit and easier to reason about.

### Negative

- The application needs a storage abstraction layer and backend-specific implementation details.
- Hybrid and cloud deployments must manage credentials, permissions, and network failures for object storage.
- Local filesystem storage and object storage do not have identical semantics, so some behavior must be normalized in the API.
- Cleanup, consistency, and eventual-read behavior for remote object stores must be handled carefully.

## Alternatives

### Store Everything on the Local Filesystem

This would keep the implementation simple, but it does not scale to hybrid or cloud deployments and makes shared, centrally managed storage difficult.

### Store Everything in Object Storage

This would unify storage semantics, but it would impose unnecessary infrastructure on local mode and conflict with the offline-first requirement.

### Use Different Storage Systems Per Artifact Type

For example, slides on filesystem, derived artifacts in cache storage, and model artifacts in a registry. This introduces avoidable complexity and makes cross-mode behavior harder to reason about.

### Use a Database as the Primary Blob Store

This would simplify transactional linking between records and blobs, but it is a poor fit for large slide and model binaries and would create storage and performance concerns.

## Notes

- This ADR intentionally follows the deployment boundaries in `docs/architecture/deployment-topologies.md`.
- The implementation should preserve a single logical API for object access even when the physical backend changes by mode.
