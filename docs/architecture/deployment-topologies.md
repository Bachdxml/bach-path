# Deployment Topologies

This document defines the supported deployment boundaries for Bach Path and the components that may sit on either side of those boundaries.

## Scope

Bach Path is a desktop-first pathology workflow system. The default experience is a single-user desktop app that launches a local API on the same machine. Other topologies are supported only when explicitly configured.

## Supported Modes

### Local

Use when the desktop app and all stateful services run on one workstation.

- Electron desktop app
- Local FastAPI service
- Local slide files on the same machine or mounted storage
- Local database and cache
- Local model artifacts

Properties:

- Lowest operational complexity
- No network dependency for core workflows
- Best fit for development, individual use, and offline operation

### Hybrid

Use when the desktop app stays local but one or more backend dependencies are provided by a trusted internal environment.

- Desktop app remains on the user machine
- API may remain local or be reached over a trusted LAN/VPN path
- Database, object storage, auth provider, or queue may live in a private environment
- Slide files may remain local while metadata, jobs, or model references are shared remotely

Properties:

- Intended for managed lab or team environments
- Requires explicit network and identity configuration
- Preserves local workstation control over file access

### Cloud

Use when the service plane runs in a hosted environment and the client connects over the network.

- Desktop or thin client connects to remote API endpoints
- Persistence and model storage are remote
- Identity, audit, and access control must be enforced centrally

Properties:

- Highest operational overhead
- Requires stronger security and availability controls
- Only suitable when cloud hosting and data handling are approved

## Architecture Components

- `Electron main process`: starts and supervises the API process in local mode, manages trusted OS integration, and enforces launcher-level checks.
- `Renderer UI`: presents import, gallery, viewer, model, and settings workflows.
- `Preload bridge`: exposes a narrow, whitelisted IPC surface to the renderer.
- `FastAPI service`: owns business logic, validation, persistence, and read/write APIs.
- `Slide storage`: source slide files and derived artifacts such as tiles, previews, or caches.
- `Database`: stores slide metadata, job state, model selection, and other durable records.
- `Queue or job runner`: handles long-running work such as import, inference, and model maintenance where applicable.
- `Object/model storage`: holds deployable model artifacts and generated outputs when not stored locally.
- `Auth provider`: supplies identity and access control in hybrid or cloud modes.

## Trust Boundaries

- Browser renderer to preload bridge: untrusted UI code may only call approved IPC methods.
- Renderer or client to API: every request is validated at the service boundary.
- API to local filesystem: file paths must be normalized and constrained to approved roots or explicit user-selected locations.
- API to database and queue: all state changes are server-side decisions, not client-driven writes.
- API to external services: treat network calls as untrusted and authenticate each dependency explicitly.

## Data Flow Summary

### Import

1. User selects slides in the desktop UI.
2. The API validates paths, file types, and access.
3. Metadata and import state are persisted.
4. Derived previews, caches, or tiles are generated as needed.

### Inference

1. User selects a slide and model.
2. The API loads the slide and resolves the active deploy model.
3. A job is queued or executed.
4. Progress, results, and overlays are written back to durable storage for the UI to read.

### Model Selection and Deployment

1. The API enumerates available deploy artifacts.
2. The user selects a model or default inference checkpoint.
3. The chosen artifact becomes the active inference input for subsequent jobs.

## Security Assumptions

- Local mode assumes the workstation is trusted by the user and protected by OS login and disk controls.
- Local API traffic is loopback-only unless the deployment topology explicitly opens a network boundary.
- Slide paths, filenames, and imported metadata are untrusted and must be validated.
- Cloud and hybrid deployments require explicit authentication, authorization, transport protection, and audit logging.
- Secrets must come from environment or managed secret storage, never from source control.

## Operational Constraints

- The desktop app expects a backend service to be available before most workflows can complete.
- Local mode must not depend on external internet access for core user flows.
- Any remote service used in hybrid or cloud mode must have stable latency and clear failure handling.
- Large slide files and long-running inference tasks may take significant local CPU, memory, disk, or GPU resources.
- The system should degrade safely when derived caches or optional services are unavailable.

## Non-Goals

- Not a multi-tenant public SaaS by default.
- Not a browser-only application.
- Not a general-purpose file manager or annotation platform.
- Not a storage system for unrestricted public uploads.
- Not a guarantee of regulatory compliance in any mode without separate review and controls.
- Not a requirement that all components be cloud-hosted.

