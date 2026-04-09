# Desktop App Architecture

This Electron app uses a layered renderer architecture to keep responsibilities clear and testable.

## Layers

- `index.js`: Electron main process (window lifecycle, local API process management, trusted IPC handlers).
- `preload.js`: secure bridge that exposes whitelisted IPC methods to the renderer.
- `js/core.bach-path.js`: shared runtime namespace and event bus (`window.BachPath`).
- `js/api.js`: API service contract registered as `BachPath.services.slidesApi`.
- Feature modules:
  - `js/app.js`: shell orchestration (tabs, theme, API health, settings).
  - `js/import.js`: slide import workflow.
  - `js/gallery.js`: slide browsing, filtering, selection, batch actions.
  - `js/viewer.js`: slide viewer overlay, inference overlays, export actions.
  - `js/models-tab.js`: training/deploy info surface.

## Contracts

Use `window.BachPath` for cross-module communication:

- Services: `BachPath.services.*` for shared dependencies (for example `slidesApi`).
- Features: `BachPath.features.*` for feature capabilities (`refresh`, `open`, etc.).
- Events: `BachPath.emit(eventName, detail)` and `BachPath.on(eventName, listener)`.

Legacy globals are still exposed for compatibility, but new code should prefer `BachPath` contracts.

## Standards

- Keep each module scoped to one domain concern.
- Avoid creating new renderer globals unless required for backward compatibility.
- Resolve feature-to-feature calls through `BachPath.features` instead of direct global function calls.
- Register shared services once and consume via dependency lookup (`BachPath.services`).
