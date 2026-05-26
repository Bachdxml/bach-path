# QA Report - Bach Path

Date: 2026-05-26  
Scope: Repository inspection, setup validation, backend/API tests, Electron startup smoke, direct API workflow testing for import, gallery data, model discovery, inference, review status, and deletion. No source files were intentionally modified; only this report was added.

## Summary of App Health

Bach Path is a local pathology desktop workflow app. The Electron desktop shell starts a local FastAPI backend, imports whole-slide/raster slide files, displays imported slides in a gallery/viewer, discovers deployed model weights, and runs fungal inference through the `wsi-fungal-segmentation` scripts.

Overall health: **usable for core local workflows, with setup and startup rough edges**.

Post-QA remediation update: the confirmed code-level findings from this report were rechecked with subagents and fixed where appropriate. The inference run-listing finding was determined to be a false positive in the current code. The broken `.venv` remains a local environment repair item, not a source-code fix.

The backend API test suite passes, desktop packaging succeeds, and a direct end-to-end API workflow succeeded: import PNG slide, list it, read metadata, fetch thumbnail, discover model, run inference, persist result/regions, update review status, and delete the slide. The main risks found are a stale project venv, missing migration assets, renderer API calls that happen before the generated API key is installed, placeholder desktop tests, and inconsistent error response shapes for validation errors.

## Repository Structure and Workflows

- `apps/desktop`: Electron desktop app, renderer HTML/CSS/JS, local API process launcher.
- `services/local-api`: FastAPI app, SQLite persistence, slide import/storage, inference run orchestration, health/training/model APIs.
- `wsi-fungal-segmentation`: model training/evaluation/export and inference scripts.
- `training_data`: expected local training-data dropzone.
- `docs`: ADRs, roadmap, deployment notes.

Core user workflow:

1. Start desktop app with `cd apps/desktop && npm start`.
2. Electron starts the local API with `APP_DATA_DIR`, `APP_LOG_DIR`, generated `APP_API_KEY`, and query API-key support for images.
3. User imports `.svs`, `.tif`, `.tiff`, or `.png` files.
4. User opens Gallery, selects a slide, reviews metadata/thumbnail/viewer.
5. User selects deployed `.pth.gz` or `.pt.gz` model.
6. User runs inference and reviews persisted overlay regions/results.

## Commands Run

```bash
pwd
rg --files -g '!*node_modules*' -g '!*.png' -g '!*.jpg' -g '!*.jpeg' -g '!*.gif' -g '!*.svg' | head -200
find . -maxdepth 3 -name package.json -o -name pyproject.toml -o -name requirements.txt -o -name README.md -o -name .env.example -o -name docker-compose.yml -o -name Dockerfile
git status --short
sed -n '1,240p' README.md
sed -n '1,220p' apps/desktop/package.json
sed -n '1,240p' services/local-api/requirements.txt
sed -n '1,260p' apps/desktop/index.js
sed -n '1,260p' apps/desktop/js/api.js
sed -n '1,320p' apps/desktop/js/app.js
sed -n '1,260p' services/local-api/app/main.py
sed -n '1,260p' services/local-api/app/settings.py
sed -n '1,320p' apps/desktop/index.html
sed -n '1,340p' apps/desktop/js/import.js
sed -n '1,340p' services/local-api/app/api/routes/slides.py
sed -n '1,340p' services/local-api/app/api/routes/inference.py
test -d apps/desktop/node_modules && echo desktop_node_modules=yes || echo desktop_node_modules=no
test -x services/local-api/.venv/bin/python && services/local-api/.venv/bin/python --version || python3 --version
test -x services/local-api/.venv/bin/pytest && echo api_pytest=yes || echo api_pytest=no
find . -maxdepth 3 -name '.env*' -o -name 'alembic.ini' -o -path './services/local-api/alembic/*'
npm test
npm run pack
.venv/bin/pytest
python3 scripts/qa_smoke.py
python3.13 -m pytest
python3.13 -m pip show fastapi pillow pytest
python3.13 -m app.cli --host 127.0.0.1 --port 8876 --data-dir /private/tmp/bach-path-qa-api --log-dir /private/tmp/bach-path-qa-api/logs
curl -s -i -H 'x-api-key: qa-key' http://127.0.0.1:8876/health
curl -s -i http://127.0.0.1:8876/health
curl -s -i -H 'x-api-key: qa-key' http://127.0.0.1:8876/slides
curl -s -i http://127.0.0.1:8876/slides
curl -s -i -H 'x-api-key: qa-key' http://127.0.0.1:8876/inference/models
python3.13 -c "from PIL import Image; import pathlib; p=pathlib.Path('/private/tmp/bach-path-qa-slide.png'); Image.new('RGB',(640,480),(220,230,240)).save(p); print(p)"
curl -s -i -H 'x-api-key: qa-key' -H 'content-type: application/json' -d '{"file_paths":["/private/tmp/bach-path-qa-slide.png"],"title":"QA collection","source_type":"files"}' http://127.0.0.1:8876/slides/import-collection
curl -s -i -H 'x-api-key: qa-key' http://127.0.0.1:8876/slides/1/metadata
curl -s -i -H 'x-api-key: qa-key' 'http://127.0.0.1:8876/slides/1/thumbnail?size=120'
curl -s -i -H 'x-api-key: qa-key' http://127.0.0.1:8876/slides/1/deepzoom.dzi
curl -s -i -H 'x-api-key: qa-key' -H 'content-type: application/json' -d '{"model_file":"models/deploy-fungus.pth.gz","threshold":0.1}' http://127.0.0.1:8876/inference/slides/1/run
curl -s -i -H 'x-api-key: qa-key' http://127.0.0.1:8876/inference/runs/1
curl -s -i -H 'x-api-key: qa-key' http://127.0.0.1:8876/inference/runs/1/regions
curl -s -i -H 'x-api-key: qa-key' -X DELETE http://127.0.0.1:8876/slides/1
npm install --dry-run
python3.13 -m pip install --dry-run -r services/local-api/requirements.txt
python3.13 -m pip install --dry-run -r wsi-fungal-segmentation/requirements.txt
npm start
```

## Validation Results

### Passing

- `apps/desktop/node_modules` exists.
- `npm install --dry-run` reports desktop dependencies are up to date.
- `npm run pack` succeeds and creates a macOS package directory. Code signing is skipped because no signing identity is configured.
- `python3.13 -m pytest` from `services/local-api` passes: **49 passed, 3 warnings**.
- API starts successfully when launched with required `--data-dir` and `--log-dir`.
- Protected API routes reject missing API key with 401.
- Public `/health` returns 200 without auth.
- `/inference/models` returns deployed model `models/deploy-fungus.pth.gz`.
- Direct API import/list/metadata/thumbnail/review/inference/regions/delete flow works.
- Inference completed on a generated 640x480 PNG:
  - queued -> running -> succeeded
  - summary: `{"total":1,"fungus_positive":1,"fungus_negative":0}`
  - region persisted and returned from `/inference/runs/1/regions`

### Blocked or Partial

- Full visual click-through in the Electron renderer was limited because the available Browser plugin did not expose the required Node REPL `js` execution tool. I performed Electron startup smoke and direct API workflow testing instead.
- `python3 scripts/qa_smoke.py` from repo root failed in the default `python3` environment because `PIL` was missing there, even though `Pillow` is installed in the Python 3.13 environment.
- The local `services/local-api/.venv` is stale/broken because its scripts point to `/Users/ryanvu/bach-path/...` instead of `/Users/ryanvu/Documents/bach-path/...`.

## Bugs and Findings

### High - Project-local API venv is broken after checkout move

Severity: High  
Fix status: Needs local environment repair  
Area: Setup and tests

Reproduction steps:

1. From `services/local-api`, run `.venv/bin/pytest`.

Expected behavior:

- The documented local venv runs pytest successfully.

Actual behavior:

```text
zsh:1: .venv/bin/pytest: bad interpreter: /Users/ryanvu/bach-path/services/local-api/.venv/bin/python: no such file or directory
```

Root cause:

- The checked-in/local `.venv` scripts were created at a different filesystem path and contain absolute shebangs.

Recommended fix:

- Recreate the venv at the current checkout path:
  `cd services/local-api && rm -rf .venv && python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Do not rely on a moved/cached `.venv` for onboarding.
- Source-level follow-up completed: `scripts/qa_smoke.py` now performs clearer environment checks, and `QA.md` documents `python3.13 scripts/qa_smoke.py`.

### High - Electron renderer makes protected API calls before API key is installed

Severity: High  
Fix status: Fixed  
Area: Desktop startup/auth/frontend contract

Reproduction steps:

1. Run `cd apps/desktop && npm start`.
2. Watch terminal/API logs during renderer startup.

Expected behavior:

- Renderer waits for `api-ready`, sets API base/key, then calls protected routes.
- Startup logs should not contain auth failures from first-party renderer calls.

Actual behavior:

```text
HTTPException status=401 path=/training/info
HTTPException status=401 path=/slides
```

Root cause:

- `apps/desktop/js/models-tab.js` calls `loadDeployInfo()` on `DOMContentLoaded`.
- `apps/desktop/js/gallery.js` calls `loadGallery()` inside `initGallery()`.
- Both can run before `apps/desktop/js/app.js` receives `window.electronAPI.onApiReady(...)` and calls `slidesApi.setApiKey(...)`.

User impact:

- Initial Models/Gallery panels can show transient failure or stale error messaging even though the API is running.
- Logs are noisy and can mask real auth issues.

Recommended fix:

- Gate protected API calls behind a shared API-ready event/promise.
- Alternatively remove the automatic `DOMContentLoaded` calls from feature modules and let `app.js` trigger them after `api-ready`.

Fix applied:

- Added API-ready coordination helpers in `apps/desktop/js/core.bach-path.js`.
- Guarded Gallery and Models protected API loads in `apps/desktop/js/gallery.js` and `apps/desktop/js/models-tab.js`.
- Electron startup smoke confirmed the API becomes healthy on port 8765 without the previous startup 401 log noise.

### Medium - Validation errors return raw FastAPI `detail` instead of normalized app error shape

Severity: Medium  
Fix status: Fixed  
Area: API error handling/frontend contract

Reproduction steps:

```bash
curl -s -i -H 'x-api-key: qa-key' -H 'content-type: application/json' \
  -d '{"file_paths":[],"title":"Empty"}' \
  http://127.0.0.1:8876/slides/import-collection
```

Expected behavior:

- Error response matches the app-wide format consumed by `parseErrorResponse()`:
  `{"error":{"code":"...","message":"..."}}`

Actual behavior:

```json
{"detail":[{"type":"too_short","loc":["body","file_paths"],"msg":"List should have at least 1 item after validation, not 0","input":[],"ctx":{"field_type":"List","min_length":1,"actual_length":0}}]}
```

Root cause:

- Request validation exceptions are not normalized by the registered API error handlers.

User impact:

- Frontend can display raw JSON text instead of a friendly message for validation failures.

Recommended fix:

- Add a FastAPI `RequestValidationError` handler in `app/api/errors.py` that returns the same `error.code/message/request_id` shape as other API errors.

Fix applied:

- Added a normalized `RequestValidationError` handler in `services/local-api/app/api/errors.py`.
- Added `services/local-api/tests/test_validation_errors.py` covering body, query, and path validation.

### Medium - Alembic is listed but migration assets are missing

Severity: Medium  
Fix status: Fixed  
Area: Database/deployment

Reproduction steps:

1. Start API.
2. Observe startup logs.
3. Run `find services/local-api -maxdepth 3 -type d -name alembic -o -name versions -o -name 'alembic.ini'`.

Expected behavior:

- Migrations are present and run, or docs clearly state metadata bootstrap is intentional.

Actual behavior:

```text
Alembic assets not found; falling back to metadata.create_all().
```

Root cause:

- `alembic` is in requirements, but no Alembic migration directory/config was found.

User impact:

- Existing user databases may not upgrade safely as schemas evolve.
- Deployment behavior is less predictable than migration-based startup.

Recommended fix:

- Add Alembic config and initial migration, or remove Alembic dependency and explicitly document `create_all()` as the development-only strategy.

Fix applied:

- Added `services/local-api/alembic.ini`, `services/local-api/alembic/env.py`, and initial revision `20260526_0001_initial_schema.py`.
- Added `services/local-api/tests/test_migrations_assets.py`.
- Startup now runs Alembic instead of the missing-assets fallback while preserving the existing compatibility bootstrap path.

### Medium - Desktop test script always fails

Severity: Medium  
Fix status: Fixed  
Area: Automated tests

Reproduction steps:

```bash
cd apps/desktop
npm test
```

Expected behavior:

- Runs meaningful desktop unit/smoke tests, or exits successfully if no tests are configured.

Actual behavior:

```text
Error: no test specified
```

Root cause:

- `package.json` contains the default placeholder test script.

Recommended fix:

- Add at least a smoke test for renderer modules and IPC preload shape, or change the script to a documented no-op only if intentional.

Fix applied:

- Replaced the placeholder `npm test` with a lightweight `node --check` syntax pass over Electron entrypoints and renderer JS.

### Medium - Repo smoke script depends on default `python3` environment instead of documented Python 3.13 setup

Severity: Medium  
Fix status: Fixed  
Area: QA tooling/setup

Reproduction steps:

```bash
python3 scripts/qa_smoke.py
```

Expected behavior:

- Smoke script either runs with documented environment instructions or emits a clear dependency/setup message.

Actual behavior:

```text
ModuleNotFoundError: No module named 'PIL'
```

Root cause:

- The default `python3` environment lacks `Pillow`; the README documents Python 3.13 for API setup.

Recommended fix:

- Document running it with `python3.13` after installing API requirements, or add a small wrapper/check that explains missing dependencies.

Fix applied:

- Moved Pillow import behind explicit environment checks in `scripts/qa_smoke.py`.
- Updated `QA.md` to document `python3.13 scripts/qa_smoke.py`.

### Low - DeepZoom DZI returns 400 for imported raster PNG without pre-generated tiles

Severity: Low  
Fix status: Fixed  
Area: Viewer/API edge case

Reproduction steps:

1. Import a PNG slide.
2. Request `/slides/1/deepzoom.dzi`.

Expected behavior:

- Raster PNG viewer path either works with generated DZI/tiles or the frontend uses the raster tile endpoint and does not attempt DeepZoom DZI.

Actual behavior:

```json
{"error":{"code":"not_found","message":"DeepZoom tiles not pre-generated for this slide", ...}}
```

Root cause:

- PNG metadata/thumbnails/tiles are supported, but DZI is not generated for this imported raster in the tested path.

User impact:

- Potential viewer failure for PNG imports if the renderer always chooses the DZI URL.

Recommended fix:

- Confirm viewer fallback behavior for raster slides visually.
- Add an automated test that opens a PNG slide through the same viewer code path.

Fix applied:

- Verified the viewer code intentionally falls back to raster tiles when DZI is unavailable.
- Changed missing DeepZoom artifact responses to HTTP 404 and added an API regression test confirming raster tile fallback still returns JPEG.

### Low - Immediate run listing can briefly lag after inference enqueue

Severity: Low  
Fix status: False positive  
Area: Async inference consistency

Reproduction steps:

1. POST `/inference/slides/1/run`.
2. Immediately GET `/inference/slides/1/runs`.

Expected behavior:

- The newly queued run appears immediately.

Actual behavior:

- One immediate call returned `{"runs":[]}` even though the run creation response returned run `id: 1`; a subsequent call showed the run as `running`.

Root cause:

- Likely transaction/session timing around background task commit and read ordering.

Recommended fix:

- No production fix needed. Subagent verification confirmed the run row is committed before the job is enqueued/submitted, and existing queue/lifecycle tests pass.

## Workflow Results

### Setup and boot

- README documents prerequisites and setup steps.
- Desktop dependency dry-run: pass.
- Local API dependency dry-run: pass in current Python 3.13 environment.
- ML dependency dry-run: initially failed under sandbox network restrictions, passed after network approval.
- API direct boot: pass with required `--data-dir` and `--log-dir`.
- Electron boot: starts API and window process; startup logs include renderer 401s described above.
- Migrations: missing Alembic assets, `metadata.create_all()` fallback.

### Core slide workflow

Tested through direct API:

- Empty slides list: pass, returns `{"slides":[]}`.
- Import generated PNG: pass.
- List imported slide: pass.
- Metadata: pass, dimensions `[640,480]`.
- Thumbnail: pass with header auth and query API key.
- Review status update: pass with `PATCH /slides/{id}/review`.
- Delete slide: pass.

### Model and inference workflow

- Model discovery: pass, found `models/deploy-fungus.pth.gz`.
- Missing model request: pass, returns normalized app error.
- Inference run: pass on generated PNG.
- Run status persisted: pass.
- Regions persisted: pass.
- Slide list reflects positive result after inference: pass.

### Auth and permissions

- Protected routes without key: pass, 401.
- Protected routes with `x-api-key`: pass.
- Query API key for image URLs: pass when `APP_ALLOW_QUERY_API_KEY=true`.
- No user login/signup flow found; app uses local generated API key for desktop/API protection.

### Frontend/UI

Code-inspected major controls:

- Home, Import, Gallery, Models, Settings tabs.
- Import drag/drop, file picker, recursive folder import, cancel button.
- Gallery search/sort/filter/favorites/select/delete/batch inference.
- Viewer close, metadata, export view, export regions, inference, threshold, overlay opacity, review decisions.
- Settings theme and API port.

Runtime UI click-through was limited by browser automation availability. Electron startup was tested and revealed protected API calls before key initialization.

## Recommended Next Steps

1. Recreate the local API `.venv`; the checked-out venv has stale absolute shebangs and should be rebuilt rather than patched.
2. Add fuller desktop/Electron UI E2E coverage beyond the syntax smoke.
3. Add E2E coverage for:
   - first launch with empty DB
   - import invalid extension
   - import valid PNG/SVS
   - gallery refresh/search/filter/select/delete
   - viewer opens raster PNG successfully
   - model discovery empty state and populated state
   - inference success and failure states
   - API key timing on startup

## Post-Remediation Validation

```bash
cd services/local-api && python3.13 -m pytest
# 56 passed, 3 warnings

cd apps/desktop && npm test
# passed

python3.13 scripts/qa_smoke.py
# passed

cd apps/desktop && npm start
# API health confirmed on http://127.0.0.1:8765/health
```

## Notes

- `git status` showed existing modified desktop files. Initial status showed `apps/desktop/package.json` and `apps/desktop/package-lock.json`; later status also showed `apps/desktop/index.html`, `apps/desktop/css/styles.css`, and `apps/desktop/js/viewer.js`. I did not intentionally edit or revert these files.
- `npm run pack` generated/updated build artifacts under `apps/desktop/dist/`, which appear ignored by git.
