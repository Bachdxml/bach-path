# Clinical Hardening Validated Findings

## Confirmed And Fixed

- `CH-001` bug high confidence: collection rename did not commit, so renamed import collections could revert after session close. Fixed in `services/local-api/app/api/routes/slides.py` with persistence test.
- `CH-002` performance high confidence: inference run summaries loaded all `Region` ORM rows. Fixed in `services/local-api/app/api/routes/inference.py` with SQL aggregate summaries.
- `CH-003` performance high confidence: gallery thumbnails regenerated on every request. Fixed with thumbnail cache under `tiles_cache_dir`.
- `CH-004` security high confidence: desktop-launched local API could run without an API key. Fixed by generating a per-session key in `apps/desktop/index.js`.
- `CH-005` security high confidence: OpenSlide metadata returned arbitrary vendor properties that may contain PHI. Fixed by allowlisting safe properties.
- `CH-006` reliability high confidence: malformed inference regions could persist invalid geometry/scores. Fixed with output validation and safe failed-run transition.
- `CH-007` reliability high confidence: failed single-slide import could leave an empty auto-created collection. Fixed with compensating cleanup.
- `CH-008` reliability high confidence: oversized raster inference checked pixel limits after RGB decode. Fixed by checking dimensions before conversion.

## Confirmed, Deferred

- `CH-009` performance high confidence: gallery polling sends one request per pending inference run. Recommended follow-up: batch status endpoint and shared polling tracker.
- `CH-010` performance high confidence: `/regions` returns all detections and viewer renders them synchronously. Recommended follow-up: pagination/top-N plus server-side heatmap tiles.
- `CH-011` performance high confidence: OpenSlide handle is reopened per dynamic tile request. Recommended follow-up: bounded slide-handle cache or background DeepZoom generation.
- `CH-012` clinical safety high confidence: fast auto-level/background-skip inference changes need golden-slide validation before clinical use. Recommended follow-up: curated positive/negative/sparse-positive regression set.
- `CH-013` clinical provenance high confidence: review decisions lack reviewer, timestamp, rationale, run linkage, and history. Recommended follow-up: review audit event model.
