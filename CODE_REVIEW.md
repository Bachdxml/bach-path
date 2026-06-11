# Code Review — Bach Path

> Reviewed: 2026-06-10
> Branch: code-review-fixes
> **Remediation: 2026-06-10 — all 12 findings resolved and verified (see Resolution Status below). Finding texts preserved as originally written for reference.**

## Resolution Status (2026-06-10)

| # | Status | Verification evidence |
|---|--------|----------------------|
| 1 | **Resolved** | `preload.js` exposes `webUtils.getPathForFile` bridge; drop handler routes through main-process `validateDroppedPaths`. Diff-reviewed + syntax-checked; runtime drag-drop smoke test still recommended on next app launch. |
| 2 | **Resolved** | `set-config` no longer mutates `apiReadyState`; `app.js` no longer writes the persisted key into `slidesApi`. Connection state flows solely from `api-ready`. Diff-reviewed. |
| 3 | **Resolved** | Cheap level/negative checks moved before cache lookup; regression test `test_raster_tile_status_codes_are_cache_state_independent` passes (404/400 identical pre- and post-cache). |
| 4 | **Resolved (mitigation)** | Cancel messaging is now honest ("slides already processed may still appear") and the gallery reconciles. **Deferred follow-up:** job-based import with true server-side cancellation. |
| 5 | **Resolved** | In-flight guard + `Promise.allSettled` fan-out in `startInferencePolling`; completion semantics preserved. Diff-reviewed + syntax-checked. |
| 6 | **Resolved** | Batch/folder endpoints validate checkpoint + all slide paths before enqueuing anything; regression test `test_batch_inference_with_missing_slide_file_queues_nothing` passes (zero runs, zero enqueues on failure). |
| 7 | **Resolved** | Opacity `input` updates mounted overlay opacity in place (zero geometry recompute); outline geometry memoized per regions-array identity. Diff-reviewed + syntax-checked. |
| 8 | **Resolved** | Cache-hit path returns bytes without `Image.open`; covered by the same regression test as #3. |
| 9 | **Resolved** | Single `classify_inference_result` in `app/inference/results.py`; both routes delegate; 6 new unit tests pass. |
| 10 | **Resolved** | Lazy TTL eviction under the existing lock; tests confirm no thread growth across 20 completions and TTL purge. |
| 11 | **Resolved** | One canonical `get_app_session_factory` + single `_app_engine_lock` in `db/session.py`; both `get_db` and the inference worker delegate to it. Diff-reviewed. |
| 12 | **Resolved** | `npm test` → `node scripts/check-syntax.js`; verified passing on Windows ("Syntax check passed for 10 file(s)."). |

Final verification: backend `pytest` = **66 passed, 2 failed** — the 2 failures
are pre-existing at clean HEAD (independently verified by two workers via
stash/worktree baselines) and unrelated to these findings:
`test_inference_subprocess_invocation_includes_model_metadata_and_optional_threshold`
(stale `--level auto` assertion) and `test_delete_slide_removes_file_and_tile_cache`
(tile-cache dir naming expectation). These remain open as known pre-existing
test debt, not regressions.

Scope: FastAPI local API (`services/local-api/app`), the Electron desktop app
(`apps/desktop` — main process, preload, and renderer modules), and the
inference subprocess (`wsi-fungal-segmentation/scripts/run_inference_api.py`).
The branch has already absorbed two prior code-review passes — auth
(`require_api_key` wired on every router in `router.py`), subprocess env
allow-listing, secret redaction, weights-only checkpoint loading, loopback-only
host validation, IPC sender verification, and startup reconciliation of
orphaned runs are all in place and correct. The findings below are what remains.

## Critical Correctness Bugs

### 1. Drag-and-drop import is silently broken: `File.path` no longer exists on Electron 40
**File:** `apps/desktop/js/import.js:21` (also `apps/desktop/package.json:58`)

```js
if (WSI_EXTENSIONS.includes(ext) && f.path) {
  paths.push(f.path);
}
```

The drop handler reads `f.path` off dropped `File` objects. Electron **removed
`File.path` in v32** (replaced by `webUtils.getPathForFile()` in the preload),
and this project pins `electron: ^40.9.0` with `sandbox: true`. So `f.path` is
always `undefined`, `getPathsFromFiles` always returns `[]`, and every drop
shows "No valid files (SVS, TIF, TIFF, PNG) dropped" regardless of what was
dropped. A primary import flow is dead. Related: the main process already has a
hardened validation handler for exactly this — `get-dropped-file-paths`
(`index.js:534`, `validateDroppedPaths` at `index.js:156`) exposed as
`getDroppedFilePaths` (`preload.js:49`) — but **no renderer code ever calls
it**; it is dead code, which is why the regression went unnoticed.

**Fix:** In `preload.js`, import `webUtils` and expose
`getPathForFiles: (files) => files.map((f) => webUtils.getPathForFile(f))`.
In the drop handler, convert `e.dataTransfer.files` to paths via that bridge,
then pass them through the existing `getDroppedFilePaths` IPC so the main
process's validation (absolute path, extension, tile-like filtering, existence)
actually runs. This both fixes the bug and un-deadens the validation chain.

---

### 2. `set-config` desyncs the renderer's API connection state from the running server
**File:** `apps/desktop/index.js:527-532`, root cause vs `index.js:349`

```js
handleTrusted("set-config", (_, config) => {
  const nextConfig = validateConfigPayload(config);
  saveConfig(nextConfig);
  apiReadyState = buildApiReadyState(nextConfig);   // <-- rebuilt from disk config
  return loadConfig();
});
```

`startApi()` launches the server with `config.apiKey || crypto.randomBytes(32)`
(`index.js:349`) — when no key is configured (the default), the key exists
**only in memory**, captured in the `apiReadyState` returned by `startApi()`.
Any later `set-config` call (e.g. saving a port in Settings, `app.js:281`)
rebuilds `apiReadyState` from the **persisted** config, which has no key — and
possibly a new port the running server is not listening on. The running API
still requires the old generated key. On macOS, `activate` re-creates the
window with this stale state (`index.js:516-519`), so every authenticated
request 401s (or hits a dead port) until full app restart. The same conflation
appears in `app.js:256-264`, which pushes the *persisted* key into `slidesApi`
at startup, racing the correct key delivered by `api-ready`.

**Fix:** Separate live process state from persisted config. Keep
`apiReadyState` owned exclusively by `startApi()` (it describes the process
that is actually running); `set-config` should only persist, and the UI already
tells users a restart is required. Remove the `getConfig().then(setApiKey(...))`
write in `app.js` — the renderer should receive connection state from the
`api-ready` event alone.

---

### 3. Raster tile out-of-bounds returns 400 on a cache hit but 404 on a cache miss
**File:** `services/local-api/app/api/routes/slides.py:710-714` (cache hit) vs `services/local-api/app/api/routes/slides.py:327-328` (generation path)

The raster tile endpoint validates coordinates in two divergent places. On a
**cache hit** it reopens the image and rejects an out-of-bounds tile with
`SLIDE_INVALID` → HTTP 400:

```python
if x < 0 or y < 0 or x * tile_size >= iw or y * tile_size >= ih or level != 0:
    raise AppError(ErrorCode.SLIDE_INVALID, ...)   # 400
```

On a **cache miss**, `_raster_tile_jpeg` rejects the same out-of-bounds tile
with `NOT_FOUND` → HTTP 404 (`slides.py:327-328`). So the identical request
returns 400 or 404 purely depending on whether the tile happened to be cached.
OpenSeadragon treats 404 as "no tile here" (normal at slide edges) but surfaces
other 4xx as load errors, so edge tiles flicker as errors only after the cache
warms. Root cause: bounds validation is duplicated and the two copies disagree
on the semantics of out-of-bounds. The OpenSlide branch (`slides.py:744-749`)
already gets it right: negative coords → 400, past-the-edge → 404.

**Fix:** Extract a single `_validate_raster_tile_coords(level, x, y, iw, ih,
tile_size)` helper that raises `SLIDE_INVALID` for negative coords / wrong
level and `NOT_FOUND` for past-the-edge, and call it from both paths. This also
removes the duplicated rule.

---

## Significant Reliability Issues

### 4. Import "Cancel" aborts only the HTTP request; the server finishes the import anyway
**File:** `apps/desktop/js/import.js:73-79` with `services/local-api/app/api/routes/slides.py:513`

The cancel button calls `importAbortController.abort()`, which closes the fetch.
But `import_slide_collection` is a sync (`def`) FastAPI handler running in the
threadpool — client disconnect does **not** cancel it. The server keeps copying
potentially multi-gigabyte slide files and commits the whole collection. The UI
then reports "Import canceled." while the follow-up gallery refresh
(`import.js:141-142`) displays the slides that were supposedly canceled —
contradictory state presented to the user, and gigabytes of "canceled" data on
disk.

**Fix:** Make collection import job-based like inference already is: the
endpoint enqueues an import job and returns a job id; the client polls and a
`DELETE /import-jobs/{id}` cancels between files (per-file granularity is
natural since `_import_slide_file` is already the unit of work, and the
existing rollback in `import_slide_collection` shows how to clean up). Short of
that, after an abort the client must reconcile with the server (poll the
collection) instead of claiming cancellation succeeded.

---

### 5. Gallery inference polling is sequential per-run and can overlap itself
**File:** `apps/desktop/js/gallery.js:471-499`

```js
inferencePollTimer = setInterval(async () => {
  const ids = [...pendingInferenceRunIds];
  ...
  for (const runId of ids) {
    const run = await gallerySlidesApi.getInferenceRun(runId);  // serial awaits
    ...
  }
}, 2000);
```

Two problems compound. (a) Each pending run is polled with a serial `await`, so
a batch or folder run (up to 64 / 512 slides) issues up to N sequential GETs
per cycle. (b) `setInterval` does **not** wait for the previous async callback,
so if one cycle's N serial requests take longer than 2 s, the next cycle starts
on top of it — overlapping cycles that each re-poll the whole set, multiplying
load exactly when the server is busiest running inference.

**Fix:** Guard against re-entrancy with an in-flight flag (skip a tick while
the previous cycle runs), and fan the per-run polls out with `Promise.all` —
or, better, add a single `GET /inference/runs/status?ids=...` batch endpoint
and poll that once per cycle.

---

### 6. Batch and folder inference leave partial side effects when one slide fails mid-loop
**File:** `services/local-api/app/api/routes/inference.py:688-698` (batch), `services/local-api/app/api/routes/inference.py:740-749` (folder)

Both endpoints loop over slides calling `_queue_inference_run_for_slide`, which
commits its `InferenceRun` row and submits the worker **inside** the loop
(`inference.py:584-606`). If slide *k* raises — e.g.
`_resolve_managed_slide_path` throws `STORAGE_INCONSISTENT` because that
slide's file is missing (`access.py:23-24`) — slides `0..k-1` are already
committed, queued, and executing, but the endpoint propagates the error and
returns 4xx/5xx. The client sees "batch failed" while N jobs are in fact
running, and the response contains no `run_ids`, so the gallery never tracks
them and they silently finish in the background.

**Fix:** Validate everything that can fail up front (resolve each slide path
and the model checkpoint before enqueuing anything), then enqueue in a second
pass so the operation is all-or-nothing. If partial success is acceptable
product behavior, change the response to report per-slide
`{queued: [...], failed: [...]}` instead of aborting the whole request.

---

## Performance Issues

### 7. Dragging the overlay-opacity slider recomputes the entire positive-outline clustering
**File:** `apps/desktop/js/viewer.js:1311-1313`

```js
viewerOverlayOpacity?.addEventListener("input", () => {
  addRegionOverlays(lastRegions, viewerShowNegative?.checked || false);
});
```

`addRegionOverlays` calls `computePositiveOutlineByRegion` (`viewer.js:801`),
which runs union-find clustering and, per cluster, two-pass chamfer distance
fields over a `gridW × gridH` canvas (`computeChamferDistanceField`,
`viewer.js:557`; `computeClusterOutlineLayers`, `viewer.js:711`). That work
depends only on the **regions and the outline color**, never on opacity. The
`input` event fires continuously while the slider is dragged, so every pixel of
slider travel re-clusters and re-runs chamfer over potentially thousands of
positive tiles, then rebuilds every overlay DOM node. On a positive-heavy slide
this makes the opacity control visibly janky.

**Fix:** Separate "what to draw" from "how opaque." Compute `outlineByRegion`
once per run/color and cache it (invalidate on new run or
`positive-outline-color-changed`). The opacity handler should only set
`container.style.opacity` on the already-mounted overlays (or a CSS variable
the overlays read) — no recompute, no DOM teardown.

---

### 8. Raster tile cache hits re-open the full source image just to validate bounds
**File:** `services/local-api/app/api/routes/slides.py:710-712`

```python
if tile_path.exists():
    with Image.open(slide_path) as img:
        iw, ih = img.size
    ...
    return Response(content=tile_path.read_bytes(), ...)
```

Even when the JPEG tile is already on disk, every request opens and parses the
source PNG/TIFF header to read `.size`. Tile requests are the hottest path in
the app (the viewer fires many per second during pan/zoom), and the OpenSlide
branch deliberately avoids this via the handle cache. For raster slides each
cached tile still pays a filesystem open + image-header parse.

**Fix:** The tile was only written if it was valid, so a cache hit can return
the bytes immediately without re-validating. If validation on hit is still
wanted, cache the slide's `(width, height)` per slide id instead of reopening
the image each call. Natural to fix together with Finding #3 (shared
validator).

---

## Design / Maintainability

### 9. Inference-result classification is duplicated and can drift
**File:** `services/local-api/app/api/routes/slides.py:427-435` and `services/local-api/app/api/routes/inference.py:870-878`

`list_slides._inference_result` and `_inference_result_from_run` both map
(positive count, negative count) → `"positive" | "negative" | "needs_review" |
"unchecked"`, with the same precedence rules expressed twice over different
inputs. A future change to the rule (e.g. a minimum positive count) must be
made in both places or the gallery badge and the viewer status will disagree.

**Fix:** Extract one helper, e.g. `classify_inference_result(positive: int,
negative: int, *, succeeded: bool) -> str`, and call it from both sites.

---

### 10. InMemoryQueue spawns one `threading.Timer` thread per completed job
**File:** `services/local-api/app/queue/in_memory.py:80-83`

```python
def _schedule_eviction(self, job_id: str) -> None:
    timer = threading.Timer(_COMPLETED_JOB_TTL_SECS, self._evict, args=(job_id,))
    timer.daemon = True
    timer.start()
```

Every `ack`/`fail`/`cancel` starts a 300-second daemon timer thread. A folder
run that completes hundreds of jobs in a burst creates hundreds of idle timer
threads that linger for five minutes. It works, but it scales thread count with
job throughput for no benefit on a single-user local app.

**Fix:** Replace per-job timers with one periodic sweep (a single background
thread, or lazy eviction that drops expired entries whenever `_jobs` is
touched), bounding threads at O(1) regardless of job volume.

---

### 11. `app.state.engine` lazy init is guarded by two different locks (latent race)
**File:** `services/local-api/app/api/deps.py:11,16-20` and `services/local-api/app/api/routes/inference.py:48,556-563`

`get_db` double-checks `app.state.engine` under `_db_init_lock`, while
`_get_session_factory` double-checks the same attribute under a *different*
lock, `_engine_init_lock`. Double-checked locking only works when all writers
share one lock; two locks are not mutually exclusive, so concurrent first-time
initialization could construct two engines and leak one SQLite connection pool.
Currently unreachable in practice — every inference route also depends on
`get_db`, which initializes the engine first — so this is latent/defensive, not
an active bug.

**Fix:** Move the lazy engine/session-factory creation into a single helper
(e.g. `get_session_factory(app)` in `db/session.py`) guarded by one
module-level lock, and call it from both `get_db` and the inference worker
path. This also deletes the near-duplicate init block.

---

### 12. `npm test` cannot run on Windows
**File:** `apps/desktop/package.json:12`

```json
"test": "node --check index.js && node --check preload.js && find js -name '*.js' -print0 | xargs -0 -n1 node --check"
```

The script shells out to Unix `find`/`xargs`, which do not exist in
`cmd.exe`/PowerShell — `npm test` fails outright on Windows, one of the
project's three supported dev platforms (README line 21). The syntax-check gate
is therefore never run by Windows contributors.

**Fix:** Replace with a cross-platform invocation — e.g. a tiny Node script
that globs `js/**/*.js` and runs `node --check`, or `node --check` via a
`glob`-based npm package — so the same command works on all three platforms.

---

## Summary Table

All findings below are resolved as of 2026-06-10 — see Resolution Status at the top for per-finding evidence.

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | Bug | `import.js:21` | Drag-and-drop import dead: `File.path` removed in Electron ≥32 (project pins ^40); validation IPC exists but is uncalled |
| 2 | Bug | `index.js:527-532` | `set-config` rebuilds connection state from disk, dropping the in-memory generated API key the running server requires |
| 3 | Bug | `slides.py:710-714` / `:327-328` | Out-of-bounds raster tile returns 400 on cache hit, 404 on miss |
| 4 | Reliability | `import.js:73-79`, `slides.py:513` | "Cancel" aborts the request but the server finishes and commits the import |
| 5 | Reliability | `gallery.js:471-499` | Inference polling is serial per-run and overlapping `setInterval` cycles pile up |
| 6 | Reliability | `inference.py:688-698`, `:740-749` | Batch/folder runs leave queued+running jobs when a later slide fails |
| 7 | Performance | `viewer.js:1311-1313` | Opacity slider recomputes full chamfer/clustering outline on every input event |
| 8 | Performance | `slides.py:710-712` | Cached raster tiles re-open the source image to validate bounds |
| 9 | Design | `slides.py:427-435`, `inference.py:870-878` | Duplicated inference-result classification logic |
| 10 | Design | `in_memory.py:80-83` | One timer thread per completed job |
| 11 | Design | `deps.py:11`, `inference.py:48` | Engine lazy-init guarded by two different locks (latent race) |
| 12 | Design | `package.json:12` | `npm test` uses Unix-only `find`/`xargs`; fails on Windows |

## Recommended Fix Order

1. **#1 — Drag-and-drop import broken** — a primary user flow silently does
   nothing on the shipped Electron version; fixing it also activates the
   dormant path-validation IPC.
2. **#2 — set-config connection desync** — any settings save poisons the live
   session's auth state; small fix (stop rebuilding live state from disk).
3. **#3 — Raster tile status inconsistency** — small, self-contained
   correctness fix that also produces the shared validator needed for #8.
4. **#5 — Gallery polling overlap** — most likely to cause real load problems
   today, since batch/folder runs are a primary workflow and the request storm
   hits the server while it is already busy.
5. **#4 — Cancel doesn't cancel** — contradictory UX plus gigabytes of
   unwanted data; job-based import is the durable fix.
6. **#7 — Opacity recompute** — clear UX regression on positive-heavy slides;
   mechanical fix once outline caching is separated from opacity.
7. **#6 — Batch/folder partial failure** — validate-then-enqueue or report
   partial success.
8. **#8 — Raster cache-hit reopen** — fold into #3's shared validator.
9. **#9–#12 — Design cleanups** — do alongside the above when touching the
   same files; #11 is defensive only.
