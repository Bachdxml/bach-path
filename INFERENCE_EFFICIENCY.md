# WSI Inference Efficiency — Analysis & Plan

**Problem.** On large whole-slide images (WSIs), inference accuracy drops because
tiles end up "too large": each 512 px tile covers more physical tissue at lower
resolution than the model was trained on, so fungal structures shrink and blur.
We want to preserve accuracy without an unbounded runtime cost. Longer inference
is acceptable if the accuracy is there.

This document diagnoses *why* the pipeline downsamples on large slides, then lists
independent efficiency techniques that each buy back "magnification headroom" —
compute saved can be spent running at a finer pyramid level instead of a coarser
one. Each technique below maps 1:1 to a spec in `specs/` and is meant to ship as
its own commit.

All line references are to
[`run_inference_api.py`](wsi-fungal-segmentation/scripts/run_inference_api.py)
and [`model.py`](wsi-fungal-segmentation/src/model.py) unless stated otherwise.

---

## Root cause: level selection ignores the tissue mask

`_select_openslide_level` ([lines 175–194](wsi-fungal-segmentation/scripts/run_inference_api.py#L175-L194))
chooses the pyramid level by counting **every grid position on the whole level**
via `_count_tile_positions` — background glass included. It returns the first
(finest) level whose *full-grid* tile count is `<= target_tiles` (default 30000).

A large WSI produces far more than 30000 full-grid tiles at level 0, so the
selector steps down to level 1 (2× downsample), level 2 (4×), etc. At level `L`
each 512 px tile spans `2^L ×` more tissue at `1/2^L` the resolution. The model
was trained on 512 px tiles at native magnification, so its input distribution
shifts and accuracy falls. **This is the "tiles too large" effect.**

The tissue mask (`_build_openslide_tissue_mask` /
`_build_effective_tissue_mask`) already exists, but it is only consulted *after*
the level is chosen, to skip background tiles during the two inference passes
([lines 1343–1351](wsi-fungal-segmentation/scripts/run_inference_api.py#L1343-L1351)).
It never influences which level is picked. So a slide that is 90% glass is forced
to a coarse level as if all that glass had to be processed at full resolution.

**Consequence and the lever.** Real tissue is usually a small fraction of a WSI.
If we budget by *tissue* tiles rather than *all* tiles, most large slides can stay
at a finer level for the same amount of real work. And anything that makes each
tile cheaper to process lets us raise the effective tile budget — i.e. run finer —
within the same wall-clock. Every technique below is in service of that.

---

## The current inference path (for reference)

For each slide the pipeline runs a **two-pass, neighborhood-conditioned**
inference (`_run_passes`,
[lines 940–1010](wsi-fungal-segmentation/scripts/run_inference_api.py#L940-L1010)):

1. **Pass 1 — density classification.** Every tile goes through `model(batch)`
   with no `density_label`. Only `density_logits.argmax` is kept per tile; the
   full segmentation decoder output is computed and **thrown away**. Decoded
   pixels are cached to a memmap for pass 2.
2. **Consensus.** Each tile's density class is replaced by the majority class in
   its `k`-neighborhood grid.
3. **Pass 2 — segmentation.** Every tile goes through `model(batch,
   density_label=consensus)` again; `sigmoid(seg_logits)` is scored into regions.

So the model runs a **full encoder+decoder forward twice per tile**. Pass 1 uses
`torch.autocast(fp16)`; pass 2 does **not** (plain `torch.no_grad`). Tiles are
read from the slide serially on the CPU between GPU batches.

---

## Techniques (each = one spec, one commit)

Ordered by leverage. E1 fixes the accuracy loss directly; E2–E5 cut per-tile cost
so a finer level is affordable within the same runtime.

### E1 — Tissue-aware level selection  *(accuracy-critical)*
**Spec:** `specs/tissue-aware-level-selection.md`

Build the tissue mask (already available at the low-res thumbnail) *before*
choosing the level, and in `_select_openslide_level` count only tile positions
that pass the tissue check, instead of the full grid. Pick the finest level whose
**tissue-tile** count is `<= target_tiles`.

- **Why it helps:** a sparse large slide stays at native (or finer) magnification
  because the glass no longer inflates the count. Directly addresses the
  root-cause accuracy drop.
- **Cost/risk:** builds the thumbnail mask once per candidate level (cheap — it's
  thumbnail-resolution). Must fall back to full-grid counting if the mask can't be
  built (mask=None), preserving today's behavior. Keep the `max_tiles` hard cap
  and `--no-skip-background` semantics intact.
- **Sketch:** thread the tissue-mask + `min_tissue_fraction` into
  `_count_tile_positions` (or a tissue-aware variant) and call it from
  `_select_openslide_level`. Reuse the already-selected level's mask for the tile
  loop so it isn't computed twice.

### E2 — Skip the decoder in pass 1 (density-only forward)
**Spec:** `specs/pass1-density-only-forward.md`

Pass 1 needs only `density_logits`, which comes straight off the bottleneck
(`density_head(b)`, [model.py:141](wsi-fungal-segmentation/src/model.py#L141)) —
yet the current forward runs the entire decoder
([model.py:154–175](wsi-fungal-segmentation/src/model.py#L154-L175)) and discards
it. Add a `density_only=True` path to `ResidualAttentionUNet.forward` that returns
after the density head, and call it in pass 1.

- **Why it helps:** the decoder (4 upconv + 4 residual + 4 attention blocks) is
  roughly half the network's compute. Skipping it in pass 1 cuts pass-1 cost
  substantially with **bit-identical** density predictions (same code path up to
  the bottleneck).
- **Cost/risk:** low. Pure additive branch; existing 4-tuple return is unchanged
  when `density_only` is not set. Verify density labels are identical to the old
  path on a fixed input.

### E3 — GPU throughput parity with training (fp16 + channels_last + TF32)
**Spec:** `specs/inference-gpu-throughput-parity.md`

Training already runs `amp`, `channels_last`, and `tf32`
([`profile-lab-cuda.yaml`](wsi-fungal-segmentation/configs/profile-lab-cuda.yaml#L42-L45)).
Inference only enables `cudnn.benchmark` and pass-1 autocast. Bring inference to
parity:
- wrap **pass 2** in `torch.autocast` (it currently runs fp32),
- put the model and tile batches in `channels_last` memory format,
- enable TF32 (`torch.backends.cuda.matmul.allow_tf32`, `cudnn.allow_tf32`).

- **Why it helps:** pass 2 is the expensive full-decoder pass and is currently the
  only fp32 forward. Tensor-core fp16 + channels_last conv is the standard ~1.5–2×
  throughput win on NVIDIA. TF32 accelerates the fp32 ops that remain.
- **Cost/risk:** low and CUDA-only — all three must be no-ops / guarded off on CPU
  and MPS so the existing CPU test path is unchanged. Segmentation is thresholded
  (`sigmoid >= threshold`), so fp16 rounding on logits is immaterial; confirm
  region output is unchanged within tolerance on CPU (unaffected) and that CUDA
  paths are guarded.

### E4 — Overlap slide I/O with GPU compute (tile prefetch)
**Spec:** `specs/tile-io-prefetch.md`

`load_tensor` reads and decodes each tile from OpenSlide on the CPU, serially,
in the batch loop ([lines 948–951](wsi-fungal-segmentation/scripts/run_inference_api.py#L948-L951)).
The GPU idles during slide reads. Use a stdlib `concurrent.futures.ThreadPoolExecutor`
to decode the *next* batch while the GPU processes the current one (OpenSlide
releases the GIL during reads, so threads actually overlap).

- **Why it helps:** on large slides, slide decode is a real fraction of wall-clock;
  overlapping it with compute raises the affordable tile budget → finer level.
- **Cost/risk:** moderate. Bounded prefetch depth to keep memory flat; order of
  results must be preserved (positions drive coordinates). Keep a
  `--prefetch-workers 0` escape hatch that restores today's serial path. stdlib
  only — no new dependency.

### E5 — Compile the model once (`torch.compile`)
**Spec:** `specs/inference-torch-compile.md`

Training sets `compile: true`; inference runs eager. Wrap the model in
`torch.compile` after load, guarded by a `try/except` that falls back to eager if
compilation is unavailable or fails (older PyTorch, no compiler backend, CPU/MPS).

- **Why it helps:** kernel fusion + reduced Python overhead, amortized over
  thousands of near-identical forward passes. Fixed 512×512 tile shape means one
  compiled graph is reused for every batch.
- **Cost/risk:** low with the fallback. First batch pays a one-time compile cost;
  gate behind a flag (e.g. `--compile/--no-compile`, default off until measured)
  so it never regresses small-slide latency. Must not break the CPU test path.

---

## Deliberately deferred (documented, not specced)

- **Two-stage coarse→fine adaptive magnification.** Run a fast coarse pass over
  the whole slide at a low level to find candidate tissue, then re-run only those
  regions at native magnification. This is the ultimate accuracy/efficiency
  trade but a much larger change (region bookkeeping, two coordinate systems,
  overlap handling). E1 captures most of the benefit for a fraction of the risk;
  revisit this only if E1–E5 still leave large slides under-magnified.
- **Auto batch-size sizing to VRAM.** A tuning knob, not a correctness win; the
  `--batch-size` flag already exists. Add only if a fixed default proves limiting.

## Verification note

The test harness
([`test_run_inference_api.py`](wsi-fungal-segmentation/tests/test_run_inference_api.py))
runs on CPU with a `FakeModel` and monkeypatched OpenSlide — no GPU or real
checkpoint required. Every technique must keep `pytest` green on CPU; CUDA-only
paths (E3, E5) must be guarded so they are inert off-GPU.
