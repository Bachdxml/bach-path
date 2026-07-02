#!/usr/bin/env python3
"""
Run fungus segmentation inference on a whole slide image.
Outputs a JSON file with segmentation-derived fungus_positive regions.

Usage:
  python run_inference_api.py --slide-path /path/to/slide.svs --output-json out.json --checkpoint checkpoints/best_model.pth
  python run_inference_api.py --slide-path slide.svs --output-json out.json --checkpoint best.pth --device cuda:0
"""

import argparse
import base64
from collections.abc import Mapping
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure we can import from src (run from wsi-fungal-segmentation or bach-path)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.model import ResidualAttentionUNet
from src.inference_utils import infer_with_neighborhood

# Formats OpenSlide cannot open - run tile inference from a raster in memory.
RASTER_SLIDE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# ImageNet normalization (matches training)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EXIT_SUCCESS = 0
EXIT_ARGS = 1
EXIT_SLIDE = 2
EXIT_INFERENCE = 3
EXIT_OUTPUT = 4

MAX_INFERENCE_TILES_DEFAULT = 200_000
MAX_RASTER_PIXELS_DEFAULT = 150_000_000
TISSUE_MASK_MAX_DIMENSION = 2048
TARGET_INFERENCE_TILES_DEFAULT = 30000

# Isolation-context filter defaults. OFF by default
# (opt-in via --tissue-context-radius / --tissue-context-fraction) so existing
# inference behavior is unchanged unless explicitly enabled.
TISSUE_CONTEXT_RADIUS_DEFAULT = 0
TISSUE_CONTEXT_FRACTION_DEFAULT = 0.0


def _safe_torch_load(source, *, map_location):
    """Refuse to deserialize checkpoints unless PyTorch supports weights-only loading."""
    try:
        return torch.load(source, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This script requires torch.load(..., weights_only=True). "
            "Upgrade PyTorch or re-export the checkpoint; refusing unsafe deserialization."
        ) from exc


def _is_tensor_state_dict(candidate: Mapping) -> bool:
    return bool(candidate) and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in candidate.items()
    )


def _load_checkpoint_dict(path: Path, device: torch.device):
    """Load torch checkpoint from plain or gzip-compressed file (.pth.gz)."""
    name = path.name.lower()
    if path.suffix.lower() == ".gz" or name.endswith((".pth.gz", ".pt.gz")):
        with gzip.open(path, "rb") as f:
            return _safe_torch_load(f, map_location=device)
    return _safe_torch_load(path, map_location=device)


def _select_device(requested: str) -> torch.device:
    value = (requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value.startswith("cuda"):
        return torch.device(value if torch.cuda.is_available() else "cpu")
    if value == "mps":
        mps = getattr(torch.backends, "mps", None)
        return torch.device("mps" if mps is not None and mps.is_available() else "cpu")
    return torch.device(value)


def _extract_model_state_dict(checkpoint) -> Tuple[Mapping, dict]:
    # Checkpoints cross a trust boundary, so only accept the minimal structures
    # produced by this project's training/export pipeline.
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping loaded in weights-only mode.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, Mapping) or not _is_tensor_state_dict(state_dict):
            raise ValueError("Checkpoint 'model_state_dict' must be a non-empty tensor state dict.")
        return state_dict, checkpoint

    if _is_tensor_state_dict(checkpoint):
        return checkpoint, {}

    raise ValueError(
        "Checkpoint must contain a 'model_state_dict' entry or be a plain tensor state dict."
    )


def preprocess_tile(pil_img: Image.Image, target_size: int = 512) -> torch.Tensor:
    """Convert PIL tile to normalized tensor [1, 3, H, W]."""
    img = pil_img.convert("RGB")
    arr = np.array(img)
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        img = img.resize((target_size, target_size), Image.BILINEAR)
        arr = np.array(img)
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN)) / np.array(STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def _max_inference_tiles() -> int:
    raw = (os.environ.get("BACH_MAX_INFERENCE_TILES") or "").strip()
    if not raw:
        return MAX_INFERENCE_TILES_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return MAX_INFERENCE_TILES_DEFAULT
    return max(1, value)


def _max_raster_pixels() -> int:
    raw = (os.environ.get("BACH_MAX_RASTER_PIXELS") or "").strip()
    if not raw:
        return MAX_RASTER_PIXELS_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return MAX_RASTER_PIXELS_DEFAULT
    return max(1, value)


def _iter_tile_positions(level_w: int, level_h: int, tile_size: int, stride: int):
    for y in range(0, level_h, stride):
        for x in range(0, level_w, stride):
            w = min(tile_size, level_w - x)
            h = min(tile_size, level_h - y)
            if w < tile_size // 2 or h < tile_size // 2:
                continue
            yield (x, y, w, h)


def _count_tile_positions(level_w: int, level_h: int, tile_size: int, stride: int) -> int:
    return sum(1 for _ in _iter_tile_positions(level_w, level_h, tile_size, stride))


def _select_openslide_level(slide, requested_level: str, tile_size: int, stride: int, target_tiles: int) -> int:
    value = (requested_level or "auto").strip().lower()
    if value != "auto":
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError("--level must be an integer or 'auto'") from exc

    best_level = 0
    best_tiles = None
    for level_idx, (level_w, level_h) in enumerate(slide.level_dimensions):
        tile_count = _count_tile_positions(level_w, level_h, tile_size, stride)
        if tile_count <= 0:
            continue
        if tile_count <= target_tiles:
            return level_idx
        if best_tiles is None or tile_count < best_tiles:
            best_level = level_idx
            best_tiles = tile_count
    return best_level


def _otsu_threshold(values_uint8: np.ndarray) -> int:
    """Return the Otsu threshold (0-255) that best separates a bimodal histogram.

    Standard between-class-variance maximization over a 256-bin histogram.
    Operates on the small thumbnail-resolution image, so the O(256) scan is
    negligible next to a single GPU forward pass.
    """
    hist, _ = np.histogram(values_uint8, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128

    bin_centers = np.arange(256, dtype=np.float64)
    sum_total = float(np.dot(bin_centers, hist))

    weight_bg = np.cumsum(hist)
    sum_bg = np.cumsum(bin_centers * hist)

    weight_fg = total - weight_bg
    # Avoid divide-by-zero for thresholds where one side is empty.
    safe_weight_bg = np.where(weight_bg > 0, weight_bg, 1)
    safe_weight_fg = np.where(weight_fg > 0, weight_fg, 1)

    mean_bg = sum_bg / safe_weight_bg
    mean_fg = (sum_total - sum_bg) / safe_weight_fg

    between_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_class_var = np.where((weight_bg > 0) & (weight_fg > 0), between_class_var, -1.0)

    return int(np.argmax(between_class_var))


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a boolean mask with a (2*radius+1)^2 square structuring element.

    Out-of-bounds neighbors are treated as background, so tissue regions
    touching the thumbnail edge shrink inward rather than wrapping/padding
    as tissue.
    """
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.ones((h, w), dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out &= padded[dy:dy + h, dx:dx + w]
    return out


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask with a (2*radius+1)^2 square structuring element."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def _morphological_cleanup(mask: np.ndarray, radius: int) -> np.ndarray:
    """Opening (erode->dilate) to drop isolated specks, then closing
    (dilate->erode) to fill small holes inside real tissue regions.
    """
    if radius <= 0:
        return mask
    opened = _binary_dilate(_binary_erode(mask, radius), radius)
    closed = _binary_erode(_binary_dilate(opened, radius), radius)
    return closed


def _tissue_mask_from_rgb_image(
    image: Image.Image,
    *,
    saturation_threshold: int = 38,
    use_otsu: bool = True,
    morph_radius: int = 2,
) -> np.ndarray:
    """Return a coarse mask of non-background tissue from an RGB-ish image.

    Combines two independent signals so each covers the other's blind spot:
      - Otsu threshold on luminosity: catches tissue that reads as darker
        than glass, with a per-slide threshold instead of a fixed constant,
        so it tracks staining-intensity variation slide to slide.
      - HSV saturation threshold: catches tissue that is faintly stained
        (similar luminosity to background) but still has real hue/chroma,
        while staying tight enough to reject the diffuse low-saturation
        blue tint that AB counterstain can leave across blank glass.
    A pixel counts as tissue if either signal fires. Morphological opening
    then closing removes isolated dye specks and fills small holes.

    Note: this is still a per-pixel heuristic. It cannot distinguish real
    tissue from a saturated, sharp-bordered artifact (e.g. dye precipitate)
    sitting on otherwise-blank glass, since both present as colored/dark
    pixels in isolation. See _tissue_context_mask for the complementary
    spatial check that disambiguates using neighborhood context instead of
    per-pixel color.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=bool)

    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)

    # HSV saturation in 0-255 scale: S = (max-min)/max, with max=0 -> S=0.
    safe_max = np.where(max_channel > 0, max_channel, 1.0)
    saturation = np.where(max_channel > 0, (max_channel - min_channel) / safe_max * 255.0, 0.0)
    saturated = saturation > saturation_threshold

    if use_otsu:
        # Standard luminosity weighting; tissue is darker than glass, so
        # tissue pixels fall below the Otsu threshold.
        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
        threshold = _otsu_threshold(gray_uint8)
        darker_than_glass = gray_uint8 < threshold
    else:
        darker_than_glass = np.zeros_like(saturated)

    tissue = saturated | darker_than_glass
    return _morphological_cleanup(tissue, morph_radius)


# =====================================================================
# BEGIN TEMP DIAGNOSTIC TOOLING — remove this block before merging.
# Lets you diff the current _tissue_mask_from_rgb_image against the old
# loose heuristic on a real slide, to check whether real tissue (not just
# background specks) is being clipped. Invoke with --diagnose-tissue-mask.
# =====================================================================

def _old_tissue_mask_from_rgb_image(
    image: Image.Image,
    *,
    background_threshold: int = 245,
    min_channel_delta: int = 8,
) -> np.ndarray:
    """The original (pre-Otsu/saturation) heuristic, kept only for diffing
    against the current mask. Not used anywhere in the inference path.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=bool)
    min_channel = arr.min(axis=2)
    max_channel = arr.max(axis=2)
    saturation_proxy = max_channel.astype(np.int16) - min_channel.astype(np.int16)
    dark_or_colored = (min_channel < background_threshold) | (saturation_proxy >= min_channel_delta)
    return dark_or_colored.astype(bool)


def _label_connected_components(mask: np.ndarray):
    """4-connected component labeling. Uses scipy if available, else falls
    back to a pure-numpy BFS. Returns (labels, num_components).
    """
    try:
        from scipy.ndimage import label as scipy_label
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labels, n = scipy_label(mask, structure=structure)
        return labels, n
    except ImportError:
        pass

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current_label += 1
            stack = [(y, x)]
            labels[y, x] = current_label
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current_label
                        stack.append((ny, nx))
    return labels, current_label


def _diagnostic_get_thumbnail(slide_path: Path, level_hint):
    """Load a flat RGB thumbnail for diagnostics, capped at the same
    TISSUE_MASK_MAX_DIMENSION used in production so the diff is
    apples-to-apples with what inference actually sees.
    """
    suffix = slide_path.suffix.lower()
    if suffix in RASTER_SLIDE_EXTENSIONS or suffix in (".tif", ".tiff", ".bmp"):
        img = Image.open(slide_path).convert("RGB")
        scale = min(1.0, TISSUE_MASK_MAX_DIMENSION / max(img.size))
        if scale < 1.0:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
        return img

    import openslide
    slide = openslide.OpenSlide(str(slide_path))
    level = level_hint if level_hint is not None else slide.level_count - 1
    level = max(0, min(level, slide.level_count - 1))
    level_w, level_h = slide.level_dimensions[level]
    scale = min(1.0, TISSUE_MASK_MAX_DIMENSION / max(level_w, level_h))
    thumb_size = (max(1, int(level_w * scale)), max(1, int(level_h * scale)))
    if hasattr(slide, "get_thumbnail"):
        return slide.get_thumbnail(thumb_size).convert("RGB")
    return slide.read_region((0, 0), level, thumb_size).convert("RGB")


def _run_tissue_mask_diagnostics(
    slide_path: Path,
    *,
    out_dir: Path,
    level_hint=None,
    top_n: int = 20,
    crop_margin: int = 20,
    context_radius: int = 0,
    context_fraction: float = 0.0,
) -> int:
    """Diff old vs new tissue mask on one slide; write an overlay PNG, the
    top-N largest 'removed' (old-tissue-now-background) blob crops sorted
    biggest first, and a text summary. Returns an EXIT_* code.

    If context_radius/context_fraction are set (nonzero), ALSO applies
    _build_effective_tissue_mask on top of the new base mask and reports a
    second overlay + removed-blob crops for base-vs-effective. This is the
    stage that actually gates inference (_build_effective_tissue_mask =
    base_mask AND context_mask), so diffing only the base color mask can
    look fine while the context filter is silently doing the real damage
    downstream -- pass the same --tissue-context-radius/--tissue-context-
    fraction you use (or would use) for real inference to see what it does.
    """
    from PIL import ImageDraw

    name = slide_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading thumbnail for {slide_path.name} ...")
    try:
        thumb = _diagnostic_get_thumbnail(slide_path, level_hint)
    except Exception as exc:
        print(f"Error: could not load thumbnail: {exc}", file=sys.stderr)
        return EXIT_SLIDE
    thumb_arr = np.asarray(thumb)

    print("Computing old and new tissue masks ...")
    old_mask = _old_tissue_mask_from_rgb_image(thumb)
    new_mask = _tissue_mask_from_rgb_image(thumb)
    if old_mask.shape != new_mask.shape:
        print("Error: mask shapes differ unexpectedly; aborting.", file=sys.stderr)
        return EXIT_SLIDE

    def _diff_and_report(stage_name: str, before_mask: np.ndarray, after_mask: np.ndarray):
        kept = before_mask & after_mask
        removed = before_mask & ~after_mask   # before said tissue, after disagrees -> scrutinize
        added = after_mask & ~before_mask     # after said tissue, before disagreed -> harmless, FYI

        total_before = int(before_mask.sum())
        total_after = int(after_mask.sum())
        pct_removed = (removed.sum() / total_before * 100) if total_before else 0.0

        overlay = thumb_arr.copy()
        overlay[kept] = (0.55 * overlay[kept] + 0.45 * np.array([60, 200, 60])).astype(np.uint8)
        overlay[removed] = (0.35 * overlay[removed] + 0.65 * np.array([230, 30, 30])).astype(np.uint8)
        overlay[added] = (0.6 * overlay[added] + 0.4 * np.array([40, 120, 230])).astype(np.uint8)
        overlay_path = out_dir / f"{name}_{stage_name}_diff_overlay.png"
        Image.fromarray(overlay).save(overlay_path)

        labels, n_components = _label_connected_components(removed)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        order = np.argsort(sizes)[::-1]
        order = order[sizes[order] > 0][:top_n]

        crop_dir = out_dir / f"{name}_{stage_name}_removed_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"[{stage_name}] before coverage: {total_before / before_mask.size:.1%}",
            f"[{stage_name}] after coverage: {total_after / after_mask.size:.1%}",
            f"[{stage_name}] removed pixels: {removed.sum()} ({pct_removed:.1f}% of before)",
            f"[{stage_name}] added pixels: {added.sum()}",
            f"[{stage_name}] removed connected components: {n_components}",
            "",
            f"[{stage_name}] largest removed blobs (review these first):",
            f"{'rank':>4}  {'pixels':>8}  {'bbox (y0,y1,x0,x1)':>26}  crop_file",
        ]
        for rank, label_id in enumerate(order, start=1):
            ys, xs = np.where(labels == label_id)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            size = int(sizes[label_id])
            m = crop_margin
            cy0, cy1 = max(0, y0 - m), min(thumb_arr.shape[0], y1 + m)
            cx0, cx1 = max(0, x0 - m), min(thumb_arr.shape[1], x1 + m)
            crop = Image.fromarray(thumb_arr[cy0:cy1, cx0:cx1].copy())
            draw = ImageDraw.Draw(crop)
            draw.rectangle([x0 - cx0, y0 - cy0, x1 - cx0 - 1, y1 - cy0 - 1], outline=(230, 30, 30), width=2)
            crop_name = f"{rank:03d}_size{size}_y{y0}-{y1}_x{x0}-{x1}.png"
            crop.save(crop_dir / crop_name)
            lines.append(f"{rank:>4}  {size:>8}  ({y0},{y1},{x0},{x1})  {crop_name}")
        return lines, overlay_path, crop_dir, len(order)

    all_lines = [f"Slide: {slide_path.name}", f"Thumbnail size: {thumb.size[0]}x{thumb.size[1]}", ""]

    stage1_lines, stage1_overlay, stage1_crops, stage1_n = _diff_and_report("base_mask", old_mask, new_mask)
    all_lines += ["=== STAGE 1: base color mask (old heuristic vs new Otsu+saturation) ==="] + stage1_lines
    print("\n".join(all_lines))
    print(f"\nOverlay: {stage1_overlay}")
    print(f"Crops:   {stage1_crops}/ ({stage1_n} saved)")

    if context_radius > 0 and context_fraction > 0:
        effective_mask = _build_effective_tissue_mask(
            new_mask, context_radius=context_radius, context_fraction=context_fraction
        )
        stage2_lines, stage2_overlay, stage2_crops, stage2_n = _diff_and_report(
            "context_filter", new_mask, effective_mask
        )
        all_lines += [
            "",
            f"=== STAGE 2: isolation-context filter (radius={context_radius}, "
            f"min_fraction={context_fraction}) applied on top of new base mask ===",
        ] + stage2_lines
        print("\n" + "\n".join(all_lines[-len(stage2_lines) - 3:]))
        print(f"\nOverlay: {stage2_overlay}")
        print(f"Crops:   {stage2_crops}/ ({stage2_n} saved)")
        print(
            "\nNOTE: this is the mask that actually gates inference "
            "(_build_effective_tissue_mask = base_mask AND context_mask). "
            "If stage 1 looked clean but you're still missing tissue in the "
            "app, stage 2 is almost certainly where it's happening."
        )
    else:
        print(
            "\nSkipped context-filter stage (context_radius/context_fraction not "
            "provided or <= 0). Pass --diagnose-context-radius / "
            "--diagnose-context-fraction matching what real inference uses "
            "(defaults: "
            f"{TISSUE_CONTEXT_RADIUS_DEFAULT}, {TISSUE_CONTEXT_FRACTION_DEFAULT}) "
            "to see its effect -- that filter is ON by default and is not "
            "covered by the stage-1 diff above."
        )

    summary_path = out_dir / f"{name}_diff_summary.txt"
    summary_path.write_text("\n".join(all_lines) + "\n")
    print(f"\nSummary: {summary_path}")
    return EXIT_SUCCESS

# =====================================================================
# END TEMP DIAGNOSTIC TOOLING
# =====================================================================


def _box_filter_mean(mask: np.ndarray, radius: int) -> np.ndarray:
    """Vectorized local-mean filter over a boolean mask using an integral image.

    Returns, for each cell, the fraction of True cells in a (2*radius+1)^2
    window centered on that cell (edges clipped to the array bounds, so the
    denominator shrinks near the border rather than padding with zeros).
    This is the building block for the isolation-context tissue check: it is
    O(mask size) regardless of radius, computed once per slide on the small
    thumbnail-resolution mask, so its cost is unmeasurable next to a single
    GPU forward pass.
    """
    if radius <= 0:
        return mask.astype(np.float32)

    h, w = mask.shape
    # Integral image with a leading zero row/col for easy windowed sums.
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)

    row_idx = np.arange(h)
    col_idx = np.arange(w)
    r0 = np.clip(row_idx - radius, 0, h)
    r1 = np.clip(row_idx + radius + 1, 0, h)
    c0 = np.clip(col_idx - radius, 0, w)
    c1 = np.clip(col_idx + radius + 1, 0, w)

    # Broadcast row/col window bounds into full 2D index grids.
    R0 = r0[:, None]
    R1 = r1[:, None]
    C0 = c0[None, :]
    C1 = c1[None, :]

    window_sum = (
        integral[R1, C1] - integral[R0, C1] - integral[R1, C0] + integral[R0, C0]
    )
    window_area = (R1 - R0) * (C1 - C0)
    window_area = np.maximum(window_area, 1)
    return (window_sum / window_area).astype(np.float32)


def _tissue_context_mask(
    base_mask: np.ndarray,
    *,
    radius: int,
    min_fraction: float,
) -> np.ndarray | None:
    """Return a mask of cells that have sufficient real tissue nearby.

    `base_mask` is the per-pixel/per-cell tissue-or-dye mask from
    _tissue_mask_from_rgb_image. A cell passes the context check if the
    fraction of True cells within a (2*radius+1)^2 neighborhood meets
    min_fraction. An isolated dye speck on blank glass has ~0% true-tissue
    neighbors and fails; a tile at the genuine periphery of a real tissue
    region still has many true-tissue neighbors and passes.

    Returns None if the filter is disabled (radius <= 0 or min_fraction <= 0),
    signaling callers to skip the context gate entirely.
    """
    if radius <= 0 or min_fraction <= 0:
        return None
    local_fraction = _box_filter_mean(base_mask, radius)
    return local_fraction >= min_fraction


def _tile_has_tissue(
    mask: np.ndarray | None,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    level_w: int,
    level_h: int,
    min_fraction: float,
) -> bool:
    if mask is None or min_fraction <= 0:
        return True
    mask_h, mask_w = mask.shape
    if mask_h <= 0 or mask_w <= 0 or level_w <= 0 or level_h <= 0:
        return True

    x0 = int(np.floor(x * mask_w / level_w))
    y0 = int(np.floor(y * mask_h / level_h))
    x1 = int(np.ceil((x + w) * mask_w / level_w))
    y1 = int(np.ceil((y + h) * mask_h / level_h))
    x0 = max(0, min(mask_w - 1, x0))
    y0 = max(0, min(mask_h - 1, y0))
    x1 = max(x0 + 1, min(mask_w, x1))
    y1 = max(y0 + 1, min(mask_h, y1))

    tile_mask = mask[y0:y1, x0:x1]
    if tile_mask.size == 0:
        return True
    return float(tile_mask.mean()) >= min_fraction


def _build_openslide_tissue_mask(slide, *, level: int) -> np.ndarray | None:
    """Build a small background/tissue mask for the selected inference level."""
    try:
        level_w, level_h = slide.level_dimensions[level]
        scale = min(1.0, TISSUE_MASK_MAX_DIMENSION / max(level_w, level_h))
        thumb_size = (max(1, int(level_w * scale)), max(1, int(level_h * scale)))
        if hasattr(slide, "get_thumbnail"):
            thumbnail = slide.get_thumbnail(thumb_size)
        else:
            thumbnail = slide.read_region((0, 0), level, thumb_size)
        return _tissue_mask_from_rgb_image(thumbnail)
    except Exception as exc:
        print(f"Warning: could not build tissue mask; processing all tiles: {exc}", file=sys.stderr)
        return None


def _build_effective_tissue_mask(
    base_mask: np.ndarray | None,
    *,
    context_radius: int,
    context_fraction: float,
) -> np.ndarray | None:
    """Combine the per-pixel tissue mask with the isolation-context gate.

    A cell is admitted only if it passes BOTH checks: it looks like tissue/dye
    by color (base_mask), AND it has enough real-tissue neighbors nearby
    (context mask). This is what excludes an isolated dye speck on blank
    glass while leaving true tissue -- including tissue near its own edge --
    unaffected, since edge tiles still have plenty of true-tissue neighbors
    within the context radius.
    """
    if base_mask is None:
        return None
    context_mask = _tissue_context_mask(
        base_mask, radius=context_radius, min_fraction=context_fraction
    )
    if context_mask is None:
        return base_mask
    return base_mask & context_mask


def _encode_binary_mask_bitpack(mask: np.ndarray) -> str:
    """Encode a boolean mask as row-major packed bits for compact JSON output."""
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).ravel(), bitorder="big")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _decode_binary_mask_bitpack(data: str, width: int, height: int) -> np.ndarray:
    """Inverse of _encode_binary_mask_bitpack: row-major packed bits -> bool [H, W]."""
    raw = np.frombuffer(base64.b64decode(data.encode("ascii")), dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="big")
    total = max(0, int(width)) * max(0, int(height))
    if bits.size < total:
        bits = np.concatenate([bits, np.zeros(total - bits.size, dtype=np.uint8)])
    return bits[:total].reshape(int(height), int(width)).astype(bool)


def _downsample_mask_array(mask: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a boolean mask by an integer factor using max (any) pooling.

    Maximizes retained detail: a coarse cell is positive if any covered native
    pixel was positive. Output dimensions never drop below 1x1.
    """
    if factor <= 1:
        return mask
    h, w = mask.shape
    new_h = max(1, (h + factor - 1) // factor)
    new_w = max(1, (w + factor - 1) // factor)
    pad_h = new_h * factor - h
    pad_w = new_w * factor - w
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)
    return mask.reshape(new_h, factor, new_w, factor).any(axis=(1, 3))


def _estimate_output_bytes(output: dict) -> int:
    """Serialized size of the output JSON, matching how it is written to disk."""
    return len(json.dumps(output, separators=(",", ":")).encode("utf-8"))


def _iter_mask_regions(regions):
    for region in regions:
        mask = region.get("prediction_mask")
        if isinstance(mask, dict) and mask.get("encoding") == "bitpack":
            yield region, mask


def _apply_mask_degradation(output: dict, budget: int | None) -> tuple[str, int | None]:
    """Coarsen per-tile masks so the serialized output fits ``budget`` bytes.

    Follows the degradation ladder (full -> adaptive downsample -> drop),
    stopping at the first step whose estimated output fits the budget. Detection
    box geometry and scores are never altered or dropped. Records the chosen step
    in ``output["mask_degradation"]`` and returns ``(status, factor)``.
    """
    regions = output.get("regions", [])
    output["mask_degradation"] = {"status": "full", "factor": 1}
    if not budget or budget <= 0:
        return "full", 1
    if _estimate_output_bytes(output) <= budget:
        return "full", 1

    mask_regions = list(_iter_mask_regions(regions))
    if not mask_regions:
        # Nothing left to coarsen (box-only already); the API safety net and the
        # region-count bound govern this residual case.
        return "full", 1

    # Snapshot native-resolution masks so each factor recomputes from source,
    # maximizing detail rather than compounding lossy passes.
    originals = [
        (region, _decode_binary_mask_bitpack(mask["data"], int(mask["width"]), int(mask["height"])))
        for region, mask in mask_regions
    ]
    max_dim = max(max(int(m["width"]), int(m["height"])) for _, m in mask_regions)

    # Progressively halve resolution (factor 2, 4, 8, ...) until it fits.
    factor = 2
    while True:
        for region, decoded in originals:
            reduced = _downsample_mask_array(decoded, factor)
            new_h, new_w = reduced.shape
            mask = region["prediction_mask"]
            mask["width"] = int(new_w)
            mask["height"] = int(new_h)
            mask["data"] = _encode_binary_mask_bitpack(reduced)
        output["mask_degradation"] = {"status": "downsampled", "factor": factor}
        if _estimate_output_bytes(output) <= budget:
            return "downsampled", factor
        if factor >= max_dim:
            # Masks are already 1x1 everywhere; no finer step remains.
            break
        factor *= 2

    # Last resort: drop all mask payloads, keep every detection box/score.
    for region, _ in originals:
        region.pop("prediction_mask", None)
    output["mask_degradation"] = {"status": "dropped", "factor": None}
    return "dropped", None


def _prediction_tile_region_from_prob_map(
    prob_map: np.ndarray,
    *,
    tile_w: int,
    tile_h: int,
    threshold: float,
) -> list[dict]:
    h, w = prob_map.shape
    if h <= 0 or w <= 0:
        return []
    mask = prob_map >= threshold
    if not bool(mask.any()):
        return []

    coverage = float(mask.mean())
    positive_values = prob_map[mask]
    score = float(positive_values.mean()) if positive_values.size else float(prob_map.mean())
    return [
        {
            "x": 0,
            "y": 0,
            "w": int(tile_w),
            "h": int(tile_h),
            "score": round(max(0.0, min(1.0, score)), 4),
            "payload": {
                "kind": "segmentation",
                "source": "prediction_tile",
                "coverage": round(max(0.0, min(1.0, coverage)), 4),
                "prediction_mask": {
                    "encoding": "bitpack",
                    "width": int(w),
                    "height": int(h),
                    "threshold": round(float(threshold), 4),
                    "data": _encode_binary_mask_bitpack(mask),
                },
            },
        }
    ]


def _score_seg_masks(seg_masks, tile_dims, args, tile_size, downsample=1.0):
    """
    Convert seg_masks dict from infer_with_neighborhood into a regions list.

    seg_masks  : {(x, y): prob_tensor [1, 1, H, W]}  - CPU tensors, probs 0-1
    tile_dims  : {(x, y): (w, h)}  - original tile dimensions before resize
    downsample : level-0 scale factor (1.0 for raster images)
    """
    regions = []
    for (x, y), prob_tensor in seg_masks.items():
        w, h = tile_dims[(x, y)]
        # prob_tensor is [1, 1, H, W]; squeeze to [H, W] for numpy ops
        prob_map = prob_tensor[0, 0].numpy()
        score = float(prob_map.mean())
        # Scale to level-0 coordinates (no-op for raster where downsample=1.0)
        x0 = int(round(x * downsample))
        y0 = int(round(y * downsample))
        x1 = int(round((x + w) * downsample))
        y1 = int(round((y + h) * downsample))
        w0 = max(1, x1 - x0)
        h0 = max(1, y1 - y0)

        localized = _prediction_tile_region_from_prob_map(
            prob_map,
            tile_w=w0,
            tile_h=h0,
            threshold=max(args.threshold, 0.20),
        )
        if localized:
            for loc in localized:
                payload = dict(loc.get("payload") or {})
                regions.append(
                    {
                        "x": int(x0 + loc["x"]),
                        "y": int(y0 + loc["y"]),
                        "w": int(loc["w"]),
                        "h": int(loc["h"]),
                        "score": float(loc["score"]),
                        "label": "fungus_positive",
                        **payload,
                    }
                )
            continue

        label = "fungus_positive" if score >= args.threshold else "fungus_negative"

        if args.positive_only and label != "fungus_positive":
            continue

        region = {"x": x0, "y": y0, "w": w0, "h": h0, "score": round(score, 4), "label": label}
        if label == "fungus_positive":
            region.update(
                {
                    "kind": "segmentation",
                    "source": "segmentation_tile",
                    "coverage": round(float((prob_map >= args.threshold).mean()), 4),
                }
            )
        regions.append(region)

    return regions


def _infer_regions_with_neighborhood(
    *,
    model,
    positions,
    load_tensor,
    device,
    args,
    tile_size,
    stride,
    downsample=1.0,
    k=1,
):
    if not positions:
        return []

    def to_grid(x, y):
        return y // stride, x // stride

    # First pass: classify density for each tile. Only hard class labels are
    # retained, so memory stays bounded by the current batch.
    density_preds = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(positions), args.batch_size):
            batch_positions = positions[i:i + args.batch_size]
            batch = torch.cat([load_tensor(pos) for pos in batch_positions], dim=0).to(device)
            _, density_logits, _, _ = model(batch)
            labels = density_logits.argmax(dim=1).cpu().tolist()
            for (x, y, _w, _h), label in zip(batch_positions, labels):
                density_preds[to_grid(x, y)] = label

    from collections import Counter
    label_counts = Counter(density_preds.values())
    print(f"[DIAG] Density distribution: {dict(label_counts)} (0=low,1=med,2=high,3=neg)", file=sys.stderr)


    def consensus_label(row, col):
        neighbors = [
            density_preds[(row + dr, col + dc)]
            for dr in range(-k, k + 1)
            for dc in range(-k, k + 1)
            if (row + dr, col + dc) in density_preds
        ]
        # Bias toward non-negative - if any neighbor is non-negative,
        # use majority vote among non-negative neighbors only.
        # Only return negative if every neighbor is negative.
        non_neg = [l for l in neighbors if l != 3]
        if non_neg:
            return max(sorted(set(non_neg)), key=non_neg.count)
        return 3

    regions = []
    with torch.no_grad():
        for i in range(0, len(positions), args.batch_size):
            batch_positions = positions[i:i + args.batch_size]
            batch = torch.cat([load_tensor(pos) for pos in batch_positions], dim=0).to(device)
            
            

            batch_labels = torch.tensor(
                [consensus_label(*to_grid(x, y)) for x, y, _w, _h in batch_positions],
                dtype=torch.long,
                device=device,
            )
            
            seg_logits, _, _, _ = model(batch, density_label=batch_labels)
            probs = torch.sigmoid(seg_logits).cpu()
            seg_masks = {
                (x, y): probs[j:j + 1]
                for j, (x, y, _w, _h) in enumerate(batch_positions)
            }
            tile_dims = {
                (x, y): (w, h)
                for x, y, w, h in batch_positions
            }
            # Score each batch immediately so segmentation masks do not
            # accumulate for the whole slide.
            regions.extend(_score_seg_masks(seg_masks, tile_dims, args, tile_size, downsample=downsample))

    return regions


def main():
    parser = argparse.ArgumentParser(description="Run fungus inference on a WSI")
    parser.add_argument("--slide-path", required=True, help="Path to slide (SVS/TIF/TIFF)")
    parser.add_argument("--output-json", required=False, default=None, help="Path to write output JSON")
    parser.add_argument("--checkpoint", required=False, default=None, help="Path to model checkpoint")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--level", default="auto")
    parser.add_argument("--target-tiles", type=int, default=TARGET_INFERENCE_TILES_DEFAULT)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.02)
    parser.add_argument("--no-skip-background", action="store_true")
    parser.add_argument(
        "--tissue-context-radius",
        type=int,
        default=TISSUE_CONTEXT_RADIUS_DEFAULT,
        help=(
            "Neighborhood radius (in mask cells) used for the isolation-context "
            "tissue filter, which excludes tiles with no real tissue nearby "
            "(e.g. dye artifacts on otherwise-blank glass). Only active if "
            "--tissue-context-fraction > 0."
        ),
    )
    parser.add_argument(
        "--tissue-context-fraction",
        type=float,
        default=TISSUE_CONTEXT_FRACTION_DEFAULT,
        help=(
            "Minimum fraction of tissue-positive cells required within the "
            "--tissue-context-radius neighborhood for a tile to be admitted. "
            "0 (default) disables the isolation-context filter entirely, "
            "preserving prior behavior. Start low (e.g. 0.05-0.10) and tune up; "
            "too high will clip genuine tissue-edge tiles."
        ),
    )
    parser.add_argument("--positive-only", action="store_true", help="Only output fungus_positive regions")
    parser.add_argument("--model-name", default="ResidualAttentionUNet")
    parser.add_argument("--model-version", default="1.0")
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=None,
        help="Coarsen per-tile masks so the written JSON stays within this byte budget",
    )
    # === BEGIN TEMP DIAGNOSTIC TOOLING — remove before merging ===
    parser.add_argument(
        "--diagnose-tissue-mask",
        default=None,
        metavar="OUT_DIR",
        help=(
            "Diff the current tissue mask against the old loose heuristic for "
            "--slide-path and write an overlay + largest-removed-blob crops to "
            "OUT_DIR, then exit without running inference. No --checkpoint or "
            "--output-json required in this mode."
        ),
    )
    parser.add_argument("--diagnose-top-n", type=int, default=20, help="Largest removed blobs to crop/save")
    parser.add_argument("--diagnose-crop-margin", type=int, default=20, help="Context padding (px) around each crop")
    parser.add_argument(
        "--diagnose-context-radius",
        type=int,
        default=None,
        help="Context radius to test in the diagnostic (default: same as --tissue-context-radius / module default)",
    )
    parser.add_argument(
        "--diagnose-context-fraction",
        type=float,
        default=None,
        help="Context fraction to test in the diagnostic (default: same as --tissue-context-fraction / module default)",
    )
    # === END TEMP DIAGNOSTIC TOOLING ===
    args = parser.parse_args()

    slide_path = Path(args.slide_path)
    if not slide_path.exists():
        print(f"Error: Slide not found: {slide_path}", file=sys.stderr)
        return EXIT_ARGS

    # === BEGIN TEMP DIAGNOSTIC TOOLING — remove before merging ===
    if args.diagnose_tissue_mask:
        level_hint = None
        if args.level not in (None, "auto"):
            try:
                level_hint = int(args.level)
            except ValueError:
                level_hint = None
        diag_radius = args.diagnose_context_radius
        if diag_radius is None:
            diag_radius = args.tissue_context_radius if args.tissue_context_radius is not None else TISSUE_CONTEXT_RADIUS_DEFAULT
        diag_fraction = args.diagnose_context_fraction
        if diag_fraction is None:
            diag_fraction = args.tissue_context_fraction if args.tissue_context_fraction is not None else TISSUE_CONTEXT_FRACTION_DEFAULT
        return _run_tissue_mask_diagnostics(
            slide_path,
            out_dir=Path(args.diagnose_tissue_mask),
            level_hint=level_hint,
            top_n=args.diagnose_top_n,
            crop_margin=args.diagnose_crop_margin,
            context_radius=diag_radius,
            context_fraction=diag_fraction,
        )
    # === END TEMP DIAGNOSTIC TOOLING ===

    if args.output_json is None:
        print("Error: --output-json is required (unless using --diagnose-tissue-mask)", file=sys.stderr)
        return EXIT_ARGS
    if args.checkpoint is None:
        print("Error: --checkpoint is required (unless using --diagnose-tissue-mask)", file=sys.stderr)
        return EXIT_ARGS

    output_path = Path(args.output_json)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return EXIT_ARGS
    if args.tile_size <= 0:
        print("Error: --tile-size must be > 0", file=sys.stderr)
        return EXIT_ARGS
    if args.stride <= 0:
        print("Error: --stride must be > 0", file=sys.stderr)
        return EXIT_ARGS
    if args.target_tiles <= 0:
        print("Error: --target-tiles must be > 0", file=sys.stderr)
        return EXIT_ARGS
    if args.batch_size <= 0:
        print("Error: --batch-size must be > 0", file=sys.stderr)
        return EXIT_ARGS
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: --threshold must be in [0.0, 1.0]", file=sys.stderr)
        return EXIT_ARGS
    if not (0.0 <= args.min_tissue_fraction <= 1.0):
        print("Error: --min-tissue-fraction must be in [0.0, 1.0]", file=sys.stderr)
        return EXIT_ARGS
    if not (0.0 <= args.tissue_context_fraction <= 1.0):
        print("Error: --tissue-context-fraction must be in [0.0, 1.0]", file=sys.stderr)
        return EXIT_ARGS
    if args.tissue_context_radius < 0:
        print("Error: --tissue-context-radius must be >= 0", file=sys.stderr)
        return EXIT_ARGS
    if args.tile_size > 8192:
        print("Error: --tile-size must be <= 8192", file=sys.stderr)
        return EXIT_ARGS
    if args.stride > 8192:
        print("Error: --stride must be <= 8192", file=sys.stderr)
        return EXIT_ARGS

    # Load model
    try:
        device = _select_device(args.device)
        ckpt = _load_checkpoint_dict(checkpoint_path, device)
        state_dict, ckpt_meta = _extract_model_state_dict(ckpt)

        model_kwargs = {"in_ch": 3, "out_ch": 1, "num_density_classes": 4, "dropout_p": 0.3}
        ckpt_cfg = ckpt_meta.get("cfg") if isinstance(ckpt_meta, Mapping) else None
        if isinstance(ckpt_cfg, dict):
            ckpt_model_cfg = ckpt_cfg.get("model")
            if isinstance(ckpt_model_cfg, dict):
                model_kwargs.update(ckpt_model_cfg)
        model = ResidualAttentionUNet(**model_kwargs)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return EXIT_ARGS

    tile_size = args.tile_size
    stride = args.stride
    max_tiles = _max_inference_tiles()

    # ----------------------------------------------------------------
    # Raster image (PNG/JPEG): no OpenSlide
    # ----------------------------------------------------------------
    if slide_path.suffix.lower() in RASTER_SLIDE_EXTENSIONS:
        try:
            full_img = Image.open(slide_path)
            level_w, level_h = full_img.size
            if (level_w * level_h) > _max_raster_pixels():
                full_img.close()
                print(
                    f"Error: Raster image too large ({level_w}x{level_h}); "
                    "refusing to load into memory. Use WSI format or raise BACH_MAX_RASTER_PIXELS.",
                    file=sys.stderr,
                )
                return EXIT_SLIDE
            full_img = full_img.convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}", file=sys.stderr)
            return EXIT_SLIDE

        if str(args.level).strip().lower() not in {"0", "auto"}:
            print("Error: Raster images only support --level 0 or auto", file=sys.stderr)
            return EXIT_SLIDE
        level = 0
        dims = (level_w, level_h)

        total_tiles = _count_tile_positions(level_w, level_h, tile_size, stride)
        if total_tiles <= 0:
            print("Error: No tiles to process", file=sys.stderr)
            return EXIT_SLIDE
        if total_tiles > max_tiles:
            print(
                f"Error: Refusing to process {total_tiles} tiles "
                f"(limit={max_tiles}; override with BACH_MAX_INFERENCE_TILES).",
                file=sys.stderr,
            )
            return EXIT_SLIDE

        tissue_mask = None
        if not args.no_skip_background:
            scale = min(1.0, TISSUE_MASK_MAX_DIMENSION / max(level_w, level_h))
            mask_size = (max(1, int(level_w * scale)), max(1, int(level_h * scale)))
            base_mask = _tissue_mask_from_rgb_image(full_img.resize(mask_size, Image.BILINEAR))
            tissue_mask = _build_effective_tissue_mask(
                base_mask,
                context_radius=args.tissue_context_radius,
                context_fraction=args.tissue_context_fraction,
            )

        # Store coordinates only; tile tensors are loaded batch-by-batch during
        # the two inference passes to avoid materializing the whole slide.
        positions = []
        skipped_background = 0

        try:
            for x, y, w, h in _iter_tile_positions(level_w, level_h, tile_size, stride):
                if not _tile_has_tissue(
                    tissue_mask, x=x, y=y, w=w, h=h,
                    level_w=level_w, level_h=level_h,
                    min_fraction=args.min_tissue_fraction,
                ):
                    skipped_background += 1
                    continue
                positions.append((x, y, w, h))
        except Exception as e:
            print(f"Error collecting tiles: {e}", file=sys.stderr)
            return EXIT_SLIDE

        inferred_tiles = len(positions)

        def load_tensor(pos):
            x, y, w, h = pos
            region = full_img.crop((x, y, x + w, y + h))
            if w != tile_size or h != tile_size:
                region = region.resize((tile_size, tile_size), Image.BILINEAR)
            return preprocess_tile(region, tile_size)

        try:
            regions = _infer_regions_with_neighborhood(
                model=model,
                positions=positions,
                load_tensor=load_tensor,
                device=device,
                args=args,
                tile_size=tile_size,
                stride=stride,
                downsample=1.0,
            )
        except Exception as e:
            print(f"Inference error: {e}", file=sys.stderr)
            return EXIT_INFERENCE

    # ----------------------------------------------------------------
    # OpenSlide WSI
    # ----------------------------------------------------------------
    else:
        try:
            import openslide
        except Exception as e:
            print(f"Error importing OpenSlide: {e}", file=sys.stderr)
            return EXIT_SLIDE
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as e:
            print(f"Error opening slide: {e}", file=sys.stderr)
            return EXIT_SLIDE

        try:
            try:
                level = _select_openslide_level(
                    slide, str(args.level), tile_size, stride, args.target_tiles,
                )
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                slide.close()
                return EXIT_ARGS

            if level < 0 or level >= slide.level_count:
                print(f"Error: Invalid level {level} (slide has {slide.level_count} levels)", file=sys.stderr)
                slide.close()
                return EXIT_SLIDE

            level_w, level_h = slide.level_dimensions[level]
            dims = slide.dimensions
            total_tiles = _count_tile_positions(level_w, level_h, tile_size, stride)
            if total_tiles <= 0:
                print("Error: No tiles to process", file=sys.stderr)
                slide.close()
                return EXIT_SLIDE
            if total_tiles > max_tiles:
                print(
                    f"Error: Refusing to process {total_tiles} tiles "
                    f"(limit={max_tiles}; override with BACH_MAX_INFERENCE_TILES).",
                    file=sys.stderr,
                )
                slide.close()
                return EXIT_SLIDE

            downsample = float(slide.level_downsamples[level])
            tissue_mask = None
            if not args.no_skip_background:
                base_mask = _build_openslide_tissue_mask(slide, level=level)
                tissue_mask = _build_effective_tissue_mask(
                    base_mask,
                    context_radius=args.tissue_context_radius,
                    context_fraction=args.tissue_context_fraction,
                )
            _debug_base_mask_cells = int(base_mask.sum()) if base_mask is not None else None
            _debug_effective_mask_cells = int(tissue_mask.sum()) if tissue_mask is not None else None
            _debug_total_mask_cells = int(base_mask.size) if base_mask is not None else None
            # Store coordinates only; tile tensors are loaded batch-by-batch
            # during each pass, which keeps memory tied to batch size.
            positions = []
            skipped_background = 0

            for x, y, w, h in _iter_tile_positions(level_w, level_h, tile_size, stride):
                if not _tile_has_tissue(
                    tissue_mask, x=x, y=y, w=w, h=h,
                    level_w=level_w, level_h=level_h,
                    min_fraction=args.min_tissue_fraction,
                ):
                    skipped_background += 1
                    continue
                positions.append((x, y, w, h))

            inferred_tiles = len(positions)

            def load_tensor(pos):
                x, y, w, h = pos
                x0, y0 = int(x * downsample), int(y * downsample)
                region = slide.read_region((x0, y0), level, (w, h)).convert("RGB")
                if w != tile_size or h != tile_size:
                    region = region.resize((tile_size, tile_size), Image.BILINEAR)
                return preprocess_tile(region, tile_size)

            try:
                regions = _infer_regions_with_neighborhood(
                    model=model,
                    positions=positions,
                    load_tensor=load_tensor,
                    device=device,
                    args=args,
                    tile_size=tile_size,
                    stride=stride,
                    downsample=downsample,
                )
            except Exception as e:
                slide.close()
                print(f"Inference error: {e}", file=sys.stderr)
                return EXIT_INFERENCE

            slide.close()

        except Exception as e:
            try:
                slide.close()
            except Exception:
                pass
            print(f"Error processing slide: {e}", file=sys.stderr)
            return EXIT_SLIDE

    n_pos = sum(1 for r in regions if r["label"] == "fungus_positive")
    n_neg = sum(1 for r in regions if r["label"] == "fungus_negative")

    output = {
        "model_name": args.model_name,
        "model_version": args.model_version,
        "slide_path": str(slide_path),
        "tile_size": tile_size,
        "stride": stride,
        "level": level,
        "requested_level": args.level,
        "target_tiles": args.target_tiles,
        "threshold": args.threshold,
        "slide_dimensions": list(dims),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_tiles": total_tiles,
            "inferred_tiles": inferred_tiles,
            "skipped_background": skipped_background,
            "fungus_positive": n_pos,
            "fungus_negative": n_neg,
       },
       "debug": {
           "tissue_context_radius": args.tissue_context_radius,
           "tissue_context_fraction": args.tissue_context_fraction,
           "no_skip_background": args.no_skip_background,
           "base_mask_tissue_cells": _debug_base_mask_cells,
           "effective_mask_tissue_cells": _debug_effective_mask_cells,
           "total_mask_cells": _debug_total_mask_cells,
        },
        "regions": regions,
    }

    # Coarsen masks at the source so the file written to disk already fits the
    # budget handed down from the API worker (90% of the API hard limit).
    _apply_mask_degradation(output, args.max_output_bytes)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, separators=(",", ":"))
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return EXIT_OUTPUT

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())