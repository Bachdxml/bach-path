#!/usr/bin/env python3
"""
Run fungus segmentation inference on a whole slide image.
Outputs a JSON file with per-tile fungus_positive/fungus_negative labels.

Usage:
  python run_inference_api.py --slide-path /path/to/slide.svs --output-json out.json --checkpoint checkpoints/best_model.pth
  python run_inference_api.py --slide-path slide.svs --output-json out.json --checkpoint best.pth --device cuda:0
"""

import argparse
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

# Formats OpenSlide cannot open — run tile inference from a raster in memory.
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


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components over a small boolean grid."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    stack.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    stack.append((cy, cx + 1))
            components.append(cells)
    return components


def _tile_regions_from_prob_map(
    prob_map: np.ndarray,
    *,
    tile_w: int,
    tile_h: int,
    tile_size: int,
    threshold: float,
    max_components: int = 8,
    min_cells: int = 2,
) -> list[dict]:
    """
    Build localized subregions from a tile segmentation probability map.
    Uses a coarse cell grid to keep region count bounded while preserving locality.
    """
    h, w = prob_map.shape
    if h <= 0 or w <= 0:
        return []

    cell_size = 32
    grid_h = max(1, int(np.ceil(h / cell_size)))
    grid_w = max(1, int(np.ceil(w / cell_size)))
    grid_scores = np.zeros((grid_h, grid_w), dtype=np.float32)

    for gy in range(grid_h):
        y0 = gy * cell_size
        y1 = min(h, y0 + cell_size)
        for gx in range(grid_w):
            x0 = gx * cell_size
            x1 = min(w, x0 + cell_size)
            patch = prob_map[y0:y1, x0:x1]
            grid_scores[gy, gx] = float(patch.mean()) if patch.size else 0.0

    adaptive_threshold = float(np.quantile(grid_scores, 0.80))
    component_threshold = max(threshold, adaptive_threshold)

    mask = grid_scores >= component_threshold
    components = _connected_components(mask)
    if not components and component_threshold > threshold:
        mask = grid_scores >= threshold
        components = _connected_components(mask)

    regions = []
    for cells in components:
        if len(cells) < min_cells:
            continue
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        gy0, gy1 = min(ys), max(ys) + 1
        gx0, gx1 = min(xs), max(xs) + 1
        score = float(np.mean([grid_scores[cy, cx] for cy, cx in cells]))

        # Map coarse-grid bbox back into original (possibly edge-clipped) tile dimensions.
        x0 = int(round((gx0 * cell_size) * tile_w / tile_size))
        y0 = int(round((gy0 * cell_size) * tile_h / tile_size))
        x1 = int(round((min(w, gx1 * cell_size)) * tile_w / tile_size))
        y1 = int(round((min(h, gy1 * cell_size)) * tile_h / tile_size))
        x0 = max(0, min(tile_w - 1, x0))
        y0 = max(0, min(tile_h - 1, y0))
        x1 = max(x0 + 1, min(tile_w, x1))
        y1 = max(y0 + 1, min(tile_h, y1))

        regions.append(
            {
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "score": round(max(0.0, min(1.0, score)), 4),
            }
        )

    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions[:max_components]


def _compute_hotspot(prob_map: torch.Tensor, *, x: int, y: int, w: int, h: int) -> dict | None:
    """
    Estimate a fungus hotspot inside a tile from the segmentation probability map.
    Returns absolute slide-space coordinates plus a localized bbox for rendering.
    """
    if w <= 0 or h <= 0:
        return None

    data = prob_map.detach().float()
    if data.ndim == 3:
        data = data.squeeze(0)
    if data.ndim != 2 or data.numel() == 0:
        return None

    map_h, map_w = data.shape
    mean_prob = float(data.mean().item())
    cutoff = max(0.05, min(0.8, mean_prob * 1.25))
    weights = torch.clamp(data - cutoff, min=0.0)
    if float(weights.sum().item()) <= 1e-6:
        weights = torch.clamp(data - mean_prob, min=0.0)
    if float(weights.sum().item()) <= 1e-6:
        weights = torch.clamp(data, min=0.0)

    total = float(weights.sum().item())
    if total <= 1e-6:
        return None

    active = weights > 0
    if not bool(active.any().item()):
        active = data >= max(float(data.max().item()) * 0.85, mean_prob)

    xs = torch.arange(map_w, device=data.device, dtype=torch.float32).unsqueeze(0).expand(map_h, map_w)
    ys = torch.arange(map_h, device=data.device, dtype=torch.float32).unsqueeze(1).expand(map_h, map_w)
    cx = float((weights * xs).sum().item() / total)
    cy = float((weights * ys).sum().item() / total)

    active_ys, active_xs = torch.where(active)
    if active_xs.numel() == 0 or active_ys.numel() == 0:
        x_min = x_max = int(round(cx))
        y_min = y_max = int(round(cy))
    else:
        x_min = int(active_xs.min().item())
        x_max = int(active_xs.max().item())
        y_min = int(active_ys.min().item())
        y_max = int(active_ys.max().item())

    scale_x = w / map_w
    scale_y = h / map_h
    hotspot_x = x + (x_min * scale_x)
    hotspot_y = y + (y_min * scale_y)
    hotspot_w = max(scale_x, (x_max - x_min + 1) * scale_x)
    hotspot_h = max(scale_y, (y_max - y_min + 1) * scale_y)
    hotspot_cx = x + ((cx + 0.5) * scale_x)
    hotspot_cy = y + ((cy + 0.5) * scale_y)

    return {
        "hotspot": {
            "cx": round(hotspot_cx, 2),
            "cy": round(hotspot_cy, 2),
            "x": round(hotspot_x, 2),
            "y": round(hotspot_y, 2),
            "w": round(hotspot_w, 2),
            "h": round(hotspot_h, 2),
            "coverage": round(float(active.float().mean().item()), 4),
            "source": "segmentation_centroid",
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run fungus inference on a WSI")
    parser.add_argument("--slide-path", required=True, help="Path to slide (SVS/TIF/TIFF)")
    parser.add_argument("--output-json", required=True, help="Path to write output JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--positive-only", action="store_true", help="Only output fungus_positive regions")
    parser.add_argument("--model-name", default="ResidualAttentionUNet")
    parser.add_argument("--model-version", default="1.0")
    args = parser.parse_args()

    slide_path = Path(args.slide_path)
    output_path = Path(args.output_json)
    checkpoint_path = Path(args.checkpoint)

    if not slide_path.exists():
        print(f"Error: Slide not found: {slide_path}", file=sys.stderr)
        return EXIT_ARGS
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return EXIT_ARGS
    if args.tile_size <= 0:
        print("Error: --tile-size must be > 0", file=sys.stderr)
        return EXIT_ARGS
    if args.stride <= 0:
        print("Error: --stride must be > 0", file=sys.stderr)
        return EXIT_ARGS
    if args.batch_size <= 0:
        print("Error: --batch-size must be > 0", file=sys.stderr)
        return EXIT_ARGS
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: --threshold must be in [0.0, 1.0]", file=sys.stderr)
        return EXIT_ARGS
    if args.tile_size > 8192:
        print("Error: --tile-size must be <= 8192", file=sys.stderr)
        return EXIT_ARGS
    if args.stride > 8192:
        print("Error: --stride must be <= 8192", file=sys.stderr)
        return EXIT_ARGS

    # Load model
    try:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    level = args.level
    max_tiles = _max_inference_tiles()

    # --- Raster image (PNG/JPEG): no OpenSlide ---
    if slide_path.suffix.lower() in RASTER_SLIDE_EXTENSIONS:
        try:
            full_img = Image.open(slide_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}", file=sys.stderr)
            return EXIT_SLIDE

        if level != 0:
            print("Error: Raster images only support --level 0", file=sys.stderr)
            return EXIT_SLIDE

        level_w, level_h = full_img.size
        if (level_w * level_h) > _max_raster_pixels():
            print(
                f"Error: Raster image too large ({level_w}x{level_h}); "
                "refusing to load into memory. Use WSI format or raise BACH_MAX_RASTER_PIXELS.",
                file=sys.stderr,
            )
            return EXIT_SLIDE
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

        regions = []
        batch_size = args.batch_size
        density_label = torch.tensor([3] * batch_size, dtype=torch.long, device=device)
        tile_iter = _iter_tile_positions(level_w, level_h, tile_size, stride)

        try:
            for i in range(0, total_tiles, batch_size):
                batch_positions = []
                for _ in range(batch_size):
                    pos = next(tile_iter, None)
                    if pos is None:
                        break
                    batch_positions.append(pos)
                if not batch_positions:
                    break
                batch_tensors = []

                for x, y, w, h in batch_positions:
                    region = full_img.crop((x, y, x + w, y + h))
                    if w != tile_size or h != tile_size:
                        region = region.resize((tile_size, tile_size), Image.BILINEAR)
                    tensor = preprocess_tile(region, tile_size)
                    batch_tensors.append(tensor)

                batch = torch.cat(batch_tensors, dim=0).to(device)
                density_batch = density_label[: len(batch_positions)]

                try:
                    with torch.no_grad():
                        seg_logits, _, _, _ = model(batch, density_batch)
                    probs = torch.sigmoid(seg_logits)
                except Exception as e:
                    print(f"Inference error at batch {i // batch_size}: {e}", file=sys.stderr)
                    return EXIT_INFERENCE

                for j, (x, y, w, h) in enumerate(batch_positions):
                    prob_map = probs[j, 0].detach().cpu().numpy()
                    score = float(prob_map.mean())
                    label = "fungus_positive" if score >= args.threshold else "fungus_negative"
                    if label == "fungus_positive":
                        localized = _tile_regions_from_prob_map(
                            prob_map,
                            tile_w=w,
                            tile_h=h,
                            tile_size=tile_size,
                            threshold=max(args.threshold, 0.20),
                        )
                        if localized:
                            for loc in localized:
                                regions.append(
                                    {
                                        "x": int(x + loc["x"]),
                                        "y": int(y + loc["y"]),
                                        "w": int(loc["w"]),
                                        "h": int(loc["h"]),
                                        "score": float(loc["score"]),
                                        "label": "fungus_positive",
                                    }
                                )
                            continue
                    if args.positive_only and label != "fungus_positive":
                        continue
                    region = {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "score": round(score, 4),
                        "label": label,
                    }
                    hotspot_payload = _compute_hotspot(probs[j], x=int(x), y=int(y), w=int(w), h=int(h))
                    if hotspot_payload:
                        region.update(hotspot_payload)
                    regions.append(region)
        except Exception as e:
            print(f"Error processing image: {e}", file=sys.stderr)
            return EXIT_SLIDE

    else:
        # --- OpenSlide WSI ---
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

            regions = []
            batch_size = args.batch_size

            # Density label: 3 = negative (unknown at inference)
            density_label = torch.tensor([3] * batch_size, dtype=torch.long, device=device)
            downsample = float(slide.level_downsamples[level])
            tile_iter = _iter_tile_positions(level_w, level_h, tile_size, stride)

            for i in range(0, total_tiles, batch_size):
                batch_positions = []
                for _ in range(batch_size):
                    pos = next(tile_iter, None)
                    if pos is None:
                        break
                    batch_positions.append(pos)
                if not batch_positions:
                    break
                batch_tensors = []

                for x, y, w, h in batch_positions:
                    # read_region uses level-0 coordinates
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    region = slide.read_region((x0, y0), level, (w, h))
                    region = region.convert("RGB")
                    # Pad/resize to tile_size if needed
                    if w != tile_size or h != tile_size:
                        region = region.resize((tile_size, tile_size), Image.BILINEAR)
                    tensor = preprocess_tile(region, tile_size)
                    batch_tensors.append(tensor)

                batch = torch.cat(batch_tensors, dim=0).to(device)
                density_batch = density_label[: len(batch_positions)]

                try:
                    with torch.no_grad():
                        seg_logits, _, _, _ = model(batch, density_batch)
                    probs = torch.sigmoid(seg_logits)
                except Exception as e:
                    print(f"Inference error at batch {i // batch_size}: {e}", file=sys.stderr)
                    slide.close()
                    return EXIT_INFERENCE

                for j, (x, y, w, h) in enumerate(batch_positions):
                    prob_map = probs[j, 0].detach().cpu().numpy()
                    score = float(prob_map.mean())
                    label = "fungus_positive" if score >= args.threshold else "fungus_negative"
                    # Convert to level-0 coordinates for API/Region
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    w0 = int(w * downsample)
                    h0 = int(h * downsample)
                    if label == "fungus_positive":
                        localized = _tile_regions_from_prob_map(
                            prob_map,
                            tile_w=w0,
                            tile_h=h0,
                            tile_size=tile_size,
                            threshold=max(args.threshold, 0.20),
                        )
                        if localized:
                            for loc in localized:
                                regions.append(
                                    {
                                        "x": int(x0 + loc["x"]),
                                        "y": int(y0 + loc["y"]),
                                        "w": int(loc["w"]),
                                        "h": int(loc["h"]),
                                        "score": float(loc["score"]),
                                        "label": "fungus_positive",
                                    }
                                )
                            continue
                    if args.positive_only and label != "fungus_positive":
                        continue
                    region = {
                        "x": x0,
                        "y": y0,
                        "w": w0,
                        "h": h0,
                        "score": round(score, 4),
                        "label": label,
                    }
                    hotspot_payload = _compute_hotspot(probs[j], x=x0, y=y0, w=w0, h=h0)
                    if hotspot_payload:
                        region.update(hotspot_payload)
                    regions.append(region)

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
        "threshold": args.threshold,
        "slide_dimensions": list(dims),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_tiles": total_tiles,
            "fungus_positive": n_pos,
            "fungus_negative": n_neg,
        },
        "regions": regions,
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return EXIT_OUTPUT

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
