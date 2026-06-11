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


def _tissue_mask_from_rgb_image(
    image: Image.Image,
    *,
    background_threshold: int = 245,
    min_channel_delta: int = 8,
) -> np.ndarray:
    """Return a coarse mask of non-background tissue from an RGB-ish image."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=bool)
    min_channel = arr.min(axis=2)
    max_channel = arr.max(axis=2)
    saturation_proxy = max_channel.astype(np.int16) - min_channel.astype(np.int16)
    dark_or_colored = (min_channel < background_threshold) | (saturation_proxy >= min_channel_delta)
    return dark_or_colored.astype(bool)


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


def _encode_binary_mask_bitpack(mask: np.ndarray) -> str:
    """Encode a boolean mask as row-major packed bits for compact JSON output."""
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).ravel(), bitorder="big")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _prediction_tile_region_from_prob_map(
    prob_map: np.ndarray,
    *,
    tile_w: int,
    tile_h: int,
    threshold: float,
) -> list[dict]:
    """Return prediction-tile region dicts.

    Each returned dict carries the raw boolean mask under the private key
    ``_mask`` so the caller can write it to the npz sidecar; the caller must
    strip ``_mask`` before placing the region in the JSON payload.
    """
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
            "_mask": np.ascontiguousarray(mask, dtype=bool),
            "payload": {
                "kind": "segmentation",
                "source": "prediction_tile",
                "coverage": round(max(0.0, min(1.0, coverage)), 4),
                "threshold": round(float(threshold), 4),
                "has_mask": True,
            },
        }
    ]


def _score_seg_masks(seg_masks, tile_dims, args, tile_size, downsample=1.0):
    """
    Convert seg_masks dict from infer_with_neighborhood into a regions list.

    seg_masks  : {(x, y): prob_tensor [1, 1, H, W]}  - CPU tensors, probs 0-1
    tile_dims  : {(x, y): (w, h)}  - original tile dimensions before resize
    downsample : level-0 scale factor (1.0 for raster images)

    Returns (regions, masks_by_key) where masks_by_key maps the final
    "{x}_{y}" level-0 region key to the 2D boolean prediction mask array.
    """
    regions = []
    masks_by_key = {}
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
                fx = int(x0 + loc["x"])
                fy = int(y0 + loc["y"])
                loc_mask = loc.get("_mask")
                if loc_mask is not None:
                    masks_by_key[f"{fx}_{fy}"] = loc_mask
                regions.append(
                    {
                        "x": fx,
                        "y": fy,
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

    return regions, masks_by_key


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
        return [], {}

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
    masks_by_key = {}
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
            batch_regions, batch_masks = _score_seg_masks(
                seg_masks, tile_dims, args, tile_size, downsample=downsample
            )
            regions.extend(batch_regions)
            masks_by_key.update(batch_masks)

    return regions, masks_by_key


def main():
    parser = argparse.ArgumentParser(description="Run fungus inference on a WSI")
    parser.add_argument("--slide-path", required=True, help="Path to slide (SVS/TIF/TIFF)")
    parser.add_argument("--output-json", required=True, help="Path to write output JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--level", default="auto")
    parser.add_argument("--target-tiles", type=int, default=TARGET_INFERENCE_TILES_DEFAULT)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.02)
    parser.add_argument("--no-skip-background", action="store_true")
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
            tissue_mask = _tissue_mask_from_rgb_image(full_img.resize(mask_size, Image.BILINEAR))

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
            regions, masks_by_key = _infer_regions_with_neighborhood(
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
            tissue_mask = None if args.no_skip_background else _build_openslide_tissue_mask(slide, level=level)

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
                regions, masks_by_key = _infer_regions_with_neighborhood(
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
        "regions": regions,
    }

    mask_threshold = max(args.threshold, 0.20)
    sidecar_path = output_path.with_name(output_path.stem + "_masks.npz")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, separators=(",", ":"))
        # Masks live in a compact compressed sidecar so the JSON stays small.
        npz_arrays = {key: mask for key, mask in masks_by_key.items()}
        npz_arrays["__threshold__"] = np.float32(mask_threshold)
        np.savez_compressed(sidecar_path, **npz_arrays)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return EXIT_OUTPUT

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
