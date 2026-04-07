#!/usr/bin/env python3
"""
Run fungus segmentation inference on a whole slide image.
Outputs a JSON file with per-tile fungus_positive/fungus_negative labels.

Usage:
  python run_inference_api.py --slide-path /path/to/slide.svs --output-json out.json --checkpoint checkpoints/best_model.pth
  python run_inference_api.py --slide-path slide.svs --output-json out.json --checkpoint best.pth --device cuda:0
"""

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def _load_checkpoint_dict(path: Path, device: torch.device) -> dict:
    """Load torch checkpoint from plain or gzip-compressed file (.pth.gz)."""
    name = path.name.lower()
    if path.suffix.lower() == ".gz" or name.endswith((".pth.gz", ".pt.gz")):
        with gzip.open(path, "rb") as f:
            try:
                return torch.load(f, map_location=device, weights_only=True)
            except TypeError:
                return torch.load(f, map_location=device)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


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

    # Load model
    try:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        ckpt = _load_checkpoint_dict(checkpoint_path, device)

        model_kwargs = {"in_ch": 3, "out_ch": 1, "num_density_classes": 4, "dropout_p": 0.3}
        ckpt_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
        if isinstance(ckpt_cfg, dict):
            ckpt_model_cfg = ckpt_cfg.get("model")
            if isinstance(ckpt_model_cfg, dict):
                model_kwargs.update(ckpt_model_cfg)
        model = ResidualAttentionUNet(**model_kwargs)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return EXIT_ARGS

    tile_size = args.tile_size
    stride = args.stride
    level = args.level

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
        dims = (level_w, level_h)

        tiles_x = list(range(0, level_w, stride))
        tiles_y = list(range(0, level_h, stride))
        tile_positions = []
        for y in tiles_y:
            for x in tiles_x:
                w = min(tile_size, level_w - x)
                h = min(tile_size, level_h - y)
                if w < tile_size // 2 or h < tile_size // 2:
                    continue
                tile_positions.append((x, y, w, h))

        if not tile_positions:
            print("Error: No tiles to process", file=sys.stderr)
            return EXIT_SLIDE

        regions = []
        total_tiles = len(tile_positions)
        batch_size = args.batch_size
        density_label = torch.tensor([3] * batch_size, dtype=torch.long, device=device)

        try:
            for i in range(0, total_tiles, batch_size):
                batch_positions = tile_positions[i : i + batch_size]
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
                    score = float(probs[j].mean().item())
                    label = "fungus_positive" if score >= args.threshold else "fungus_negative"
                    if args.positive_only and label != "fungus_positive":
                        continue
                    regions.append({
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "score": round(score, 4),
                        "label": label,
                    })
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

            # Build tile grid (level coordinates)
            tiles_x = list(range(0, level_w, stride))
            tiles_y = list(range(0, level_h, stride))
            tile_positions = []
            for y in tiles_y:
                for x in tiles_x:
                    w = min(tile_size, level_w - x)
                    h = min(tile_size, level_h - y)
                    if w < tile_size // 2 or h < tile_size // 2:
                        continue
                    tile_positions.append((x, y, w, h))

            if not tile_positions:
                print("Error: No tiles to process", file=sys.stderr)
                slide.close()
                return EXIT_SLIDE

            regions = []
            total_tiles = len(tile_positions)
            batch_size = args.batch_size

            # Density label: 3 = negative (unknown at inference)
            density_label = torch.tensor([3] * batch_size, dtype=torch.long, device=device)

            for i in range(0, total_tiles, batch_size):
                batch_positions = tile_positions[i : i + batch_size]
                batch_tensors = []

                for x, y, w, h in batch_positions:
                    # read_region uses level-0 coordinates
                    downsample = float(slide.level_downsamples[level])
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
                    score = float(probs[j].mean().item())
                    label = "fungus_positive" if score >= args.threshold else "fungus_negative"
                    if args.positive_only and label != "fungus_positive":
                        continue
                    # Convert to level-0 coordinates for API/Region
                    downsample = float(slide.level_downsamples[level])
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    w0 = int(w * downsample)
                    h0 = int(h * downsample)
                    regions.append({
                        "x": x0,
                        "y": y0,
                        "w": w0,
                        "h": h0,
                        "score": round(score, 4),
                        "label": label,
                    })

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
