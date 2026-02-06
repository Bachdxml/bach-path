"""
Generate heatmap overlay per slide from tile probabilities.
Background: downscaled tile mosaic; overlay: probability heatmap.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from train import PhikonMLPHead, PHIKON_MODEL_NAME, HIDDEN_DIM, THRESHOLD

# Config
DEFAULT_TILE_SIZE = 512
DEFAULT_DOWNSCALE = 32  # each tile -> 32x32 in background
DEFAULT_ALPHA = 0.5
DEFAULT_STRIDE = 512  # stride for inference (512 = no overlap; 256 = overlap)


def load_encoder_and_head(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    processor = AutoImageProcessor.from_pretrained(ckpt["model_name"])
    encoder = AutoModel.from_pretrained(ckpt["model_name"])
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)
    encoder.eval()
    head_hidden = ckpt.get("head_hidden", 256)
    dropout = ckpt.get("dropout", 0.2)
    head = PhikonMLPHead(ckpt["hidden_dim"], head_hidden=head_hidden, dropout=dropout).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return processor, encoder, head


def run_inference(
    tile_paths,
    coords,
    processor,
    encoder,
    head,
    device,
    batch_size=16,
):
    """Run inference on tiles; return (paths, x, y, prob)."""
    probs = []
    for i in tqdm(range(0, len(tile_paths), batch_size), desc="Inference", leave=False):
        batch_paths = tile_paths[i : i + batch_size]
        batch_coords = coords[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", (512, 512), (128, 128, 128)))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            enc_out = encoder(pixel_values=pixel_values)
            feat = enc_out.last_hidden_state[:, 0, :]
            logits = head(feat)
            p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
    return probs


def build_heatmap_and_background(
    slide_df,
    tile_size,
    stride,
    downscale,
    probs,
):
    """Build 2D heatmap grid and background mosaic from slide tiles. stride controls grid step (e.g. 256 for denser heatmaps)."""
    xs = slide_df["x"].values
    ys = slide_df["y"].values
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    # Grid indices (stride units; stride 256 gives denser grid when tiles are exported at 256)
    x_idx = ((xs - min_x) // stride).astype(int)
    y_idx = ((ys - min_y) // stride).astype(int)
    cols = (max_x - min_x) // stride + 1
    rows = (max_y - min_y) // stride + 1

    heatmap = np.full((rows, cols), np.nan)
    for k, (i, j) in enumerate(zip(y_idx, x_idx)):
        heatmap[i, j] = probs[k]
    # Fill missing with 0 for display
    heatmap = np.nan_to_num(heatmap, nan=0.0)

    # Background: mosaic of downscaled tiles
    bg_h, bg_w = rows * downscale, cols * downscale
    background = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
    paths = slide_df["tile_path"].values
    for k, (i, j) in enumerate(zip(y_idx, x_idx)):
        r0, r1 = i * downscale, (i + 1) * downscale
        c0, c1 = j * downscale, (j + 1) * downscale
        try:
            img = np.array(Image.open(paths[k]).convert("RGB"))
            small = np.array(
                Image.fromarray(img).resize((downscale, downscale), Image.Resampling.LANCZOS)
            )
            background[r0:r1, c0:c1] = small
        except Exception:
            background[r0:r1, c0:c1] = 128
    return heatmap, background, (min_x, min_y), (rows, cols)


def main():
    parser = argparse.ArgumentParser(description="Generate heatmap overlay per slide")
    parser.add_argument("--labels_csv", type=str, default="outputs/labels.csv", help="Labels CSV (for tile list)")
    parser.add_argument("--ckpt", type=str, default="outputs/models/phikon_head_best.pt", help="Model checkpoint")
    parser.add_argument("--out_dir", type=str, default="outputs/heatmaps", help="Output directory for heatmaps")
    parser.add_argument("--slide_id", type=str, default=None, help="Slide ID (default: first in CSV)")
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE, help="Tile size in pixels")
    parser.add_argument("--downscale", type=int, default=DEFAULT_DOWNSCALE, help="Tile size in background mosaic")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Heatmap overlay alpha")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Stride for tile grid (512 or 256)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.labels_csv)
    if df.empty:
        raise ValueError("Labels CSV is empty")

    slide_id = args.slide_id
    if slide_id is None:
        slide_id = df["slide_id"].iloc[0]
    slide_df = df[df["slide_id"] == slide_id].copy()
    if slide_df.empty:
        raise ValueError(f"No tiles found for slide_id={slide_id}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint {args.ckpt}...")
    processor, encoder, head = load_encoder_and_head(Path(args.ckpt), device)

    tile_paths = slide_df["tile_path"].tolist()
    coords = list(zip(slide_df["x"], slide_df["y"]))
    probs = run_inference(tile_paths, coords, processor, encoder, head, device)

    heatmap, background, _origin, (rows, cols) = build_heatmap_and_background(
        slide_df, args.tile_size, args.stride, args.downscale, probs
    )

    # Resize heatmap to same pixel size as background for overlay
    heatmap_resized = np.repeat(np.repeat(heatmap, args.downscale, axis=0), args.downscale, axis=1)
    # Mask so only valid region gets colormap
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(background)
    im = ax.imshow(heatmap_resized, alpha=args.alpha, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="P(fungus)")
    ax.set_title(f"Heatmap: {slide_id}")
    ax.axis("off")
    out_path = out_dir / f"{slide_id}_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
