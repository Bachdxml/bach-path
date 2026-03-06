"""
classify_tiles.py
=================
Reads tile_coverage.csv produced by TileExport_WithCoverage.groovy,
applies neighbourhood-aware density classification, and moves tiles
into high / medium / low / negative subfolders.

Run in two modes:
  1. Preview mode (default): plots coverage histogram, prints stats,
     does NOT move any files. Use this to calibrate thresholds first.
  2. Apply mode (--apply): moves files into density folders.

Usage:
  python classify_tiles.py --export_dir path/to/exports_ml/ImageName
  python classify_tiles.py --export_dir path/to/exports_ml/ImageName --apply
  python classify_tiles.py --export_dir path/to/exports_ml/ImageName --apply --low 0.003 --medium 0.03
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# =======================
# DEFAULTS
# =======================
DEFAULT_LOW_THRESHOLD    = 0.005   # < 0.5%  foreground → low
DEFAULT_MEDIUM_THRESHOLD = 0.05    # < 5%    foreground → medium, else high
DEFAULT_OWN_WEIGHT       = 0.6
DEFAULT_NEIGHBOR_WEIGHT  = 0.4
TILE_SIZE                = 512


# =======================
# NEIGHBOURHOOD SCORING
# =======================
OFFSETS = [(-1,-1),(0,-1),(1,-1),
           (-1, 0),       (1, 0),
           (-1, 1),(0, 1),(1, 1)]

def build_coverage_map(df):
    return {(int(row.x), int(row.y)): float(row.coverage)
            for _, row in df.iterrows()}

def context_score(x, y, coverage_map, own_weight, neighbor_weight):
    own = coverage_map.get((x, y), 0.0)
    neighbors = [
        coverage_map[(x + dx * TILE_SIZE, y + dy * TILE_SIZE)]
        for dx, dy in OFFSETS
        if (x + dx * TILE_SIZE, y + dy * TILE_SIZE) in coverage_map
    ]
    neighbor_mean = sum(neighbors) / len(neighbors) if neighbors else 0.0
    return (own_weight * own) + (neighbor_weight * neighbor_mean)

def classify(score, low_thresh, medium_thresh):
    if score >= medium_thresh:
        return "high"
    elif score >= low_thresh:
        return "medium"
    else:
        return "low"


# =======================
# HISTOGRAM / PREVIEW
# =======================
def plot_histogram(df, low_thresh, medium_thresh, scores):
    positive_scores = [s for s, neg in zip(scores, df.is_negative) if not neg]

    if not positive_scores:
        print("⚠️  No positive tiles found — nothing to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tile Coverage Distribution", fontsize=14, fontweight="bold")

    # --- Raw coverage histogram ---
    ax1 = axes[0]
    raw = df[df.is_negative == False]["coverage"].values
    ax1.hist(raw, bins=80, color="#4C9BE8", edgecolor="white", linewidth=0.3)
    ax1.set_yscale("log")
    ax1.set_xlabel("Raw foreground coverage")
    ax1.set_ylabel("Tile count (log scale)")
    ax1.set_title("Raw coverage (positive tiles only)")
    ax1.axvline(low_thresh,    color="#F5A623", linewidth=1.5,
                label=f"low cutoff ({low_thresh:.4f})")
    ax1.axvline(medium_thresh, color="#E74C3C", linewidth=1.5,
                label=f"medium cutoff ({medium_thresh:.4f})")
    ax1.legend(fontsize=8)

    # Percentile annotations
    for p in [50, 75, 90, 95, 99]:
        v = np.percentile(raw, p)
        ax1.axvline(v, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax1.text(v, ax1.get_ylim()[1] * 0.5, f"p{p}", fontsize=6,
                 color="gray", rotation=90, va="top")

    # --- Context score histogram ---
    ax2 = axes[1]
    ax2.hist(positive_scores, bins=80, color="#5CB85C", edgecolor="white", linewidth=0.3)
    ax2.set_yscale("log")
    ax2.set_xlabel("Context-weighted score")
    ax2.set_ylabel("Tile count (log scale)")
    ax2.set_title("Context-weighted score (positive tiles only)")
    ax2.axvline(low_thresh,    color="#F5A623", linewidth=1.5,
                label=f"low cutoff ({low_thresh:.4f})")
    ax2.axvline(medium_thresh, color="#E74C3C", linewidth=1.5,
                label=f"medium cutoff ({medium_thresh:.4f})")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("coverage_distribution.png", dpi=150, bbox_inches="tight")
    print("📊 Histogram saved to: coverage_distribution.png")
    plt.show()


def print_stats(df, scores, labels, low_thresh, medium_thresh):
    positive_mask = df.is_negative == False
    positive_scores = [s for s, m in zip(scores, positive_mask) if m]
    raw = df[positive_mask]["coverage"].values

    print("\n" + "=" * 55)
    print("COVERAGE STATISTICS (positive tiles only)")
    print("=" * 55)
    print(f"  Total tiles:     {len(df)}")
    print(f"  Positive tiles:  {int(positive_mask.sum())}")
    print(f"  Negative tiles:  {int((~positive_mask).sum())}")
    print()
    print(f"  Raw coverage percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"    p{p:>2}: {np.percentile(raw, p):.5f}")
    print()
    print(f"  Thresholds in use:")
    print(f"    low    ≥ {low_thresh:.5f}")
    print(f"    medium ≥ {medium_thresh:.5f}")
    print()
    print("  Projected class distribution:")
    for density in ["high", "medium", "low", "negative"]:
        count = labels.count(density)
        pct   = 100 * count / len(labels) if labels else 0
        print(f"    {density:<10} {count:>6} tiles  ({pct:.1f}%)")
    print("=" * 55)
    print()
    print("💡 Tip: adjust --low and --medium so that your class")
    print("   distribution matches your annotation density mix.")
    print("   Re-run without --apply until the split looks right,")
    print("   then add --apply to move the files.\n")


# =======================
# MAIN
# =======================
def main():
    parser = argparse.ArgumentParser(description="Classify exported tiles by density.")
    parser.add_argument("--export_dir", required=True,
                        help="Path to the exported image folder (contains tile_coverage.csv)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually move files. Without this flag, preview only.")
    parser.add_argument("--low",    type=float, default=DEFAULT_LOW_THRESHOLD,
                        help=f"Low density threshold (default {DEFAULT_LOW_THRESHOLD})")
    parser.add_argument("--medium", type=float, default=DEFAULT_MEDIUM_THRESHOLD,
                        help=f"Medium density threshold (default {DEFAULT_MEDIUM_THRESHOLD})")
    parser.add_argument("--own_weight",      type=float, default=DEFAULT_OWN_WEIGHT)
    parser.add_argument("--neighbor_weight", type=float, default=DEFAULT_NEIGHBOR_WEIGHT)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    csv_path   = export_dir / "tile_coverage.csv"

    if not csv_path.exists():
        print(f"❌ tile_coverage.csv not found in {export_dir}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["is_negative"] = df["is_negative"].astype(str).str.lower().isin(["true", "1"])

    print(f"📄 Loaded {len(df)} tiles from {csv_path}")

    # Build coverage map and compute scores
    coverage_map = build_coverage_map(df)

    scores = []
    labels = []
    for _, row in df.iterrows():
        if row.is_negative:
            scores.append(0.0)
            labels.append("negative")
        else:
            score = context_score(
                int(row.x), int(row.y),
                coverage_map,
                args.own_weight, args.neighbor_weight
            )
            scores.append(score)
            labels.append(classify(score, args.low, args.medium))

    # Always show stats
    print_stats(df, scores, labels, args.low, args.medium)
    plot_histogram(df, args.low, args.medium, scores)

    if not args.apply:
        print("ℹ️  Preview mode — no files moved.")
        print("   Add --apply to the command when thresholds look good.\n")
        return

    # =======================
    # MOVE FILES
    # =======================
    unclassified_img  = export_dir / "unclassified" / "images"
    unclassified_mask = export_dir / "unclassified" / "masks"

    if not unclassified_img.exists():
        print(f"❌ unclassified/images folder not found in {export_dir}")
        sys.exit(1)

    # Create destination directories
    for density in ["high", "medium", "low", "negative"]:
        (export_dir / density / "images").mkdir(parents=True, exist_ok=True)
        (export_dir / density / "masks").mkdir(parents=True, exist_ok=True)

    counts  = {"high": 0, "medium": 0, "low": 0, "negative": 0}
    missing = 0

    for (_, row), density in zip(df.iterrows(), labels):
        img_src  = unclassified_img  / (row.filename + ".png")
        mask_src = unclassified_mask / (row.filename + "_mask.png")

        img_dst  = export_dir / density / "images" / (row.filename + ".png")
        mask_dst = export_dir / density / "masks"  / (row.filename + "_mask.png")

        moved = 0
        if img_src.exists():
            shutil.move(str(img_src), str(img_dst))
            moved += 1
        else:
            print(f"⚠️  Missing image: {img_src.name}")
            missing += 1

        if mask_src.exists():
            shutil.move(str(mask_src), str(mask_dst))
            moved += 1
        else:
            print(f"⚠️  Missing mask: {mask_src.name}")
            missing += 1

        if moved > 0:
            counts[density] += 1

    print("\n" + "=" * 55)
    print("✅ Classification complete:")
    for k, v in counts.items():
        print(f"   {k:<12} {v} tiles")
    if missing:
        print(f"\n⚠️  {missing} missing files — check warnings above")
    print(f"\n📁 Output: {export_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
