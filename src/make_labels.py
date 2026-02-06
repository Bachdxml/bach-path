"""
Generate tile-level labels from QuPath GeoJSON annotations.
Labels tiles as positive (fungus) if overlap with target class >= POS_OVERLAP_FRAC.
"""
import argparse
import sys
from pathlib import Path

# Allow running as python src/make_labels.py from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from tqdm import tqdm

from utils import (
    DEFAULT_TILE_SIZE,
    parse_tile_xy,
    load_fungus_union_polygon,
    tile_box,
)

# Configurable defaults
TILE_SIZE = 512
POS_OVERLAP_FRAC = 0.02
FUNGUS_CLASS_NAME = "fungus"
TILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def process_slide(
    slide_id: str,
    tiles_dir: Path,
    geojson_path: Path,
    fungus_class_name: str,
    tile_size: int,
    pos_overlap_frac: float,
) -> list[dict]:
    """Process one slide: list tiles and assign labels from GeoJSON overlap."""
    fungus_union = load_fungus_union_polygon(geojson_path, fungus_class_name)
    slide_tiles_dir = tiles_dir / slide_id
    if not slide_tiles_dir.is_dir():
        return []

    rows = []
    for path in slide_tiles_dir.iterdir():
        if path.suffix.lower() not in TILE_EXTENSIONS:
            continue
        try:
            x, y = parse_tile_xy(path.name)
        except ValueError as e:
            raise RuntimeError(f"Slide {slide_id}: {e}") from e

        box_geom = tile_box(x, y, tile_size)
        if fungus_union is None:
            overlap_frac = 0.0
        else:
            try:
                inter = box_geom.intersection(fungus_union)
                overlap_area = inter.area if not inter.is_empty else 0.0
            except Exception:
                overlap_area = 0.0
            tile_area = tile_size * tile_size
            overlap_frac = overlap_area / tile_area if tile_area else 0.0

        label = 1 if overlap_frac >= pos_overlap_frac else 0
        rows.append({
            "tile_path": str(path.resolve()),
            "slide_id": slide_id,
            "x": x,
            "y": y,
            "label": label,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate tile labels from QuPath GeoJSON")
    parser.add_argument("--tiles_root", type=str, default="tiles_512", help="Root folder for tiles (per-slide subdirs)")
    parser.add_argument("--geojson_root", type=str, default="qupath_geojson", help="Root folder for GeoJSON files")
    parser.add_argument("--out", type=str, default="outputs/labels.csv", help="Output CSV path")
    parser.add_argument("--tile_size", type=int, default=TILE_SIZE, help="Tile size in pixels")
    parser.add_argument("--pos_overlap_frac", type=float, default=POS_OVERLAP_FRAC, help="Overlap fraction threshold for positive label")
    parser.add_argument("--fungus_class", type=str, default=FUNGUS_CLASS_NAME, help="QuPath class name for fungus (e.g. Fungus)")
    args = parser.parse_args()

    tiles_root = Path(args.tiles_root)
    geojson_root = Path(args.geojson_root)
    out_path = Path(args.out)

    if not tiles_root.is_dir():
        raise FileNotFoundError(f"Tiles root not found: {tiles_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    slide_ids = [d.name for d in tiles_root.iterdir() if d.is_dir()]
    all_rows = []
    for slide_id in tqdm(sorted(slide_ids), desc="Slides"):
        geojson_path = geojson_root / f"{slide_id}.geojson"
        if not geojson_path.exists():
            geojson_path = geojson_root / f"{slide_id}.geo.json"
        rows = process_slide(
            slide_id=slide_id,
            tiles_dir=tiles_root,
            geojson_path=geojson_path,
            fungus_class_name=args.fungus_class,
            tile_size=args.tile_size,
            pos_overlap_frac=args.pos_overlap_frac,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df[["tile_path", "slide_id", "x", "y", "label"]]
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    if not df.empty:
        print(f"  Labels: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
