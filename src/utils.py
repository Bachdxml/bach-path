"""
Shared utilities for tile labeling, training, and heatmap generation.
"""
import json
import re
from pathlib import Path
from typing import Optional

from shapely.geometry import shape, box
from shapely.ops import unary_union


# Default tile size (pixels)
DEFAULT_TILE_SIZE = 512

# Regex to parse x,y from tile filename: <SLIDE_ID>_x12345_y67890.png (or .jpg, .tif)
# Also allow: x12345_y67890.png, slide_x12345_y67890.jpg, etc.
TILE_XY_PATTERN = re.compile(
    r"_x(\d+)_y(\d+)\.(?:png|jpg|jpeg|tif|tiff)$",
    re.IGNORECASE,
)


def parse_tile_xy(filename: str) -> tuple[int, int]:
    """
    Parse (x, y) top-left coordinates from a tile filename.
    Expected format: ..._x12345_y67890.png (or .jpg, .tif).
    Fails loudly (raises ValueError) if not parseable.
    """
    name = Path(filename).name
    m = TILE_XY_PATTERN.search(name)
    if m is None:
        raise ValueError(
            f"Cannot parse x,y from tile filename: {filename!r}. "
            "Expected pattern: ..._x<num>_y<num>.png|.jpg|.tif (e.g. SLIDE_x12345_y67890.png)"
        )
    return int(m.group(1)), int(m.group(2))


def _get_class_name_from_properties(properties: dict) -> Optional[str]:
    """Extract class name from QuPath GeoJSON feature properties."""
    if not properties:
        return None
    # QuPath can use pathClass, classification, class, or name
    for key in ("pathClass", "classification", "class", "name"):
        val = properties.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            return val.get("name") or val.get("classification")
        if isinstance(val, str):
            return val
    return None


def load_fungus_union_polygon(
    geojson_path: str | Path,
    fungus_class_name: str,
) -> Optional["shapely.Geometry"]:
    """
    Load GeoJSON and return the union of all polygons for the given class.
    Class name comparison is case-insensitive.
    Returns None if file missing, invalid, or no matching polygons.
    """
    path = Path(geojson_path)
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    target = fungus_class_name.strip().lower()
    polygons = []

    # GeoJSON can be FeatureCollection or single Feature
    features = []
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]

    for feat in features:
        if feat.get("type") != "Feature":
            continue
        prop = feat.get("properties") or {}
        name = _get_class_name_from_properties(prop)
        if name is None:
            continue
        if name.strip().lower() != target:
            continue
        geom = feat.get("geometry")
        if geom is None:
            continue
        try:
            shp = shape(geom)
            if shp.is_empty:
                continue
            if shp.geom_type == "Polygon":
                polygons.append(shp)
            elif shp.geom_type == "MultiPolygon":
                polygons.extend(shp.geoms)
        except Exception:
            continue

    if not polygons:
        return None
    return unary_union(polygons)


def tile_box(x: int, y: int, tile_size: int = DEFAULT_TILE_SIZE):
    """Return Shapely box for tile at (x, y) with given tile size."""
    return box(x, y, x + tile_size, y + tile_size)
