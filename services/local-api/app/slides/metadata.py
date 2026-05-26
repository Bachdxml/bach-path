from __future__ import annotations
from pathlib import Path
from typing import Any

from PIL import Image

RASTER_EXTENSIONS = {".png"}
SAFE_OPENSLIDE_PROPERTY_NAMES = {
    "openslide.vendor",
    "openslide.mpp-x",
    "openslide.mpp-y",
    "openslide.objective-power",
}


def read_raster_metadata(slide_path: Path) -> dict[str, Any]:
    """Metadata for flat raster images (PNG, etc.) used when OpenSlide does not apply."""
    with Image.open(slide_path) as im:
        w, h = im.size
    w, h = int(w), int(h)
    return {
        "vendor": None,
        "level_count": 1,
        "dimensions": (w, h),
        "level_dimensions": [(w, h)],
        "mpp_x": None,
        "mpp_y": None,
        "properties": {},
    }


def read_openslide_metadata(slide_path: Path) -> dict[str, Any]:
    import openslide

    with openslide.OpenSlide(str(slide_path)) as s:
        props = dict(s.properties)  # OpenSlideProperties
        safe_props = {
            key: value
            for key, value in props.items()
            if key in SAFE_OPENSLIDE_PROPERTY_NAMES
        }
        dims = (int(s.dimensions[0]), int(s.dimensions[1]))
        level_dims = [(int(w), int(h)) for (w, h) in s.level_dimensions]
        vendor = props.get(openslide.PROPERTY_NAME_VENDOR)

        def _to_float(k: str) -> float | None:
            v = props.get(k)
            if v is None:
                return None
            try:
                return float(v)
            except ValueError:
                return None

        return {
            "vendor": vendor,
            "level_count": int(s.level_count),
            "dimensions": dims,
            "level_dimensions": level_dims,
            "mpp_x": _to_float(openslide.PROPERTY_NAME_MPP_X),
            "mpp_y": _to_float(openslide.PROPERTY_NAME_MPP_Y),
            "properties": safe_props,
        }
