from __future__ import annotations
from pathlib import Path
from typing import Any

import openslide

def read_openslide_metadata(slide_path: Path) -> dict[str, Any]:
    with openslide.OpenSlide(str(slide_path)) as s:
        props = dict(s.properties)  # OpenSlideProperties
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
            "properties": props,
        }
