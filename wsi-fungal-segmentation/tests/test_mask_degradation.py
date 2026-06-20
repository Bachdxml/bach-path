from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_inference_api.py"
MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "model.py"


def _load_module():
    src_package = types.ModuleType("src")
    src_package.__path__ = [str(MODEL_PATH.parent)]
    model_spec = importlib.util.spec_from_file_location("src.model", MODEL_PATH)
    model_module = importlib.util.module_from_spec(model_spec)
    sys.modules["src"] = src_package
    sys.modules["src.model"] = model_module
    model_spec.loader.exec_module(model_module)
    src_package.model = model_module

    spec = importlib.util.spec_from_file_location("mask_degradation_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inference_module():
    return _load_module()


def _mask_region(mod, *, x, y, size, fill=True):
    arr = np.ones((size, size), dtype=bool) if fill else np.zeros((size, size), dtype=bool)
    return {
        "x": x,
        "y": y,
        "w": size,
        "h": size,
        "score": 0.9,
        "label": "fungus_positive",
        "kind": "segmentation",
        "source": "prediction_tile",
        "coverage": 1.0,
        "prediction_mask": {
            "encoding": "bitpack",
            "width": size,
            "height": size,
            "threshold": 0.5,
            "data": mod._encode_binary_mask_bitpack(arr),
        },
    }


def _output_with(regions):
    return {"model_name": "m", "regions": regions}


def test_bitpack_roundtrip_is_lossless(inference_module):
    mod = inference_module
    arr = np.zeros((8, 5), dtype=bool)
    arr[0, 0] = True
    arr[7, 4] = True
    arr[3, 2] = True
    data = mod._encode_binary_mask_bitpack(arr)
    decoded = mod._decode_binary_mask_bitpack(data, 5, 8)
    assert decoded.shape == (8, 5)
    assert np.array_equal(decoded, arr)


def test_downsample_uses_any_pooling_and_never_degenerates(inference_module):
    mod = inference_module
    arr = np.zeros((4, 4), dtype=bool)
    arr[0, 0] = True  # one positive pixel in the top-left 2x2 block
    reduced = mod._downsample_mask_array(arr, 2)
    assert reduced.shape == (2, 2)
    assert bool(reduced[0, 0]) is True
    assert bool(reduced[1, 1]) is False
    # A 1x1 mask cannot shrink further but stays valid (never 0-sized).
    tiny = np.ones((1, 1), dtype=bool)
    assert mod._downsample_mask_array(tiny, 8).shape == (1, 1)


def test_full_status_when_under_budget_leaves_masks_untouched(inference_module):
    mod = inference_module
    regions = [_mask_region(mod, x=0, y=0, size=16)]
    output = _output_with([r.copy() for r in regions])
    original_data = output["regions"][0]["prediction_mask"]["data"]

    status, factor = mod._apply_mask_degradation(output, budget=10_000_000)

    assert status == "full"
    assert factor == 1
    assert output["mask_degradation"] == {"status": "full", "factor": 1}
    assert output["regions"][0]["prediction_mask"]["data"] == original_data
    assert output["regions"][0]["prediction_mask"]["width"] == 16


def test_no_budget_keeps_full(inference_module):
    mod = inference_module
    output = _output_with([_mask_region(mod, x=0, y=0, size=64)])
    status, factor = mod._apply_mask_degradation(output, budget=None)
    assert status == "full"
    assert factor == 1


def test_downsampled_status_fits_budget_and_preserves_boxes(inference_module):
    mod = inference_module
    regions = [
        _mask_region(mod, x=10, y=20, size=64),
        _mask_region(mod, x=80, y=20, size=64),
        _mask_region(mod, x=10, y=90, size=64),
    ]
    output = _output_with(regions)
    full_size = mod._estimate_output_bytes(output)
    # Force coarsening by setting the budget just below the full-resolution size.
    budget = full_size - 1

    status, factor = mod._apply_mask_degradation(output, budget=budget)

    assert status == "downsampled"
    assert factor == 2
    assert output["mask_degradation"] == {"status": "downsampled", "factor": 2}
    # Written-size estimate is at or below the budget.
    assert mod._estimate_output_bytes(output) <= budget
    for region in output["regions"]:
        mask = region["prediction_mask"]
        assert mask["width"] == 32  # 64 -> halved
        assert mask["height"] == 32
        # Box geometry/score/label untouched.
        assert region["w"] == 64
        assert region["h"] == 64
        assert region["score"] == 0.9
        assert region["label"] == "fungus_positive"


def test_dropped_status_keeps_boxes_without_masks(inference_module):
    mod = inference_module
    regions = [
        _mask_region(mod, x=1, y=2, size=32),
        _mask_region(mod, x=40, y=2, size=32),
    ]
    output = _output_with(regions)

    status, factor = mod._apply_mask_degradation(output, budget=10)

    assert status == "dropped"
    assert factor is None
    assert output["mask_degradation"] == {"status": "dropped", "factor": None}
    for region in output["regions"]:
        assert "prediction_mask" not in region
        # Every detection box/score is preserved.
        assert region["x"] in (1, 40)
        assert region["w"] == 32
        assert region["score"] == 0.9
        assert region["label"] == "fungus_positive"


def test_no_masks_over_budget_reports_full(inference_module):
    mod = inference_module
    # Box-only regions (no prediction_mask) that nominally exceed a tiny budget;
    # there are no masks to coarsen, so the run is reported full and the API
    # safety net / region-count bound governs this residual case.
    regions = [
        {"x": i, "y": 0, "w": 4, "h": 4, "score": 0.1, "label": "fungus_negative"}
        for i in range(20)
    ]
    output = _output_with(regions)
    status, factor = mod._apply_mask_degradation(output, budget=10)
    assert status == "full"
    assert factor == 1
