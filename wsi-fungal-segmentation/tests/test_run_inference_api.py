from __future__ import annotations

import importlib.util
import base64
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_inference_api.py"
MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "model.py"


def _load_module():
    src_package = types.ModuleType("src")
    src_package.__path__ = [str(MODEL_PATH.parent)]
    model_spec = importlib.util.spec_from_file_location("src.model", MODEL_PATH)
    assert model_spec is not None
    assert model_spec.loader is not None
    model_module = importlib.util.module_from_spec(model_spec)
    sys.modules["src"] = src_package
    sys.modules["src.model"] = model_module
    model_spec.loader.exec_module(model_module)
    src_package.model = model_module

    spec = importlib.util.spec_from_file_location("test_run_inference_api_module", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bitpack_foreground_count(data: str, total_pixels: int) -> int:
    raw = base64.b64decode(data.encode("ascii"))
    count = 0
    for i in range(total_pixels):
        count += (raw[i // 8] >> (7 - (i % 8))) & 1
    return count


@pytest.fixture
def inference_module():
    return _load_module()


def test_main_runs_two_pass_neighborhood_inference_for_raster(tmp_path, monkeypatch, inference_module):
    slide_path = tmp_path / "slide.png"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    Image.new("RGB", (64, 64), color=(90, 160, 110)).save(slide_path)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    recorded_calls: list[tuple[tuple[torch.Tensor, ...], dict]] = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            recorded_calls.append((args, kwargs))
            batch = args[0]
            batch_size, _, height, width = batch.shape
            return (
                torch.zeros((batch_size, 1, height, width), dtype=torch.float32),
                torch.zeros((batch_size, 4), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 8), max(1, width // 8)), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 4), max(1, width // 4)), dtype=torch.float32),
            )

    monkeypatch.setattr(inference_module, "ResidualAttentionUNet", FakeModel)
    monkeypatch.setattr(inference_module, "_load_checkpoint_dict", lambda path, device: {"mock": "checkpoint"})
    monkeypatch.setattr(
        inference_module,
        "_extract_model_state_dict",
        lambda checkpoint: ({"weight": torch.ones(1)}, {"cfg": {"model": {}}}),
    )
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference_api.py",
            "--slide-path",
            str(slide_path),
            "--output-json",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
            "--tile-size",
            "64",
            "--stride",
            "64",
            "--batch-size",
            "1",
        ],
    )

    assert inference_module.main() == 0
    assert output_path.exists()
    assert len(recorded_calls) == 2
    assert recorded_calls[0][1] == {"density_only": True}
    assert "density_label" in recorded_calls[1][1]
    assert recorded_calls[0][0][0].shape == (1, 3, 64, 64)
    assert recorded_calls[1][0][0].shape == (1, 3, 64, 64)


def test_raster_size_guard_runs_before_rgb_decode(tmp_path, monkeypatch, inference_module):
    slide_path = tmp_path / "huge.png"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    slide_path.write_text("stub image", encoding="utf-8")
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    class FakeImage:
        size = (10_000, 10_000)

        def convert(self, mode):
            raise AssertionError("large raster should be rejected before RGB decode")

        def close(self):
            return None

    monkeypatch.setattr(inference_module.Image, "open", lambda path: FakeImage())
    monkeypatch.setattr(inference_module, "_max_raster_pixels", lambda: 1_000)
    class FakeModel:
        def load_state_dict(self, state_dict):
            return None

        def to(self, *args, **kwargs):
            # Accept device and the CUDA-only memory_format kwarg alike.
            return self

        def eval(self):
            return self

    monkeypatch.setattr(inference_module, "ResidualAttentionUNet", lambda **kwargs: FakeModel())
    monkeypatch.setattr(inference_module, "_load_checkpoint_dict", lambda path, device: {"mock": "checkpoint"})
    monkeypatch.setattr(
        inference_module,
        "_extract_model_state_dict",
        lambda checkpoint: ({"weight": torch.ones(1)}, {"cfg": {"model": {}}}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference_api.py",
            "--slide-path",
            str(slide_path),
            "--output-json",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ],
    )

    assert inference_module.main() == inference_module.EXIT_SLIDE
    assert not output_path.exists()


def test_openslide_path_runs_two_pass_neighborhood_inference(tmp_path, monkeypatch, inference_module):
    slide_path = tmp_path / "slide.svs"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    slide_path.write_text("stub slide", encoding="utf-8")
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    recorded_calls: list[tuple[tuple[torch.Tensor, ...], dict]] = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            recorded_calls.append((args, kwargs))
            batch = args[0]
            batch_size, _, height, width = batch.shape
            return (
                torch.zeros((batch_size, 1, height, width), dtype=torch.float32),
                torch.zeros((batch_size, 4), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 8), max(1, width // 8)), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 4), max(1, width // 4)), dtype=torch.float32),
            )

    class FakeOpenSlide:
        level_count = 1
        level_dimensions = [(64, 64)]
        dimensions = (64, 64)
        level_downsamples = [1.0]

        def __init__(self, path):
            self.path = path

        def read_region(self, location, level, size):
            return Image.new("RGB", size, color=(120, 90, 150))

        def close(self):
            return None

    monkeypatch.setattr(inference_module, "ResidualAttentionUNet", FakeModel)
    monkeypatch.setattr(inference_module, "_load_checkpoint_dict", lambda path, device: {"mock": "checkpoint"})
    monkeypatch.setattr(
        inference_module,
        "_extract_model_state_dict",
        lambda checkpoint: ({"weight": torch.ones(1)}, {"cfg": {"model": {}}}),
    )
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=FakeOpenSlide))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference_api.py",
            "--slide-path",
            str(slide_path),
            "--output-json",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
            "--tile-size",
            "64",
            "--stride",
            "64",
            "--batch-size",
            "1",
        ],
    )

    assert inference_module.main() == 0
    assert output_path.exists()
    assert len(recorded_calls) == 2
    assert recorded_calls[0][1] == {"density_only": True}
    assert "density_label" in recorded_calls[1][1]
    assert recorded_calls[0][0][0].shape == (1, 3, 64, 64)
    assert recorded_calls[1][0][0].shape == (1, 3, 64, 64)


def test_openslide_path_skips_background_tiles_before_model(tmp_path, monkeypatch, inference_module):
    slide_path = tmp_path / "slide.svs"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    slide_path.write_text("stub slide", encoding="utf-8")
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    recorded_calls: list[tuple[tuple[torch.Tensor, ...], dict]] = []
    read_locations: list[tuple[int, int]] = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            recorded_calls.append((args, kwargs))
            batch = args[0]
            batch_size, _, height, width = batch.shape
            return (
                torch.zeros((batch_size, 1, height, width), dtype=torch.float32),
                torch.zeros((batch_size, 4), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 8), max(1, width // 8)), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 4), max(1, width // 4)), dtype=torch.float32),
            )

    class FakeOpenSlide:
        level_count = 1
        level_dimensions = [(128, 64)]
        dimensions = (128, 64)
        level_downsamples = [1.0]

        def __init__(self, path):
            self.path = path

        def get_thumbnail(self, size):
            thumb = Image.new("RGB", (128, 64), color=(255, 255, 255))
            for x in range(64, 128):
                for y in range(64):
                    thumb.putpixel((x, y), (150, 90, 160))
            return thumb

        def read_region(self, location, level, size):
            read_locations.append(location)
            return Image.new("RGB", size, color=(120, 90, 150))

        def close(self):
            return None

    monkeypatch.setattr(inference_module, "ResidualAttentionUNet", FakeModel)
    monkeypatch.setattr(inference_module, "_load_checkpoint_dict", lambda path, device: {"mock": "checkpoint"})
    monkeypatch.setattr(
        inference_module,
        "_extract_model_state_dict",
        lambda checkpoint: ({"weight": torch.ones(1)}, {"cfg": {"model": {}}}),
    )
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=FakeOpenSlide))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference_api.py",
            "--slide-path",
            str(slide_path),
            "--output-json",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
            "--tile-size",
            "64",
            "--stride",
            "64",
            "--batch-size",
            "1",
        ],
    )

    assert inference_module.main() == 0
    assert read_locations == [(64, 0), (64, 0)]
    assert len(recorded_calls) == 2
    assert recorded_calls[0][1] == {"density_only": True}
    assert "density_label" in recorded_calls[1][1]


def test_openslide_auto_level_uses_tile_budget_and_level_zero_coordinates(
    tmp_path,
    monkeypatch,
    inference_module,
):
    slide_path = tmp_path / "slide.svs"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    slide_path.write_text("stub slide", encoding="utf-8")
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    read_calls: list[tuple[tuple[int, int], int, tuple[int, int]]] = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            batch = args[0]
            batch_size, _, height, width = batch.shape
            return (
                torch.ones((batch_size, 1, height, width), dtype=torch.float32),
                torch.zeros((batch_size, 4), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 8), max(1, width // 8)), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 4), max(1, width // 4)), dtype=torch.float32),
            )

    class FakeOpenSlide:
        level_count = 3
        level_dimensions = [(4096, 4096), (1024, 1024), (256, 256)]
        dimensions = (4096, 4096)
        level_downsamples = [1.0, 4.0, 16.3]

        def __init__(self, path):
            self.path = path

        def get_thumbnail(self, size):
            return Image.new("RGB", size, color=(120, 90, 150))

        def read_region(self, location, level, size):
            read_calls.append((location, level, size))
            return Image.new("RGB", size, color=(120, 90, 150))

        def close(self):
            return None

    monkeypatch.setattr(inference_module, "ResidualAttentionUNet", FakeModel)
    monkeypatch.setattr(inference_module, "_load_checkpoint_dict", lambda path, device: {"mock": "checkpoint"})
    monkeypatch.setattr(
        inference_module,
        "_extract_model_state_dict",
        lambda checkpoint: ({"weight": torch.ones(1)}, {"cfg": {"model": {}}}),
    )
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=FakeOpenSlide))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference_api.py",
            "--slide-path",
            str(slide_path),
            "--output-json",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
            "--tile-size",
            "128",
            "--stride",
            "128",
            "--target-tiles",
            "10",
        ],
    )

    assert inference_module.main() == 0
    output = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    assert output["level"] == 2
    assert output["summary"]["total_tiles"] == 4
    assert all(call[1] == 2 for call in read_calls)
    assert output["regions"][0]["w"] == 2086


def test_infer_with_neighborhood_uses_neighbor_consensus_for_second_pass(inference_module):
    tiles = {
        (0, 0): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        (8, 0): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        (16, 0): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        (0, 8): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        (8, 8): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        (16, 8): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        (0, 16): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        (8, 16): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        (16, 16): torch.zeros((1, 3, 8, 8), dtype=torch.float32),
    }
    density_by_coord = {
        coord: (0 if coord == (8, 8) else 2)
        for coord in tiles
    }
    coords = list(tiles)
    second_pass_labels = []

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, batch, density_label=None):
            batch_size, _, height, width = batch.shape
            if density_label is not None:
                second_pass_labels.extend(density_label.cpu().tolist())
                return (
                    torch.zeros((batch_size, 1, height, width), dtype=torch.float32),
                    torch.zeros((batch_size, 4), dtype=torch.float32),
                    torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32),
                    torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32),
                )

            start = len(second_pass_labels)
            logits = torch.zeros((batch_size, 4), dtype=torch.float32)
            for row, coord in enumerate(coords[start:start + batch_size]):
                logits[row, density_by_coord[coord]] = 1.0
            return (
                torch.zeros((batch_size, 1, height, width), dtype=torch.float32),
                logits,
                torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32),
                torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32),
            )

    masks = inference_module.infer_with_neighborhood(
        FakeModel(),
        tiles,
        torch.device("cpu"),
        tile_size=8,
        stride=8,
        batch_size=len(tiles),
    )

    assert set(masks) == set(tiles)
    assert second_pass_labels[coords.index((8, 8))] == 2


def test_prediction_tile_region_from_prob_map_returns_mask_payload(inference_module):
    prob_map = torch.zeros((64, 64), dtype=torch.float32).numpy()
    prob_map[16:48, 16:48] = 1.0

    regions = inference_module._prediction_tile_region_from_prob_map(
        prob_map,
        tile_w=80,
        tile_h=80,
        threshold=0.5,
    )

    assert len(regions) == 1
    region = regions[0]
    assert region["x"] == 0
    assert region["y"] == 0
    assert region["w"] == 80
    assert region["h"] == 80
    assert region["score"] == pytest.approx(1.0, abs=0.01)
    assert region["payload"]["kind"] == "segmentation"
    assert region["payload"]["source"] == "prediction_tile"
    assert region["payload"]["coverage"] == pytest.approx(0.25, abs=0.01)
    mask = region["payload"]["prediction_mask"]
    assert mask["encoding"] == "bitpack"
    assert mask["width"] == 64
    assert mask["height"] == 64
    assert _bitpack_foreground_count(mask["data"], 64 * 64) == 32 * 32


def test_prediction_tile_region_from_prob_map_keeps_disconnected_mask_islands(inference_module):
    prob_map = torch.zeros((128, 128), dtype=torch.float32).numpy()
    prob_map[0:32, 0:32] = 1.0
    prob_map[96:128, 96:128] = 1.0

    regions = inference_module._prediction_tile_region_from_prob_map(
        prob_map,
        tile_w=128,
        tile_h=128,
        threshold=0.5,
    )

    assert len(regions) == 1
    mask = regions[0]["payload"]["prediction_mask"]
    assert mask["encoding"] == "bitpack"
    assert _bitpack_foreground_count(mask["data"], 128 * 128) == (32 * 32) * 2


# ---------------------------------------------------------------------------
# Tissue-aware level selection (specs/tissue-aware-level-selection.md)
# ---------------------------------------------------------------------------

def _patch_fake_model(monkeypatch, inference_module):
    """Patch in a CPU FakeModel + checkpoint/device stubs shared by the
    tissue-aware level-selection tests."""

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            batch = args[0]
            batch_size, _, height, width = batch.shape
            return (
                torch.zeros((batch_size, 1, height, width), dtype=torch.float32),
                torch.zeros((batch_size, 4), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 8), max(1, width // 8)), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 4), max(1, width // 4)), dtype=torch.float32),
            )

    monkeypatch.setattr(inference_module, "ResidualAttentionUNet", FakeModel)
    monkeypatch.setattr(inference_module, "_load_checkpoint_dict", lambda path, device: {"mock": "checkpoint"})
    monkeypatch.setattr(
        inference_module,
        "_extract_model_state_dict",
        lambda checkpoint: ({"weight": torch.ones(1)}, {"cfg": {"model": {}}}),
    )
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    return FakeModel


class _SparseTissueSlide:
    """Two-level slide whose tissue occupies only the top-left quarter.

    At level 0 (400x400, tile 100) the full grid is 16 tiles but only the 4
    top-left tiles contain tissue; at level 1 (100x100) the full grid is 1 tile.
    """

    level_count = 2
    level_dimensions = [(400, 400), (100, 100)]
    dimensions = (400, 400)
    level_downsamples = [1.0, 4.0]

    def __init__(self, path):
        self.path = path

    def get_thumbnail(self, size):
        w, h = size
        thumb = Image.new("RGB", (w, h), color=(255, 255, 255))
        for x in range(w // 2):
            for y in range(h // 2):
                thumb.putpixel((x, y), (150, 90, 160))
        return thumb

    def read_region(self, location, level, size):
        return Image.new("RGB", size, color=(150, 90, 160))

    def close(self):
        return None


def _run_level_selection_slide(tmp_path, monkeypatch, inference_module, openslide_cls, extra_argv=()):
    slide_path = tmp_path / "slide.svs"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    slide_path.write_text("stub slide", encoding="utf-8")
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    _patch_fake_model(monkeypatch, inference_module)
    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=openslide_cls))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference_api.py",
            "--slide-path", str(slide_path),
            "--output-json", str(output_path),
            "--checkpoint", str(checkpoint_path),
            "--tile-size", "100",
            "--stride", "100",
            "--target-tiles", "10",
            "--batch-size", "4",
            *extra_argv,
        ],
    )
    assert inference_module.main() == 0
    import json
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_tissue_aware_selection_picks_finer_level_than_full_grid(tmp_path, monkeypatch, inference_module):
    # Full-grid selection would step down to the coarse level (level 1) because
    # the level-0 grid (16 tiles) exceeds target_tiles=10. Tissue-aware counting
    # sees only 4 tissue tiles at level 0, so it stays at the finer level 0.
    output = _run_level_selection_slide(tmp_path, monkeypatch, inference_module, _SparseTissueSlide)
    assert output["level"] == 0
    assert output["summary"]["total_tiles"] == 16      # full grid at selected level
    assert output["summary"]["inferred_tiles"] == 4     # only tissue tiles inferred
    assert output["summary"]["skipped_background"] == 12

    # Contrast: the plain full-grid selector would have chosen the coarse level.
    plain_level = inference_module._select_openslide_level(
        _SparseTissueSlide("x"), "auto", 100, 100, 10,
    )
    assert plain_level == 1


def test_tissue_aware_selection_falls_back_to_full_grid_when_mask_is_none(tmp_path, monkeypatch, inference_module):
    # If the mask cannot be built, level selection must fall back to full-grid
    # counting, which steps down to the coarse level 1.
    monkeypatch.setattr(inference_module, "_build_openslide_tissue_mask", lambda slide, level: None)
    output = _run_level_selection_slide(tmp_path, monkeypatch, inference_module, _SparseTissueSlide)
    assert output["level"] == 1
    assert output["debug"]["base_mask_tissue_cells"] is None


def test_no_skip_background_uses_full_grid_level_selection(tmp_path, monkeypatch, inference_module):
    # --no-skip-background disables the tissue mask entirely, so selection uses
    # full-grid counting (coarse level 1) and no tiles are skipped.
    output = _run_level_selection_slide(
        tmp_path, monkeypatch, inference_module, _SparseTissueSlide,
        extra_argv=("--no-skip-background",),
    )
    assert output["level"] == 1
    assert output["summary"]["skipped_background"] == 0
    assert output["debug"]["base_mask_tissue_cells"] is None


# ---------------------------------------------------------------------------
# GPU throughput parity (specs/inference-gpu-throughput-parity.md)
# ---------------------------------------------------------------------------

def test_autocast_ctx_is_noop_on_cpu_and_autocast_on_cuda(inference_module):
    from contextlib import nullcontext

    cpu_ctx = inference_module._autocast_ctx(torch.device("cpu"))
    assert isinstance(cpu_ctx, nullcontext)
    with cpu_ctx:
        assert torch.is_autocast_enabled() is False

    cuda_ctx = inference_module._autocast_ctx(torch.device("cuda"))
    assert isinstance(cuda_ctx, torch.autocast)


def test_cpu_path_takes_no_cuda_autocast_or_channels_last(tmp_path, monkeypatch, inference_module):
    slide_path = tmp_path / "slide.svs"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    slide_path.write_text("stub slide", encoding="utf-8")
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    autocast_flags: list[bool] = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict):
            return None

        def to(self, device):
            # Only accepts a device positional arg: if the CUDA-only
            # channels_last conversion (model.to(memory_format=...)) were run on
            # CPU, this would raise TypeError and main() would fail.
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            autocast_flags.append(torch.is_autocast_enabled())
            batch = args[0]
            batch_size, _, height, width = batch.shape
            return (
                torch.zeros((batch_size, 1, height, width), dtype=torch.float32),
                torch.zeros((batch_size, 4), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 8), max(1, width // 8)), dtype=torch.float32),
                torch.zeros((batch_size, 1, max(1, height // 4), max(1, width // 4)), dtype=torch.float32),
            )

    class FakeOpenSlide:
        level_count = 1
        level_dimensions = [(64, 64)]
        dimensions = (64, 64)
        level_downsamples = [1.0]

        def __init__(self, path):
            self.path = path

        def get_thumbnail(self, size):
            return Image.new("RGB", size, color=(150, 90, 160))

        def read_region(self, location, level, size):
            return Image.new("RGB", size, color=(150, 90, 160))

        def close(self):
            return None

    monkeypatch.setattr(inference_module, "ResidualAttentionUNet", FakeModel)
    monkeypatch.setattr(inference_module, "_load_checkpoint_dict", lambda path, device: {"mock": "checkpoint"})
    monkeypatch.setattr(
        inference_module,
        "_extract_model_state_dict",
        lambda checkpoint: ({"weight": torch.ones(1)}, {"cfg": {"model": {}}}),
    )
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=FakeOpenSlide))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference_api.py",
            "--slide-path", str(slide_path),
            "--output-json", str(output_path),
            "--checkpoint", str(checkpoint_path),
            "--tile-size", "64",
            "--stride", "64",
            "--batch-size", "1",
        ],
    )

    assert inference_module.main() == 0            # channels_last guard not tripped
    assert autocast_flags                          # model was actually called
    assert all(flag is False for flag in autocast_flags)  # no CUDA autocast on CPU


# ---------------------------------------------------------------------------
# Tile I/O prefetch (specs/tile-io-prefetch.md)
# ---------------------------------------------------------------------------

def test_iter_decoded_batches_serial_and_prefetch_preserve_order(inference_module):
    positions = [(i, 0, 1, 1) for i in range(10)]

    def load_tensor(pos):
        return torch.full((1, 1, 1, 1), float(pos[0]))

    for workers in (0, 1, 3):
        out = list(inference_module._iter_decoded_batches(positions, 2, load_tensor, workers))
        # Two positions per batch -> 5 batches, consumed in order.
        assert [start for start, _, _ in out] == [0, 2, 4, 6, 8]
        first_values = [float(batch[0, 0, 0, 0]) for _, _, batch in out]
        assert first_values == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_iter_decoded_batches_bounds_inflight(inference_module):
    import threading
    import time

    positions = [(i, 0, 1, 1) for i in range(12)]
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def load_tensor(pos):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.01)
        with lock:
            state["active"] -= 1
        return torch.zeros((1, 1, 1, 1))

    out = list(inference_module._iter_decoded_batches(positions, 1, load_tensor, 2))
    assert len(out) == 12
    # Concurrency is bounded by the pool size (2 workers).
    assert state["max_active"] <= 2


def test_iter_decoded_batches_propagates_decode_error(inference_module):
    positions = [(0, 0, 1, 1), (1, 0, 1, 1), (2, 0, 1, 1)]

    def load_tensor(pos):
        if pos[0] == 1:
            raise ValueError("boom")
        return torch.zeros((1, 1, 1, 1))

    with pytest.raises(ValueError, match="boom"):
        list(inference_module._iter_decoded_batches(positions, 1, load_tensor, 2))


def test_prefetch_output_matches_serial(tmp_path, monkeypatch, inference_module):
    # Same slide, prefetch (default 2) vs serial (0), single-tile batches so
    # multiple batches are decoded. Region output must be identical.
    prefetch_out = _run_level_selection_slide(
        tmp_path, monkeypatch, inference_module, _SparseTissueSlide,
        extra_argv=("--batch-size", "1"),
    )
    serial_out = _run_level_selection_slide(
        tmp_path, monkeypatch, inference_module, _SparseTissueSlide,
        extra_argv=("--batch-size", "1", "--prefetch-workers", "0"),
    )
    assert prefetch_out["regions"] == serial_out["regions"]
    assert prefetch_out["summary"] == serial_out["summary"]
