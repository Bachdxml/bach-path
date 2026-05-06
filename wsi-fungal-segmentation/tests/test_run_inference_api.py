from __future__ import annotations

import importlib.util
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


@pytest.fixture
def inference_module():
    return _load_module()


def test_main_calls_model_without_explicit_density_label(tmp_path, monkeypatch, inference_module):
    slide_path = tmp_path / "slide.png"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    Image.new("RGB", (64, 64), color=(90, 160, 110)).save(slide_path)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    recorded_calls: list[tuple[torch.Tensor, ...]] = []

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

        def __call__(self, *args):
            recorded_calls.append(args)
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
    assert len(recorded_calls) == 1
    assert len(recorded_calls[0]) == 1
    assert recorded_calls[0][0].shape == (1, 3, 64, 64)


def test_openslide_path_calls_model_without_explicit_density_label(tmp_path, monkeypatch, inference_module):
    slide_path = tmp_path / "slide.svs"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "checkpoint.pth.gz"
    slide_path.write_text("stub slide", encoding="utf-8")
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    recorded_calls: list[tuple[torch.Tensor, ...]] = []

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

        def __call__(self, *args):
            recorded_calls.append(args)
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
    assert len(recorded_calls) == 1
    assert len(recorded_calls[0]) == 1
    assert recorded_calls[0][0].shape == (1, 3, 64, 64)


def test_compute_hotspot_stays_centered_for_centered_synthetic_map(inference_module):
    prob_map = torch.zeros((8, 8), dtype=torch.float32)
    prob_map[3:5, 3:5] = 1.0

    hotspot = inference_module._compute_hotspot(prob_map, x=100, y=200, w=80, h=80)

    assert hotspot is not None
    payload = hotspot["hotspot"]
    assert payload["cx"] == pytest.approx(140.0, abs=0.01)
    assert payload["cy"] == pytest.approx(240.0, abs=0.01)
    assert payload["x"] == pytest.approx(130.0, abs=0.01)
    assert payload["y"] == pytest.approx(230.0, abs=0.01)
    assert payload["w"] == pytest.approx(20.0, abs=0.01)
    assert payload["h"] == pytest.approx(20.0, abs=0.01)


def test_compute_hotspot_shifts_off_center_for_off_center_synthetic_map(inference_module):
    prob_map = torch.zeros((8, 8), dtype=torch.float32)
    prob_map[1:3, 5:7] = 1.0

    hotspot = inference_module._compute_hotspot(prob_map, x=100, y=200, w=80, h=80)

    assert hotspot is not None
    payload = hotspot["hotspot"]
    assert payload["cx"] > 140.0
    assert payload["cy"] < 240.0
    assert payload["x"] >= 150.0
    assert payload["y"] <= 210.0
    assert payload["x"] + payload["w"] <= 180.0
    assert payload["y"] + payload["h"] <= 230.0
