"""Tests for the density-only forward path (specs/pass1-density-only-forward.md)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "model.py"


def _load_model_module():
    spec = importlib.util.spec_from_file_location("wsi_model_under_test", MODEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_model(mod):
    return mod.ResidualAttentionUNet(
        in_ch=3, out_ch=1, num_density_classes=4, dropout_p=0.3
    ).eval()


def test_density_only_returns_none_seg_and_density_shape():
    mod = _load_model_module()
    model = _build_model(mod)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        seg, density, aux3, aux2 = model(x, density_only=True)
    assert seg is None and aux3 is None and aux2 is None
    assert density.shape == (2, 4)


def test_density_only_matches_full_forward_density_logits():
    mod = _load_model_module()
    model = _build_model(mod)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        _, density_full, _, _ = model(x)
        _, density_only, _, _ = model(x, density_only=True)
    assert torch.allclose(density_full, density_only, atol=1e-6)


def test_density_only_skips_decoder():
    mod = _load_model_module()
    model = _build_model(mod)
    x = torch.randn(1, 3, 64, 64)

    calls = {"decoder_ran": False}
    original = model.upconv4.forward

    def spy(*args, **kwargs):
        calls["decoder_ran"] = True
        return original(*args, **kwargs)

    model.upconv4.forward = spy
    with torch.no_grad():
        model(x, density_only=True)
    assert calls["decoder_ran"] is False

    # Sanity: the full forward does run the decoder through upconv4.
    with torch.no_grad():
        model(x)
    assert calls["decoder_ran"] is True


def test_density_only_ignores_density_label_and_still_skips_decoder():
    mod = _load_model_module()
    model = _build_model(mod)
    x = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 2], dtype=torch.long)
    with torch.no_grad():
        seg, density, _, _ = model(x, density_label=labels, density_only=True)
    assert seg is None
    assert density.shape == (2, 4)
