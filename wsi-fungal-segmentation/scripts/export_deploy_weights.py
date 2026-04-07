#!/usr/bin/env python3
"""
Export a gzip-compressed inference-only checkpoint for Bach Path (desktop).

Train on a separate machine or environment using the repo layout:
  wsi-fungal-segmentation/scripts/train.py
  wsi-fungal-segmentation/configs/default.yaml

Then run this script and copy the resulting .pth.gz into:
  wsi-fungal-segmentation/models/

Example:
  python scripts/export_deploy_weights.py \\
    --checkpoint checkpoints/best_model.pth \\
    --output models/deploy-fungus.pth.gz
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gzip
import io
import sys
from pathlib import Path

import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _safe_torch_load(source, *, map_location):
    """Refuse to deserialize checkpoints unless PyTorch supports weights-only loading."""
    try:
        return torch.load(source, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This script requires torch.load(..., weights_only=True). "
            "Upgrade PyTorch or re-export the checkpoint; refusing unsafe deserialization."
        ) from exc


def _is_tensor_state_dict(candidate: Mapping) -> bool:
    return bool(candidate) and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in candidate.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export compressed deploy weights for Bach Path")
    parser.add_argument("--checkpoint", required=True, help="Full training checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="Output path, e.g. models/deploy.pth.gz")
    args = parser.parse_args()

    src = Path(args.checkpoint).resolve()
    dest = Path(args.output).resolve()
    if not src.is_file():
        print(f"Checkpoint not found: {src}", file=sys.stderr)
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        ckpt = _safe_torch_load(src, map_location="cpu")
    except Exception as exc:
        print(f"Failed to load checkpoint safely: {exc}", file=sys.stderr)
        return 1

    # Export only accepts training checkpoints from this repo's pipeline.
    if not isinstance(ckpt, Mapping):
        print("Expected a mapping checkpoint loaded in weights-only mode.", file=sys.stderr)
        return 1

    state_dict = ckpt.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not _is_tensor_state_dict(state_dict):
        print("Expected a non-empty tensor 'model_state_dict' in the checkpoint.", file=sys.stderr)
        return 1

    deploy = {
        "model_state_dict": state_dict,
        "cfg": ckpt.get("cfg"),
    }

    buf = io.BytesIO()
    torch.save(deploy, buf)
    raw = buf.getvalue()
    with gzip.open(dest, "wb", compresslevel=9) as gz:
        gz.write(raw)

    ratio = (100.0 * len(raw)) / max(src.stat().st_size, 1)
    print(f"Wrote {dest} ({dest.stat().st_size} bytes, raw pickle ~{len(raw)} B, ~{ratio:.0f}% of source size)")
    print(f"Copy into: {_project_root() / 'models'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
