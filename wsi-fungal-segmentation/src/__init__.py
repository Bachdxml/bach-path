from __future__ import annotations

from .losses import AsymmetricSimilarityLoss, CombinedLoss
from .metrics import compute_all_metrics
from .model import ResidualAttentionUNet

_DATASET_EXPORTS = {
    "AugmentedWSI_Dataset",
    "TilePair",
    "WSI_Dataset",
    "WSIDatasetIndex",
    "make_stratified_sampler",
}

__all__ = [
    "AsymmetricSimilarityLoss",
    "CombinedLoss",
    "compute_all_metrics",
    "ResidualAttentionUNet",
    *_DATASET_EXPORTS,
]


def __getattr__(name: str):
    if name in _DATASET_EXPORTS:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
