"""Re-export dataset classes from wsi_dataset_correct."""
from .wsi_dataset_correct import (
    AugmentedWSI_Dataset,
    DENSITY_FOLDERS,
    DENSITY_LABELS,
    TilePair,
    WSI_Dataset,
    WSIDatasetIndex,
    make_stratified_sampler,
)

__all__ = [
    "AugmentedWSI_Dataset",
    "DENSITY_FOLDERS",
    "DENSITY_LABELS",
    "TilePair",
    "WSI_Dataset",
    "WSIDatasetIndex",
    "make_stratified_sampler",
]
