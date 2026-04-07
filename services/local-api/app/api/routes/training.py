from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/training", tags=["training"])


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent.parent.parent


class PathEntry(BaseModel):
    """Repo-relative path plus a redacted server-side identifier."""

    repo_relative: str
    resolved: str


class TrainingInfoResponse(BaseModel):
    """
    Training runs outside the desktop app. This endpoint documents canonical paths
    (aligned with wsi-fungal-segmentation on main) and where to place deploy weights.
    """

    training_in_app: bool = False
    train_script: PathEntry
    config_default: PathEntry
    export_deploy_script: PathEntry
    models_dir: PathEntry
    qupath_export_script: PathEntry
    notes: list[str]


@router.get("/info", response_model=TrainingInfoResponse)
def training_info() -> TrainingInfoResponse:
    train_rel = "wsi-fungal-segmentation/scripts/train.py"
    cfg_rel = "wsi-fungal-segmentation/configs/default.yaml"
    export_rel = "wsi-fungal-segmentation/scripts/export_deploy_weights.py"
    models_rel = "wsi-fungal-segmentation/models"
    qp_rel = "wsi-fungal-segmentation/qu-path-scripts/export_tiles.groovy"

    def entry(rel: str) -> PathEntry:
        return PathEntry(repo_relative=rel, resolved=f"<workspace>/{rel}")

    notes = [
        "Train in a Python environment with GPU support using the paths above; do not rely on the desktop app to run training.",
        "After training, export a small gzip checkpoint with export_deploy_weights.py and copy the .pth.gz file into models/.",
        "Inference loads deployed gzip checkpoints (.pth.gz / .pt.gz) from models/.",
    ]
    if os.environ.get("INFERENCE_CHECKPOINT"):
        notes.append("INFERENCE_CHECKPOINT is set; inference may ignore the models directory.")

    return TrainingInfoResponse(
        train_script=entry(train_rel),
        config_default=entry(cfg_rel),
        export_deploy_script=entry(export_rel),
        models_dir=entry(models_rel),
        qupath_export_script=entry(qp_rel),
        notes=notes,
    )
