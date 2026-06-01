"""
Load a saved checkpoint and run final evaluation + prediction visualization.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
"""

import argparse
from collections.abc import Mapping
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src import (
    AugmentedWSI_Dataset,
    CombinedLoss,
    ResidualAttentionUNet,
    WSIDatasetIndex,
    compute_all_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _load_checkpoint(path: Path, device: torch.device) -> Mapping:
    # Checkpoints cross a trust boundary; only accept the dict layout emitted by
    # this project's training/export scripts and loaded in weights-only mode.
    checkpoint = _safe_torch_load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping loaded in weights-only mode.")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not _is_tensor_state_dict(state_dict):
        raise ValueError("Checkpoint must contain a non-empty 'model_state_dict' tensor mapping.")

    return checkpoint

def denormalize(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


def overlay_mask(image_tensor, mask_tensor, color=(1, 0, 0), alpha=0.4,
                 bg_tint=None, bg_alpha=0.2):
    img = denormalize(image_tensor)

    if mask_tensor.shape != image_tensor.shape[1:]:
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=image_tensor.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    mask    = mask_tensor.cpu().numpy()
    overlay = img.copy()

    for c in range(3):
        if bg_tint is not None:
            overlay[:, :, c] = np.where(
                mask > 0,
                img[:, :, c] * (1 - alpha)    + color[c]    * alpha,
                img[:, :, c] * (1 - bg_alpha) + bg_tint[c]  * bg_alpha,
            )
        else:
            overlay[:, :, c] = np.where(
                mask > 0,
                img[:, :, c] * (1 - alpha) + color[c] * alpha,
                img[:, :, c],
            )
    return overlay


def plot_training_curves(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    def _plot_series(ax, values, label, color=None):
        valid_points = [
            (epoch_idx, value)
            for epoch_idx, value in enumerate(values, start=1)
            if value is not None
        ]
        if not valid_points:
            return False

        xs, ys = zip(*valid_points)
        # A single history point is otherwise invisible with a line-only plot.
        marker = "o"
        linestyle = "-" if len(xs) > 1 else "None"
        ax.plot(xs, ys, label=label, color=color, marker=marker, linestyle=linestyle)
        ax.set_xticks(list(xs))
        return True

    def _plot_panel(ax, train_values, val_values):
        plotted = _plot_series(ax, train_values, "Train")
        plotted = _plot_series(ax, val_values, "Val") or plotted
        return plotted

    plotted_any = [
        _plot_panel(axes[0], history.get("train_loss", []), history.get("val_loss", [])),
        _plot_panel(axes[1], history.get("train_dice", []), history.get("val_dice", [])),
        _plot_panel(axes[2], history.get("train_iou", []), history.get("val_iou", [])),
        _plot_series(axes[3], history.get("lr", []), "LR", color="purple"),
    ]

    titles = ["Loss", "Dice", "IoU", "Learning Rate"]
    for idx, ax in enumerate(axes):
        ax.set_title(titles[idx])
        ax.grid(True)
        if plotted_any[idx]:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No history data", ha="center", va="center", transform=ax.transAxes)

    axes[3].set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(checkpoint_path: str, config_path: str = "configs/default.yaml",
         visualize: bool = False, n_vis: int = 2):

    with open(config_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = _load_checkpoint(Path(checkpoint_path), device)
    epoch = ckpt.get("epoch", "unknown")
    best_dice = ckpt.get("best_dice")
    best_dice_text = f"{best_dice:.4f}" if isinstance(best_dice, (int, float)) else "n/a"
    print(f"  Epoch {epoch}  |  best dice = {best_dice_text}")

    img_size = ckpt.get("cfg", cfg)["data"]["img_size"]

    model = ResidualAttentionUNet(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = CombinedLoss(loss_cfg=cfg["loss"])

    # ---- Build val loader ----
    index = WSIDatasetIndex(
        cfg["data"]["export_root"],
        strict_mode=False,
        allow_size_mismatch=True,
        flat_format=cfg["data"].get("flat_format", False),
        skip_validation=True,
    )
    index.build_index()
    _, val_pairs = index.get_train_val_split(
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=cfg["data"]["random_seed"],
    )

    val_ds = AugmentedWSI_Dataset(val_pairs, img_size=img_size, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["loader"]["num_workers"],
    )

    # ---- Evaluate ----
    model.eval()
    running = {"loss": 0, "dice": 0, "iou": 0, "precision": 0, "recall": 0}
    n = len(val_loader)
    if n == 0:
        print("Validation split is empty; skipping metric computation and visualization.")
        return

    with torch.no_grad():
        for imgs, masks, density_labels in val_loader:
            imgs           = imgs.to(device)
            masks          = masks.to(device)
            density_labels = density_labels.to(device)
            seg_logits, density_logits, aux3, aux2 = model(imgs, density_labels)
            m = compute_all_metrics(torch.sigmoid(seg_logits), masks)
            total, l_seg, l_density = criterion(seg_logits, density_logits, masks, density_labels, aux3, aux2)
            running["loss"] += total.item()
            running["dice"]      += m["dice"]
            running["iou"]       += m["iou"]
            running["precision"] += m["precision"]
            running["recall"]    += m["recall"]

    print("\nFinal Validation Metrics:")
    for k, v in running.items():
        print(f"  {k:<10}: {v/n:.4f}")

    # ---- Plot training curves (if stored in checkpoint) ----
    if "history" in ckpt:
        history = ckpt["history"]
        epoch_count = max(
            (len(series) for series in history.values() if isinstance(series, list)),
            default=0,
        )
        if epoch_count <= 1:
            print(
                f"\nHistory contains {epoch_count} epoch"
                f"{'' if epoch_count == 1 else 's'}; plots will show points rather than lines."
            )
        plot_training_curves(history)

    # ---- Visualize predictions ----
    if visualize:
        imgs, masks, density_labels = next(iter(val_loader))
        imgs           = imgs.to(device)
        masks          = masks.to(device)
        density_labels = density_labels.to(device)
        with torch.no_grad():
            seg_logits, _, _, _ = model(imgs)  # density_label=None triggers self-prediction
            preds_bin = (torch.sigmoid(seg_logits) > 0.5).float()

        for i in range(min(n_vis, imgs.size(0))):
            gt   = overlay_mask(imgs[i], masks[i, 0],     color=(0, 1, 0), alpha=0.4)
            pred = overlay_mask(imgs[i], preds_bin[i, 0], color=(1, 0, 0), alpha=0.4,
                                bg_tint=(0, 0, 1), bg_alpha=0.2)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(denormalize(imgs[i])); axes[0].set_title("Image")
            axes[1].imshow(gt);   axes[1].set_title("Ground Truth (green)")
            axes[2].imshow(pred); axes[2].set_title("Prediction (red / blue bg)")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--visualize",  action="store_true")
    parser.add_argument("--n_vis",      type=int, default=2)
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.visualize, args.n_vis)
