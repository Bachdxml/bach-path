"""
Load a saved checkpoint and run final evaluation + prediction visualization.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src import (
    AugmentedWSI_Dataset,
    FocalTverskyLoss,
    ResidualAttentionUNet,
    WSIDatasetIndex,
    compute_all_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train")
    if history["val_loss"]:
        axes[0].plot(epochs_range, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs_range, history["train_dice"], label="Train")
    if history["val_dice"]:
        axes[1].plot(epochs_range, history["val_dice"], label="Val")
    axes[1].set_title("Dice"); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs_range, history["train_iou"], label="Train")
    if history["val_iou"]:
        axes[2].plot(epochs_range, history["val_iou"], label="Val")
    axes[2].set_title("IoU"); axes[2].legend(); axes[2].grid(True)

    axes[3].plot(epochs_range, history["lr"], color="purple")
    axes[3].set_title("Learning Rate"); axes[3].set_yscale("log"); axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(checkpoint_path: str, config_path: str = "configs/default.yaml",
         visualize: bool = False, n_vis: int = 2):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"  Epoch {ckpt['epoch']}  |  best dice = {ckpt['best_dice']:.4f}")

    img_size = ckpt.get("img_size", cfg["data"]["img_size"])

    model = ResidualAttentionUNet(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = FocalTverskyLoss(**cfg["loss"])

    # ---- Build val loader ----
    index = WSIDatasetIndex(cfg["data"]["export_root"], strict_mode=True)
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

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            m = compute_all_metrics(torch.sigmoid(logits), masks)
            running["loss"]      += criterion(logits, masks).item()
            running["dice"]      += m["dice"]
            running["iou"]       += m["iou"]
            running["precision"] += m["precision"]
            running["recall"]    += m["recall"]

    print("\nFinal Validation Metrics:")
    for k, v in running.items():
        print(f"  {k:<10}: {v/n:.4f}")

    # ---- Plot training curves (if stored in checkpoint) ----
    if "history" in ckpt:
        plot_training_curves(ckpt["history"])

    # ---- Visualize predictions ----
    if visualize:
        imgs, masks = next(iter(val_loader))
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            preds_bin = (torch.sigmoid(model(imgs)) > 0.5).float()

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
