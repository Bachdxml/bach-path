"""
Training entry point.

Usage:
    python train.py                          # uses configs/default.yaml
    python train.py --config configs/my.yaml
"""

import argparse
import sys
from pathlib import Path

# scripts/ is not the package root; ensure wsi-fungal-segmentation is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gc
import os

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

# Reduce PyTorch memory overhead
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
torch.set_num_threads(2)

from src import (
    AugmentedWSI_Dataset,
    AsymmetricSimilarityLoss,
    ResidualAttentionUNet,
    WSIDatasetIndex,
    compute_all_metrics,
    make_stratified_sampler,
)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch_num, clip_grad=1.0):
    """
    Runs one full pass over the training DataLoader.
    Expects loader to yield (imgs, masks, density_labels).
    Returns dict with avg loss, dice, iou for the epoch.
    """
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    n_batches    = len(loader)
    log_interval = max(1, n_batches // 10)

    for batch_idx, (imgs, masks, density_labels) in enumerate(loader):
        imgs           = imgs.to(device, non_blocking=True)
        masks          = masks.to(device, non_blocking=True)
        density_labels = density_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(imgs, density_labels)
        loss   = criterion(logits, masks, density_labels)  # density-aware loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            m = compute_all_metrics(probs, masks)

        running_loss += loss.item()
        running_dice += m['dice']
        running_iou  += m['iou']

        if batch_idx % log_interval == 0:
            print(f"  [Epoch {epoch_num} | Batch {batch_idx+1}/{n_batches}] "
                  f"loss={loss.item():.4f}  dice={m['dice']:.4f}  "
                  f"iou={m['iou']:.4f}")

    return {
        "loss": running_loss / n_batches,
        "dice": running_dice / n_batches,
        "iou":  running_iou  / n_batches,
    }

def evaluate(model, loader, criterion, device):
    """
    Runs full validation pass.
    Expects loader to yield (imgs, masks, density_labels).
    Returns dict with avg loss, dice, iou, precision, recall — or None.
    """
    if loader is None:
        return None

    model.eval()

    running_loss      = 0.0
    running_dice      = 0.0
    running_iou       = 0.0
    running_precision = 0.0
    running_recall    = 0.0
    n_batches         = len(loader)

    with torch.no_grad():
        for imgs, masks, density_labels in loader:
            imgs           = imgs.to(device,  non_blocking=True)
            masks          = masks.to(device, non_blocking=True)
            density_labels = density_labels.to(device, non_blocking=True)

            logits = model(imgs, density_labels)
            loss   = criterion(logits, masks, density_labels)
            probs  = torch.sigmoid(logits)

            m = compute_all_metrics(probs, masks)
            running_loss      += loss.item()
            running_dice      += m['dice']
            running_iou       += m['iou']
            running_precision += m['precision']
            running_recall    += m['recall']

    return {
        "loss":      running_loss      / n_batches,
        "dice":      running_dice      / n_batches,
        "iou":       running_iou       / n_batches,
        "precision": running_precision / n_batches,
        "recall":    running_recall    / n_batches,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main_with_args(cfg_path: str = "configs/default.yaml",
                   export_root: str | None = None,
                   flat_format: bool | None = None,
                   progress_file: str | None = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if export_root:
        cfg["data"]["export_root"] = str(export_root)
    if flat_format is not None:
        cfg["data"]["flat_format"] = flat_format
    return _run_training(cfg, progress_file)


def _run_training(cfg: dict, progress_file: str | None = None):
    """Run training with config dict. Optionally write progress to JSON file."""
    # ---- Data ----
    export_root = Path(cfg["data"]["export_root"])
    if not export_root.exists():
        raise FileNotFoundError(f"Export root not found: {export_root}")
    flat_format = cfg["data"].get("flat_format", False)

    index = WSIDatasetIndex(
        export_root,
        strict_mode=False,
        allow_size_mismatch=True,
        flat_format=flat_format,
        skip_validation=True,
    )
    index.build_index()
    index.save_index(Path("dataset_index.json"))

    train_pairs, val_pairs = index.get_train_val_split(
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=cfg["data"]["random_seed"],
    )

    train_wsis = {p.wsi_id for p in train_pairs}
    val_wsis   = {p.wsi_id for p in val_pairs}
    assert not (train_wsis & val_wsis), "WSI leakage detected!"
    print("✅ No WSI leakage")

    # ---- Loaders ----
    img_size   = cfg["data"]["img_size"]
    batch_size = cfg["loader"]["batch_size"]
    n_workers  = cfg["loader"]["num_workers"]
    pin        = torch.cuda.is_available()

    train_ds = AugmentedWSI_Dataset(train_pairs, img_size=img_size, augment=True)
    val_ds   = AugmentedWSI_Dataset(val_pairs,   img_size=img_size, augment=False) \
               if val_pairs else None

    train_sampler = make_stratified_sampler(train_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=n_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=pin) \
                   if val_ds else None

    print(f"Train: {len(train_ds)} tiles  |  Val: {len(val_ds) if val_ds else 0} tiles")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = ResidualAttentionUNet(**cfg["model"]).to(device)
    criterion = AsymmetricSimilarityLoss(**cfg["loss"])
    optimizer = optim.AdamW(model.parameters(), **cfg["optimizer"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", **cfg["scheduler"]
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    t_cfg           = cfg["training"]
    checkpoint_path = t_cfg["checkpoint_path"]
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss":   [], "val_dice":   [], "val_iou":   [],
        "lr":         [],
    }
    best_val_dice    = -1.0
    no_improve_count = 0

    def write_progress(status: str, epoch: int = 0, **kwargs):
        if progress_file:
            import json
            data = {"status": status, "epoch": epoch, **kwargs}
            with open(progress_file, "w") as f:
                json.dump(data, f, indent=2)

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    write_progress("running", epoch=0)

    for epoch in range(1, t_cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{t_cfg['epochs']}   |   LR = {current_lr:.2e}")
        print("=" * 60)

        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch_num=epoch, clip_grad=t_cfg["clip_grad"]
        )
        gc.collect()
        val_m = evaluate(model, val_loader, criterion, device)
        gc.collect()

        sched_metric = -(val_m["dice"] if val_m else train_m["dice"])
        scheduler.step(sched_metric)

        history["lr"].append(current_lr)
        history["train_loss"].append(train_m["loss"])
        history["train_dice"].append(train_m["dice"])
        history["train_iou"].append(train_m["iou"])
        if val_m:
            history["val_loss"].append(val_m["loss"])
            history["val_dice"].append(val_m["dice"])
            history["val_iou"].append(val_m["iou"])

        write_progress("running", epoch=epoch,
                      train_loss=train_m["loss"], train_dice=train_m["dice"],
                      val_loss=val_m["loss"] if val_m else None,
                      val_dice=val_m["dice"] if val_m else None,
                      best_dice=best_val_dice)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train → loss={train_m['loss']:.4f}  dice={train_m['dice']:.4f}  iou={train_m['iou']:.4f}")
        if val_m:
            print(f"  Val   → loss={val_m['loss']:.4f}  dice={val_m['dice']:.4f}  iou={val_m['iou']:.4f}")
            print(f"           precision={val_m['precision']:.4f}  recall={val_m['recall']:.4f}")

        monitor_dice = val_m["dice"] if val_m else train_m["dice"]
        if monitor_dice > best_val_dice:
            best_val_dice    = monitor_dice
            no_improve_count = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice":            best_val_dice,
                "history":              history,
                "img_size":             img_size,
                "batch_size":           batch_size,
            }, checkpoint_path)
            print(f"  ✅  New best dice={best_val_dice:.4f} — checkpoint saved.")
        else:
            no_improve_count += 1
            print(f"  ⏳  No improvement ({no_improve_count}/{t_cfg['early_stop_patience']}). Best={best_val_dice:.4f}")

        if no_improve_count >= t_cfg["early_stop_patience"]:
            print(f"\n🛑 Early stopping at epoch {epoch}.")
            break

    print("\n" + "=" * 60)
    print(f"Training complete. Best dice: {best_val_dice:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    write_progress("succeeded", epoch=epoch, best_dice=best_val_dice,
                   checkpoint_path=str(checkpoint_path))
    return history


def main(cfg_path: str = "configs/default.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return _run_training(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--export-root", help="Path to QuPath-exported tiles (slide/images, slide/masks)")
    parser.add_argument("--progress-file", help="JSON file to write training progress for API polling")
    parser.add_argument("--flat-format", action="store_true",
                        help="Use flat format: <slide>/images, <slide>/masks (default in config)")
    args = parser.parse_args()
    main_with_args(
        cfg_path=args.config,
        export_root=args.export_root or None,
        flat_format=args.flat_format if args.flat_format else None,
        progress_file=args.progress_file,
    )
