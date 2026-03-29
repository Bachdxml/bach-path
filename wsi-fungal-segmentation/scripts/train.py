"""
Training entry point.

Usage:
    python train.py                          # uses configs/default.yaml
    python train.py --config configs/my.yaml
"""

import argparse
import sys
import platform
import signal
import time
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
    CombinedLoss,
    ResidualAttentionUNet,
    WSIDatasetIndex,
    compute_all_metrics,
    make_stratified_sampler,
)

def log(message: str = "") -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch_num, clip_grad=1.0, should_stop=None):
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
    processed_batches = 0

    for batch_idx, (imgs, masks, density_labels) in enumerate(loader):
        if callable(should_stop) and should_stop():
            break
        imgs           = imgs.to(device, non_blocking=True)
        masks          = masks.to(device, non_blocking=True)
        density_labels = density_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        seg_logits, density_logits, aux3, aux2 = model(imgs, density_labels)
        total, l_seg, l_density = criterion(seg_logits, density_logits, masks, density_labels, aux3, aux2)

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(seg_logits)
            m = compute_all_metrics(probs, masks)

        running_loss += total.item()
        running_dice += m['dice']
        running_iou  += m['iou']
        processed_batches += 1

        if batch_idx % log_interval == 0:
            pct = ((batch_idx + 1) / max(1, n_batches)) * 100.0
            log(
                f"  [Epoch {epoch_num} | Batch {batch_idx+1}/{n_batches} | {pct:5.1f}%] "
                f"loss={total.item():.4f}  dice={m['dice']:.4f}  "
                f"iou={m['iou']:.4f}"
            )

    denom = max(1, processed_batches)
    return {
        "loss": running_loss / denom,
        "dice": running_dice / denom,
        "iou":  running_iou  / denom,
        "stopped_early": processed_batches < n_batches,
    }

def evaluate(model, loader, criterion, device, epoch_num: int | None = None):
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
    log_interval      = max(1, n_batches // 10)
    started_at        = time.time()

    with torch.no_grad():
        for batch_idx, (imgs, masks, density_labels) in enumerate(loader):
            imgs           = imgs.to(device,  non_blocking=True)
            masks          = masks.to(device, non_blocking=True)
            density_labels = density_labels.to(device, non_blocking=True)

            seg_logits, density_logits, aux3, aux2 = model(imgs, density_labels)
            total, l_seg, l_density = criterion(seg_logits, density_logits, masks, density_labels, aux3, aux2)
            probs = torch.sigmoid(seg_logits)
            running_loss += total.item()

            m = compute_all_metrics(probs, masks)
            running_dice      += m['dice']
            running_iou       += m['iou']
            running_precision += m['precision']
            running_recall    += m['recall']

            if batch_idx % log_interval == 0:
                pct = ((batch_idx + 1) / max(1, n_batches)) * 100.0
                elapsed = time.time() - started_at
                epoch_part = f"Epoch {epoch_num} | " if epoch_num is not None else ""
                log(
                    f"  [Val {epoch_part}Batch {batch_idx+1}/{n_batches} | {pct:5.1f}% | "
                    f"{elapsed:5.1f}s] loss={total.item():.4f} dice={m['dice']:.4f}"
                )

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
    log("✅ No WSI leakage")

    # ---- Loaders ----
    img_size   = cfg["data"]["img_size"]
    batch_size = cfg["loader"]["batch_size"]
    n_workers  = cfg["loader"]["num_workers"]
    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    pin = has_cuda

    # Desktop training on macOS/CPU can be killed by OS memory pressure (exit -9).
    # Use a conservative batch size for large tiles when CUDA is unavailable.
    if not has_cuda and img_size >= 512 and batch_size > 1:
        log(
            f"⚠️  Reducing batch size from {batch_size} to 1 for stability "
            f"(device={'mps' if has_mps else 'cpu'}, img_size={img_size}, os={platform.system()})."
        )
        batch_size = 1

    train_ds = AugmentedWSI_Dataset(train_pairs, img_size=img_size, augment=True)
    val_ds   = AugmentedWSI_Dataset(val_pairs,   img_size=img_size, augment=False) \
               if val_pairs else None

    train_sampler = make_stratified_sampler(train_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=n_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=pin) \
                   if val_ds else None

    log(f"Train: {len(train_ds)} tiles  |  Val: {len(val_ds) if val_ds else 0} tiles")
    if has_cuda:
        device = torch.device("cuda")
    elif has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log(f"Device: {device}")

    model     = ResidualAttentionUNet(**cfg["model"]).to(device)
    criterion = CombinedLoss(loss_cfg=cfg["loss"])
    optimizer = optim.AdamW(model.parameters(), **cfg["optimizer"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, **cfg["scheduler"]
    )
    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

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
    stop_requested = {"value": False}

    def write_progress(status: str, epoch: int = 0, **kwargs):
        if progress_file:
            import json
            import os
            import tempfile
            data = {"status": status, "epoch": epoch, **kwargs}
            progress_path = Path(progress_file)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=progress_path.parent,
                prefix=f"{progress_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_path = Path(tmp_file.name)
            os.replace(tmp_path, progress_path)

    def handle_stop_signal(signum, frame):
        stop_requested["value"] = True
        log(f"\n⚠️  Stop requested (signal {signum}). Will save checkpoint and exit after current step.")

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGINT, handle_stop_signal)

    log("\n" + "=" * 60)
    log("Starting Training")
    log("=" * 60)
    write_progress("running", epoch=0)

    epoch = 0
    try:
        for epoch in range(1, t_cfg["epochs"] + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            log(f"\n{'='*60}")
            log(f"Epoch {epoch}/{t_cfg['epochs']}   |   LR = {current_lr:.2e}")
            log("=" * 60)

            train_m = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch_num=epoch, clip_grad=t_cfg["clip_grad"],
                should_stop=lambda: stop_requested["value"],
            )
            gc.collect()
            if not stop_requested["value"] and val_loader is not None:
                log(f"  [Epoch {epoch}] Starting validation...")
            val_m = (
                evaluate(model, val_loader, criterion, device, epoch_num=epoch)
                if not stop_requested["value"]
                else None
            )
            gc.collect()

            sched_metric = val_m["dice"] if val_m else train_m["dice"]
            scheduler.step(sched_metric)

            history["lr"].append(current_lr)
            history["train_loss"].append(train_m["loss"])
            history["train_dice"].append(train_m["dice"])
            history["train_iou"].append(train_m["iou"])

            
            history["val_loss"].append(val_m["loss"] if val_m else None)
            history["val_dice"].append(val_m["dice"] if val_m else None)
            history["val_iou"].append(val_m["iou"]  if val_m else None)

            write_progress("running", epoch=epoch,
                        train_loss=train_m["loss"], train_dice=train_m["dice"],
                        val_loss=val_m["loss"] if val_m else None,
                        val_dice=val_m["dice"] if val_m else None,
                        best_dice=best_val_dice)

            log(f"\nEpoch {epoch} Summary:")
            log(f"  Train → loss={train_m['loss']:.4f}  dice={train_m['dice']:.4f}  iou={train_m['iou']:.4f}")
            if val_m:
                log(f"  Val   → loss={val_m['loss']:.4f}  dice={val_m['dice']:.4f}  iou={val_m['iou']:.4f}")
                log(f"           precision={val_m['precision']:.4f}  recall={val_m['recall']:.4f}")

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
                    "cfg": cfg,
                }, checkpoint_path)
                log(f"  ✅  New best dice={best_val_dice:.4f} — checkpoint saved.")
            else:
                no_improve_count += 1
                log(f"  ⏳  No improvement ({no_improve_count}/{t_cfg['early_stop_patience']}). Best={best_val_dice:.4f}")

            if stop_requested["value"]:
                log(f"\n🛑 Stop requested by user at epoch {epoch}.")
                break

            if no_improve_count >= t_cfg["early_stop_patience"]:
                log(f"\n🛑 Early stopping at epoch {epoch}.")
                break

        if stop_requested["value"]:
            stopped_checkpoint = str(Path(checkpoint_path).with_name("stopped_model.pth"))
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice":            best_val_dice,
                "history":              history,
                "cfg": cfg,
                "stopped_by_user":      True,
            }, stopped_checkpoint)
            log("\n" + "=" * 60)
            log(f"Training stopped by user. Saved checkpoint: {stopped_checkpoint}")
            log("=" * 60)
            write_progress("stopped", epoch=epoch, best_dice=best_val_dice,
                        checkpoint_path=str(stopped_checkpoint))
            return history

        log("\n" + "=" * 60)
        log(f"Training complete. Best dice: {best_val_dice:.4f}")
        log(f"Checkpoint: {checkpoint_path}")
        log("=" * 60)
        write_progress("succeeded", epoch=epoch, best_dice=best_val_dice,
                    checkpoint_path=str(checkpoint_path))
        return history
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)


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
