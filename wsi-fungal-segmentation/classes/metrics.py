import torch


def compute_all_metrics(preds_sigmoid, targets, threshold=0.5, smooth=1e-6):
    """
    Single-pass metric computation. Use this in your eval loop.
    Returns dict with dice, iou, precision, recall.
    """
    preds_bin = (preds_sigmoid > threshold).float()
    preds_flat  = preds_bin.reshape(preds_bin.size(0), -1)
    target_flat = targets.reshape(targets.size(0), -1)

    tp = (preds_flat * target_flat).sum(dim=1)
    fp = (preds_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * target_flat).sum(dim=1)

    union = preds_flat.sum(dim=1) + target_flat.sum(dim=1) - tp
    intersection = tp

    dice      = (2.0 * intersection + smooth) / (preds_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    iou       = (intersection + smooth) / (union + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)

    return {
        "dice":      dice.mean().item(),
        "iou":       iou.mean().item(),
        "precision": precision.mean().item(),
        "recall":    recall.mean().item(),
    }
