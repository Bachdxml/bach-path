import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricSimilarityLoss(nn.Module):
    """
    Density-aware Asymmetric Similarity Loss.

    alpha : FN penalty weight - higher alpha = more sensitive (don't miss positives)
    beta  : FP penalty weight - higher beta = more conservative (avoid false alarms)
    alpha + beta should sum to 1.0

    Density behaviour:
        low density     -> balanced alpha/beta (conservative - structures less likely real)
        medium density  -> slightly higher alpha (mild recall bias)
        high density    -> higher alpha, lower beta (sensitive - don't miss structures in dense regions)
        negative        -> balanced alpha/beta (all-background tiles)
    """

    _DENSITY_IDX = {"low": 0, "medium": 1, "high": 2, "negative": 3}

    def __init__(self, density_params: dict, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.DENSITY_PARAMS = {
            self._DENSITY_IDX[name]: tuple(ab)
            for name, ab in density_params.items()
        }

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor,
                density_labels: torch.Tensor) -> torch.Tensor:
        """
        logits         : [B, 1, H, W]  raw model output (pre-sigmoid)
        targets        : [B, 1, H, W]  binary masks
        density_labels : [B]           long tensor, density class per tile
        """
        probs = torch.sigmoid(logits)

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        dims = (1, 2, 3)
        tp = (probs * targets).sum(dims)
        fp = (probs * (1 - targets)).sum(dims)
        fn = ((1 - probs) * targets).sum(dims)

        # Build per-sample alpha/beta tensors from density labels
        alpha = torch.zeros(logits.size(0), device=logits.device)
        beta  = torch.zeros(logits.size(0), device=logits.device)

        for density_idx, (a, b) in self.DENSITY_PARAMS.items():
            mask      = (density_labels == density_idx)
            alpha[mask] = a
            beta[mask]  = b

        # Asymmetric similarity index per sample
        similarity = (tp + self.smooth) / (
            tp + alpha * fn + beta * fp + self.smooth
        )

        loss = (1.0 - similarity)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combines segmentation loss (AsymmetricSimilarityLoss), density classification
    loss (CrossEntropy), and auxiliary deep-supervision losses.
    """

    def __init__(self, loss_cfg: dict):
        super().__init__()
        self.seg_loss = AsymmetricSimilarityLoss(
            density_params=loss_cfg["density_params"],
            smooth=loss_cfg.get("smooth", 1e-6),
        )
        self.density_loss = nn.CrossEntropyLoss()
        self.density_weight = loss_cfg["density_weight"]
        self.aux3_weight = loss_cfg["aux3_weight"]
        self.aux2_weight = loss_cfg["aux2_weight"]

    def forward(self, seg_logits, density_logits, targets, density_labels, aux3, aux2):
        l_seg = self.seg_loss(seg_logits, targets, density_labels)
        l_density = self.density_loss(density_logits, density_labels)

        aux3_up = F.interpolate(aux3, size=targets.shape[2:],
                                mode='bilinear', align_corners=False)
        aux2_up = F.interpolate(aux2, size=targets.shape[2:],
                                mode='bilinear', align_corners=False)
        l_aux = (
            self.aux3_weight * self.seg_loss(aux3_up, targets, density_labels)
            + self.aux2_weight * self.seg_loss(aux2_up, targets, density_labels)
        )

        total = l_seg + l_aux + self.density_weight * l_density
        return total, l_seg, l_density
