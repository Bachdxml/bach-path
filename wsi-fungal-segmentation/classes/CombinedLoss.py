class CombinedLoss(nn.Module):
    """
    Combines segmentation loss and density classification loss.
    Weights are read from config.yaml → loss section.
    """
 
    def __init__(self, loss_cfg: dict):
        """
        loss_cfg : cfg["loss"]
        """
        super().__init__()
        self.seg_loss       = AsymmetricSimilarityLoss(
            density_params=loss_cfg["density_params"]
        )
        self.density_loss   = nn.CrossEntropyLoss()
        self.density_weight = loss_cfg["density_weight"]
        self.aux3_weight    = loss_cfg["aux3_weight"]
        self.aux2_weight    = loss_cfg["aux2_weight"]
 
    def forward(self, seg_logits, density_logits, targets, density_labels, aux3, aux2):
        l_seg     = self.seg_loss(seg_logits, targets, density_labels)
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
