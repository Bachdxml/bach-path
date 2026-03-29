import numpy as np
import torch.nn.functional as F


def overlay_mask(image_tensor, mask_tensor, color=[1,0,0], alpha=0.4, bg_tint=None, bg_alpha=0.2):
    """
    Overlay a binary mask on an image.
    
    Args:
        image_tensor: Input image tensor
        mask_tensor: Binary mask tensor
        color: RGB color for the mask overlay
        alpha: Opacity of the mask overlay
        bg_tint: Optional RGB color to tint the background (non-mask areas)
        bg_alpha: Opacity of the background tint
    """
    # Denormalize the image
    img = image_tensor.cpu().permute(1,2,0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    # Upsample mask if needed
    if mask_tensor.shape != image_tensor.shape[1:]:
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=image_tensor.shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    mask = mask_tensor.cpu().numpy()
    overlay = img.copy()
    
    for c in range(3):
        if bg_tint is not None:
            # Apply mask color where mask > 0, and background tint where mask == 0
            overlay[:,:,c] = np.where(
                mask > 0, 
                img[:,:,c] * (1 - alpha) + color[c] * alpha,  # Mask areas
                img[:,:,c] * (1 - bg_alpha) + bg_tint[c] * bg_alpha  # Background areas
            )
        else:
            # Original behavior - only overlay the mask
            overlay[:,:,c] = np.where(
                mask > 0, 
                img[:,:,c] * (1 - alpha) + color[c] * alpha, 
                img[:,:,c]
            )
    
    return overlay
