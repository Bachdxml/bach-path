# Put model in evaluation mode
model.eval()

# Get one batch from TRAINING set (if no validation)
imgs, masks = next(iter(train_loader))
imgs = imgs.to(device)
masks = masks.to(device)

# Run inference
with torch.no_grad():
    preds = model(imgs)  # UNet outputs [B,1,H,W] directly
    preds_sigmoid = torch.sigmoid(preds)

# Threshold at 0.5
preds_bin = (preds_sigmoid > 0.5).float() #lowered for testing purposes

# Visualize both samples
for i in range(min(2, imgs.size(0))):
    # Denormalize image
    img_np = imgs[i].cpu().permute(1,2,0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    mask_np = masks[i,0].cpu().numpy()
    pred_np = preds_bin[i,0].cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
