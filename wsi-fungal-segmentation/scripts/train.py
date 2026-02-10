# TEMPORARY FIX: Use all data for training (only 2 samples available)
print("⚠️ WARNING: Using all data for training (no validation split)")
print(f"   Total tiles available: {len(index.tile_pairs)}")

# Define image size
IMG_SIZE = 512  

# Use ALL data for training
train_pairs = index.tile_pairs
train_dataset = WSI_Dataset(train_pairs, img_size=IMG_SIZE)

print(f"\nTrain dataset: {len(train_dataset)} tiles")

# Create only train loader
BATCH_SIZE = 2  # Set to match your data size (you have 2 tiles)
NUM_WORKERS = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# No validation loader for now
val_loader = None

# Quick check
train_imgs, train_masks = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  Images: {train_imgs.shape}")
print(f"  Masks: {train_masks.shape}")

#---------------------------------------#

# Visualize first 2 samples from training set
for i in range(min(2, train_imgs.size(0))):
    # Denormalize image for visualization
    img_np = train_imgs[i].permute(1,2,0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    mask_np = train_masks[i,0].numpy()
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#---------------------------------------#

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = ResidualAttentionUNet(in_ch=3, out_ch=1).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

#---------------------------------------#

EPOCHS = 10

print("="*60)
print("Starting Training...")
print("="*60)

for epoch in range(EPOCHS):
    # ===== TRAINING =====
    model.train()
    train_loss = 0.0
    
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * imgs.size(0)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    train_loss /= len(train_dataset)
    
    # Print epoch summary (no validation)
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{EPOCHS} Summary")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"{'='*60}\n")

print("Training complete!")

#---------------------------------------#
