class WSI_Dataset(Dataset):
    """
    PyTorch Dataset that operates on validated TilePair records.
    GUARANTEED to have correct image-mask pairing.
    """
    
    def __init__(self, tile_pairs: List[TilePair], img_size: int = 512):
        self.tile_pairs = tile_pairs
        
        # Ensure img_size is divisible by 16 for UNet
        if img_size % 16 != 0:
            img_size = ((img_size // 16) + 1) * 16
            print(f"Adjusting image size to {img_size} (must be divisible by 16)")
        
        self.img_size = img_size
        
        # Image transforms with ImageNet normalization
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Mask transforms (nearest neighbor to preserve labels)
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.tile_pairs)
    
    def __getitem__(self, idx):
        pair = self.tile_pairs[idx]
        
        # Load image and mask (Hard Pair)
        img = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("L")
        
        # Apply transforms
        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_transform(mask)
        
        # Binarize mask (handles both 0/255 and 0/1 encodings)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return img_tensor, mask_tensor
