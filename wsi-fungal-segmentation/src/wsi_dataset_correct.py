@dataclass
class TilePair:
    """Immutable record of a single image-mask pair with WSI identity"""
    image_path: Path
    mask_path: Path
    wsi_id: str
    tile_id: str
    
    def __post_init__(self):
        """Validate that paths exist"""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {self.mask_path}")


class WSIDatasetIndex:
    """
    Builds and validates an immutable index of WSI tile pairs.
    This is the single source of truth for what data exists.
    """
    
    def __init__(self, export_root: Path, strict_mode: bool = True, 
                 allow_size_mismatch: bool = False):  # <-- NEW PARAMETER
        self.export_root = Path(export_root)
        self.strict_mode = strict_mode
        self.allow_size_mismatch = allow_size_mismatch  # <-- NEW
        self.tile_pairs: List[TilePair] = []
        self.wsi_groups: Dict[str, List[TilePair]] = {}
        self.validation_report: Dict = {
            'total_wsis_found': 0,
            'valid_wsis': 0,
            'skipped_wsis': [],
            'total_pairs': 0,
            'issues': []
        }
        
    def build_index(self) -> 'WSIDatasetIndex':
        """Build the dataset index with full validation"""
        print("="*60)
        print("Building WSI Dataset Index...")
        print("="*60)
        
        if not self.export_root.exists():
            raise FileNotFoundError(f"Export root not found: {self.export_root}")
        
        wsi_folders = [d for d in self.export_root.iterdir() if d.is_dir()]
        self.validation_report['total_wsis_found'] = len(wsi_folders)
        
        for wsi_folder in wsi_folders:
            try:
                pairs = self._process_wsi_folder(wsi_folder)
                if pairs:
                    self.tile_pairs.extend(pairs)
                    self.wsi_groups[wsi_folder.name] = pairs
                    self.validation_report['valid_wsis'] += 1
            except Exception as e:
                if self.strict_mode:
                    raise
                else:
                    self.validation_report['skipped_wsis'].append({
                        'wsi_id': wsi_folder.name,
                        'reason': str(e)
                    })
                    print(f"⚠️  Skipping {wsi_folder.name}: {e}")
        
        self.validation_report['total_pairs'] = len(self.tile_pairs)
        self._print_summary()
        
        if self.strict_mode and len(self.validation_report['issues']) > 0:
            raise ValueError(f"Validation failed with {len(self.validation_report['issues'])} issues")
        
        return self
    
    def _process_wsi_folder(self, wsi_folder: Path) -> List[TilePair]:
        """Process a single WSI folder and return validated tile pairs"""
        wsi_id = wsi_folder.name
        img_dir = wsi_folder / "images"
        msk_dir = wsi_folder / "masks"
        
        # Check directory structure
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing images directory in {wsi_id}")
        if not msk_dir.exists():
            raise FileNotFoundError(f"Missing masks directory in {wsi_id}")
        
        # Get all image files
        image_files = list(img_dir.glob("*.[pP][nN][gG]"))
        
        if len(image_files) == 0:
            raise ValueError(f"No PNG files found in {wsi_id}/images")
        
        # Check for duplicates
        if len(image_files) != len(set(image_files)):
            duplicates = [f for f in image_files if image_files.count(f) > 1]
            raise ValueError(f"Duplicate image files detected in {wsi_id}: {duplicates[:5]}")
        
        # Build pairs by explicit matching
        pairs = []
        unpaired_images = []
        
        for img_path in natsorted(image_files):
            mask_path = self._find_corresponding_mask(img_path, msk_dir)
            
            if mask_path is None:
                unpaired_images.append(img_path.name)
                continue
            
            # Validate this specific pair
            self._validate_tile_pair(img_path, mask_path, wsi_id)
            
            # Create immutable pair record
            tile_id = img_path.stem
            pair = TilePair(
                image_path=img_path,
                mask_path=mask_path,
                wsi_id=wsi_id,
                tile_id=tile_id
            )
            pairs.append(pair)
        
        # Report unpaired images
        if unpaired_images:
            issue = f"{wsi_id}: {len(unpaired_images)} images without masks"
            self.validation_report['issues'].append(issue)
            
            if self.strict_mode:
                raise ValueError(f"{issue}. First 5: {unpaired_images[:5]}")
            else:
                print(f"⚠️  {issue}")
        
        print(f"✓ {wsi_id}: {len(pairs)} valid pairs")
        return pairs
    
    def _find_corresponding_mask(self, image_path: Path, mask_dir: Path) -> Path:
        """Find mask using explicit naming contract: tile_001.png -> tile_001_mask.png"""
        expected_mask_name = f"{image_path.stem}_mask.png"
        mask_path = mask_dir / expected_mask_name
        
        if mask_path.exists():
            return mask_path
        
        # Try case-insensitive variants
        for suffix in ["_mask.png", "_mask.PNG", "_MASK.png", "_MASK.PNG"]:
            alt_mask_path = mask_dir / f"{image_path.stem}{suffix}"
            if alt_mask_path.exists():
                return alt_mask_path
        
        return None
    
    def _validate_tile_pair(self, image_path: Path, mask_path: Path, wsi_id: str):
        """Validate semantic properties of an image-mask pair"""
        try:
            img = Image.open(image_path)
            mask = Image.open(mask_path)
            
            # Check 1: Shape match (MODIFIED TO ALLOW MISMATCH)
            if img.size != mask.size:
                issue = (
                    f"Shape mismatch in {wsi_id}/{image_path.name}: "
                    f"image={img.size}, mask={mask.size}"
                )
                
                if self.allow_size_mismatch:
                    # Just warn, don't fail
                    self.validation_report['issues'].append(issue)
                    print(f"⚠️  {issue} (will be resized during training)")
                else:
                    # Fail if not allowing mismatches
                    raise ValueError(issue)
            
            # Check 2: Image is RGB
            if img.mode != 'RGB':
                raise ValueError(f"Image is not RGB in {wsi_id}/{image_path.name}: mode={img.mode}")
            
            # Check 3: Mask is grayscale
            if mask.mode not in ['L', '1']:
                raise ValueError(f"Mask is not grayscale in {wsi_id}/{mask_path.name}: mode={mask.mode}")
            
            # Check 4: Mask value range
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            
            if not (set(unique_values).issubset({0, 1}) or set(unique_values).issubset({0, 255})):
                issue = f"Unexpected mask values in {wsi_id}/{mask_path.name}: unique values = {unique_values}"
                self.validation_report['issues'].append(issue)
                
                if self.strict_mode:
                    raise ValueError(issue)
                else:
                    print(f"⚠️  {issue}")
            
        except Exception as e:
            if self.strict_mode and not self.allow_size_mismatch:
                raise
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("Dataset Index Summary")
        print("="*60)
        print(f"WSIs found:        {self.validation_report['total_wsis_found']}")
        print(f"WSIs valid:        {self.validation_report['valid_wsis']}")
        print(f"WSIs skipped:      {len(self.validation_report['skipped_wsis'])}")
        print(f"Total tile pairs:  {self.validation_report['total_pairs']}")
        print(f"Validation issues: {len(self.validation_report['issues'])}")
        
        if self.wsi_groups:
            print(f"\nPairs per WSI:")
            for wsi_id, pairs in sorted(self.wsi_groups.items()):
                print(f"  {wsi_id}: {len(pairs)} pairs")
        
        print("="*60 + "\n")
    
    def get_train_val_split(self, val_ratio: float = 0.2, random_seed: int = 42) -> Tuple[List[TilePair], List[TilePair]]:
        """Split dataset by WSI (no leakage!)"""
        import random
        random.seed(random_seed)
        
        wsi_ids = list(self.wsi_groups.keys())
        random.shuffle(wsi_ids)
        
        n_val = max(1, int(len(wsi_ids) * val_ratio))
        val_wsi_ids = set(wsi_ids[:n_val])
        train_wsi_ids = set(wsi_ids[n_val:])
        
        train_pairs = [p for p in self.tile_pairs if p.wsi_id in train_wsi_ids]
        val_pairs = [p for p in self.tile_pairs if p.wsi_id in val_wsi_ids]
        
        print(f"\nTrain/Val Split (by WSI):")
        print(f"  Train WSIs: {len(train_wsi_ids)} ({len(train_pairs)} tiles)")
        print(f"  Val WSIs:   {len(val_wsi_ids)} ({len(val_pairs)} tiles)")
        print(f"  Train WSI IDs: {sorted(train_wsi_ids)}")
        print(f"  Val WSI IDs:   {sorted(val_wsi_ids)}")
        
        return train_pairs, val_pairs
    
    def save_index(self, output_path: Path):
        """Save index to JSON for reproducibility"""
        index_data = {
            'export_root': str(self.export_root),
            'strict_mode': self.strict_mode,
            'validation_report': self.validation_report,
            'tile_pairs': [
                {
                    'image_path': str(p.image_path),
                    'mask_path': str(p.mask_path),
                    'wsi_id': p.wsi_id,
                    'tile_id': p.tile_id
                }
                for p in self.tile_pairs
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"Index saved to {output_path}")
