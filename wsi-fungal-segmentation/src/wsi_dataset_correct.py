import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

DENSITY_LABELS = {"low": 0, "medium": 1, "high": 2, "negative": 3}
DENSITY_FOLDERS = list(DENSITY_LABELS.keys())  # folder names to scan

@dataclass
class TilePair:
    """Immutable record of a single image-mask pair with WSI identity and density class"""
    image_path: Path
    mask_path:  Path
    wsi_id:     str
    tile_id:    str
    density:    str = "unknown"   # low | medium | high | negative

# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

class WSIDatasetIndex:
    """
    Builds and validates an index of WSI tile pairs.

    Two formats supported:
    1. Density-stratified: <wsi_folder>/<density>/images/*.png, masks/*_mask.png
       where <density> is high, medium, low, or negative
    2. Flat (QuPath export): <wsi_folder>/images/*.png, <wsi_folder>/masks/*_mask.png
       All tiles assigned density=medium. Set flat_format=True to use this.
    """

    def __init__(self, export_root: Path, strict_mode: bool = True,
                 allow_size_mismatch: bool = False, flat_format: bool = False,
                 skip_validation: bool = False):
        self.export_root       = Path(export_root)
        self.strict_mode       = strict_mode
        self.allow_size_mismatch = allow_size_mismatch
        self.flat_format       = flat_format
        self.skip_validation   = skip_validation
        self.tile_pairs: List[TilePair] = []
        self.wsi_groups: Dict[str, List[TilePair]] = {}
        self.validation_report: Dict = {
            'total_wsis_found': 0,
            'valid_wsis':       0,
            'skipped_wsis':     [],
            'total_pairs':      0,
            'density_counts':   {d: 0 for d in DENSITY_FOLDERS},
            'issues':           [],
            'unpaired_skipped': 0,  # images with no matching mask (skipped, not fatal)
        }


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self) -> "WSIDatasetIndex":
        """Build and validate the full dataset index."""
        print("=" * 60)
        print("Building WSI Dataset Index...")
        print("=" * 60)

        if not self.export_root.exists():
            raise FileNotFoundError(f"Export root not found: {self.export_root}")

        wsi_folders = [d for d in self.export_root.iterdir() if d.is_dir()]
        self.validation_report["total_wsis_found"] = len(wsi_folders)

        for wsi_folder in wsi_folders:
            try:
                pairs = self._process_wsi_folder(wsi_folder)
                if pairs:
                    self.tile_pairs.extend(pairs)
                    self.wsi_groups[wsi_folder.name] = pairs
                    self.validation_report["valid_wsis"] += 1
                    for p in pairs:
                        self.validation_report['density_counts'][p.density] += 1
            except Exception as e:
                if self.strict_mode:
                    raise
                self.validation_report["skipped_wsis"].append(
                    {"wsi_id": wsi_folder.name, "reason": str(e)}
                )
                print(f"⚠️  Skipping {wsi_folder.name}: {e}")

        self.validation_report["total_pairs"] = len(self.tile_pairs)
        self._print_summary()

        if self.strict_mode and self.validation_report["issues"]:
            raise ValueError(
                f"Validation failed with {len(self.validation_report['issues'])} issues"
            )

        return self


    def get_train_val_split(self, val_ratio: float = 0.2,
                            random_seed: int = 42) -> Tuple[List[TilePair], List[TilePair]]:
        """Split dataset by WSI (no leakage)"""
        wsi_ids = list(sorted(self.wsi_groups.keys()))
        rng = np.random.default_rng(random_seed)
        wsi_ids = rng.permutation(wsi_ids).tolist()

        n_val = max(1, int(len(wsi_ids) * val_ratio))
        val_wsi_ids   = set(wsi_ids[:n_val])
        train_wsi_ids = set(wsi_ids[n_val:])

        train_pairs = [p for p in self.tile_pairs if p.wsi_id in train_wsi_ids]
        val_pairs   = [p for p in self.tile_pairs if p.wsi_id in val_wsi_ids]

        print(f"\nTrain/Val Split (by WSI):")
        print(f"  Train WSIs: {len(train_wsi_ids)} ({len(train_pairs)} tiles)")
        print(f"  Val WSIs:   {len(val_wsi_ids)} ({len(val_pairs)} tiles)")
        print(f"  Train WSI IDs: {sorted(train_wsi_ids)}")
        print(f"  Val WSI IDs:   {sorted(val_wsi_ids)}")

        return train_pairs, val_pairs

    def save_index(self, output_path: Path):
        """Save index to JSON"""
        index_data = {
            'export_root':       str(self.export_root),
            'strict_mode':       self.strict_mode,
            'validation_report': self.validation_report,
            'tile_pairs': [
                {
                    'image_path': str(p.image_path),
                    'mask_path':  str(p.mask_path),
                    'wsi_id':     p.wsi_id,
                    'tile_id':    p.tile_id,
                    'density':    p.density
                }
                for p in self.tile_pairs
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        print(f"Index saved to {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_wsi_folder(self, wsi_folder: Path) -> List[TilePair]:
        """Process one WSI folder — flat format or density-stratified."""
        if self.flat_format:
            return self._process_wsi_folder_flat(wsi_folder)
        return self._process_wsi_folder_density(wsi_folder)

    def _process_wsi_folder_flat(self, wsi_folder: Path) -> List[TilePair]:
        """Flat QuPath format: <wsi>/images/*.png, <wsi>/masks/*_mask.png. All density=medium."""
        wsi_id = wsi_folder.name
        img_dir = wsi_folder / "images"
        msk_dir = wsi_folder / "masks"
        if not img_dir.exists() or not msk_dir.exists():
            raise FileNotFoundError(
                f"Flat format requires {wsi_id}/images/ and {wsi_id}/masks/. "
                f"Found images={img_dir.exists()}, masks={msk_dir.exists()}")
        all_pairs = []
        image_files = list(img_dir.glob("*.[pP][nN][gG]"))
        if not image_files:
            raise ValueError(f"No PNG images in {wsi_id}/images/")
        for img_path in natsorted(image_files):
            mask_path = self._find_corresponding_mask(img_path, msk_dir)
            if mask_path is None:
                alt = msk_dir / f"{img_path.stem}.png"
                if alt.exists():
                    mask_path = alt
                else:
                    continue
            self._validate_tile_pair(img_path, mask_path, wsi_id)
            all_pairs.append(TilePair(
                image_path=img_path,
                mask_path=mask_path,
                wsi_id=wsi_id,
                tile_id=img_path.stem,
                density="medium"
            ))
            self.validation_report["density_counts"]["medium"] += 1
        print(f"✓ {wsi_id}: {len(all_pairs)} pairs (flat format, density=medium)")
        return all_pairs

    def _process_wsi_folder_density(self, wsi_folder: Path) -> List[TilePair]:
        """Process one WSI folder — iterates over density subfolders"""
        wsi_id = wsi_folder.name
        all_pairs = []

        found_any_density = False
        for density in DENSITY_FOLDERS:
            img_dir = wsi_folder / density / "images"
            msk_dir = wsi_folder / density / "masks"

            if not img_dir.exists():
                continue  # this density class not present for this WSI — fine

            if not msk_dir.exists():
                raise FileNotFoundError(
                    f"images/ exists but masks/ missing for {wsi_id}/{density}")

            image_files = list(img_dir.glob("*.[pP][nN][gG]"))
            if not image_files:
                continue

            found_any_density = True

            # Duplicate check within this density folder
            stems = [f.stem for f in image_files]
            if len(stems) != len(set(stems)):
                from collections import Counter
                duplicates = [s for s, n in Counter(stems).items() if n > 1]
                raise ValueError(
                    f"Duplicate files in {wsi_id}/{density}: {duplicates[:5]}")

            unpaired = []
            for img_path in natsorted(image_files):
                mask_path = self._find_corresponding_mask(img_path, msk_dir)
                if mask_path is None:
                    unpaired.append(img_path.name)
                    continue

                self._validate_tile_pair(img_path, mask_path, wsi_id)

                all_pairs.append(TilePair(
                    image_path=img_path,
                    mask_path=mask_path,
                    wsi_id=wsi_id,
                    tile_id=img_path.stem,
                    density=density
                ))

            if unpaired:
                # Exports often omit masks for some tiles; skip them instead of failing the build
                n = len(unpaired)
                self.validation_report['unpaired_skipped'] += n
                print(
                    f"\u26a0\ufe0f  {wsi_id}/{density}: skipped {n} image(s) without matching "
                    f"_mask.png (e.g. {unpaired[:3]})"
                )

        if not found_any_density:
            raise FileNotFoundError(
                f"No density subfolders (high/medium/low/negative) found in {wsi_id}. "
                f"Run classify_tiles.py first.")

        density_summary = {d: sum(1 for p in all_pairs if p.density == d)
                           for d in DENSITY_FOLDERS}
        summary_str = "  ".join(f"{d}={n}" for d, n in density_summary.items() if n > 0)
        print(f"\u2713 {wsi_id}: {len(all_pairs)} pairs  [{summary_str}]")
        return all_pairs

    def _find_corresponding_mask(self, image_path: Path, mask_dir: Path) -> Path:
        """Find mask: tile_x512_y1024.png -> tile_x512_y1024_mask.png"""
        expected = mask_dir / f"{image_path.stem}_mask.png"
        if expected.exists():
            return expected
        for p in mask_dir.iterdir():
            if p.stem.lower() == f"{image_path.stem}_mask":
                return p
        return None

    def _validate_tile_pair(self, image_path: Path, mask_path: Path, wsi_id: str):
        """Validate semantic properties of an image-mask pair"""
        if self.skip_validation:
            return
        try:
            img  = Image.open(image_path)
            mask = Image.open(mask_path)

            if img.size != mask.size:
                issue = (f"Shape mismatch in {wsi_id}/{image_path.name}: "
                         f"image={img.size}, mask={mask.size}")
                if self.allow_size_mismatch:
                    self.validation_report['issues'].append(issue)
                    print(f"\u26a0\ufe0f  {issue} (will be resized during training)")
                else:
                    raise ValueError(issue)

            if img.mode != 'RGB':
                raise ValueError(
                    f"Image not RGB in {wsi_id}/{image_path.name}: mode={img.mode}")

            if mask.mode not in ['L', '1']:
                raise ValueError(
                    f"Mask not grayscale in {wsi_id}/{mask_path.name}: mode={mask.mode}")

            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            if not (set(unique_values).issubset({0, 1}) or
                    set(unique_values).issubset({0, 255})):
                issue = (f"Unexpected mask values in {wsi_id}/{mask_path.name}: "
                         f"unique={unique_values}")
                self.validation_report['issues'].append(issue)
                if self.strict_mode:
                    raise ValueError(issue)
                else:
                    print(f"\u26a0\ufe0f  {issue}")

        except (ValueError, AssertionError):
            if self.strict_mode:
                raise
        except Exception as e:
            raise RuntimeError(f"Failed to validate {image_path.name}") from e

    def _print_summary(self):
        print("\n" + "="*60)
        print("Dataset Index Summary")
        print("="*60)
        print(f"WSIs found:        {self.validation_report['total_wsis_found']}")
        print(f"WSIs valid:        {self.validation_report['valid_wsis']}")
        print(f"WSIs skipped:      {len(self.validation_report['skipped_wsis'])}")
        print(f"Total tile pairs:  {self.validation_report['total_pairs']}")
        us = self.validation_report.get('unpaired_skipped', 0)
        if us:
            print(f"Unpaired skipped:  {us} (images with no matching mask)")
        print(f"Validation issues: {len(self.validation_report['issues'])}")
        print(f"\nDensity breakdown:")
        for d, n in self.validation_report['density_counts'].items():
            print(f"  {d:<12} {n} tiles")
        if self.wsi_groups:
            print(f"\nPairs per WSI:")
            for wsi_id, pairs in sorted(self.wsi_groups.items()):
                print(f"  {wsi_id}: {len(pairs)} pairs")
        print("="*60 + "\n")

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class WSI_Dataset(Dataset):
    """
    PyTorch Dataset that operates on validated TilePair records.
    Returns (image_tensor, mask_tensor, density_label) per sample.
    density_label is a long integer: low=0, medium=1, high=2, negative=3
    """

    def __init__(self, tile_pairs: List[TilePair], img_size: int = 512):
        self.tile_pairs = tile_pairs

        if img_size % 16 != 0:
            img_size = ((img_size // 16) + 1) * 16
            print(f"Adjusting image size to {img_size} (must be divisible by 16)")

        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.tile_pairs)

    def __getitem__(self, idx):
        pair = self.tile_pairs[idx]

        img  = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("L")

        img_tensor  = self.img_transform(img)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        if pair.density not in DENSITY_LABELS:
            raise ValueError(
                f"Unknown density '{pair.density}' for tile {pair.tile_id}")
        density_label = torch.tensor(
            DENSITY_LABELS[pair.density], dtype=torch.long)

        return img_tensor, mask_tensor, density_label



class AugmentedWSI_Dataset(WSI_Dataset):
    """
    Extends WSI_Dataset with joint image+mask augmentation for training.
    Returns (image_tensor, mask_tensor, density_label).
    """
    def __init__(self, tile_pairs, img_size=512, augment=True):
        super().__init__(tile_pairs, img_size=img_size)
        self.augment = augment
        self.jitter  = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

    def __getitem__(self, idx):
        pair = self.tile_pairs[idx]

        img  = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("L")

        img  = TF.resize(img,  [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            k    = random.randint(0, 3)
            img  = TF.rotate(img,  90 * k)
            mask = TF.rotate(mask, 90 * k,
                             interpolation=TF.InterpolationMode.NEAREST)
            img  = self.jitter(img)

        img_tensor  = TF.to_tensor(img)
        img_tensor  = TF.normalize(img_tensor,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        mask_tensor = TF.to_tensor(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        if pair.density not in DENSITY_LABELS:
            raise ValueError(
                f"Unknown density '{pair.density}' for tile {pair.tile_id}")
        density_label = torch.tensor(
            DENSITY_LABELS[pair.density], dtype=torch.long)

        return img_tensor, mask_tensor, density_label


def make_stratified_sampler(tile_pairs: list) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that gives each density class
    equal expected frequency in every batch, regardless of how many
    tiles exist per class.
    """
    if not tile_pairs:
        raise ValueError("Cannot create stratified sampler for an empty dataset.")

    labels = [DENSITY_LABELS.get(p.density, 0) for p in tile_pairs]
    class_counts = [0] * len(DENSITY_LABELS)
    for lbl in labels:
        class_counts[lbl] += 1

    # Weight per class = 1 / count (zero-safe)
    class_weights = [1.0 / max(c, 1) for c in class_counts]

    # Assign per-sample weight
    sample_weights = [class_weights[l] for l in labels]

    print("Stratified sampler class counts:")
    for name, idx in DENSITY_LABELS.items():
        print(f"  {name:<12} {class_counts[idx]} tiles  "
              f"(weight={class_weights[idx]:.4f})")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
