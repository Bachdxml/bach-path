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


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TilePair:
    """Immutable record of a single image-mask pair with WSI identity."""
    image_path: Path
    mask_path:  Path
    wsi_id:     str
    tile_id:    str


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

class WSIDatasetIndex:
    """
    Walks an export directory, validates every image-mask pair, and builds
    a clean index that can be split into train/val without WSI leakage.

    Expected directory layout:
        export_root/
            <wsi_id>/
                images/   <tile_id>.png
                masks/    <tile_id>_mask.png
    """

    def __init__(self, export_root: Path, strict_mode: bool = True,
                 allow_size_mismatch: bool = False):
        self.export_root       = Path(export_root)
        self.strict_mode       = strict_mode
        self.allow_size_mismatch = allow_size_mismatch
        self.tile_pairs:  List[TilePair]           = []
        self.wsi_groups:  Dict[str, List[TilePair]] = {}
        self.validation_report: Dict = {
            "total_wsis_found": 0,
            "valid_wsis":       0,
            "skipped_wsis":     [],
            "total_pairs":      0,
            "issues":           [],
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

    def get_train_val_split(
        self, val_ratio: float = 0.2, random_seed: int = 42
    ) -> Tuple[List[TilePair], List[TilePair]]:
        """Split by WSI to prevent data leakage."""
        wsi_ids = sorted(self.wsi_groups.keys())
        rng = np.random.default_rng(random_seed)
        wsi_ids = rng.permutation(wsi_ids).tolist()

        n_val = max(1, int(len(wsi_ids) * val_ratio))
        val_wsi_ids   = set(wsi_ids[:n_val])
        train_wsi_ids = set(wsi_ids[n_val:])

        train_pairs = [p for p in self.tile_pairs if p.wsi_id in train_wsi_ids]
        val_pairs   = [p for p in self.tile_pairs if p.wsi_id in val_wsi_ids]

        print(f"\nTrain/Val Split (by WSI):")
        print(f"  Train: {len(train_wsi_ids)} WSIs  ({len(train_pairs)} tiles)")
        print(f"  Val:   {len(val_wsi_ids)} WSIs  ({len(val_pairs)} tiles)")

        return train_pairs, val_pairs

    def save_index(self, output_path: Path):
        """Serialize index to JSON for reproducibility."""
        data = {
            "export_root":       str(self.export_root),
            "strict_mode":       self.strict_mode,
            "validation_report": self.validation_report,
            "tile_pairs": [
                {
                    "image_path": str(p.image_path),
                    "mask_path":  str(p.mask_path),
                    "wsi_id":     p.wsi_id,
                    "tile_id":    p.tile_id,
                }
                for p in self.tile_pairs
            ],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Index saved to {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_wsi_folder(self, wsi_folder: Path) -> List[TilePair]:
        wsi_id  = wsi_folder.name
        img_dir = wsi_folder / "images"
        msk_dir = wsi_folder / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing images/ in {wsi_id}")
        if not msk_dir.exists():
            raise FileNotFoundError(f"Missing masks/ in {wsi_id}")

        image_files = list(img_dir.glob("*.[pP][nN][gG]"))
        if not image_files:
            raise ValueError(f"No PNG files in {wsi_id}/images")

        stems = [f.stem for f in image_files]
        seen, duplicates = set(), []
        for s in stems:
            if s in seen:
                duplicates.append(s)
            seen.add(s)
        if duplicates:
            raise ValueError(f"Duplicate tiles in {wsi_id}: {duplicates[:5]}")

        pairs, unpaired = [], []
        for img_path in natsorted(image_files):
            mask_path = self._find_mask(img_path, msk_dir)
            if mask_path is None:
                unpaired.append(img_path.name)
                continue
            self._validate_pair(img_path, mask_path, wsi_id)
            pairs.append(TilePair(
                image_path=img_path,
                mask_path=mask_path,
                wsi_id=wsi_id,
                tile_id=img_path.stem,
            ))

        if unpaired:
            issue = f"{wsi_id}: {len(unpaired)} images without masks"
            self.validation_report["issues"].append(issue)
            if self.strict_mode:
                raise ValueError(f"{issue}. First 5: {unpaired[:5]}")
            print(f"⚠️  {issue}")

        print(f"✓ {wsi_id}: {len(pairs)} valid pairs")
        return pairs

    def _find_mask(self, image_path: Path, mask_dir: Path):
        candidate = mask_dir / f"{image_path.stem}_mask.png"
        if candidate.exists():
            return candidate
        for p in mask_dir.iterdir():
            if p.stem.lower() == f"{image_path.stem}_mask":
                return p
        return None

    def _validate_pair(self, image_path: Path, mask_path: Path, wsi_id: str):
        try:
            img  = Image.open(image_path)
            mask = Image.open(mask_path)

            if img.size != mask.size:
                issue = (
                    f"Size mismatch in {wsi_id}/{image_path.name}: "
                    f"image={img.size} mask={mask.size}"
                )
                if self.allow_size_mismatch:
                    self.validation_report["issues"].append(issue)
                    print(f"⚠️  {issue} (will be resized during training)")
                else:
                    raise ValueError(issue)

            if img.mode != "RGB":
                raise ValueError(f"Image not RGB in {wsi_id}/{image_path.name}: {img.mode}")
            if mask.mode not in ("L", "1"):
                raise ValueError(f"Mask not grayscale in {wsi_id}/{mask_path.name}: {mask.mode}")

            unique = np.unique(np.array(mask))
            if not (set(unique).issubset({0, 1}) or set(unique).issubset({0, 255})):
                issue = f"Unexpected mask values in {wsi_id}/{mask_path.name}: {unique}"
                self.validation_report["issues"].append(issue)
                if self.strict_mode:
                    raise ValueError(issue)
                print(f"⚠️  {issue}")

        except (ValueError, AssertionError):
            if self.strict_mode:
                raise
        except Exception as e:
            raise RuntimeError(f"Failed to validate {image_path.name}") from e

    def _print_summary(self):
        r = self.validation_report
        print("\n" + "=" * 60)
        print("Dataset Index Summary")
        print("=" * 60)
        print(f"WSIs found:        {r['total_wsis_found']}")
        print(f"WSIs valid:        {r['valid_wsis']}")
        print(f"WSIs skipped:      {len(r['skipped_wsis'])}")
        print(f"Total tile pairs:  {r['total_pairs']}")
        print(f"Validation issues: {len(r['issues'])}")
        if self.wsi_groups:
            print("\nPairs per WSI:")
            for wsi_id, pairs in sorted(self.wsi_groups.items()):
                print(f"  {wsi_id}: {len(pairs)}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class WSI_Dataset(Dataset):
    """
    PyTorch Dataset built from validated TilePair records.
    Guaranteed correct image-mask pairing.
    """

    def __init__(self, tile_pairs: List[TilePair], img_size: int = 512):
        self.tile_pairs = tile_pairs

        if img_size % 16 != 0:
            img_size = ((img_size // 16) + 1) * 16
            print(f"Adjusting img_size to {img_size} (must be divisible by 16)")
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.tile_pairs)

    def __getitem__(self, idx):
        pair = self.tile_pairs[idx]
        img  = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("L")
        img_t  = self.img_transform(img)
        mask_t = self.mask_transform(mask)
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t


class AugmentedWSI_Dataset(WSI_Dataset):
    """
    Extends WSI_Dataset with joint image+mask augmentation for training.
    Augmentations are applied identically to both so labels stay aligned.
    """

    def __init__(self, tile_pairs: List[TilePair], img_size: int = 512,
                 augment: bool = True):
        super().__init__(tile_pairs, img_size=img_size)
        self.augment = augment
        self._jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

    def __getitem__(self, idx):
        pair = self.tile_pairs[idx]
        img  = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("L")

        img  = TF.resize(img,  [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            k = random.randint(0, 3)
            img  = TF.rotate(img,  90 * k)
            mask = TF.rotate(mask, 90 * k,
                             interpolation=TF.InterpolationMode.NEAREST)
            img = self._jitter(img)

        img_t  = TF.normalize(TF.to_tensor(img),
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        mask_t = (TF.to_tensor(mask) > 0.5).float()

        return img_t, mask_t
