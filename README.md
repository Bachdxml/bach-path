# Tile-level fungus classifier pipeline

End-to-end pipeline that:

1. Reads 512×512 tiles exported from whole-slide images
2. Reads QuPath annotation GeoJSON per slide
3. Generates tile-level labels (fungus vs not) by polygon overlap
4. Trains a tile classifier using **Phikon-v2** as a frozen encoder + small MLP head
5. Generates a heatmap overlay image per slide from tile probabilities

First version is focused on correctness, leakage-safe splits (by slide), and usable heatmaps.

---

## Project structure

```
project/
  tiles_512/               # input: per-slide tile folders
  qupath_geojson/          # input: one GeoJSON per slide
  outputs/
    labels.csv
    models/
      phikon_head_best.pt
    heatmaps/
      <SLIDE_ID>_heatmap.png
  src/
    make_labels.py
    train.py
    make_heatmap.py
    utils.py
  requirements.txt
  README.md
```

---

## Environment setup

- **Python 3.10+**
- Create and use a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Input assumptions

- **Tiles**: already exported under `tiles_512/<SLIDE_ID>/*.png` (or `.jpg`/`.tif`).  
  Filenames must include top-left coordinates in slide pixel space, e.g.  
  `SLIDE_001_x12345_y67890.png`.
- **QuPath GeoJSON**: one file per slide at `qupath_geojson/<SLIDE_ID>.geojson`.  
  Fungus class name in QuPath is configurable (default `"fungus"` / `"Fungus"`).

---

## 1. Label generation

Generates `outputs/labels.csv` with columns: `tile_path`, `slide_id`, `x`, `y`, `label` (0/1).  
A tile is positive if its overlap with the target class polygons ≥ `--pos_overlap_frac` (default 0.02).

```bash
python src/make_labels.py --tiles_root tiles_512 --geojson_root qupath_geojson --out outputs/labels.csv --fungus_class Fungus
```

Configurable: `--tile_size` (512), `--pos_overlap_frac` (0.02), `--fungus_class`, paths.

---

## 2. Training

Train Phikon-v2 (frozen) + MLP head. Split is **by slide** (80% train / 20% val) to avoid leakage.  
Saves best checkpoint when validation AUC improves.

```bash
python src/train.py --labels_csv outputs/labels.csv --out_dir outputs/models
```

Optional: `--val_frac`, `--batch_size`, `--epochs`, `--lr`, `--random_state`.

---

## 3. Heatmap generation

Builds a per-slide heatmap: background = downscaled tile mosaic; overlay = tile probabilities.

```bash
python src/make_heatmap.py --labels_csv outputs/labels.csv --ckpt outputs/models/phikon_head_best.pt --out_dir outputs/heatmaps --slide_id SLIDE_001
```

If `--slide_id` is omitted, the first slide in the CSV is used.  
Optional: `--tile_size`, `--downscale` (tile size in mosaic), `--alpha`, `--stride` (512 or 256 for denser heatmaps if tiles were exported at 256 stride).

---

## Example commands (full run)

```bash
# 1. Generate labels
python src/make_labels.py --tiles_root tiles_512 --geojson_root qupath_geojson --out outputs/labels.csv --fungus_class Fungus

# 2. Train
python src/train.py --labels_csv outputs/labels.csv --out_dir outputs/models

# 3. Heatmap for one slide
python src/make_heatmap.py --labels_csv outputs/labels.csv --ckpt outputs/models/phikon_head_best.pt --out_dir outputs/heatmaps --slide_id SLIDE_001
```

---

## Common failure modes

| Issue | Cause | Fix |
|------|--------|-----|
| **x/y parsing error** | Tile filename doesn’t match `..._x12345_y67890.png` | Use the expected naming; implement a custom parser in `utils.parse_tile_xy` if your convention differs. |
| **GeoJSON class name mismatch** | QuPath uses a different class name (e.g. `"Fungus"` vs `"fungus"`) | Pass `--fungus_class "Fungus"` (or the exact string from QuPath). Script compares case-insensitively. |
| **Missing GeoJSON for some slides** | No `qupath_geojson/<SLIDE_ID>.geojson` | Those slides get all tiles labeled 0 (no fungus). Add GeoJSON or ignore those slides. |
| **All labels 0 or all 1** | No polygons for target class, or overlap threshold too high/low | Check GeoJSON and `--pos_overlap_frac`; ensure at least one slide has the target class. |
| **CUDA out of memory** | Batch size too large | Use `--batch_size 16` (or smaller) in `train.py`. |

---

## Optional features

- **Stride for heatmaps**: Use `--stride 256` when tiles were exported at 256-pixel stride for a denser, smoother heatmap.
- **Hard negative mining**: Stub in `train.py` (`collect_hard_negatives_stub`) for collecting top false positives on negative slides; can be wired in for curriculum or review.
