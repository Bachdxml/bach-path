# WSI Fungal Segmentation — Residual Attention UNet

Binary segmentation of fungal elements in whole slide images (WSI) using a Residual Attention UNet. Optimized for PAS/AB stained histopathology with sparse foreground detection. Built around QuPath-exported tile datasets with strict data integrity validation and WSI-level train/val splitting.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Architecture](https://img.shields.io/badge/architecture-ResAttUNet-orange.svg)

---

## Table of Contents

- [Quick Start](#quick-start)
- [QuPath Export](#qupath-export)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Quick Start

```bash
pip install -r requirements.txt
```

Edit the data path in `configs/default.yaml`:
```yaml
data:
  export_root: "/path/to/exports_ml"
```

```bash
python train.py
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
```

To run a different config without editing files:
```bash
python train.py --config configs/my_experiment.yaml
```

---

## QuPath Export

Export tiles and masks from QuPath using the Groovy script below. Place it in your QuPath scripts folder and run via **Scripts > export_tiles.groovy**.

```groovy
// ==============================================
// QuPath 0.6.x – Tile & Mask Export with NEGATIVES
// ==============================================

import qupath.lib.images.servers.ImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.objects.PathAnnotationObject

import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.Graphics2D
import java.awt.geom.AffineTransform
import java.io.File

// =======================
// USER SETTINGS
// =======================
int TILE_SIZE = 512
double DOWNSAMPLE = 1.0
String OUTPUT_DIR = "exports_ml"

// Mask values
int BACKGROUND = 0
int FOREGROUND = 255

// Classification name for negative examples
String NEGATIVE_CLASS_NAME = "Negative"

// =======================
// SETUP
// =======================
def imageData = getCurrentImageData()
def server = imageData.getServer()
def annotations = getAnnotationObjects()

if (annotations.isEmpty()) {
    print "❌ No annotations found!"
    return
}

// Separate positive and negative annotations
def positiveAnnotations = []
def negativeAnnotations = []

annotations.each { ann ->
    def pathClass = ann.getPathClass()
    if (pathClass != null && pathClass.getName() == NEGATIVE_CLASS_NAME) {
        negativeAnnotations << ann
    } else {
        positiveAnnotations << ann
    }
}

println "📊 Found ${positiveAnnotations.size()} positive annotations"
println "📊 Found ${negativeAnnotations.size()} negative annotations"

if (positiveAnnotations.isEmpty() && negativeAnnotations.isEmpty()) {
    println "❌ No valid annotations found!"
    return
}

// Safe way to get image filename
def fullPath = server.getPath()
def imageName = new File(fullPath).getName()

// Create output directories
def projectDir = getProject().getBaseDirectory()
def outDir = new File(projectDir, OUTPUT_DIR + "/" + imageName)
def imgDir = new File(outDir, "images")
def maskDir = new File(outDir, "masks")

if (outDir.exists()) {
    println "❌ Export directory already exists for image:"
    println outDir.getAbsolutePath()
    println "❌ Aborting to prevent overwriting."
    return
}

imgDir.mkdirs()
maskDir.mkdirs()
println "✅ Export directory created: " + outDir.getAbsolutePath()

// =======================
// SPATIAL INDEXING (BOTH POSITIVE AND NEGATIVE)
// =======================
println "🔍 Building spatial index..."

def GRID_SIZE = TILE_SIZE * 2
def positiveSpatialIndex = [:].withDefault { [] }
def negativeSpatialIndex = [:].withDefault { [] }

def overallBounds = null

// Index positive annotations
positiveAnnotations.each { ann ->
    def roi = ann.getROI()
    double roiX = roi.getBoundsX()
    double roiY = roi.getBoundsY()
    double roiW = roi.getBoundsWidth()
    double roiH = roi.getBoundsHeight()
    
    if (overallBounds == null) {
        overallBounds = [minX: roiX, minY: roiY, 
                        maxX: roiX + roiW, maxY: roiY + roiH]
    } else {
        overallBounds.minX = Math.min(overallBounds.minX, roiX)
        overallBounds.minY = Math.min(overallBounds.minY, roiY)
        overallBounds.maxX = Math.max(overallBounds.maxX, roiX + roiW)
        overallBounds.maxY = Math.max(overallBounds.maxY, roiY + roiH)
    }
    
    int minGridX = (int)(roiX / GRID_SIZE)
    int maxGridX = (int)((roiX + roiW) / GRID_SIZE)
    int minGridY = (int)(roiY / GRID_SIZE)
    int maxGridY = (int)((roiY + roiH) / GRID_SIZE)
    
    for (int gy = minGridY; gy <= maxGridY; gy++) {
        for (int gx = minGridX; gx <= maxGridX; gx++) {
            positiveSpatialIndex["${gx}_${gy}"] << ann
        }
    }
}

// Index negative annotations
negativeAnnotations.each { ann ->
    def roi = ann.getROI()
    double roiX = roi.getBoundsX()
    double roiY = roi.getBoundsY()
    double roiW = roi.getBoundsWidth()
    double roiH = roi.getBoundsHeight()
    
    if (overallBounds == null) {
        overallBounds = [minX: roiX, minY: roiY, 
                        maxX: roiX + roiW, maxY: roiY + roiH]
    } else {
        overallBounds.minX = Math.min(overallBounds.minX, roiX)
        overallBounds.minY = Math.min(overallBounds.minY, roiY)
        overallBounds.maxX = Math.max(overallBounds.maxX, roiX + roiW)
        overallBounds.maxY = Math.max(overallBounds.maxY, roiY + roiH)
    }
    
    int minGridX = (int)(roiX / GRID_SIZE)
    int maxGridX = (int)((roiX + roiW) / GRID_SIZE)
    int minGridY = (int)(roiY / GRID_SIZE)
    int maxGridY = (int)((roiY + roiH) / GRID_SIZE)
    
    for (int gy = minGridY; gy <= maxGridY; gy++) {
        for (int gx = minGridX; gx <= maxGridX; gx++) {
            negativeSpatialIndex["${gx}_${gy}"] << ann
        }
    }
}

println "✅ Spatial index built. Processing tiles..."

// =======================
// IMAGE BOUNDS
// =======================
def width = server.getWidth()
def height = server.getHeight()

// =======================
// TILE LOOP
// =======================
int positiveTileCount = 0
int negativeTileCount = 0
int skippedOutOfBounds = 0
int skippedNoAnnotations = 0

for (int y = 0; y < height; y += TILE_SIZE) {
    for (int x = 0; x < width; x += TILE_SIZE) {

        // Skip tiles completely outside annotation bounds
        if (x + TILE_SIZE < overallBounds.minX || x > overallBounds.maxX ||
            y + TILE_SIZE < overallBounds.minY || y > overallBounds.maxY) {
            skippedOutOfBounds++
            continue
        }

        int w = Math.min(TILE_SIZE, width - x)
        int h = Math.min(TILE_SIZE, height - y)

        // Get nearby annotations
        int gridX = (int)(x / GRID_SIZE)
        int gridY = (int)(y / GRID_SIZE)
        def nearbyPositive = positiveSpatialIndex["${gridX}_${gridY}"] ?: []
        def nearbyNegative = negativeSpatialIndex["${gridX}_${gridY}"] ?: []

        if (nearbyPositive.isEmpty() && nearbyNegative.isEmpty()) {
            skippedNoAnnotations++
            continue
        }

        // Check for actual intersections
        boolean hasPositiveIntersection = false
        boolean hasNegativeIntersection = false
        
        for (PathAnnotationObject ann : nearbyPositive) {
            def roi = ann.getROI()
            double roiX = roi.getBoundsX()
            double roiY = roi.getBoundsY()
            double roiW = roi.getBoundsWidth()
            double roiH = roi.getBoundsHeight()

            if (!(roiX + roiW < x || roiX > x + w ||
                  roiY + roiH < y || roiY > y + h)) {
                hasPositiveIntersection = true
                break
            }
        }
        
        if (!hasPositiveIntersection) {
            for (PathAnnotationObject ann : nearbyNegative) {
                def roi = ann.getROI()
                double roiX = roi.getBoundsX()
                double roiY = roi.getBoundsY()
                double roiW = roi.getBoundsWidth()
                double roiH = roi.getBoundsHeight()

                if (!(roiX + roiW < x || roiX > x + w ||
                      roiY + roiH < y || roiY > y + h)) {
                    hasNegativeIntersection = true
                    break
                }
            }
        }

        if (!hasPositiveIntersection && !hasNegativeIntersection) {
            skippedNoAnnotations++
            continue
        }

        // Positive tiles take precedence over negative tiles
        boolean isNegativeTile = !hasPositiveIntersection && hasNegativeIntersection

        // =======================
        // READ TILE IMAGE
        // =======================
        def region = RegionRequest.createInstance(
                server.getPath(),
                DOWNSAMPLE,
                x, y, w, h
        )

        BufferedImage tileImage = server.readRegion(region)
        if (tileImage == null) {
            println "⚠️ Skipping tile x=${x} y=${y}: image server returned null"
            continue
        }

        // =======================
        // CREATE MASK
        // =======================
        BufferedImage mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        Graphics2D g = mask.createGraphics()
        g.setColor(new Color(BACKGROUND, BACKGROUND, BACKGROUND))
        g.fillRect(0, 0, w, h)

        if (isNegativeTile) {
            // Negative tile: mask is all background (already filled above)
            g.dispose()
        } else {
            // Positive tile: fill in the positive annotations
            g.setColor(new Color(FOREGROUND, FOREGROUND, FOREGROUND))
            
            nearbyPositive.each { PathAnnotationObject ann ->
                def roi = ann.getROI()
                double roiX = roi.getBoundsX()
                double roiY = roi.getBoundsY()
                double roiW = roi.getBoundsWidth()
                double roiH = roi.getBoundsHeight()

                if (roiX + roiW < x || roiX > x + w ||
                    roiY + roiH < y || roiY > y + h) {
                    return
                }

                def shape = roi.getShape()
                AffineTransform transform = new AffineTransform()
                transform.translate(-x, -y)
                def tileShape = transform.createTransformedShape(shape)
                g.fill(tileShape)
            }
            
            g.dispose()
        }

        // =======================
        // SAVE FILES
        // =======================
        String prefix = isNegativeTile ? "neg_tile" : "tile"
        String baseName = String.format("${prefix}_x%d_y%d", x, y)

        File imgFile = new File(imgDir, baseName + ".png")
        ImageIO.write(tileImage, "PNG", imgFile)

        File maskFile = new File(maskDir, baseName + "_mask.png")
        ImageIO.write(mask, "PNG", maskFile)

        if (isNegativeTile) {
            negativeTileCount++
        } else {
            positiveTileCount++
        }
        
        int totalCount = positiveTileCount + negativeTileCount
        if (totalCount % 50 == 0) {
            println "📊 Progress: ${positiveTileCount} positive, ${negativeTileCount} negative tiles exported..."
        }
    }
}

println ""
println "=" * 50
println "✅ Finished exporting tiles:"
println "   ${positiveTileCount} positive tiles (with foreground masks)"
println "   ${negativeTileCount} negative tiles (all-background masks)"
println "⏩ Skipped ${skippedOutOfBounds} tiles outside annotation bounds"
println "⏩ Skipped ${skippedNoAnnotations} tiles with no annotations"
println "📁 Output: ${outDir.getAbsolutePath()}"
println "=" * 50
```

---

## Data Format

### Directory Layout

```
exports_ml/
    <wsi_id>/
        images/   tile_x0_y0.png
                  tile_x512_y0.png
                  ...
        masks/    tile_x0_y0_mask.png
                  tile_x512_y0_mask.png
                  ...
```

### Naming Contract

Pairing is done by filename stem — the mask name must be exactly `<tile_id>_mask.png`:

```
tile_x512_y512.png  →  tile_x512_y512_mask.png
```

The indexer will fail on any missing mask, duplicate tile, or naming mismatch.

### Mask Requirements

- Mode: grayscale (`L`)
- Values: `0` and `255` only (binarized to 0/1 at load time)
- Size: must match the corresponding image — size mismatches usually indicate a QuPath export error and should be fixed upstream rather than silenced with `allow_size_mismatch`

### Train/Val Split

Split is **by WSI**, not by tile. No WSI appears in both sets. With `val_ratio=0.2` and `random_seed=42` the split is deterministic. A `dataset_index.json` is written each run for reproducibility.

### Data Integrity Checks

The indexer validates every pair before training starts:

| Issue | Detection | Behavior |
|-------|-----------|----------|
| Missing mask | Filename lookup fails | Error (strict) / warning (non-strict) |
| Duplicate tiles | Stem collision check | Always errors |
| Image not RGB | Mode check | Always errors |
| Mask not grayscale | Mode check | Always errors |
| Wrong mask values | Unique value check | Error (strict) / warning (non-strict) |
| Size mismatch | Dimension check | Error by default |
| Case sensitivity | Case-insensitive fallback | Auto-resolved |

### Minimum Data Requirements

- 2+ WSIs for a train/val split
- 50–100+ tiles per WSI to avoid severe overfitting
- Multiple fungal morphologies represented (hyphae, yeast, pseudohyphae) for generalization

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Copy it to run experiments without touching code:

```yaml
data:
  export_root: "/path/to/exports_ml"
  img_size: 512        # must be divisible by 16
  val_ratio: 0.2
  random_seed: 42

loss:
  alpha: 0.7           # FN penalty — higher = better recall (good for sparse detection)
  beta: 0.3            # FP penalty
  gamma: 0.75          # focal exponent

training:
  epochs: 50
  early_stop_patience: 15
  checkpoint_path: "checkpoints/best_model.pth"
```

---

## Architecture

Residual Attention UNet — vanilla UNet with two additions:

**Residual blocks** replace plain double-conv blocks. The skip connection allows better gradient flow and lets the network learn complex fungal morphology (hyphae topology, yeast wall texture) without degradation.

**Attention gates** are applied to each skip connection before concatenation. They learn to suppress irrelevant background tissue and focus on fungal regions — critical when fungi occupy less than 1% of the tile.

```
Input (3, 512, 512)
    ↓ enc1–enc4 (residual blocks + maxpool)
Bottleneck (1024 ch)
    ↓ upconv + attention-gated skip + residual decode (×4)
Output (1, 512, 512)
```

~34M parameters vs ~31M for vanilla UNet.

**Loss — Focal Tversky** (`alpha=0.7, beta=0.3`): penalizes false negatives more than false positives. Appropriate for sparse fungal detection where missing a positive region matters more than a false alarm.

---

## Troubleshooting

**Size mismatch error**
```
ValueError: Size mismatch: image=(512,512) mask=(256,256)
```
The most common cause is `DOWNSAMPLE != 1.0` in the QuPath script. Fix the export. If you need to proceed temporarily: set `allow_size_mismatch: true` in `configs/default.yaml`.

---

**Only 1 WSI found, can't split**
```
Need at least 2 WSIs for train/val split
```
Export more slides. For quick testing only, set `val_ratio: 0.0` to skip validation entirely.

---

**Model predicts all zeros**
```python
# Diagnose first
print(preds_sigmoid.min(), preds_sigmoid.max(), preds_sigmoid.mean())
```
If max is near 0, the model hasn't learned yet. Try more epochs, a lower threshold (`> 0.3` instead of `> 0.5`), or verify your masks actually contain positive pixels.

---

**CUDA out of memory**

Reduce `batch_size` to 2 or 1, or reduce `img_size` to 256.

---

**Training restarts from scratch on re-run**

The model reinitializes on every run unless you load a checkpoint first. `evaluate.py` handles this automatically. To resume training manually:
```python
ckpt = torch.load("checkpoints/best_model.pth")
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```

---

**Unpaired images**
```
Warning: 12 images without masks
```
Check QuPath export. Every image needs an exact `_mask.png` counterpart. The indexer prints which files are missing pairs.

---

## Roadmap

- [ ] Multi-class segmentation for fungal morphology (yeast / narrow hyphae / broad hyphae)
- [ ] Transfer learning from binary → multi-class using PCR labels
- [ ] Attention map visualization
- [ ] Whole-slide inference pipeline (tile → stitch predictions)
- [ ] Multi-stain support (H&E, GMS)
- [ ] Cross-validation

---

## References

- UNet: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Attention UNet: [Oktay et al., 2018](https://arxiv.org/abs/1804.03999)
- Residual blocks: [He et al., 2016](https://arxiv.org/abs/1512.03385)
