// ==============================================
// QuPath 0.6.x – Tile & Mask Export with Coverage CSV
// Exports tiles to unclassified/ folder with per-tile
// foreground coverage logged to tile_coverage.csv
// Run classify_tiles.py afterwards to sort into density folders
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

// -------------------------------------------------------
// MASTER OUTPUT DIRECTORY
// -------------------------------------------------------
String MASTER_OUTPUT_DIR = "PASTE_YOUR_MASTER_OUTPUT_PATH_HERE"

// Mask values
int BACKGROUND = 0
int FOREGROUND = 255

// Classification name for negative examples
String NEGATIVE_CLASS_NAME = "Negative"

// =======================
// VALIDATE MASTER DIR
// =======================
if (MASTER_OUTPUT_DIR == "PASTE_YOUR_MASTER_OUTPUT_PATH_HERE" || MASTER_OUTPUT_DIR.trim().isEmpty()) {
    println "❌ ERROR: You must set MASTER_OUTPUT_DIR before running this script."
    println "   Open the script and replace the placeholder with your actual output path."
    return
}

def masterDir = new File(MASTER_OUTPUT_DIR)
if (!masterDir.exists()) {
    boolean created = masterDir.mkdirs()
    if (!created) {
        println "❌ ERROR: Could not create master output directory:"
        println "   ${MASTER_OUTPUT_DIR}"
        println "   Check that the path is valid and you have write permissions."
        return
    }
    println "✅ Master output directory created: ${masterDir.getAbsolutePath()}"
} else {
    println "✅ Master output directory found: ${masterDir.getAbsolutePath()}"
}

// =======================
// COVERAGE FUNCTION
// =======================
double computeCoverage(BufferedImage mask) {
    def raster = mask.getRaster()
    int w = mask.getWidth()
    int h = mask.getHeight()
    int fg = 0
    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            if (raster.getSample(px, py, 0) > 127) fg++
        }
    }
    return fg / (double)(w * h)
}

// =======================
// SETUP
// =======================
def imageData = getCurrentImageData()
def server = imageData.getServer()
def annotations = getAnnotationObjects()

if (annotations.isEmpty()) {
    println "❌ No annotations found!"
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

// =======================
// PER-WSI OUTPUT DIR
// Created inside the master output directory
// =======================
def outDir = new File(masterDir, imageName)

if (outDir.exists()) {
    println "⚠️  WARNING: Export directory already exists for image: ${imageName}"
    println "   Path: ${outDir.getAbsolutePath()}"
    println "   Skipping this WSI to avoid overwriting existing data."
    println "   Delete or rename the existing folder if you want to re-export."
    return
}

def imgDir  = new File(outDir, "unclassified/images")
def maskDir = new File(outDir, "unclassified/masks")
imgDir.mkdirs()
maskDir.mkdirs()
println "✅ Export directory created: " + outDir.getAbsolutePath()

// CSV writer
def csvFile = new File(outDir, "tile_coverage.csv")
def csvWriter = csvFile.newWriter()
csvWriter.writeLine("filename,x,y,coverage,is_negative")

// =======================
// SPATIAL INDEXING
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
def width  = server.getWidth()
def height = server.getHeight()

// =======================
// TILE LOOP
// =======================
int positiveTileCount   = 0
int negativeTileCount   = 0
int skippedOutOfBounds  = 0
int skippedNoAnnotations = 0

for (int y = 0; y < height; y += TILE_SIZE) {
    for (int x = 0; x < width; x += TILE_SIZE) {

        // Skip tiles completely outside annotation bounds
        if (x + TILE_SIZE < overallBounds.minX || x > overallBounds.maxX ||
            y + TILE_SIZE < overallBounds.minY || y > overallBounds.maxY) {
            skippedOutOfBounds++
            continue
        }

        int w = Math.min(TILE_SIZE, width  - x)
        int h = Math.min(TILE_SIZE, height - y)

        // Get nearby annotations via spatial index
        int gridX = (int)(x / GRID_SIZE)
        int gridY = (int)(y / GRID_SIZE)
        def nearbyPositive = positiveSpatialIndex["${gridX}_${gridY}"] ?: []
        def nearbyNegative = negativeSpatialIndex["${gridX}_${gridY}"] ?: []

        if (nearbyPositive.isEmpty() && nearbyNegative.isEmpty()) {
            skippedNoAnnotations++
            continue
        }

        // Check for actual bounding box intersections
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
            // Negative tile: mask stays all background
            g.dispose()
        } else {
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
        // COMPUTE COVERAGE & SAVE
        // =======================
        double coverage = isNegativeTile ? 0.0 : computeCoverage(mask)
        String baseName = String.format("tile_x%d_y%d", x, y)

        File imgFile  = new File(imgDir,  baseName + ".png")
        File maskFile = new File(maskDir, baseName + "_mask.png")

        ImageIO.write(tileImage, "PNG", imgFile)
        ImageIO.write(mask,      "PNG", maskFile)

        csvWriter.writeLine("${baseName},${x},${y},${coverage},${isNegativeTile}")

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

// =======================
// FINALISE
// =======================
csvWriter.flush()
csvWriter.close()

println ""
println "=" * 50
println "✅ Finished exporting tiles for: ${imageName}"
println "   ${positiveTileCount} positive tiles (with foreground masks)"
println "   ${negativeTileCount} negative tiles (all-background masks)"
println "⏩ Skipped ${skippedOutOfBounds} tiles outside annotation bounds"
println "⏩ Skipped ${skippedNoAnnotations} tiles with no annotations"
println "📄 Coverage CSV: ${csvFile.getAbsolutePath()}"
println "📁 Output: ${outDir.getAbsolutePath()}"
println "=" * 50
println ""
println "➡️  Next step: run classify_tiles.py on the master output directory"
println "    to sort tiles into high / medium / low / negative folders"
println "=" * 50
