// ==============================================
// QuPath 0.6.x – Tile & Mask Export (OPTIMIZED)
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
int TILE_SIZE = 512              // pixels
double DOWNSAMPLE = 1.0          // safe downsample
String OUTPUT_DIR = "exports_ml" // folder inside QuPath project

// Mask values
int BACKGROUND = 0
int FOREGROUND = 255             // white

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
// SPATIAL INDEXING
// =======================
println "🔍 Building spatial index for ${annotations.size()} annotations..."

// Build spatial index to quickly find relevant annotations per tile
def GRID_SIZE = TILE_SIZE * 2  // adjust this for performance tuning
def spatialIndex = [:].withDefault { [] }

// Also track overall bounds to skip empty regions
def overallBounds = null

annotations.each { ann ->
    def roi = ann.getROI()
    double roiX = roi.getBoundsX()
    double roiY = roi.getBoundsY()
    double roiW = roi.getBoundsWidth()
    double roiH = roi.getBoundsHeight()
    
    // Update overall bounds
    if (overallBounds == null) {
        overallBounds = [minX: roiX, minY: roiY, 
                        maxX: roiX + roiW, maxY: roiY + roiH]
    } else {
        overallBounds.minX = Math.min(overallBounds.minX, roiX)
        overallBounds.minY = Math.min(overallBounds.minY, roiY)
        overallBounds.maxX = Math.max(overallBounds.maxX, roiX + roiW)
        overallBounds.maxY = Math.max(overallBounds.maxY, roiY + roiH)
    }
    
    // Add annotation to all grid cells it overlaps
    int minGridX = (int)(roiX / GRID_SIZE)
    int maxGridX = (int)((roiX + roiW) / GRID_SIZE)
    int minGridY = (int)(roiY / GRID_SIZE)
    int maxGridY = (int)((roiY + roiH) / GRID_SIZE)
    
    for (int gy = minGridY; gy <= maxGridY; gy++) {
        for (int gx = minGridX; gx <= maxGridX; gx++) {
            spatialIndex["${gx}_${gy}"] << ann
        }
    }
}

println "✅ Spatial index built. Processing tiles in annotated regions only..."

// =======================
// IMAGE BOUNDS
// =======================
def width = server.getWidth()
def height = server.getHeight()

// =======================
// OPTIMIZED TILE LOOP
// =======================
int tileCount = 0
int skippedOutOfBounds = 0
int skippedNoAnnotations = 0

for (int y = 0; y < height; y += TILE_SIZE) {
    for (int x = 0; x < width; x += TILE_SIZE) {

        // OPTIMIZATION 1: Skip tiles completely outside annotation bounds
        if (x + TILE_SIZE < overallBounds.minX || x > overallBounds.maxX ||
            y + TILE_SIZE < overallBounds.minY || y > overallBounds.maxY) {
            skippedOutOfBounds++
            continue
        }

        int w = Math.min(TILE_SIZE, width - x)
        int h = Math.min(TILE_SIZE, height - y)

        // OPTIMIZATION 2: Get only nearby annotations using spatial index
        int gridX = (int)(x / GRID_SIZE)
        int gridY = (int)(y / GRID_SIZE)
        def nearbyAnnotations = spatialIndex["${gridX}_${gridY}"] ?: []

        if (nearbyAnnotations.isEmpty()) {
            skippedNoAnnotations++
            continue
        }

        // Check if any nearby annotations actually intersect this tile
        boolean hasIntersection = false
        for (PathAnnotationObject ann : nearbyAnnotations) {
            def roi = ann.getROI()
            double roiX = roi.getBoundsX()
            double roiY = roi.getBoundsY()
            double roiW = roi.getBoundsWidth()
            double roiH = roi.getBoundsHeight()

            if (!(roiX + roiW < x || roiX > x + w ||
                  roiY + roiH < y || roiY > y + h)) {
                hasIntersection = true
                break
            }
        }

        if (!hasIntersection) {
            skippedNoAnnotations++
            continue
        }

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
        g.setColor(new Color(FOREGROUND, FOREGROUND, FOREGROUND))

        boolean hasAnnotations = false

        // Only iterate through nearby annotations (not all annotations!)
        nearbyAnnotations.each { PathAnnotationObject ann ->
            def roi = ann.getROI()

            double roiX = roi.getBoundsX()
            double roiY = roi.getBoundsY()
            double roiW = roi.getBoundsWidth()
            double roiH = roi.getBoundsHeight()

            // Check intersection with current tile
            if (roiX + roiW < x || roiX > x + w ||
                roiY + roiH < y || roiY > y + h) {
                return
            }

            hasAnnotations = true
            def shape = roi.getShape()
            AffineTransform transform = new AffineTransform()
            transform.translate(-x, -y)
            def tileShape = transform.createTransformedShape(shape)
            g.fill(tileShape)
        }

        g.dispose()

        if (!hasAnnotations) {
            println "⏩ Skipping tile x=${x} y=${y}: no annotation pixels"
            skippedNoAnnotations++
            continue
        }

        // =======================
        // SAVE FILES
        // =======================
        String baseName = String.format("tile_x%d_y%d", x, y)

        File imgFile = new File(imgDir, baseName + ".png")
        ImageIO.write(tileImage, "PNG", imgFile)

        File maskFile = new File(maskDir, baseName + "_mask.png")
        ImageIO.write(mask, "PNG", maskFile)

        tileCount++
        if (tileCount % 50 == 0) {
            println "📊 Progress: ${tileCount} tiles exported..."
        }
    }
}

println ""
println "=" * 50
println "✅ Finished exporting tiles with masks: ${tileCount}"
println "⏩ Skipped ${skippedOutOfBounds} tiles outside annotation bounds"
println "⏩ Skipped ${skippedNoAnnotations} tiles with no annotations"
println "📁 Output: ${outDir.getAbsolutePath()}"
println "=" * 50
