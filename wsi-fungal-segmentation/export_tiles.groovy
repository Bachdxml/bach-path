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
