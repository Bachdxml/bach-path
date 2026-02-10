// ==============================================
// QuPath 0.6.x – Tile & Mask Export (Patched)
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
// IMAGE BOUNDS
// =======================
def width = server.getWidth()
def height = server.getHeight()

// =======================
// TILE LOOP
// =======================
int tileCount = 0

for (int y = 0; y < height; y += TILE_SIZE) {
    for (int x = 0; x < width; x += TILE_SIZE) {

        int w = Math.min(TILE_SIZE, width - x)
        int h = Math.min(TILE_SIZE, height - y)

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

        annotations.each { PathAnnotationObject ann ->
            def roi = ann.getROI()

            // Use QuPath 0.6.x compatible bounds
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
        println "✅ Exported tile x=${x} y=${y}"
    }
}

println "✅ Finished exporting tiles with masks: " + tileCount
