const viewerContainer = document.getElementById("viewer-container");
const viewerOverlay = document.getElementById("tab-viewer");
const viewerOverlayBackdrop = document.getElementById("viewer-overlay-backdrop");
const viewerOverlayShell = document.getElementById("viewer-overlay-shell");
const viewerPrevSlideBtn = document.getElementById("viewer-prev-slide");
const viewerNextSlideBtn = document.getElementById("viewer-next-slide");
const viewerBack = document.getElementById("viewer-back");
const viewerEmpty = document.getElementById("viewer-empty");
const runInferenceBtn = document.getElementById("viewer-run-inference");
const inferenceStatus = document.getElementById("viewer-inference-status");
const btnViewerFitSlide = document.getElementById("viewer-fit-slide");
const viewerShowNegative = document.getElementById("viewer-show-negative");
const viewerInferenceThreshold = document.getElementById("viewer-inference-threshold");
const viewerInferenceThresholdValue = document.getElementById("viewer-inference-threshold-value");
const viewerOverlayOpacity = document.getElementById("viewer-overlay-opacity");
const viewerScaleWrap = document.getElementById("viewer-scale-wrap");
const viewerScaleBar = document.getElementById("viewer-scale-bar");
const viewerScaleLabel = document.getElementById("viewer-scale-label");
const viewerMetadataAside = document.getElementById("viewer-metadata-aside");
const viewerMetaContent = document.getElementById("viewer-meta-content");
const btnViewerSlideInfo = document.getElementById("btn-viewer-slide-info");
const btnViewerMetadataClose = document.getElementById("btn-viewer-metadata-close");
const btnViewerExportView = document.getElementById("viewer-export-view");
const btnViewerExportRegions = document.getElementById("viewer-export-regions");
const viewerTileReviewContent = document.getElementById("viewer-tile-review-content");
const viewerTileReviewStatus = document.getElementById("viewer-tile-review-status");
const btnViewerReviewPositive = document.getElementById("viewer-review-positive");
const btnViewerReviewNegative = document.getElementById("viewer-review-negative");
const btnViewerReviewIndeterminate = document.getElementById("viewer-review-indeterminate");
const viewerApp = window.BachPath || null;
const viewerSlidesApi = viewerApp?.services?.slidesApi || window.slidesApi;
const viewerInferenceModelChangedEvent =
  viewerApp?.constants?.events?.inferenceModelChanged || "inference-model-changed";

if (viewerShowNegative) {
  viewerShowNegative.checked = false;
  viewerShowNegative.disabled = false;
  viewerShowNegative.title = "Show model-negative tile outlines";
}

let viewer = null;
let currentSlideId = null;
let currentMpp = null;
let currentRunId = null;
let lastRegions = [];
let currentSlideWidth = null;
let currentSlideHeight = null;
let currentSlideLabel = "";
let currentReviewStatus = "unreviewed";
let viewerRequestSeq = 0;
let activeInferencePollToken = 0;
let viewerSlideOrder = [];
let selectedReviewRegionId = null;
let viewerDisplayRotation = 0;
const INFERENCE_THRESHOLD_KEY = "inferenceThresholdByModel";
/** Rotate tall whole-slide scans so the long axis fits the viewer width. */
const VIEWER_PORTRAIT_ASPECT_RATIO = 2;
/** Padding around the slide when fitting to the viewer (fraction of viewport). */
const VIEWER_FIT_PADDING = 0.08;

function getThresholdMap() {
  try {
    const raw = localStorage.getItem(INFERENCE_THRESHOLD_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveThresholdMap(map) {
  localStorage.setItem(INFERENCE_THRESHOLD_KEY, JSON.stringify(map));
}

function getSelectedModelId() {
  const getModelFromShell =
    viewerApp?.features?.shell?.getSelectedInferenceModel || window.getSelectedInferenceModel;
  return typeof getModelFromShell === "function" ? getModelFromShell() : null;
}

function getCurrentThreshold() {
  const sliderVal = parseInt(viewerInferenceThreshold?.value || "10", 10);
  return Math.max(0.01, Math.min(0.99, sliderVal / 100));
}

function syncThresholdLabel() {
  if (!viewerInferenceThresholdValue) return;
  viewerInferenceThresholdValue.textContent = getCurrentThreshold().toFixed(2);
}

function loadThresholdForModel(modelId) {
  if (!viewerInferenceThreshold) return;
  const map = getThresholdMap();
  const saved = modelId ? map[modelId] : null;
  const t = Number.isFinite(saved) ? saved : 0.1;
  viewerInferenceThreshold.value = String(Math.round(Math.max(0.01, Math.min(0.99, t)) * 100));
  syncThresholdLabel();
}

function isViewerOpen() {
  return !!viewerOverlay && !viewerOverlay.hidden;
}

function getViewerSlideOrder() {
  const getOrderedIds =
    viewerApp?.features?.gallery?.getOrderedSlideIds || window.galleryGetOrderedSlideIds;
  if (typeof getOrderedIds !== "function") {
    return currentSlideId ? [currentSlideId] : [];
  }
  const ids = getOrderedIds().filter((id) => Number.isFinite(id));
  if (currentSlideId && !ids.includes(currentSlideId)) {
    ids.unshift(currentSlideId);
  }
  return ids;
}

function updateViewerNavButtons() {
  const idx = viewerSlideOrder.indexOf(currentSlideId);
  const hasPrev = idx > 0;
  const hasNext = idx >= 0 && idx < viewerSlideOrder.length - 1;
  if (viewerPrevSlideBtn) viewerPrevSlideBtn.disabled = !hasPrev;
  if (viewerNextSlideBtn) viewerNextSlideBtn.disabled = !hasNext;
}

async function navigateViewer(delta) {
  if (!currentSlideId) return;
  viewerSlideOrder = getViewerSlideOrder();
  const idx = viewerSlideOrder.indexOf(currentSlideId);
  if (idx < 0) return;
  const nextId = viewerSlideOrder[idx + delta];
  if (nextId == null) return;
  await showViewer(nextId);
}

function closeViewer() {
  activeInferencePollToken += 1;
  if (viewer) {
    viewer.destroy();
    viewer = null;
  }
  currentSlideId = null;
  currentMpp = null;
  currentSlideWidth = null;
  currentSlideHeight = null;
  currentSlideLabel = "";
  currentReviewStatus = "unreviewed";
  selectedReviewRegionId = null;
  viewerDisplayRotation = 0;
  lastRegions = [];
  viewerContainer.innerHTML = "";
  viewerContainer.style.display = "none";
  viewerEmpty.style.display = "flex";
  currentRunId = null;
  if (viewerMetaContent) viewerMetaContent.innerHTML = "";
  showMetadataPanel(false);
  runInferenceBtn.disabled = false;
  setInferenceStatus("");
  renderTileReviewPanel([]);
  updateReviewStatusDisplay();
  if (viewerOverlay) viewerOverlay.hidden = true;
}

function setInferenceStatus(text) {
  if (inferenceStatus) inferenceStatus.textContent = text;
}

function formatInferenceResultLabel(result) {
  if (result === "positive") return "AI positive";
  if (result === "negative") return "AI negative";
  if (result === "needs_review") return "AI needs review";
  return "AI unchecked";
}

function escapeHtml(s) {
  if (s == null || s === "") return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

function filenameFromPath(p) {
  if (!p) return "";
  return String(p).split(/[/\\]/).pop() || "";
}

function getGallerySlide(slideId) {
  const getSlideById = viewerApp?.features?.gallery?.getSlideById || window.galleryGetSlideById;
  return typeof getSlideById === "function" ? getSlideById(slideId) : null;
}

function shouldDisplaySlideHorizontally(width, height) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0) {
    return false;
  }
  return height / width >= VIEWER_PORTRAIT_ASPECT_RATIO;
}

function getPreferredViewerRotation(width, height) {
  return shouldDisplaySlideHorizontally(width, height) ? 90 : 0;
}

function getMinZoomImageRatio(width, height) {
  if (!shouldDisplaySlideHorizontally(width, height)) {
    return 0.5;
  }
  const aspect = height / width;
  if (aspect >= 6) return 0.01;
  if (aspect >= 4) return 0.02;
  return 0.05;
}

function fitSlideToViewer(immediately = true) {
  if (!viewer?.viewport) return;
  const item = viewer.world?.getItemAt?.(0);
  if (!item) {
    viewer.viewport.goHome(immediately);
    return;
  }
  const bounds = item.getBounds(true);
  if (!bounds) {
    viewer.viewport.goHome(immediately);
    return;
  }
  const pad = VIEWER_FIT_PADDING;
  const padded = bounds.clone();
  padded.x -= bounds.width * pad;
  padded.y -= bounds.height * pad;
  padded.width *= 1 + 2 * pad;
  padded.height *= 1 + 2 * pad;
  viewer.viewport.fitBounds(padded, immediately);
}

function applyViewerDisplayOrientation() {
  if (!viewer) return;
  const dims = getSlideDimensions();
  viewerDisplayRotation = dims
    ? getPreferredViewerRotation(dims.width, dims.height)
    : 0;
  viewer.viewport.setRotation(viewerDisplayRotation, true);
  fitSlideToViewer(true);
}

function updateReviewStatusDisplay() {
  if (!viewerTileReviewStatus) return;
  const label =
    currentReviewStatus === "positive"
      ? "positive"
      : currentReviewStatus === "negative"
        ? "negative"
        : currentReviewStatus === "indeterminate"
          ? "needs review"
          : "unreviewed";
  viewerTileReviewStatus.textContent = `Decision: ${label}`;
}

function renderMetadataPanel(meta) {
  if (!viewerMetaContent) return;
  if (!meta) {
    viewerMetaContent.innerHTML = '<p class="viewer-meta-empty">Metadata not available.</p>';
    return;
  }
  const dim =
    meta.dimensions && meta.dimensions.length >= 2
      ? `${meta.dimensions[0].toLocaleString()} × ${meta.dimensions[1].toLocaleString()} px`
      : "—";
  const mppX =
    meta.mpp_x != null && Number.isFinite(meta.mpp_x) ? `${meta.mpp_x.toFixed(4)} µm/px` : "—";
  const mppY =
    meta.mpp_y != null && Number.isFinite(meta.mpp_y) ? `${meta.mpp_y.toFixed(4)} µm/px` : "—";
  const levels =
    meta.level_dimensions && meta.level_dimensions.length
      ? meta.level_dimensions.map((d) => `${d[0]}×${d[1]}`).join(", ")
      : "—";
  const viewerOrientation =
    meta.dimensions && meta.dimensions.length >= 2
      ? shouldDisplaySlideHorizontally(meta.dimensions[0], meta.dimensions[1])
        ? "Rotated horizontal (tall scan)"
        : "Native orientation"
      : "—";

  const props = meta.properties || {};
  const propKeys = Object.keys(props).sort();
  let propsHtml = "";
  if (propKeys.length) {
    const rows = propKeys
      .slice(0, 40)
      .map((k) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(props[k])}</dd>`)
      .join("");
    const more =
      propKeys.length > 40 ? `<p class="viewer-meta-empty">${propKeys.length - 40} more…</p>` : "";
    propsHtml = `<details class="viewer-meta-props"><summary>Vendor properties (${propKeys.length})</summary><dl class="viewer-meta-dl">${rows}</dl>${more}</details>`;
  }

  viewerMetaContent.innerHTML = `
    <dl class="viewer-meta-dl">
      <dt>Slide ID</dt><dd>${escapeHtml(meta.slide_id)}</dd>
      <dt>Dimensions (level 0)</dt><dd>${escapeHtml(dim)}</dd>
      <dt>Pyramid levels</dt><dd>${escapeHtml(meta.level_count)}</dd>
      <dt>Level sizes</dt><dd style="font-size:11px">${escapeHtml(levels)}</dd>
      <dt>MPP X / Y</dt><dd>${escapeHtml(mppX)} / ${escapeHtml(mppY)}</dd>
      <dt>Vendor</dt><dd>${escapeHtml(meta.vendor || "—")}</dd>
      <dt>Viewer</dt><dd>${escapeHtml(viewerOrientation)}</dd>
    </dl>
    ${propsHtml}
  `;
}

function showMetadataPanel(show) {
  if (viewerMetadataAside) viewerMetadataAside.hidden = !show;
}

function niceRoundMicrons(x) {
  if (x <= 0 || !Number.isFinite(x)) return 50;
  const exp = Math.floor(Math.log10(x));
  const base = 10 ** exp;
  const f = x / base;
  let nice = 1;
  if (f <= 1.5) nice = 1;
  else if (f <= 3.5) nice = 2;
  else if (f <= 7.5) nice = 5;
  else nice = 10;
  return nice * base;
}

function updateScaleBar() {
  if (!viewerScaleWrap || !viewerScaleBar || !viewerScaleLabel) return;
  if (!viewer || !currentMpp) {
    viewerScaleWrap.hidden = true;
    return;
  }
  viewerScaleWrap.hidden = false;
  try {
    const bounds = viewer.viewport.getBounds();
    const tl = viewer.viewport.viewportToImageCoordinates(bounds.getTopLeft());
    const br = viewer.viewport.viewportToImageCoordinates(bounds.getBottomRight());
    const widthPx = Math.abs(br.x - tl.x);
    const widthUm = widthPx * currentMpp;
    const targetUm = widthUm * 0.22;
    const barUm = niceRoundMicrons(targetUm);
    const barPx = barUm / currentMpp;
    const frac = Math.min(1, barPx / widthPx);
    viewerScaleBar.style.width = `${Math.round(frac * 100)}%`;
    viewerScaleLabel.textContent = `${barUm % 1 === 0 ? barUm : barUm.toFixed(1)} µm`;
  } catch {
    viewerScaleLabel.textContent = currentMpp ? `MPP ≈ ${currentMpp.toFixed(3)} µm/px` : "";
  }
}

function getOverlayOpacity() {
  const v = parseInt(viewerOverlayOpacity?.value || "70", 10);
  return Math.min(1, Math.max(0.1, v / 100));
}

function clearOverlays() {
  if (viewer) {
    viewer.clearOverlays();
  }
}

function normalizeRegionLabel(label) {
  if (label === "fungus_positive" || label === "positive") return "fungus_positive";
  if (label === "fungus_negative" || label === "negative") return "fungus_negative";
  return String(label || "");
}

function getSlideDimensions() {
  if (Number.isFinite(currentSlideWidth) && Number.isFinite(currentSlideHeight)) {
    return { width: currentSlideWidth, height: currentSlideHeight };
  }
  try {
    const item = viewer?.world?.getItemAt?.(0);
    if (item) {
      const size = item.getContentSize();
      if (size?.x > 0 && size?.y > 0) {
        return { width: size.x, height: size.y };
      }
    }
  } catch (_) {}
  return null;
}

function getNumericRegionField(region, ...keys) {
  for (const key of keys) {
    const value = region?.[key];
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function getRegionPayload(region) {
  const payload = region?.payload;
  if (!payload) return null;
  if (typeof payload === "string") {
    try {
      const parsed = JSON.parse(payload);
      return parsed && typeof parsed === "object" ? parsed : null;
    } catch (_) {
      return null;
    }
  }
  return typeof payload === "object" ? payload : null;
}

function clampRectToSlide(x, y, w, h, dims) {
  if (![x, y, w, h].every(Number.isFinite)) return null;
  const x1 = Math.max(0, Math.min(dims.width, x));
  const y1 = Math.max(0, Math.min(dims.height, y));
  const x2 = Math.max(0, Math.min(dims.width, x + w));
  const y2 = Math.max(0, Math.min(dims.height, y + h));
  if (x2 <= x1 || y2 <= y1) return null;
  return { x1, y1, x2, y2 };
}

function clampPointToRect(x, y, rect) {
  if (![x, y].every(Number.isFinite) || !rect) return null;
  return {
    x: Math.max(rect.x1, Math.min(rect.x2, x)),
    y: Math.max(rect.y1, Math.min(rect.y2, y)),
  };
}

function buildCentroidFallbackRect(cx, cy, dims, referenceRect = null) {
  if (![cx, cy].every(Number.isFinite)) return null;
  const refW = referenceRect ? referenceRect.x2 - referenceRect.x1 : 0;
  const refH = referenceRect ? referenceRect.y2 - referenceRect.y1 : 0;
  const longSide = Math.max(dims.width, dims.height);
  const fallbackW = Math.max(24, refW > 0 ? refW * 0.35 : longSide * 0.028);
  const fallbackH = Math.max(24, refH > 0 ? refH * 0.35 : longSide * 0.028);
  return clampRectToSlide(cx - fallbackW / 2, cy - fallbackH / 2, fallbackW, fallbackH, dims);
}

function getRegionContribution(region, dims) {
  const x = getNumericRegionField(region, "x", "left");
  const y = getNumericRegionField(region, "y", "top");
  const w = getNumericRegionField(region, "w", "width");
  const h = getNumericRegionField(region, "h", "height");
  const tileRect = clampRectToSlide(x, y, w, h, dims);

  const rawCx =
    getNumericRegionField(region, "cx", "center_x") ??
    (tileRect ? (tileRect.x1 + tileRect.x2) * 0.5 : null);
  const rawCy =
    getNumericRegionField(region, "cy", "center_y") ??
    (tileRect ? (tileRect.y1 + tileRect.y2) * 0.5 : null);

  const areaRect = tileRect || buildCentroidFallbackRect(rawCx, rawCy, dims, tileRect);
  if (!areaRect) return null;

  const peak =
    clampPointToRect(rawCx, rawCy, areaRect) || {
      x: (areaRect.x1 + areaRect.x2) * 0.5,
      y: (areaRect.y1 + areaRect.y2) * 0.5,
    };

  return {
    x1: areaRect.x1,
    y1: areaRect.y1,
    x2: areaRect.x2,
    y2: areaRect.y2,
    cx: peak.x,
    cy: peak.y,
    coverage: getNumericRegionField(getRegionPayload(region), "coverage"),
    score: Number.isFinite(region.score) ? Math.max(0, Math.min(1, region.score)) : 0.5,
  };
}

function getPredictionMask(region) {
  const mask = getRegionPayload(region)?.prediction_mask;
  if (!mask || mask.encoding !== "bitpack") return null;
  const width = Number(mask.width);
  const height = Number(mask.height);
  const data = typeof mask.data === "string" ? mask.data : null;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    return null;
  }
  if (!data) return null;
  return { width, height, data };
}

function decodePredictionMask(mask) {
  const total = mask.width * mask.height;
  const pixels = new Uint8Array(total);
  let raw = "";
  try {
    raw = atob(mask.data);
  } catch (_) {
    return pixels;
  }
  for (let i = 0; i < total; i += 1) {
    const byte = raw.charCodeAt(Math.floor(i / 8));
    if (!Number.isFinite(byte)) break;
    const bit = 7 - (i % 8);
    if (((byte >> bit) & 1) === 1) {
      pixels[i] = 1;
    }
  }
  return pixels;
}

const POSITIVE_OUTLINE_GAP_PX = 3;
const POSITIVE_OUTLINE_THICKNESS_PX = 2;
const POSITIVE_OUTLINE_RGB = [255, 24, 24];
const POSITIVE_OUTLINE_MIDPOINT_TOLERANCE_PX = 0.5;
const MAX_OUTLINE_CLUSTER_CANVAS_PX = 4096;

function computeChamferDistanceField(foreground, width, height) {
  const dist = new Float32Array(foreground.length);
  const INF = width + height + 1;
  for (let i = 0; i < foreground.length; i += 1) {
    dist[i] = foreground[i] ? 0 : INF;
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      if (x > 0) dist[i] = Math.min(dist[i], dist[i - 1] + 1);
      if (y > 0) dist[i] = Math.min(dist[i], dist[i - width] + 1);
    }
  }
  for (let y = height - 1; y >= 0; y -= 1) {
    for (let x = width - 1; x >= 0; x -= 1) {
      const i = y * width + x;
      if (x < width - 1) dist[i] = Math.min(dist[i], dist[i + 1] + 1);
      if (y < height - 1) dist[i] = Math.min(dist[i], dist[i + width] + 1);
    }
  }
  return dist;
}

function preparePositiveMaskLayer(positive) {
  const mask = getPredictionMask(positive.region);
  if (!mask) return null;
  const pixels = decodePredictionMask(mask);
  const tileW = Math.max(1, positive.x2 - positive.x1);
  const tileH = Math.max(1, positive.y2 - positive.y1);
  return {
    region: positive.region,
    x1: positive.x1,
    y1: positive.y1,
    x2: positive.x2,
    y2: positive.y2,
    tileW,
    tileH,
    maskW: mask.width,
    maskH: mask.height,
    pixels,
  };
}

function expandedRectsOverlap(a, b, margin) {
  return (
    a.x1 - margin < b.x2 + margin &&
    a.x2 + margin > b.x1 - margin &&
    a.y1 - margin < b.y2 + margin &&
    a.y2 + margin > b.y1 - margin
  );
}

function clusterPositiveMaskLayers(layers) {
  const margin = POSITIVE_OUTLINE_GAP_PX + POSITIVE_OUTLINE_THICKNESS_PX;
  const parent = layers.map((_, index) => index);
  const find = (index) => {
    if (parent[index] === index) return index;
    parent[index] = find(parent[index]);
    return parent[index];
  };
  const union = (left, right) => {
    parent[find(left)] = find(right);
  };
  for (let i = 0; i < layers.length; i += 1) {
    for (let j = i + 1; j < layers.length; j += 1) {
      if (expandedRectsOverlap(layers[i], layers[j], margin)) union(i, j);
    }
  }
  const groups = new Map();
  for (let i = 0; i < layers.length; i += 1) {
    const root = find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(layers[i]);
  }
  return [...groups.values()];
}

function dilateBinaryMask(pixels, width, height, radius) {
  const dilated = new Uint8Array(pixels.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!pixels[y * width + x]) continue;
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          dilated[ny * width + nx] = 1;
        }
      }
    }
  }
  return dilated;
}

function extractIsolatedExteriorOutline(
  pixels,
  width,
  height,
  gapPx = POSITIVE_OUTLINE_GAP_PX,
  thicknessPx = POSITIVE_OUTLINE_THICKNESS_PX
) {
  const clearance = dilateBinaryMask(pixels, width, height, gapPx);
  const outer = dilateBinaryMask(pixels, width, height, gapPx + thicknessPx);
  const outline = new Uint8Array(pixels.length);
  for (let i = 0; i < pixels.length; i += 1) {
    if (outer[i] && !clearance[i]) outline[i] = 1;
  }
  return outline;
}

function stampMaskIntoClusterGrid(grid, gridW, gridH, layer, labelValue) {
  const scaleX = layer.tileW / layer.maskW;
  const scaleY = layer.tileH / layer.maskH;
  const originX = layer.clusterOriginX;
  const originY = layer.clusterOriginY;
  for (let my = 0; my < layer.maskH; my += 1) {
    for (let mx = 0; mx < layer.maskW; mx += 1) {
      if (!layer.pixels[my * layer.maskW + mx]) continue;
      const startX = Math.floor(layer.x1 + mx * scaleX - originX);
      const startY = Math.floor(layer.y1 + my * scaleY - originY);
      const endX = Math.ceil(layer.x1 + (mx + 1) * scaleX - originX);
      const endY = Math.ceil(layer.y1 + (my + 1) * scaleY - originY);
      for (let gy = startY; gy < endY; gy += 1) {
        if (gy < 0 || gy >= gridH) continue;
        for (let gx = startX; gx < endX; gx += 1) {
          if (gx < 0 || gx >= gridW) continue;
          grid[gy * gridW + gx] = labelValue;
        }
      }
    }
  }
}

function computeClusterOutlineLayers(cluster) {
  const margin = POSITIVE_OUTLINE_GAP_PX + POSITIVE_OUTLINE_THICKNESS_PX;
  const originX = Math.floor(Math.min(...cluster.map((layer) => layer.x1)) - margin);
  const originY = Math.floor(Math.min(...cluster.map((layer) => layer.y1)) - margin);
  const maxX = Math.ceil(Math.max(...cluster.map((layer) => layer.x2)) + margin);
  const maxY = Math.ceil(Math.max(...cluster.map((layer) => layer.y2)) + margin);
  const gridW = Math.max(1, maxX - originX);
  const gridH = Math.max(1, maxY - originY);

  if (gridW > MAX_OUTLINE_CLUSTER_CANVAS_PX || gridH > MAX_OUTLINE_CLUSTER_CANVAS_PX) {
    return cluster.map((layer) => ({
      layer,
      outline: extractIsolatedExteriorOutline(layer.pixels, layer.maskW, layer.maskH),
    }));
  }

  for (const layer of cluster) {
    layer.clusterOriginX = originX;
    layer.clusterOriginY = originY;
  }

  const labelGrids = cluster.map(() => new Uint8Array(gridW * gridH));
  cluster.forEach((layer, index) => {
    stampMaskIntoClusterGrid(labelGrids[index], gridW, gridH, layer, 1);
  });

  const distanceFields = labelGrids.map((grid) => computeChamferDistanceField(grid, gridW, gridH));
  const clusterOutlines = labelGrids.map(() => new Uint8Array(gridW * gridH));
  const gap = POSITIVE_OUTLINE_GAP_PX;
  const thickness = POSITIVE_OUTLINE_THICKNESS_PX;
  const tolerance = POSITIVE_OUTLINE_MIDPOINT_TOLERANCE_PX;

  for (let gy = 0; gy < gridH; gy += 1) {
    for (let gx = 0; gx < gridW; gx += 1) {
      const p = gy * gridW + gx;
      const distances = distanceFields
        .map((field, index) => ({ index, distance: field[p] }))
        .filter((entry) => entry.distance < gridW + gridH)
        .sort((a, b) => a.distance - b.distance);
      if (!distances.length || distances[0].distance <= 0) continue;

      const ringCandidates = distances.filter(
        (entry) => entry.distance >= gap && entry.distance <= gap + thickness
      );
      if (!ringCandidates.length) continue;

      if (ringCandidates.length === 1) {
        clusterOutlines[ringCandidates[0].index][p] = 1;
        continue;
      }

      const first = ringCandidates[0];
      const second = ringCandidates[1];
      if (Math.abs(first.distance - second.distance) <= tolerance) {
        clusterOutlines[first.index][p] = 1;
        if (second.index !== first.index) clusterOutlines[second.index][p] = 1;
      } else {
        clusterOutlines[first.index][p] = 1;
      }
    }
  }

  return cluster.map((layer, index) => ({
    layer,
    outline: sampleClusterOutlineToMask(
      clusterOutlines[index],
      layer,
      originX,
      originY,
      gridW,
      gridH
    ),
  }));
}

function sampleClusterOutlineToMask(clusterOutline, layer, originX, originY, gridW, gridH) {
  const outline = new Uint8Array(layer.maskW * layer.maskH);
  const scaleX = layer.tileW / layer.maskW;
  const scaleY = layer.tileH / layer.maskH;
  for (let my = 0; my < layer.maskH; my += 1) {
    for (let mx = 0; mx < layer.maskW; mx += 1) {
      const gx = Math.round(layer.x1 + (mx + 0.5) * scaleX - originX);
      const gy = Math.round(layer.y1 + (my + 0.5) * scaleY - originY);
      if (gx < 0 || gy < 0 || gx >= gridW || gy >= gridH) continue;
      if (clusterOutline[gy * gridW + gx]) outline[my * layer.maskW + mx] = 1;
    }
  }
  return outline;
}

function computePositiveOutlineByRegion(positives) {
  const layers = positives.map(preparePositiveMaskLayer).filter(Boolean);
  const outlineByRegion = new Map();
  for (const cluster of clusterPositiveMaskLayers(layers)) {
    const results =
      cluster.length === 1
        ? [
            {
              layer: cluster[0],
              outline: extractIsolatedExteriorOutline(
                cluster[0].pixels,
                cluster[0].maskW,
                cluster[0].maskH
              ),
            },
          ]
        : computeClusterOutlineLayers(cluster);
    for (const { layer, outline } of results) {
      outlineByRegion.set(layer.region, outline);
    }
  }
  return outlineByRegion;
}

function buildPredictionMaskOverlay(region, outlinePixels = null, maskW = 0, maskH = 0) {
  const container = document.createElement("div");
  container.className = "region-overlay region-overlay--segmentation";
  container.style.opacity = String(getOverlayOpacity());

  const mask = getPredictionMask(region);
  if (!mask) return container;

  const resolvedMaskW = maskW || mask.width;
  const resolvedMaskH = maskH || mask.height;
  const outline =
    outlinePixels ||
    extractIsolatedExteriorOutline(decodePredictionMask(mask), resolvedMaskW, resolvedMaskH);
  const canvas = document.createElement("canvas");
  canvas.width = resolvedMaskW;
  canvas.height = resolvedMaskH;
  canvas.className = "region-overlay-mask";
  const ctx = canvas.getContext("2d");
  if (!ctx) return container;

  const imageData = ctx.createImageData(resolvedMaskW, resolvedMaskH);
  const [r, g, b] = POSITIVE_OUTLINE_RGB;
  for (let i = 0; i < outline.length; i += 1) {
    if (!outline[i]) continue;
    const j = i * 4;
    imageData.data[j] = r;
    imageData.data[j + 1] = g;
    imageData.data[j + 2] = b;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  container.appendChild(canvas);
  return container;
}

function buildReviewTile(region, index, dims) {
  const contribution = getRegionContribution(region, dims);
  if (!contribution) return null;
  const score = Number.isFinite(region.score) ? Math.max(0, Math.min(1, region.score)) : null;
  return {
    id: Number.isFinite(region.id) ? region.id : index + 1,
    label: `Tile ${String(index + 1).padStart(3, "0")}`,
    score,
    rect: contribution,
    coordinates: {
      x: Math.round(contribution.x1),
      y: Math.round(contribution.y1),
      w: Math.round(contribution.x2 - contribution.x1),
      h: Math.round(contribution.y2 - contribution.y1),
    },
  };
}

function jumpToReviewTile(tile) {
  if (!viewer || !tile?.rect) return;
  selectedReviewRegionId = tile.id;
  const rect = new OpenSeadragon.Rect(
    tile.rect.x1,
    tile.rect.y1,
    Math.max(1, tile.rect.x2 - tile.rect.x1),
    Math.max(1, tile.rect.y2 - tile.rect.y1)
  );
  viewer.viewport.fitBounds(viewer.viewport.imageToViewportRectangle(rect), false);
  renderTileReviewPanel(lastRegions);
}

function renderTileReviewPanel(regions) {
  if (!viewerTileReviewContent) return;
  updateReviewStatusDisplay();
  const dims = getSlideDimensions();
  if (!dims) {
    viewerTileReviewContent.innerHTML =
      '<p class="tile-review-empty">Slide dimensions are loading. Tiles will appear after the slide opens.</p>';
    return;
  }

  const tiles = (regions || [])
    .filter((r) => normalizeRegionLabel(r.label) === "fungus_positive")
    .map((region, index) => buildReviewTile(region, index, dims))
    .filter(Boolean)
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  viewerTileReviewContent.innerHTML = "";
  const slideLabel = document.createElement("p");
  slideLabel.className = "tile-review-slide-label";
  slideLabel.textContent = `Slide ${currentSlideId}${currentSlideLabel ? ` · ${currentSlideLabel}` : ""}`;
  viewerTileReviewContent.appendChild(slideLabel);

  if (!tiles.length) {
    const empty = document.createElement("p");
    empty.className = "tile-review-empty";
    empty.textContent = "No model-positive tiles for the selected run. Confirm the whole slide context before marking negative.";
    viewerTileReviewContent.appendChild(empty);
    return;
  }

  for (const tile of tiles) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tile-review-card";
    if (selectedReviewRegionId === tile.id) btn.classList.add("is-active");
    const scoreText = tile.score == null ? "n/a" : tile.score.toFixed(2);
    btn.innerHTML = `
      <div class="tile-review-card-title">
        <span>${escapeHtml(tile.label)}</span>
        <span>Score ${escapeHtml(scoreText)}</span>
      </div>
      <div class="tile-review-card-meta">
        Slide ${escapeHtml(currentSlideId)} · Level-0 x=${escapeHtml(tile.coordinates.x)}, y=${escapeHtml(tile.coordinates.y)}<br />
        Region ${escapeHtml(tile.coordinates.w)}×${escapeHtml(tile.coordinates.h)} px · Click to open in slide
      </div>
    `;
    btn.addEventListener("click", () => jumpToReviewTile(tile));
    viewerTileReviewContent.appendChild(btn);
  }
}

function addRegionOverlays(regions, showNegative = false) {
  if (!viewer) return;
  viewer.clearOverlays();
  const dims = getSlideDimensions();
  if (!dims) return;

  const positives = [];
  const negatives = [];
  for (const r of regions || []) {
    const label = normalizeRegionLabel(r.label);
    if (label !== "fungus_positive" && !(showNegative && label === "fungus_negative")) continue;
    const contribution = getRegionContribution(r, dims);
    if (!contribution) continue;
    if (label === "fungus_positive") positives.push({ ...contribution, region: r });
    else negatives.push(contribution);
  }

  const outlineByRegion = computePositiveOutlineByRegion(positives);

  for (const p of positives) {
    const mask = getPredictionMask(p.region);
    const outline = outlineByRegion.get(p.region);
    const segment = buildPredictionMaskOverlay(
      p.region,
      outline,
      mask?.width,
      mask?.height
    );
    const rect = new OpenSeadragon.Rect(
      p.x1,
      p.y1,
      Math.max(1, p.x2 - p.x1),
      Math.max(1, p.y2 - p.y1)
    );
    try {
      viewer.addOverlay({
        element: segment,
        location: viewer.viewport.imageToViewportRectangle(rect),
      });
    } catch (_) {}
  }

  for (const n of negatives) {
    const box = document.createElement("div");
    box.className = "region-overlay region-overlay--negative";
    const rect = new OpenSeadragon.Rect(
      n.x1,
      n.y1,
      Math.max(1, n.x2 - n.x1),
      Math.max(1, n.y2 - n.y1)
    );
    try {
      viewer.addOverlay({
        element: box,
        location: viewer.viewport.imageToViewportRectangle(rect),
      });
    } catch (_) {}
  }
}

async function loadLatestInferenceRun(slideId, requestId = null) {
  const seq = requestId ?? viewerRequestSeq;
  try {
    const { runs } = await viewerSlidesApi.getSlideInferenceRuns(slideId);
    if (seq !== viewerRequestSeq || currentSlideId !== slideId) {
      return;
    }
    const latestSucceeded = (runs || []).find((r) => r.status === "succeeded");
    if (!latestSucceeded) {
      currentRunId = null;
      lastRegions = [];
      clearOverlays();
      renderTileReviewPanel([]);
      return;
    }
    currentRunId = latestSucceeded.id;
    await loadRegionsForRun(currentRunId, slideId, seq);
  } catch (_) {
    if (seq !== viewerRequestSeq || currentSlideId !== slideId) {
      return;
    }
    currentRunId = null;
    lastRegions = [];
    clearOverlays();
    renderTileReviewPanel([]);
  }
}

async function loadRegionsForRun(runId, slideId = currentSlideId, requestId = viewerRequestSeq) {
  if (!runId) return;
  try {
    const { regions } = await viewerSlidesApi.getInferenceRegions(runId);
    if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
    lastRegions = regions || [];
    selectedReviewRegionId = null;
    const hasPositive = lastRegions.some(
      (r) => normalizeRegionLabel(r.label) === "fungus_positive"
    );
    if (!hasPositive) {
      window.appToast?.("No positive regions found for this run.", "info", 3500);
    }
    addRegionOverlays(lastRegions, false);
    renderTileReviewPanel(lastRegions);
  } catch (_) {
    if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
    lastRegions = [];
    clearOverlays();
    renderTileReviewPanel([]);
  }
}

function applyMppFromMeta(meta) {
  currentMpp = null;
  currentSlideWidth = null;
  currentSlideHeight = null;
  if (!meta) return;
  const mx = meta.mpp_x;
  const my = meta.mpp_y;
  if (mx != null && my != null) currentMpp = (mx + my) / 2;
  else if (mx != null) currentMpp = mx;
  else if (my != null) currentMpp = my;
  if (Array.isArray(meta.dimensions) && meta.dimensions.length >= 2) {
    currentSlideWidth = Number(meta.dimensions[0]) || null;
    currentSlideHeight = Number(meta.dimensions[1]) || null;
  }
}

function parseDeepZoomDescriptor(xmlText) {
  const doc = new DOMParser().parseFromString(xmlText, "application/xml");
  if (doc.querySelector("parsererror")) {
    throw new Error("Invalid DeepZoom descriptor");
  }

  const imageEl = doc.getElementsByTagName("Image")[0];
  const sizeEl = doc.getElementsByTagName("Size")[0];
  if (!imageEl || !sizeEl) {
    throw new Error("Incomplete DeepZoom descriptor");
  }

  const width = parseInt(sizeEl.getAttribute("Width") || "0", 10);
  const height = parseInt(sizeEl.getAttribute("Height") || "0", 10);
  const tileSize = parseInt(imageEl.getAttribute("TileSize") || "256", 10);
  const overlap = parseInt(imageEl.getAttribute("Overlap") || "0", 10);
  const format = (imageEl.getAttribute("Format") || "jpg").trim() || "jpg";

  if (!Number.isFinite(width) || width < 1 || !Number.isFinite(height) || height < 1) {
    throw new Error("Invalid DeepZoom dimensions");
  }

  return {
    width,
    height,
    tileSize: Number.isFinite(tileSize) && tileSize > 0 ? tileSize : 256,
    overlap: Number.isFinite(overlap) && overlap >= 0 ? overlap : 0,
    format,
  };
}

function createDeepZoomTileSource(slideId, descriptor) {
  const width = descriptor.width;
  const height = descriptor.height;
  const tileSize = descriptor.tileSize;
  const overlap = descriptor.overlap;
  const format = descriptor.format;
  const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));

  return {
    type: "custom",
    width,
    height,
    tileSize,
    tileOverlap: overlap,
    minLevel: 0,
    maxLevel,
    getLevelScale: (level) => 1 / 2 ** (maxLevel - level),
    getNumTiles: (level) => {
      const scale = 1 / 2 ** (maxLevel - level);
      const levelWidth = Math.max(1, Math.ceil(width * scale));
      const levelHeight = Math.max(1, Math.ceil(height * scale));
      return new OpenSeadragon.Point(
        Math.max(1, Math.ceil(levelWidth / tileSize)),
        Math.max(1, Math.ceil(levelHeight / tileSize))
      );
    },
    getTileUrl: (level, x, y) =>
      viewerSlidesApi.getDeepZoomTileUrl(slideId, level, x, y, format),
  };
}

async function showViewer(slideId) {
  const requestId = ++viewerRequestSeq;
  activeInferencePollToken += 1; // cancel any in-flight inference poll from another slide
  currentSlideId = slideId;
  const gallerySlide = getGallerySlide(slideId);
  currentSlideLabel = filenameFromPath(gallerySlide?.original_path);
  currentReviewStatus = gallerySlide?.review_status || "unreviewed";
  selectedReviewRegionId = null;
  viewerSlideOrder = getViewerSlideOrder();
  updateViewerNavButtons();
  if (viewerOverlay) viewerOverlay.hidden = false;
  currentMpp = null;

  viewerEmpty.style.display = "none";
  viewerContainer.innerHTML = "";
  viewerContainer.style.display = "block";
  clearOverlays();
  setInferenceStatus("");
  renderTileReviewPanel([]);
  if (viewerScaleWrap) viewerScaleWrap.hidden = true;

  let meta = null;
  try {
    meta = await viewerSlidesApi.getSlideMetadata(slideId);
    if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
    applyMppFromMeta(meta);
    renderMetadataPanel(meta);
    const hideMetaForTallSlide =
      meta.dimensions?.length >= 2 &&
      shouldDisplaySlideHorizontally(meta.dimensions[0], meta.dimensions[1]);
    showMetadataPanel(!hideMetaForTallSlide);
  } catch (err) {
    console.warn("Slide metadata:", err);
    renderMetadataPanel(null);
    showMetadataPanel(false);
  }

  if (viewer) {
    viewer.destroy();
    viewer = null;
  }

  try {
    const dziUrl = viewerSlidesApi.getDeepZoomDziUrl(slideId);
    let tileSource = null;

    try {
      const dziCheck = await fetch(dziUrl);
      if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
      if (dziCheck.ok) {
        const descriptor = parseDeepZoomDescriptor(await dziCheck.text());
        if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
        tileSource = createDeepZoomTileSource(slideId, descriptor);
      }
    } catch (_) {}

    if (!tileSource) {
      if (!meta) {
        meta = await viewerSlidesApi.getSlideMetadata(slideId);
        if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
        applyMppFromMeta(meta);
        renderMetadataPanel(meta);
      }
      const osdLevelDimensions = (meta.level_dimensions || []).slice().reverse();
      if (!osdLevelDimensions.length) {
        osdLevelDimensions.push([meta.dimensions[0], meta.dimensions[1]]);
      }
      const maxLevel = Math.max(0, osdLevelDimensions.length - 1);
      const maxWidth = Math.max(1, osdLevelDimensions[maxLevel][0] || 1);
      const osdToOpenSlideLevel = (osdLevel) => maxLevel - osdLevel;

      tileSource = {
        type: "custom",
        width: meta.dimensions[0],
        height: meta.dimensions[1],
        tileSize: 256,
        minLevel: 0,
        maxLevel,
        getLevelScale: (level) => {
          const w = osdLevelDimensions[level]?.[0] || maxWidth;
          return w / maxWidth;
        },
        getNumTiles: (level) => {
          const w = osdLevelDimensions[level]?.[0] || osdLevelDimensions[maxLevel][0];
          const h = osdLevelDimensions[level]?.[1] || osdLevelDimensions[maxLevel][1];
          return new OpenSeadragon.Point(
            Math.max(1, Math.ceil(w / 256)),
            Math.max(1, Math.ceil(h / 256))
          );
        },
        getTileUrl: (level, x, y) =>
          viewerSlidesApi.getTileUrl(slideId, osdToOpenSlideLevel(level), x, y),
      };
    }

    const dimsForOrientation =
      meta?.dimensions?.length >= 2
        ? { width: meta.dimensions[0], height: meta.dimensions[1] }
        : getSlideDimensions();
    const minZoomImageRatio = dimsForOrientation
      ? getMinZoomImageRatio(dimsForOrientation.width, dimsForOrientation.height)
      : 0.5;

    viewer = OpenSeadragon({
      element: viewerContainer,
      tileSources: tileSource,
      prefixUrl: "node_modules/openseadragon/build/openseadragon/images/",
      showNavigator: false,
      showNavigationControl: true,
      gestureSettingsMouse: {
        clickToZoom: false,
        dblClickToZoom: false,
      },
      constrainDuringPan: true,
      visibilityRatio: 1.0,
      wrapHorizontal: false,
      wrapVertical: false,
      minZoomImageRatio,
      immediateRender: true,
      homeFillsViewer: false,
    });

    viewer.addHandler("open", () => {
      if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
      applyViewerDisplayOrientation();
      updateScaleBar();
      loadLatestInferenceRun(slideId, requestId);
    });

    viewer.addHandler("animation", updateScaleBar);
    viewer.addHandler("resize", () => {
      updateScaleBar();
      if (viewerDisplayRotation) {
        fitSlideToViewer(true);
      }
    });
    viewer.addHandler("canvas-click", (event) => {
      if (!event.quick) return;
      const homeZoom = viewer.viewport.getHomeZoom();
      const currentZoom = viewer.viewport.getZoom();
      const isZoomedIn = currentZoom > homeZoom * 1.05;
      if (isZoomedIn) {
        fitSlideToViewer(true);
      } else {
        const targetPoint = viewer.viewport.pointFromPixel(event.position);
        viewer.viewport.zoomTo(homeZoom * 2.5, targetPoint, true);
      }
      event.preventDefaultAction = true;
    });
  } catch (err) {
    viewerContainer.innerHTML = `<p class="viewer-error">Failed to load slide: ${escapeHtml(err.message)}</p>`;
    console.error(err);
  }
}

btnViewerSlideInfo?.addEventListener("click", () => {
  if (!viewerMetadataAside) return;
  viewerMetadataAside.hidden = !viewerMetadataAside.hidden;
});

btnViewerMetadataClose?.addEventListener("click", () => {
  showMetadataPanel(false);
});

viewerShowNegative?.addEventListener("change", () => {
  addRegionOverlays(lastRegions, viewerShowNegative.checked);
});

viewerOverlayOpacity?.addEventListener("input", () => {
  addRegionOverlays(lastRegions, viewerShowNegative?.checked || false);
});

viewerInferenceThreshold?.addEventListener("input", () => {
  syncThresholdLabel();
});

viewerInferenceThreshold?.addEventListener("change", () => {
  syncThresholdLabel();
  const modelId = getSelectedModelId();
  if (!modelId) return;
  const map = getThresholdMap();
  map[modelId] = getCurrentThreshold();
  saveThresholdMap(map);
});

window.addEventListener(viewerInferenceModelChangedEvent, (e) => {
  const modelId = e?.detail?.modelId || getSelectedModelId();
  loadThresholdForModel(modelId);
});

async function exportViewerViewport() {
  if (!viewerContainer || !currentSlideId) return;
  const api = window.electronAPI;
  if (!api?.saveViewerCapture) {
    window.appToast?.("Export requires the desktop app.", "error");
    return;
  }
  const r = viewerContainer.getBoundingClientRect();
  const rect = {
    x: Math.round(r.x),
    y: Math.round(r.y),
    width: Math.round(r.width),
    height: Math.round(r.height),
  };
  const defaultFilename = `slide-${currentSlideId}-view.png`;
  try {
    const res = await api.saveViewerCapture(rect, defaultFilename);
    if (res?.canceled) return;
    if (res?.ok === false) {
      window.appToast?.(res.error === "invalid_rect" ? "Invalid viewport." : res.error || "Export failed.", "error");
      return;
    }
    if (res?.path) window.appToast?.("Saved image.", "success");
  } catch (e) {
    window.appToast?.(e?.message || "Export failed.", "error");
  }
}

async function setSlideReviewStatus(reviewStatus) {
  if (!currentSlideId) return;
  try {
    const response = await viewerSlidesApi.updateSlideReview(currentSlideId, reviewStatus);
    currentReviewStatus = response.review_status || reviewStatus;
    updateReviewStatusDisplay();
    const refreshGallery = viewerApp?.features?.gallery?.refresh || window.galleryRefresh;
    if (typeof refreshGallery === "function") refreshGallery();
    window.appToast?.("Slide review decision saved.", "success", 2200);
  } catch (err) {
    window.appToast?.(err.message || "Could not save review decision.", "error");
  }
}

function exportInferenceRegionsJson() {
  if (!currentSlideId || !lastRegions?.length) {
    window.appToast?.("No regions to export. Run inference first.", "info");
    return;
  }
  const payload = {
    slide_id: currentSlideId,
    run_id: currentRunId,
    exported_at: new Date().toISOString(),
    regions: lastRegions,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `slide-${currentSlideId}-regions.json`;
  a.click();
  URL.revokeObjectURL(url);
  window.appToast?.("Regions JSON downloaded.", "success");
}

async function handleRunInference() {
  if (!currentSlideId || runInferenceBtn.disabled) return;
  const runSlideId = currentSlideId;
  const pollToken = ++activeInferencePollToken;
  runInferenceBtn.disabled = true;
  const selectedModel = getSelectedModelId();
  const threshold = getCurrentThreshold();
  console.log('[DEBUG] Running inference with threshold:', threshold);
  setInferenceStatus(selectedModel ? `Starting... (${selectedModel})` : "Starting...");

  try {
    const run = await viewerSlidesApi.runInference(currentSlideId, selectedModel, threshold);
    const trackInferenceRuns =
      viewerApp?.features?.gallery?.trackInferenceRuns || window.galleryTrackInferenceRuns;
    const refreshGallery = viewerApp?.features?.gallery?.refresh || window.galleryRefresh;
    if (typeof trackInferenceRuns === "function") {
      trackInferenceRuns([run.id]);
    } else if (typeof refreshGallery === "function") {
      refreshGallery();
    }
    setInferenceStatus("Running...");

    const poll = async () => {
      if (pollToken !== activeInferencePollToken || runSlideId !== currentSlideId) {
        runInferenceBtn.disabled = false;
        return;
      }
      const r = await viewerSlidesApi.getInferenceRun(run.id);
      if (pollToken !== activeInferencePollToken || runSlideId !== currentSlideId) {
        runInferenceBtn.disabled = false;
        return;
      }
      if (r.status === "succeeded") {
        const resultLabel = formatInferenceResultLabel(r.inference_result);
        const positiveCount = r.summary?.fungus_positive ?? 0;
        setInferenceStatus(`Done: ${resultLabel} (${positiveCount} positive)`);
        runInferenceBtn.disabled = false;
        const refreshGallery = viewerApp?.features?.gallery?.refresh || window.galleryRefresh;
        if (typeof refreshGallery === "function") {
          refreshGallery();
        }
        await loadLatestInferenceRun(runSlideId, viewerRequestSeq);
        return;
      }
      if (r.status === "failed") {
        setInferenceStatus(`Failed: ${r.error_message || "Unknown error"}`);
        runInferenceBtn.disabled = false;
        return;
      }
      setTimeout(() => {
        void poll();
      }, 2000);
    };
    await poll();
  } catch (err) {
    setInferenceStatus(`Error: ${err.message}`);
    runInferenceBtn.disabled = false;
  }
}

window.showViewer = showViewer;
if (viewerApp?.registerFeature) {
  viewerApp.registerFeature("viewer", {
    open: showViewer,
    close: closeViewer,
  });
}

if (viewerBack) {
  viewerBack.addEventListener("click", () => {
    closeViewer();
  });
}

viewerOverlayBackdrop?.addEventListener("click", () => closeViewer());
viewerOverlay?.addEventListener("click", (e) => {
  if (e.target === viewerOverlay) closeViewer();
});
viewerOverlayShell?.addEventListener("click", (e) => {
  e.stopPropagation();
});
viewerPrevSlideBtn?.addEventListener("click", () => {
  void navigateViewer(-1);
});
viewerNextSlideBtn?.addEventListener("click", () => {
  void navigateViewer(1);
});

if (runInferenceBtn) {
  runInferenceBtn.addEventListener("click", handleRunInference);
}

btnViewerFitSlide?.addEventListener("click", () => {
  applyViewerDisplayOrientation();
  updateScaleBar();
});
btnViewerExportView?.addEventListener("click", () => exportViewerViewport());
btnViewerExportRegions?.addEventListener("click", () => exportInferenceRegionsJson());
btnViewerReviewPositive?.addEventListener("click", () => setSlideReviewStatus("positive"));
btnViewerReviewNegative?.addEventListener("click", () => setSlideReviewStatus("negative"));
btnViewerReviewIndeterminate?.addEventListener("click", () => setSlideReviewStatus("indeterminate"));
loadThresholdForModel(getSelectedModelId());

document.addEventListener("keydown", (e) => {
  const t = e.target;
  const typing =
    t &&
    (t.tagName === "INPUT" ||
      t.tagName === "TEXTAREA" ||
      t.tagName === "SELECT" ||
      t.isContentEditable);
  if (typing) return;
  if (!isViewerOpen()) return;
  if (e.key === "Escape") {
    e.preventDefault();
    closeViewer();
    return;
  }
  if (e.key === "r" || e.key === "R") {
    if (!e.ctrlKey && !e.metaKey && currentSlideId) {
      e.preventDefault();
      handleRunInference();
    }
  }
  if (e.key === "e" || e.key === "E") {
    if (!e.ctrlKey && !e.metaKey && currentSlideId) {
      e.preventDefault();
      exportViewerViewport();
    }
  }
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    void navigateViewer(-1);
  }
  if (e.key === "ArrowRight") {
    e.preventDefault();
    void navigateViewer(1);
  }
});
