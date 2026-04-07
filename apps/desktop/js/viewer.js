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
const viewerRunSelect = document.getElementById("viewer-run-select");
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

if (viewerShowNegative) {
  viewerShowNegative.checked = false;
  viewerShowNegative.disabled = true;
  viewerShowNegative.title = "Negative overlays are disabled";
}

let viewer = null;
let currentSlideId = null;
let currentMpp = null;
let currentRunId = null;
let lastRegions = [];
let currentSlideWidth = null;
let currentSlideHeight = null;
let viewerRequestSeq = 0;
let activeInferencePollToken = 0;
let viewerSlideOrder = [];
const INFERENCE_THRESHOLD_KEY = "inferenceThresholdByModel";

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
  return typeof window.getSelectedInferenceModel === "function"
    ? window.getSelectedInferenceModel()
    : null;
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
  if (typeof window.galleryGetOrderedSlideIds !== "function") {
    return currentSlideId ? [currentSlideId] : [];
  }
  const ids = window.galleryGetOrderedSlideIds().filter((id) => Number.isFinite(id));
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
  lastRegions = [];
  viewerContainer.innerHTML = "";
  viewerContainer.style.display = "none";
  viewerEmpty.style.display = "flex";
  if (viewerRunSelect) {
    viewerRunSelect.innerHTML = "";
    viewerRunSelect.disabled = true;
  }
  if (viewerMetaContent) viewerMetaContent.innerHTML = "";
  showMetadataPanel(false);
  runInferenceBtn.disabled = false;
  setInferenceStatus("");
  if (viewerOverlay) viewerOverlay.hidden = true;
}

function setInferenceStatus(text) {
  if (inferenceStatus) inferenceStatus.textContent = text;
}

function escapeHtml(s) {
  if (s == null || s === "") return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
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

function addRegionOverlays(regions, showNegative = false) {
  if (!viewer) return;
  viewer.clearOverlays();
  const dims = getSlideDimensions();
  if (!dims) return;

  const positives = [];
  for (const r of regions || []) {
    const label = normalizeRegionLabel(r.label);
    if (label !== "fungus_positive") continue;
    const x = Number.isFinite(r.x) ? r.x : 0;
    const y = Number.isFinite(r.y) ? r.y : 0;
    const w = Math.max(1, Number.isFinite(r.w) ? r.w : 1);
    const h = Math.max(1, Number.isFinite(r.h) ? r.h : 1);
    const x1 = Math.max(0, Math.min(dims.width, x));
    const y1 = Math.max(0, Math.min(dims.height, y));
    const x2 = Math.max(0, Math.min(dims.width, x + w));
    const y2 = Math.max(0, Math.min(dims.height, y + h));
    if (x2 <= x1 || y2 <= y1) continue;
    positives.push({
      x1,
      y1,
      x2,
      y2,
      score: Number.isFinite(r.score) ? Math.max(0, Math.min(1, r.score)) : 0.5,
    });
  }
  if (!positives.length) return;

  // Build a full-slide heatmap canvas from positive detections.
  const longSide = Math.max(dims.width, dims.height);
  const scale = longSide > 1200 ? 1200 / longSide : 1;
  const canvasW = Math.max(64, Math.round(dims.width * scale));
  const canvasH = Math.max(64, Math.round(dims.height * scale));
  const heatCanvas = document.createElement("canvas");
  heatCanvas.width = canvasW;
  heatCanvas.height = canvasH;
  heatCanvas.style.width = "100%";
  heatCanvas.style.height = "100%";
  heatCanvas.style.display = "block";
  heatCanvas.style.pointerEvents = "none";

  const ctx = heatCanvas.getContext("2d");
  if (!ctx) return;
  const opacity = getOverlayOpacity();

  for (const p of positives) {
    const cx = ((p.x1 + p.x2) * 0.5 * canvasW) / dims.width;
    const cy = ((p.y1 + p.y2) * 0.5 * canvasH) / dims.height;
    const areaPx = (p.x2 - p.x1) * (p.y2 - p.y1);
    const radiusImagePx = Math.max(18, Math.min(180, Math.sqrt(areaPx) * 0.33));
    const radius = (radiusImagePx * Math.max(canvasW / dims.width, canvasH / dims.height));
    const peak = Math.max(0.08, Math.min(0.9, opacity * (0.22 + p.score * 0.95)));
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    g.addColorStop(0, `rgba(255, 48, 48, ${peak.toFixed(3)})`);
    g.addColorStop(0.45, `rgba(255, 120, 0, ${(peak * 0.62).toFixed(3)})`);
    g.addColorStop(1, "rgba(255, 0, 0, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
  }

  const container = document.createElement("div");
  container.className = "region-overlay region-heatmap-overlay";
  container.style.width = "100%";
  container.style.height = "100%";
  container.style.pointerEvents = "none";
  container.style.overflow = "hidden";
  container.style.mixBlendMode = "multiply";
  container.appendChild(heatCanvas);

  const imageRect = new OpenSeadragon.Rect(0, 0, dims.width, dims.height);
  const viewportRect = viewer.viewport.imageToViewportRectangle(imageRect);
  try {
    viewer.addOverlay({ element: container, location: viewportRect });
  } catch (_) {}
}

async function populateRunSelector(slideId, requestId = null) {
  if (!viewerRunSelect) return;
  viewerRunSelect.innerHTML = "";
  viewerRunSelect.disabled = true;
  try {
    const { runs } = await window.slidesApi.getSlideInferenceRuns(slideId);
    if (
      (requestId != null && requestId !== viewerRequestSeq) ||
      currentSlideId !== slideId
    ) {
      return;
    }
    const succeeded = (runs || []).filter((r) => r.status === "succeeded");
    if (succeeded.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No inference runs yet";
      viewerRunSelect.appendChild(opt);
      return;
    }
    for (const r of succeeded) {
      const opt = document.createElement("option");
      opt.value = String(r.id);
      const when = r.finished_at || r.created_at || "";
      const short = when ? ` — ${when.slice(0, 10)}` : "";
      opt.textContent = `Run #${r.id} (${r.model_version || "model"})${short}`;
      viewerRunSelect.appendChild(opt);
    }
    viewerRunSelect.disabled = false;
    viewerRunSelect.value = String(succeeded[0].id);
    currentRunId = succeeded[0].id;
    await loadRegionsForRun(currentRunId, slideId, requestId ?? viewerRequestSeq);
  } catch (_) {
    if (
      (requestId != null && requestId !== viewerRequestSeq) ||
      currentSlideId !== slideId
    ) {
      return;
    }
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Runs unavailable";
    viewerRunSelect.appendChild(opt);
  }
}

async function loadRegionsForRun(runId, slideId = currentSlideId, requestId = viewerRequestSeq) {
  if (!runId) return;
  try {
    const { regions } = await window.slidesApi.getInferenceRegions(runId);
    if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
    lastRegions = regions || [];
    const hasPositive = lastRegions.some(
      (r) => normalizeRegionLabel(r.label) === "fungus_positive"
    );
    if (!hasPositive) {
      window.appToast?.("No positive regions found for this run.", "info", 3500);
    }
    addRegionOverlays(lastRegions, false);
  } catch (_) {
    if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
    lastRegions = [];
    clearOverlays();
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
      window.slidesApi.getDeepZoomTileUrl(slideId, level, x, y, format),
  };
}

async function showViewer(slideId) {
  const requestId = ++viewerRequestSeq;
  activeInferencePollToken += 1; // cancel any in-flight inference poll from another slide
  currentSlideId = slideId;
  viewerSlideOrder = getViewerSlideOrder();
  updateViewerNavButtons();
  if (viewerOverlay) viewerOverlay.hidden = false;
  currentMpp = null;

  viewerEmpty.style.display = "none";
  viewerContainer.innerHTML = "";
  viewerContainer.style.display = "block";
  clearOverlays();
  setInferenceStatus("");
  if (viewerScaleWrap) viewerScaleWrap.hidden = true;

  let meta = null;
  try {
    meta = await window.slidesApi.getSlideMetadata(slideId);
    if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
    applyMppFromMeta(meta);
    renderMetadataPanel(meta);
    showMetadataPanel(true);
  } catch (err) {
    console.warn("Slide metadata:", err);
    renderMetadataPanel(null);
    showMetadataPanel(true);
  }

  if (viewer) {
    viewer.destroy();
    viewer = null;
  }

  try {
    const dziUrl = window.slidesApi.getDeepZoomDziUrl(slideId);
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
        meta = await window.slidesApi.getSlideMetadata(slideId);
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
          window.slidesApi.getTileUrl(slideId, osdToOpenSlideLevel(level), x, y),
      };
    }

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
      minZoomImageRatio: 0.9,
      immediateRender: true,
      homeFillsViewer: false,
    });

    viewer.addHandler("open", () => {
      if (requestId !== viewerRequestSeq || currentSlideId !== slideId) return;
      viewer.viewport.goHome(true);
      updateScaleBar();
      populateRunSelector(slideId, requestId);
    });

    viewer.addHandler("animation", updateScaleBar);
    viewer.addHandler("resize", updateScaleBar);
    viewer.addHandler("canvas-click", (event) => {
      if (!event.quick) return;
      const homeZoom = viewer.viewport.getHomeZoom();
      const currentZoom = viewer.viewport.getZoom();
      const isZoomedIn = currentZoom > homeZoom * 1.05;
      if (isZoomedIn) {
        viewer.viewport.goHome(true);
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

viewerRunSelect?.addEventListener("change", async () => {
  const v = viewerRunSelect.value;
  currentRunId = v ? parseInt(v, 10) : null;
  if (currentRunId) await loadRegionsForRun(currentRunId, currentSlideId, viewerRequestSeq);
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

window.addEventListener("inference-model-changed", (e) => {
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
  const selectedModel =
    typeof window.getSelectedInferenceModel === "function"
      ? window.getSelectedInferenceModel()
      : null;
  const threshold = getCurrentThreshold();
  setInferenceStatus(selectedModel ? `Starting... (${selectedModel})` : "Starting...");

  try {
    const run = await window.slidesApi.runInference(currentSlideId, selectedModel, threshold);
    if (typeof window.galleryTrackInferenceRuns === "function") {
      window.galleryTrackInferenceRuns([run.id]);
    } else if (typeof window.galleryRefresh === "function") {
      window.galleryRefresh();
    }
    setInferenceStatus("Running...");

    const poll = async () => {
      if (pollToken !== activeInferencePollToken || runSlideId !== currentSlideId) {
        runInferenceBtn.disabled = false;
        return;
      }
      const r = await window.slidesApi.getInferenceRun(run.id);
      if (pollToken !== activeInferencePollToken || runSlideId !== currentSlideId) {
        runInferenceBtn.disabled = false;
        return;
      }
      if (r.status === "succeeded") {
        setInferenceStatus(`Done: ${r.summary?.fungus_positive ?? 0} positive`);
        runInferenceBtn.disabled = false;
        if (typeof window.galleryRefresh === "function") {
          window.galleryRefresh();
        }
        await populateRunSelector(runSlideId, viewerRequestSeq);
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

btnViewerExportView?.addEventListener("click", () => exportViewerViewport());
btnViewerExportRegions?.addEventListener("click", () => exportInferenceRegionsJson());
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
