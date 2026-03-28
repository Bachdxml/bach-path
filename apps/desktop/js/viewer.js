const viewerContainer = document.getElementById("viewer-container");
const viewerBack = document.getElementById("viewer-back");
const viewerEmpty = document.getElementById("viewer-empty");
const runInferenceBtn = document.getElementById("viewer-run-inference");
const inferenceStatus = document.getElementById("viewer-inference-status");
const viewerRunSelect = document.getElementById("viewer-run-select");
const viewerShowNegative = document.getElementById("viewer-show-negative");
const viewerOverlayOpacity = document.getElementById("viewer-overlay-opacity");
const viewerScaleWrap = document.getElementById("viewer-scale-wrap");
const viewerScaleBar = document.getElementById("viewer-scale-bar");
const viewerScaleLabel = document.getElementById("viewer-scale-label");

let viewer = null;
let currentSlideId = null;
let currentMpp = null;
let currentRunId = null;
let lastRegions = [];

function setInferenceStatus(text) {
  if (inferenceStatus) inferenceStatus.textContent = text;
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

function addRegionOverlays(regions, showNegative = false) {
  if (!viewer) return;
  viewer.clearOverlays();
  const opacity = getOverlayOpacity();
  let width = 1;
  let height = 1;
  try {
    const item = viewer.world.getItemAt(0);
    const size = item.getContentSize();
    width = size.x;
    height = size.y;
  } catch (_) {}

  regions.forEach((r) => {
    if (r.label === "fungus_negative" && !showNegative) return;
    const rw = Math.max(1, r.w);
    const rh = Math.max(1, r.h);
    const rect = new OpenSeadragon.Rect(r.x, r.y, rw, rh);
    const el = document.createElement("div");
    el.className = "region-overlay";
    const color = r.label === "fungus_positive" ? "#e74c3c" : "#27ae60";
    el.style.border = `2px solid ${color}`;
    el.style.boxSizing = "border-box";
    el.style.pointerEvents = "none";
    el.style.opacity = String(opacity);
    el.title = `${r.label || "?"} (${((r.score || 0) * 100).toFixed(1)}%)`;
    try {
      viewer.addOverlay({ element: el, location: rect });
    } catch (_) {}
  });
}

async function populateRunSelector(slideId) {
  if (!viewerRunSelect) return;
  viewerRunSelect.innerHTML = "";
  viewerRunSelect.disabled = true;
  try {
    const { runs } = await window.slidesApi.getSlideInferenceRuns(slideId);
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
    await loadRegionsForRun(currentRunId);
  } catch (_) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Runs unavailable";
    viewerRunSelect.appendChild(opt);
  }
}

async function loadRegionsForRun(runId) {
  if (!runId) return;
  try {
    const { regions } = await window.slidesApi.getInferenceRegions(runId);
    lastRegions = regions || [];
    const showNeg = viewerShowNegative?.checked || false;
    addRegionOverlays(lastRegions, showNeg);
  } catch (_) {
    lastRegions = [];
    clearOverlays();
  }
}

async function showViewer(slideId) {
  currentSlideId = slideId;
  currentMpp = null;
  const tabImport = document.getElementById("tab-import");
  const tabGallery = document.getElementById("tab-gallery");
  const tabViewer = document.getElementById("tab-viewer");

  tabImport.classList.remove("active");
  tabGallery.classList.remove("active");
  tabViewer.classList.add("active");

  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
  document.querySelector('.tab-btn[data-tab="viewer"]').classList.add("active");

  viewerEmpty.style.display = "none";
  viewerContainer.innerHTML = "";
  viewerContainer.style.display = "block";
  clearOverlays();
  setInferenceStatus("");
  if (viewerScaleWrap) viewerScaleWrap.hidden = true;

  if (viewer) {
    viewer.destroy();
    viewer = null;
  }

  try {
    const apiBase = window.slidesApi.getApiBase();
    const dziUrl = `${apiBase}/slides/${slideId}/deepzoom.dzi`;
    let tileSource = null;

    try {
      const dziCheck = await fetch(dziUrl, { method: "HEAD" });
      if (dziCheck.ok) {
        tileSource = dziUrl;
      }
    } catch (_) {}

    if (!tileSource) {
      const meta = await window.slidesApi.getSlideMetadata(slideId);
      const mx = meta.mpp_x;
      const my = meta.mpp_y;
      if (mx != null && my != null) {
        currentMpp = (mx + my) / 2;
      } else if (mx != null) {
        currentMpp = mx;
      } else if (my != null) {
        currentMpp = my;
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
        getTileUrl: (level, x, y) => {
          const openSlideLevel = osdToOpenSlideLevel(level);
          return `${apiBase}/slides/${slideId}/tiles/${openSlideLevel}/${x}/${y}.jpg`;
        },
      };
    } else {
      try {
        const meta = await window.slidesApi.getSlideMetadata(slideId);
        const mx = meta.mpp_x;
        const my = meta.mpp_y;
        if (mx != null && my != null) currentMpp = (mx + my) / 2;
        else if (mx != null) currentMpp = mx;
        else if (my != null) currentMpp = my;
      } catch (_) {}
    }

    viewer = OpenSeadragon({
      element: viewerContainer,
      tileSources: tileSource,
      prefixUrl: "node_modules/openseadragon/build/openseadragon/images/",
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      showNavigationControl: true,
      constrainDuringPan: true,
      visibilityRatio: 1.0,
      wrapHorizontal: false,
      wrapVertical: false,
      minZoomImageRatio: 0.9,
      immediateRender: true,
      homeFillsViewer: false,
    });

    viewer.addHandler("open", () => {
      viewer.viewport.goHome(true);
      updateScaleBar();
      populateRunSelector(slideId);
    });

    viewer.addHandler("animation", updateScaleBar);
    viewer.addHandler("resize", updateScaleBar);
  } catch (err) {
    viewerContainer.innerHTML = `<p class="viewer-error">Failed to load slide: ${err.message}</p>`;
    console.error(err);
  }
}

viewerRunSelect?.addEventListener("change", async () => {
  const v = viewerRunSelect.value;
  currentRunId = v ? parseInt(v, 10) : null;
  if (currentRunId) await loadRegionsForRun(currentRunId);
});

viewerShowNegative?.addEventListener("change", () => {
  addRegionOverlays(lastRegions, viewerShowNegative.checked);
});

viewerOverlayOpacity?.addEventListener("input", () => {
  addRegionOverlays(lastRegions, viewerShowNegative?.checked || false);
});

async function handleRunInference() {
  if (!currentSlideId) return;
  runInferenceBtn.disabled = true;
  const selectedModel =
    typeof window.getSelectedInferenceModel === "function"
      ? window.getSelectedInferenceModel()
      : null;
  setInferenceStatus(selectedModel ? `Starting... (${selectedModel})` : "Starting...");

  try {
    const run = await window.slidesApi.runInference(currentSlideId, selectedModel);
    setInferenceStatus("Running...");

    const poll = async () => {
      const r = await window.slidesApi.getInferenceRun(run.id);
      if (r.status === "succeeded") {
        setInferenceStatus(`Done: ${r.summary?.fungus_positive ?? 0} positive`);
        runInferenceBtn.disabled = false;
        await populateRunSelector(currentSlideId);
        return;
      }
      if (r.status === "failed") {
        setInferenceStatus(`Failed: ${r.error_message || "Unknown error"}`);
        runInferenceBtn.disabled = false;
        return;
      }
      setTimeout(poll, 2000);
    };
    poll();
  } catch (err) {
    setInferenceStatus(`Error: ${err.message}`);
    runInferenceBtn.disabled = false;
  }
}

window.showViewer = showViewer;

if (viewerBack) {
  viewerBack.addEventListener("click", () => {
    document.getElementById("tab-viewer").classList.remove("active");
    document.getElementById("tab-gallery").classList.add("active");
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelector('.tab-btn[data-tab="gallery"]').classList.add("active");

    if (viewer) {
      viewer.destroy();
      viewer = null;
    }
    currentSlideId = null;
    currentMpp = null;
    lastRegions = [];
    viewerContainer.innerHTML = "";
    viewerContainer.style.display = "none";
    viewerEmpty.style.display = "block";
    if (viewerRunSelect) {
      viewerRunSelect.innerHTML = "";
      viewerRunSelect.disabled = true;
    }
  });
}

if (runInferenceBtn) {
  runInferenceBtn.addEventListener("click", handleRunInference);
}

document.addEventListener("keydown", (e) => {
  const t = e.target;
  const typing =
    t &&
    (t.tagName === "INPUT" ||
      t.tagName === "TEXTAREA" ||
      t.tagName === "SELECT" ||
      t.isContentEditable);
  if (typing) return;
  const tabViewer = document.getElementById("tab-viewer");
  if (!tabViewer?.classList.contains("active")) return;
  if (e.key === "r" || e.key === "R") {
    if (!e.ctrlKey && !e.metaKey && currentSlideId) {
      e.preventDefault();
      handleRunInference();
    }
  }
});
