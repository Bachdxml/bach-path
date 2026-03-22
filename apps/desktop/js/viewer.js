const viewerContainer = document.getElementById("viewer-container");
const viewerBack = document.getElementById("viewer-back");
const viewerEmpty = document.getElementById("viewer-empty");
const runInferenceBtn = document.getElementById("viewer-run-inference");
const inferenceStatus = document.getElementById("viewer-inference-status");

let viewer = null;
let currentSlideId = null;

async function showViewer(slideId) {
  currentSlideId = slideId;
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

  if (viewer) {
    viewer.destroy();
    viewer = null;
  }

  try {
    const apiBase = window.slidesApi.getApiBase();
    const dziUrl = `${apiBase}/slides/${slideId}/deepzoom.dzi`;
    let tileSource = null;

    // Prefer pre-generated DeepZoom: fastest open + smooth zoom.
    try {
      const dziCheck = await fetch(dziUrl, { method: "HEAD" });
      if (dziCheck.ok) {
        tileSource = dziUrl;
      }
    } catch (_) {}

    // Fallback to OpenSlide-backed custom tiles when DeepZoom is unavailable.
    if (!tileSource) {
      const meta = await window.slidesApi.getSlideMetadata(slideId);
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
    }

    viewer = OpenSeadragon({
      element: viewerContainer,
      tileSources: tileSource,
      prefixUrl: "node_modules/openseadragon/build/openseadragon/images/",
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      showNavigationControl: true,
      // Photo-like behavior: keep slide inside viewport bounds.
      constrainDuringPan: true,
      visibilityRatio: 1.0,
      wrapHorizontal: false,
      wrapVertical: false,
      // Keep initial fit visually larger without forcing crop.
      minZoomImageRatio: 0.9,
      immediateRender: true,
      homeFillsViewer: false,
    });

    viewer.addHandler("open", () => {
      // Start with full-slide fit in view every time.
      viewer.viewport.goHome(true);
      loadLatestInferenceOverlay(slideId);
    });
  } catch (err) {
    viewerContainer.innerHTML = `<p class="viewer-error">Failed to load slide: ${err.message}</p>`;
    console.error(err);
  }
}

function setInferenceStatus(text) {
  if (inferenceStatus) inferenceStatus.textContent = text;
}

function clearOverlays() {
  if (viewer) {
    viewer.clearOverlays();
  }
}

function addRegionOverlays(regions, showNegative = false) {
  if (!viewer) return;
  viewer.clearOverlays();
  const width = viewer.world.getItemAt(0).getContentSize().x;
  const height = viewer.world.getItemAt(0).getContentSize().y;

  regions.forEach((r) => {
    if (r.label === "fungus_negative" && !showNegative) return;
    const rect = new OpenSeadragon.Rect(r.x, r.y, r.w, r.h);
    const el = document.createElement("div");
    el.className = "region-overlay";
    el.style.border = `2px solid ${r.label === "fungus_positive" ? "#e74c3c" : "#27ae60"}`;
    el.style.boxSizing = "border-box";
    el.style.pointerEvents = "none";
    el.title = `${r.label} (${(r.score * 100).toFixed(1)}%)`;
    try {
      viewer.addOverlay({ element: el, location: rect });
    } catch (_) {}
  });
}

async function loadLatestInferenceOverlay(slideId) {
  try {
    const { runs } = await window.slidesApi.getSlideInferenceRuns(slideId);
    const succeeded = runs.find((r) => r.status === "succeeded");
    if (succeeded) {
      const { regions } = await window.slidesApi.getInferenceRegions(succeeded.id);
      addRegionOverlays(regions);
    }
  } catch (_) {}
}

async function handleRunInference() {
  if (!currentSlideId) return;
  runInferenceBtn.disabled = true;
  const selectedModel =
    typeof window.getSelectedInferenceModel === "function"
      ? window.getSelectedInferenceModel()
      : null;
  setInferenceStatus(
    selectedModel ? `Starting... (${selectedModel})` : "Starting..."
  );

  try {
    const run = await window.slidesApi.runInference(currentSlideId, selectedModel);
    setInferenceStatus("Running...");

    const poll = async () => {
      const r = await window.slidesApi.getInferenceRun(run.id);
      if (r.status === "succeeded") {
        setInferenceStatus(`Done: ${r.summary?.fungus_positive ?? 0} positive`);
        runInferenceBtn.disabled = false;
        const { regions } = await window.slidesApi.getInferenceRegions(run.id);
        addRegionOverlays(regions);
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
    viewerContainer.innerHTML = "";
    viewerContainer.style.display = "none";
    viewerEmpty.style.display = "block";
  });
}

if (runInferenceBtn) {
  runInferenceBtn.addEventListener("click", handleRunInference);
}
