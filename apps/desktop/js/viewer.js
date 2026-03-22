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
    const meta = await window.slidesApi.getSlideMetadata(slideId);
    const apiBase = window.slidesApi.getApiBase();
    const tileSource = {
      type: "custom",
      width: meta.dimensions[0],
      height: meta.dimensions[1],
      tileSize: 256,
      minLevel: 0,
      maxLevel: meta.level_count - 1,
      getTileUrl: (level, x, y) =>
        `${apiBase}/slides/${slideId}/tiles/${level}/${x}/${y}.jpg`,
    };

    viewer = OpenSeadragon({
      element: viewerContainer,
      tileSources: tileSource,
      prefixUrl: "node_modules/openseadragon/build/openseadragon/images/",
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
    });

    viewer.addHandler("open", () => {
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
