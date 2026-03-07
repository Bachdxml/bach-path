const viewerContainer = document.getElementById("viewer-container");
const viewerBack = document.getElementById("viewer-back");
const viewerEmpty = document.getElementById("viewer-empty");

let viewer = null;

async function showViewer(slideId) {
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
  } catch (err) {
    viewerContainer.innerHTML = `<p class="viewer-error">Failed to load slide: ${err.message}</p>`;
    console.error(err);
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
    viewerContainer.innerHTML = "";
    viewerContainer.style.display = "none";
    viewerEmpty.style.display = "block";
  });
}
