const galleryGrid = document.getElementById("gallery-grid");
const galleryEmpty = document.getElementById("gallery-empty");

function filenameFromPath(p) {
  if (!p) return "Unknown";
  return p.split(/[/\\]/).pop() || "Unknown";
}

function formatDate(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleDateString();
  } catch {
    return "";
  }
}

async function loadGallery() {
  galleryGrid.innerHTML = "";
  galleryEmpty.style.display = "none";
  try {
    const data = await window.slidesApi.listSlides();
    const slides = data.slides || [];
    if (slides.length === 0) {
      galleryEmpty.style.display = "block";
      return;
    }
    for (const s of slides) {
      const card = document.createElement("div");
      card.className = "gallery-card";
      card.dataset.slideId = s.id;
      const thumb = document.createElement("img");
      thumb.className = "gallery-card-thumb";
      thumb.src = window.slidesApi.getThumbnailUrl(s.id);
      thumb.alt = filenameFromPath(s.original_path);
      thumb.loading = "lazy";
      thumb.onerror = () => {
        thumb.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%23333' width='200' height='200'/%3E%3Ctext fill='%23999' x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle'%3ENo preview%3C/text%3E%3C/svg%3E";
      };
      const info = document.createElement("div");
      info.className = "gallery-card-info";
      const label = document.createElement("div");
      label.className = "gallery-card-name";
      label.textContent = filenameFromPath(s.original_path);
      const meta = document.createElement("div");
      meta.className = "gallery-card-date";
      meta.textContent = formatDate(s.created_at);
      info.appendChild(label);
      info.appendChild(meta);
      card.appendChild(thumb);
      card.appendChild(info);
      card.addEventListener("click", () => {
        window.showViewer(s.id);
      });
      galleryGrid.appendChild(card);
    }
  } catch (err) {
    galleryEmpty.textContent = "Failed to load slides. Is the API running?";
    galleryEmpty.style.display = "block";
    console.error(err);
  }
}

window.galleryRefresh = loadGallery;

function initGallery() {
  loadGallery();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initGallery);
} else {
  initGallery();
}
