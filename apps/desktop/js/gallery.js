const galleryGrid = document.getElementById("gallery-grid");
const galleryEmpty = document.getElementById("gallery-empty");
const gallerySearch = document.getElementById("gallery-search");
const gallerySort = document.getElementById("gallery-sort");
const galleryFavoritesOnly = document.getElementById("gallery-favorites-only");
const btnGalleryRefresh = document.getElementById("btn-gallery-refresh");
const btnGallerySelect = document.getElementById("btn-gallery-select");
const btnGalleryDeleteSelected = document.getElementById("btn-gallery-delete-selected");

const FAV_KEY = "galleryFavoriteIds";

let allSlides = [];
let selectionMode = false;
const selectedIds = new Set();
let galleryLoadSeq = 0;

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

function getFavorites() {
  try {
    const raw = localStorage.getItem(FAV_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(arr) ? arr.map(Number) : []);
  } catch {
    return new Set();
  }
}

function saveFavorites(set) {
  localStorage.setItem(FAV_KEY, JSON.stringify([...set]));
}

function toggleFavorite(id) {
  const fav = getFavorites();
  if (fav.has(id)) fav.delete(id);
  else fav.add(id);
  saveFavorites(fav);
  renderCards();
}

function compareSlides(a, b, sort) {
  const nameA = filenameFromPath(a.original_path).toLowerCase();
  const nameB = filenameFromPath(b.original_path).toLowerCase();
  const tA = new Date(a.created_at || 0).getTime();
  const tB = new Date(b.created_at || 0).getTime();
  switch (sort) {
    case "date-asc":
      return tA - tB;
    case "name-asc":
      return nameA.localeCompare(nameB);
    case "name-desc":
      return nameB.localeCompare(nameA);
    case "date-desc":
    default:
      return tB - tA;
  }
}

function getFilteredSlides() {
  const q = (gallerySearch?.value || "").trim().toLowerCase();
  const favOnly = galleryFavoritesOnly?.checked;
  const fav = getFavorites();
  let list = [...allSlides];
  if (q) {
    list = list.filter((s) => filenameFromPath(s.original_path).toLowerCase().includes(q));
  }
  if (favOnly) {
    list = list.filter((s) => fav.has(s.id));
  }
  const sort = gallerySort?.value || "date-desc";
  list.sort((a, b) => compareSlides(a, b, sort));
  return list;
}

function showSkeletons(n = 8) {
  if (!galleryGrid) return;
  galleryGrid.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const sk = document.createElement("div");
    sk.className = "gallery-skeleton";
    sk.setAttribute("aria-hidden", "true");
    galleryGrid.appendChild(sk);
  }
}

function renderCards() {
  if (!galleryGrid || !galleryEmpty) return;
  const slides = getFilteredSlides();
  galleryGrid.innerHTML = "";

  if (slides.length === 0) {
    galleryEmpty.style.display = "block";
    if (allSlides.length === 0) {
      galleryEmpty.textContent = "No slides yet. Import slides from the Import tab.";
    } else {
      galleryEmpty.textContent = "No slides match your filters. Try clearing search or favorites.";
    }
    return;
  }

  galleryEmpty.style.display = "none";
  const fav = getFavorites();

  for (const s of slides) {
    const card = document.createElement("div");
    card.className = "gallery-card" + (selectionMode ? " gallery-card--selectable" : "");
    card.dataset.slideId = String(s.id);

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "gallery-card-select";
    cb.checked = selectedIds.has(s.id);
    cb.title = "Select slide";
    cb.addEventListener("click", (e) => e.stopPropagation());
    cb.addEventListener("change", () => {
      if (cb.checked) selectedIds.add(s.id);
      else selectedIds.delete(s.id);
      updateSelectionUi();
    });

    const favBtn = document.createElement("button");
    favBtn.type = "button";
    favBtn.className = "gallery-card-fav";
    favBtn.setAttribute("aria-pressed", fav.has(s.id) ? "true" : "false");
    favBtn.title = fav.has(s.id) ? "Remove from favorites" : "Add to favorites";
    favBtn.textContent = fav.has(s.id) ? "★" : "☆";
    favBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFavorite(s.id);
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "gallery-card-delete";
    delBtn.textContent = "Delete";
    delBtn.title = "Remove slide from library";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteOneSlide(s.id);
    });

    const thumb = document.createElement("img");
    thumb.className = "gallery-card-thumb";
    thumb.src = window.slidesApi.getThumbnailUrl(s.id);
    thumb.alt = filenameFromPath(s.original_path);
    thumb.loading = "lazy";
    thumb.onerror = () => {
      thumb.src =
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect fill='%23333' width='200' height='200'/%3E%3Ctext fill='%23999' x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle'%3ENo preview%3C/text%3E%3C/svg%3E";
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

    card.appendChild(cb);
    card.appendChild(favBtn);
    card.appendChild(delBtn);
    card.appendChild(thumb);
    card.appendChild(info);

    card.addEventListener("click", () => {
      if (selectionMode) {
        cb.checked = !cb.checked;
        if (cb.checked) selectedIds.add(s.id);
        else selectedIds.delete(s.id);
        updateSelectionUi();
        return;
      }
      window.showViewer(s.id);
    });

    galleryGrid.appendChild(card);
  }
}

function updateSelectionUi() {
  if (btnGalleryDeleteSelected) {
    btnGalleryDeleteSelected.disabled = selectedIds.size === 0;
  }
  if (btnGallerySelect) {
    btnGallerySelect.classList.toggle("btn-primary", selectionMode);
    btnGallerySelect.classList.toggle("btn-secondary", !selectionMode);
    btnGallerySelect.textContent = selectionMode ? "Done" : "Select";
  }
}

async function deleteOneSlide(id) {
  if (!confirm("Delete this slide from the library? This cannot be undone.")) return;
  try {
    await window.slidesApi.deleteSlide(id);
    selectedIds.delete(id);
    const fav = getFavorites();
    fav.delete(id);
    saveFavorites(fav);
    if (typeof window.appToast === "function") {
      window.appToast("Slide deleted", "success");
    }
    await loadGalleryData();
  } catch (err) {
    if (typeof window.appToast === "function") {
      window.appToast(err.message || "Delete failed", "error");
    } else {
      alert(err.message);
    }
  }
}

async function deleteSelectedSlides() {
  const ids = [...selectedIds];
  if (ids.length === 0) return;
  if (!confirm(`Delete ${ids.length} slide(s)? This cannot be undone.`)) return;
  let ok = 0;
  for (const id of ids) {
    try {
      await window.slidesApi.deleteSlide(id);
      ok++;
      const fav = getFavorites();
      fav.delete(id);
      saveFavorites(fav);
    } catch (err) {
      console.error(err);
    }
  }
  selectedIds.clear();
  selectionMode = false;
  updateSelectionUi();
  if (typeof window.appToast === "function") {
    window.appToast(`Deleted ${ok} slide(s)`, ok === ids.length ? "success" : "info");
  }
  await loadGalleryData();
}

async function loadGalleryData() {
  if (!galleryGrid || !galleryEmpty) return;
  const requestId = ++galleryLoadSeq;
  galleryEmpty.style.display = "none";
  showSkeletons();
  try {
    const data = await window.slidesApi.listSlides();
    if (requestId !== galleryLoadSeq) return;
    allSlides = data.slides || [];
    renderCards();
  } catch (err) {
    if (requestId !== galleryLoadSeq) return;
    galleryGrid.innerHTML = "";
    galleryEmpty.textContent = "Failed to load slides. Is the API running?";
    galleryEmpty.style.display = "block";
    console.error(err);
  }
}

function loadGallery() {
  loadGalleryData();
}

window.galleryRefresh = loadGallery;

function initGallery() {
  btnGalleryRefresh?.addEventListener("click", () => loadGalleryData());
  btnGallerySelect?.addEventListener("click", () => {
    selectionMode = !selectionMode;
    if (!selectionMode) selectedIds.clear();
    updateSelectionUi();
    renderCards();
  });
  btnGalleryDeleteSelected?.addEventListener("click", deleteSelectedSlides);
  gallerySearch?.addEventListener("input", () => renderCards());
  gallerySort?.addEventListener("change", () => renderCards());
  galleryFavoritesOnly?.addEventListener("change", () => renderCards());

  updateSelectionUi();
  loadGallery();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initGallery);
} else {
  initGallery();
}
