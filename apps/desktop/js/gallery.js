const galleryGrid = document.getElementById("gallery-grid");
const galleryEmpty = document.getElementById("gallery-empty");
const gallerySearch = document.getElementById("gallery-search");
const gallerySort = document.getElementById("gallery-sort");
const galleryInferenceFilter = document.getElementById("gallery-inference-filter");
const galleryFolderFilter = document.getElementById("gallery-folder-filter");
const galleryFavoritesOnly = document.getElementById("gallery-favorites-only");
const btnGalleryCollapseToggle = document.getElementById("btn-gallery-collapse-toggle");
const btnGalleryRefresh = document.getElementById("btn-gallery-refresh");
const btnGallerySelect = document.getElementById("btn-gallery-select");
const btnGalleryInferSelected = document.getElementById("btn-gallery-infer-selected");
const btnGalleryDeleteSelected = document.getElementById("btn-gallery-delete-selected");
const galleryApp = window.BachPath || null;
const gallerySlidesApi = galleryApp?.services?.slidesApi || window.slidesApi;

const FAV_KEY = "galleryFavoriteIds";
const FOLDER_COLLAPSE_KEY = "galleryCollapsedFolders";

let allSlides = [];
let selectionMode = false;
const selectedIds = new Set();
let galleryLoadSeq = 0;

function waitForGalleryApiReady() {
  if (typeof galleryApp?.whenApiReady === "function") {
    return galleryApp.whenApiReady();
  }
  return Promise.resolve();
}
let inferencePollTimer = null;
const pendingInferenceRunIds = new Set();

function getCollapsedFolders() {
  try {
    const raw = localStorage.getItem(FOLDER_COLLAPSE_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(arr) ? arr : []);
  } catch {
    return new Set();
  }
}

function saveCollapsedFolders(set) {
  localStorage.setItem(FOLDER_COLLAPSE_KEY, JSON.stringify([...set]));
}

function updateCollapseToggleButton(groups) {
  if (!btnGalleryCollapseToggle) return;
  if (!groups.length) {
    btnGalleryCollapseToggle.disabled = true;
    btnGalleryCollapseToggle.textContent = "Collapse all";
    return;
  }
  btnGalleryCollapseToggle.disabled = false;
  const collapsed = getCollapsedFolders();
  const allCollapsed = groups.every((g) => collapsed.has(g.key));
  btnGalleryCollapseToggle.textContent = allCollapsed ? "Expand all" : "Collapse all";
}

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

function getImportDateValue(slide) {
  return (
    slide?.collection_imported_at ||
    slide?.collection_created_at ||
    slide?.imported_at ||
    slide?.created_at ||
    null
  );
}

function getImportDayGroup(slide) {
  const value = getImportDateValue(slide);
  if (!value) {
    return {
      key: "import-day:unknown",
      title: "Unknown import date",
      importedAt: 0,
    };
  }
  const d = new Date(value);
  const ts = d.getTime();
  if (!Number.isFinite(ts)) {
    return {
      key: "import-day:unknown",
      title: "Unknown import date",
      importedAt: 0,
    };
  }
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return {
    key: `import-day:${year}-${month}-${day}`,
    title: d.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" }),
    importedAt: new Date(year, d.getMonth(), d.getDate()).getTime(),
  };
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

function compareSlidesByName(a, b) {
  const nameA = filenameFromPath(a.original_path).toLowerCase();
  const nameB = filenameFromPath(b.original_path).toLowerCase();
  return nameA.localeCompare(nameB, undefined, { numeric: true, sensitivity: "base" });
}

function filterSlides() {
  const q = (gallerySearch?.value || "").trim().toLowerCase();
  const favOnly = galleryFavoritesOnly?.checked;
  const inferenceFilter = galleryInferenceFilter?.value || "all";
  const folderFilter = galleryFolderFilter?.value || "all";
  const fav = getFavorites();
  let list = [...allSlides];
  if (q) {
    list = list.filter((s) => filenameFromPath(s.original_path).toLowerCase().includes(q));
  }
  if (inferenceFilter !== "all") {
    list = list.filter((s) => (s.inference_result || "unchecked") === inferenceFilter);
  }
  if (folderFilter !== "all") {
    list = list.filter((s) => (s.folder_key || "uncategorized") === folderFilter);
  }
  if (favOnly) {
    list = list.filter((s) => fav.has(s.id));
  }
  return list;
}

function getVisibleGroups() {
  const filteredSlides = filterSlides();
  const groups = new Map();
  for (const slide of filteredSlides) {
    const dayGroup = getImportDayGroup(slide);
    const key = dayGroup.key;
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        collectionId: null,
        isLegacy: true,
        title: dayGroup.title,
        importedAt: dayGroup.importedAt,
        slides: [],
      });
    }
    const group = groups.get(key);
    group.slides.push(slide);
  }
  const orderedGroups = [...groups.values()].sort((a, b) => {
    const timeDelta = b.importedAt - a.importedAt;
    if (timeDelta !== 0) return timeDelta;
    return a.title.localeCompare(b.title);
  });
  for (const group of orderedGroups) {
    group.slides.sort(compareSlidesByName);
  }
  return orderedGroups;
}

function getOrderedSlides() {
  return getVisibleGroups().flatMap((group) => group.slides);
}

function getFilteredSlides() {
  return getOrderedSlides();
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

function refreshFolderFilterOptions() {
  if (!galleryFolderFilter) return;
  const selected = galleryFolderFilter.value || "all";
  const folders = new Map();
  for (const s of allSlides) {
    const key = s.folder_key || "uncategorized";
    const label = s.folder_label || "Uncategorized";
    if (!folders.has(key)) folders.set(key, label);
  }
  galleryFolderFilter.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "all";
  allOpt.textContent = "All folders";
  galleryFolderFilter.appendChild(allOpt);
  for (const [key, label] of folders.entries()) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = label;
    galleryFolderFilter.appendChild(opt);
  }
  if ([...folders.keys(), "all"].includes(selected)) {
    galleryFolderFilter.value = selected;
  }
}

function createSlideCard(s, fav) {
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
  thumb.src = gallerySlidesApi.getThumbnailUrl(s.id);
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
  const status = document.createElement("div");
  const result = s.inference_result || "unchecked";
  const review = s.review_status || "unreviewed";
  status.className = review === "unreviewed"
    ? `gallery-card-status gallery-card-status--${result}`
    : `gallery-card-status gallery-card-status--review gallery-card-status--review-${review}`;
  const resultText =
    result === "positive"
      ? "AI positive"
      : result === "negative"
        ? "AI negative"
        : result === "needs_review"
          ? "AI needs review"
          : "Unchecked";
  const reviewText =
    review === "positive"
      ? "Reviewed positive"
      : review === "negative"
        ? "Reviewed negative"
        : review === "indeterminate"
          ? "Needs review"
          : "";
  status.textContent = reviewText ? `${reviewText} · ${resultText}` : resultText;
  info.appendChild(label);
  info.appendChild(meta);
  info.appendChild(status);

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
    const openViewer = galleryApp?.features?.viewer?.open || window.showViewer;
    if (typeof openViewer === "function") openViewer(s.id);
  });

  return card;
}

function renderCards() {
  if (!galleryGrid || !galleryEmpty) return;
  const groups = getVisibleGroups();
  galleryGrid.innerHTML = "";

  if (groups.length === 0) {
    galleryEmpty.style.display = "block";
    if (allSlides.length === 0) {
      galleryEmpty.textContent = "No slides yet. Import slides from the Import tab.";
    } else {
      galleryEmpty.textContent = "No slides match your filters. Try clearing search or favorites.";
    }
    updateCollapseToggleButton([]);
    return;
  }

  galleryEmpty.style.display = "none";
  const fav = getFavorites();
  const collapsed = getCollapsedFolders();
  for (const group of groups) {
    const wrapper = document.createElement("section");
    wrapper.className = "gallery-folder-group";
    if (collapsed.has(group.key)) wrapper.classList.add("is-collapsed");
    const header = document.createElement("div");
    header.className = "gallery-folder-header";

    const headerActions = document.createElement("div");
    headerActions.className = "gallery-folder-actions";

    const toggleBtn = document.createElement("button");
    toggleBtn.type = "button";
    toggleBtn.className = "gallery-folder-toggle";
    const isCollapsed = wrapper.classList.contains("is-collapsed");
    const toggleIcon = document.createElement("span");
    toggleIcon.className = "gallery-collection-toggle-icon";
    toggleIcon.textContent = isCollapsed ? "▸" : "▾";
    const toggleTitle = document.createElement("span");
    toggleTitle.className = "gallery-collection-title";
    toggleTitle.textContent = group.title;
    const toggleMeta = document.createElement("span");
    toggleMeta.className = "gallery-collection-meta";
    const slideCountLabel = group.slides.length === 1 ? "1 slide" : `${group.slides.length} slides`;
    const groupDate = formatDate(group.importedAt);
    toggleMeta.textContent = `${slideCountLabel}${groupDate ? ` · ${groupDate}` : ""}`;
    toggleBtn.title = `${group.title}${groupDate ? ` · ${groupDate}` : ""}`;
    toggleBtn.append(toggleIcon, toggleTitle, toggleMeta);
    toggleBtn.addEventListener("click", () => {
      wrapper.classList.toggle("is-collapsed");
      const nowCollapsed = wrapper.classList.contains("is-collapsed");
      const next = getCollapsedFolders();
      if (nowCollapsed) next.add(group.key);
      else next.delete(group.key);
      saveCollapsedFolders(next);
      toggleIcon.textContent = nowCollapsed ? "▸" : "▾";
      updateCollapseToggleButton(groups);
    });
    headerActions.appendChild(toggleBtn);

    if (!group.isLegacy && group.collectionId != null) {
      const renameBtn = document.createElement("button");
      renameBtn.type = "button";
      renameBtn.className = "btn btn-secondary btn-sm gallery-folder-select-btn";
      renameBtn.textContent = "Rename";
      renameBtn.addEventListener("click", () => renameCollectionGroup(group));
      headerActions.appendChild(renameBtn);
    }

    const folderSelectBtn = document.createElement("button");
    folderSelectBtn.type = "button";
    folderSelectBtn.className = "btn btn-secondary btn-sm gallery-folder-select-btn";
    const selectedInGroup = group.slides.filter((s) => selectedIds.has(s.id)).length;
    const allSelectedInGroup = selectedInGroup === group.slides.length && group.slides.length > 0;
    folderSelectBtn.textContent = allSelectedInGroup ? "Deselect all" : "Select all";
    folderSelectBtn.addEventListener("click", () => {
      if (!selectionMode) selectionMode = true;
      const shouldSelectAll = !group.slides.every((s) => selectedIds.has(s.id));
      for (const slide of group.slides) {
        if (shouldSelectAll) selectedIds.add(slide.id);
        else selectedIds.delete(slide.id);
      }
      updateSelectionUi();
      renderCards();
    });
    headerActions.appendChild(folderSelectBtn);

    header.appendChild(headerActions);
    const grid = document.createElement("div");
    grid.className = "gallery-grid";
    for (const s of group.slides) {
      grid.appendChild(createSlideCard(s, fav));
    }
    wrapper.appendChild(header);
    wrapper.appendChild(grid);
    galleryGrid.appendChild(wrapper);
  }
  updateCollapseToggleButton(groups);
}

function updateSelectionUi() {
  if (btnGalleryDeleteSelected) {
    btnGalleryDeleteSelected.disabled = selectedIds.size === 0;
  }
  if (btnGalleryInferSelected) {
    btnGalleryInferSelected.disabled = selectedIds.size === 0;
  }
  if (btnGallerySelect) {
    btnGallerySelect.classList.toggle("btn-primary", selectionMode);
    btnGallerySelect.classList.toggle("btn-secondary", !selectionMode);
    btnGallerySelect.textContent = selectionMode ? "Done" : "Select";
  }
}

function getPreferredThreshold(modelId) {
  if (!modelId) return null;
  try {
    const raw = localStorage.getItem("inferenceThresholdByModel");
    const map = raw ? JSON.parse(raw) : {};
    const t = map?.[modelId];
    return Number.isFinite(t) ? t : null;
  } catch {
    return null;
  }
}

function startInferencePolling() {
  if (inferencePollTimer) return;
  inferencePollTimer = setInterval(async () => {
    const ids = [...pendingInferenceRunIds];
    if (!ids.length) {
      clearInterval(inferencePollTimer);
      inferencePollTimer = null;
      return;
    }
    let completedNow = 0;
    for (const runId of ids) {
      try {
        const run = await gallerySlidesApi.getInferenceRun(runId);
        if (run.status === "succeeded" || run.status === "failed") {
          pendingInferenceRunIds.delete(runId);
          completedNow++;
        }
      } catch {
        // Keep polling while API is transiently unavailable.
      }
    }
    if (completedNow > 0) {
      loadGallery();
    }
    if (pendingInferenceRunIds.size === 0) {
      clearInterval(inferencePollTimer);
      inferencePollTimer = null;
      window.appToast?.("Inference updates completed.", "success", 2200);
    }
  }, 2000);
}

function trackInferenceRuns(runIds) {
  for (const id of runIds || []) {
    if (Number.isFinite(id)) pendingInferenceRunIds.add(id);
  }
  loadGallery();
  startInferencePolling();
}

async function deleteOneSlide(id) {
  if (!confirm("Delete this slide from the library? This cannot be undone.")) return;
  try {
    await gallerySlidesApi.deleteSlide(id);
    selectedIds.delete(id);
    updateSelectionUi();
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

async function renameCollectionGroup(group) {
  if (!group || group.collectionId == null) return;
  const currentTitle = group.title || "Untitled collection";
  const nextTitle = prompt("Rename import collection", currentTitle);
  if (nextTitle === null) return;
  const trimmed = nextTitle.trim();
  if (!trimmed) {
    window.appToast?.("Collection title cannot be empty.", "info");
    return;
  }
  if (trimmed === currentTitle) return;
  try {
    await gallerySlidesApi.renameImportCollection(group.collectionId, trimmed);
    window.appToast?.("Collection renamed.", "success", 2500);
    await loadGalleryData();
  } catch (err) {
    window.appToast?.(err.message || "Rename failed", "error");
  }
}

async function deleteSelectedSlides() {
  const ids = [...selectedIds];
  if (ids.length === 0) return;
  if (!confirm(`Delete ${ids.length} slide(s)? This cannot be undone.`)) return;
  let ok = 0;
  for (const id of ids) {
    try {
      await gallerySlidesApi.deleteSlide(id);
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
  await waitForGalleryApiReady();
  const requestId = ++galleryLoadSeq;
  galleryEmpty.style.display = "none";
  showSkeletons();
  try {
    const data = await gallerySlidesApi.listSlides();
    if (requestId !== galleryLoadSeq) return;
    allSlides = data.slides || [];
    const validIds = new Set(allSlides.map((s) => s.id));
    for (const id of [...selectedIds]) {
      if (!validIds.has(id)) selectedIds.delete(id);
    }
    refreshFolderFilterOptions();
    updateSelectionUi();
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
window.galleryGetOrderedSlideIds = () => getOrderedSlides().map((s) => s.id);
window.galleryGetSlideById = (id) => allSlides.find((s) => s.id === id) || null;
window.galleryTrackInferenceRuns = trackInferenceRuns;
if (galleryApp?.registerFeature) {
  galleryApp.registerFeature("gallery", {
    refresh: loadGallery,
    getOrderedSlideIds: () => getOrderedSlides().map((s) => s.id),
    getSlideById: (id) => allSlides.find((s) => s.id === id) || null,
    trackInferenceRuns,
  });
}

function initGallery() {
  if (gallerySort) {
    gallerySort.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "import-day-name";
    opt.textContent = "Import day, A-Z";
    gallerySort.appendChild(opt);
    gallerySort.value = "import-day-name";
    gallerySort.disabled = true;
    gallerySort.title = "Slides are grouped by import day and sorted A-Z within each day";
  }
  btnGalleryRefresh?.addEventListener("click", () => loadGalleryData());
  btnGalleryCollapseToggle?.addEventListener("click", () => {
    const groups = getVisibleGroups().map((group) => group.key);
    if (!groups.length) return;
    const collapsed = getCollapsedFolders();
    const allCollapsed = groups.every((k) => collapsed.has(k));
    const next = getCollapsedFolders();
    if (allCollapsed) {
      for (const key of groups) next.delete(key);
    } else {
      for (const key of groups) next.add(key);
    }
    saveCollapsedFolders(next);
    renderCards();
  });
  btnGallerySelect?.addEventListener("click", () => {
    selectionMode = !selectionMode;
    if (!selectionMode) selectedIds.clear();
    updateSelectionUi();
    renderCards();
  });
  btnGalleryInferSelected?.addEventListener("click", async () => {
    const slideIds = [...selectedIds];
    if (!slideIds.length) {
      window.appToast?.("Select one or more slides first.", "info");
      return;
    }
    const modelId =
      typeof window.getSelectedInferenceModel === "function"
        ? window.getSelectedInferenceModel()
        : null;
    const threshold = getPreferredThreshold(modelId);
    btnGalleryInferSelected.disabled = true;
    try {
      const res = await gallerySlidesApi.runBatchInference(slideIds, modelId, threshold);
      trackInferenceRuns(res?.run_ids || []);
      window.appToast?.(`Queued inference for ${slideIds.length} slide(s).`, "success", 3500);
    } catch (err) {
      window.appToast?.(err?.message || "Batch inference failed", "error", 4500);
    } finally {
      updateSelectionUi();
    }
  });
  btnGalleryDeleteSelected?.addEventListener("click", deleteSelectedSlides);
  gallerySearch?.addEventListener("input", () => renderCards());
  gallerySort?.addEventListener("change", () => renderCards());
  galleryInferenceFilter?.addEventListener("change", () => renderCards());
  galleryFolderFilter?.addEventListener("change", () => renderCards());
  galleryFavoritesOnly?.addEventListener("change", () => renderCards());

  updateSelectionUi();
  loadGallery();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initGallery);
} else {
  initGallery();
}
