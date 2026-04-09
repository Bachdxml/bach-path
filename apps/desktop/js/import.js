const dropZone = document.getElementById("drop-zone");
const collectionTitleInput = document.getElementById("import-collection-title");
const filesBtn = document.getElementById("btn-select-files");
const folderBtn = document.getElementById("btn-select-folder");
const cancelBtn = document.getElementById("btn-cancel-import");
const importStatus = document.getElementById("import-status");
const importProgress = document.getElementById("import-progress");
const importApp = window.BachPath || null;
const importSlidesApi = importApp?.services?.slidesApi || window.slidesApi;

const WSI_EXTENSIONS = [".svs", ".tif", ".tiff", ".png"];

let importCancelled = false;
let importInProgress = false;
let importAbortController = null;

function getPathsFromFiles(files) {
  const paths = [];
  for (const f of files) {
    const ext = f.name ? "." + f.name.split(".").pop().toLowerCase() : "";
    if (WSI_EXTENSIONS.includes(ext) && f.path) {
      paths.push(f.path);
    }
  }
  return paths;
}

function filterValidPaths(paths) {
  const unique = [];
  const seen = new Set();
  for (const path of paths || []) {
    if (typeof path !== "string") continue;
    const lower = path.toLowerCase();
    if (!WSI_EXTENSIONS.some((ext) => lower.endsWith(ext))) continue;
    if (seen.has(path)) continue;
    seen.add(path);
    unique.push(path);
  }
  return unique;
}

function getCollectionTitle() {
  return collectionTitleInput?.value?.trim() || null;
}

function getImportProgressLabel(current, total) {
  if (!total) return "";
  if (current >= total) return `Importing ${total} slide(s)...`;
  return `Importing ${current}/${total}...`;
}

function toFiniteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function setStatus(text, isError = false) {
  importStatus.textContent = text;
  importStatus.className = "import-status" + (isError ? " error" : "");
}

function setProgress(current, total) {
  importProgress.textContent = getImportProgressLabel(current, total);
  importProgress.style.visibility = total > 0 ? "visible" : "hidden";
}

function setImporting(active) {
  if (cancelBtn) cancelBtn.disabled = !active;
  if (filesBtn) filesBtn.disabled = active;
  if (folderBtn) folderBtn.disabled = active;
}

cancelBtn?.addEventListener("click", () => {
  importCancelled = true;
  if (importAbortController) {
    importAbortController.abort();
  }
  setStatus("Stopping import...");
});

function normalizeImportResult(result, fallbackCount, fallbackTitle) {
  const count =
    toFiniteNumber(result?.imported_count) ??
    toFiniteNumber(result?.slide_count) ??
    toFiniteNumber(result?.count) ??
    fallbackCount;
  const title =
    typeof result?.collection_title === "string" && result.collection_title.trim()
      ? result.collection_title.trim()
      : typeof result?.title === "string" && result.title.trim()
        ? result.title.trim()
        : fallbackTitle;
  return { count, title };
}

async function importPaths(paths, sourceType) {
  const filePaths = filterValidPaths(paths);
  if (!filePaths.length) return;
  if (importInProgress) {
    setStatus("Import already running. Wait for completion or cancel first.", true);
    return;
  }
  importInProgress = true;
  importCancelled = false;
  importAbortController = new AbortController();
  setImporting(true);
  const collectionTitle = getCollectionTitle();
  setProgress(filePaths.length, filePaths.length);
  try {
    const result = await importSlidesApi.importCollection(
      filePaths,
      collectionTitle,
      sourceType,
      importAbortController.signal
    );
    const normalized = normalizeImportResult(result, filePaths.length, collectionTitle);
    const titleLabel = normalized.title ? ` into "${normalized.title}"` : "";
    setStatus(`Imported ${normalized.count} slide(s)${titleLabel} successfully.`);
    if (typeof window.appToast === "function" && normalized.count > 0) {
      window.appToast(`Imported ${normalized.count} slide(s)${titleLabel}.`, "success", 3000);
    }
  } catch (err) {
    if (importCancelled || err?.name === "AbortError") {
      setStatus("Import canceled.", true);
      if (typeof window.appToast === "function") {
        window.appToast("Import canceled.", "info", 3000);
      }
    } else {
      const message = (err && err.message ? err.message : "Import failed").trim();
      setStatus(message, true);
      if (typeof window.appToast === "function") {
        window.appToast(message, "error", 5000);
      }
    }
  } finally {
    importInProgress = false;
    setProgress(0, 0);
    setImporting(false);
    importAbortController = null;
  }
  const refreshGallery = importApp?.features?.gallery?.refresh || window.galleryRefresh;
  if (typeof refreshGallery === "function") refreshGallery();
}

function initImport() {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("drag-over");
    const paths = getPathsFromFiles(Array.from(e.dataTransfer.files || []));
    if (paths.length) {
      importPaths(paths, "files");
    } else {
      setStatus("No valid files (SVS, TIF, TIFF, PNG) dropped.", true);
    }
  });

  dropZone.addEventListener("click", async () => {
    const paths = await window.electronAPI.selectFiles();
    if (paths.length) {
      importPaths(paths, "files");
    }
  });

  filesBtn.addEventListener("click", async () => {
    const paths = await window.electronAPI.selectFiles();
    if (paths.length) {
      importPaths(paths, "files");
    } else {
      setStatus("No valid files (SVS, TIF, TIFF, PNG) selected.", true);
    }
  });

  folderBtn.addEventListener("click", async () => {
    const paths = await window.electronAPI.selectFolder();
    if (paths.length) {
      importPaths(paths, "folder");
    } else {
      setStatus("No WSI files found in the selected folder.", true);
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initImport);
} else {
  initImport();
}

if (importApp?.registerFeature) {
  importApp.registerFeature("import", {
    importPaths,
  });
}
