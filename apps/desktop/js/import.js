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
const TILE_FILENAME_RE = /(?:^|[_-])tile[_-]?x?\d+[_-]y?\d+(?:[_-]|$)/i;

let importCancelled = false;
let importInProgress = false;
let importAbortController = null;

function hasWsiExtension(name) {
  const lower = String(name || "").toLowerCase();
  return WSI_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

// A dropped File object carries only a name. Tile-looking PNG names are skipped
// client-side for a friendlier message; the backend enforces the same guard.
function looksLikeGeneratedTileName(name) {
  const lower = String(name || "").toLowerCase();
  if (!lower.endsWith(".png")) return false;
  const base = lower.replace(/\.[^.]+$/, "");
  return TILE_FILENAME_RE.test(base);
}

function filterValidFiles(files) {
  const valid = [];
  for (const file of files || []) {
    if (!file || !hasWsiExtension(file.name)) continue;
    if (looksLikeGeneratedTileName(file.name)) continue;
    valid.push(file);
  }
  return valid;
}

function filterValidPaths(paths) {
  const unique = [];
  const seen = new Set();
  for (const p of paths || []) {
    if (typeof p !== "string") continue;
    if (!hasWsiExtension(p)) continue;
    if (seen.has(p)) continue;
    seen.add(p);
    unique.push(p);
  }
  return unique;
}

function getCollectionTitle() {
  return collectionTitleInput?.value?.trim() || null;
}

function setStatus(text, isError = false) {
  importStatus.textContent = text;
  importStatus.className = "import-status" + (isError ? " error" : "");
}

function setProgress(current, total) {
  importProgress.textContent = total > 0 ? `Importing ${current}/${total}…` : "";
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
  setStatus("Stopping import after the current slide… already-imported slides are kept.");
});

function buildSummary(imported, total, failed) {
  let summary = `Imported ${imported} of ${total}`;
  if (failed > 0) {
    summary += ` — ${failed} couldn't be read`;
  }
  return summary;
}

// Upload one item. Drag-dropped File objects upload directly from the renderer;
// picked file paths are read and uploaded by the Electron main process. Both go
// through the same authenticated endpoint and produce identical results.
async function uploadOne(item, collectionId, signal) {
  if (item.kind === "file") {
    return importSlidesApi.uploadSlideFile(item.file, collectionId, false, signal);
  }
  const result = await window.electronAPI.uploadSlideFile({
    filePath: item.path,
    collectionId,
  });
  if (!result || result.ok !== true) {
    throw new Error((result && result.error) || "Upload failed");
  }
  return result;
}

async function runImport(items, sourceType) {
  if (!items.length) return;
  if (importInProgress) {
    setStatus("Import already running. Wait for completion or cancel first.", true);
    return;
  }
  importInProgress = true;
  importCancelled = false;
  importAbortController = new AbortController();
  setImporting(true);

  const total = items.length;
  const collectionTitle = getCollectionTitle();
  let collection = null;
  let imported = 0;
  let failed = 0;

  try {
    collection = await importSlidesApi.createImportCollection(collectionTitle, sourceType);

    for (let i = 0; i < total; i++) {
      if (importCancelled) break;
      setProgress(i + 1, total);
      try {
        await uploadOne(items[i], collection.id, importAbortController.signal);
        imported += 1;
      } catch (err) {
        if (importCancelled || err?.name === "AbortError") {
          break;
        }
        failed += 1;
      }
    }

    // Never leave an empty collection behind when nothing imported.
    if (imported === 0 && collection) {
      try {
        await importSlidesApi.deleteImportCollection(collection.id);
      } catch (_) {
        /* best-effort cleanup */
      }
    }

    const titleLabel = collectionTitle ? ` into "${collectionTitle}"` : "";
    if (importCancelled) {
      const msg =
        `Import canceled. ${buildSummary(imported, total, failed)}${titleLabel}. ` +
        "Already-imported slides were kept.";
      setStatus(msg);
      if (typeof window.appToast === "function") window.appToast(msg, "info", 5000);
    } else {
      const msg = `${buildSummary(imported, total, failed)}${titleLabel}.`;
      setStatus(msg, imported === 0 && total > 0);
      if (typeof window.appToast === "function" && imported > 0) {
        window.appToast(msg, "success", 3000);
      }
    }
  } catch (err) {
    const message = (err && err.message ? err.message : "Import failed").trim();
    setStatus(message, true);
    if (typeof window.appToast === "function") window.appToast(message, "error", 5000);
  } finally {
    importInProgress = false;
    setProgress(0, 0);
    setImporting(false);
    importAbortController = null;
  }

  const refreshGallery = importApp?.features?.gallery?.refresh || window.galleryRefresh;
  if (typeof refreshGallery === "function") refreshGallery();
}

function importFiles(files, sourceType) {
  const valid = filterValidFiles(files);
  if (!valid.length) {
    setStatus("No valid files (SVS, TIF, TIFF, PNG) selected.", true);
    return;
  }
  runImport(
    valid.map((file) => ({ kind: "file", file })),
    sourceType
  );
}

function importPaths(paths, sourceType) {
  const valid = filterValidPaths(paths);
  if (!valid.length) {
    setStatus("No valid files (SVS, TIF, TIFF, PNG) selected.", true);
    return;
  }
  runImport(
    valid.map((path) => ({ kind: "path", path })),
    sourceType
  );
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
    // Drag-dropped File objects are uploaded directly by the renderer.
    const files = Array.from(e.dataTransfer.files || []);
    const valid = filterValidFiles(files);
    if (valid.length) {
      importFiles(valid, "files");
    } else {
      setStatus("No valid files (SVS, TIF, TIFF, PNG) dropped.", true);
    }
  });

  dropZone.addEventListener("click", async () => {
    const paths = await window.electronAPI.selectFiles();
    if (paths.length) importPaths(paths, "files");
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
    importFiles,
  });
}
