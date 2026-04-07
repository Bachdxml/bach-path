const dropZone = document.getElementById("drop-zone");
const filesBtn = document.getElementById("btn-select-files");
const folderBtn = document.getElementById("btn-select-folder");
const cancelBtn = document.getElementById("btn-cancel-import");
const importStatus = document.getElementById("import-status");
const importProgress = document.getElementById("import-progress");

const WSI_EXTENSIONS = [".svs", ".tif", ".tiff", ".png"];

let importCancelled = false;
let importInProgress = false;

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

function setStatus(text, isError = false) {
  importStatus.textContent = text;
  importStatus.className = "import-status" + (isError ? " error" : "");
}

function setProgress(current, total) {
  importProgress.textContent = total > 0 ? `Importing ${current}/${total}...` : "";
  importProgress.style.visibility = total > 0 ? "visible" : "hidden";
}

function setImporting(active) {
  if (cancelBtn) cancelBtn.disabled = !active;
}

cancelBtn?.addEventListener("click", () => {
  importCancelled = true;
  setStatus("Stopping after current file…");
});

async function importPaths(paths) {
  if (!paths.length) return;
  if (importInProgress) {
    setStatus("Import already running. Wait for completion or cancel first.", true);
    return;
  }
  importInProgress = true;
  importCancelled = false;
  setImporting(true);
  setProgress(0, paths.length);
  let success = 0;
  let failed = 0;
  const failureReasons = new Map();
  try {
    for (let i = 0; i < paths.length; i++) {
      if (importCancelled) break;
      setProgress(i + 1, paths.length);
      try {
        await window.slidesApi.importSlide(paths[i]);
        success++;
      } catch (err) {
        failed++;
        const msg = (err && err.message ? err.message : "Unknown import error").trim();
        failureReasons.set(msg, (failureReasons.get(msg) || 0) + 1);
        console.error("Import failed:", paths[i], err);
      }
    }
  } finally {
    importInProgress = false;
    setProgress(0, 0);
    setImporting(false);
  }

  if (importCancelled) {
    setStatus(`Import stopped. ${success} slide(s) imported before cancel.`, failed > 0);
    if (typeof window.appToast === "function") {
      window.appToast(`Import stopped. ${success} slide(s) added.`, "info", 5000);
    }
  } else if (failed > 0) {
    const topReason = [...failureReasons.entries()].sort((a, b) => b[1] - a[1])[0];
    const reasonText = topReason ? ` Top error: ${topReason[0]} (${topReason[1]}).` : "";
    setStatus(`Imported ${success} slide(s). ${failed} failed.${reasonText}`, true);
    if (typeof window.appToast === "function") {
      window.appToast(`${failed} file(s) failed to import.${topReason ? ` ${topReason[0]}` : ""}`, "error", 6000);
    }
  } else {
    setStatus(`Imported ${success} slide(s) successfully.`);
    if (typeof window.appToast === "function" && success > 0) {
      window.appToast(`Imported ${success} slide(s).`, "success", 3000);
    }
  }

  if (typeof window.galleryRefresh === "function") {
    window.galleryRefresh();
  }
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
      importPaths(paths);
    } else {
      setStatus("No valid files (SVS, TIF, TIFF, PNG) dropped.", true);
    }
  });

  dropZone.addEventListener("click", async () => {
    const paths = await window.electronAPI.selectFiles();
    if (paths.length) {
      importPaths(paths);
    }
  });

  filesBtn.addEventListener("click", async () => {
    const paths = await window.electronAPI.selectFiles();
    if (paths.length) {
      importPaths(paths);
    } else {
      setStatus("No valid files (SVS, TIF, TIFF, PNG) selected.", true);
    }
  });

  folderBtn.addEventListener("click", async () => {
    const paths = await window.electronAPI.selectFolder();
    if (paths.length) {
      importPaths(paths);
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
