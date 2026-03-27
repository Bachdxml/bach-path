const dropZone = document.getElementById("drop-zone");
const filesBtn = document.getElementById("btn-select-files");
const folderBtn = document.getElementById("btn-select-folder");
const importStatus = document.getElementById("import-status");
const importProgress = document.getElementById("import-progress");

const WSI_EXTENSIONS = [".svs", ".tif", ".tiff", ".png"];

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

async function importPaths(paths) {
  if (!paths.length) return;
  setProgress(0, paths.length);
  let success = 0;
  let failed = 0;
  for (let i = 0; i < paths.length; i++) {
    setProgress(i + 1, paths.length);
    try {
      await window.slidesApi.importSlide(paths[i]);
      success++;
    } catch (err) {
      failed++;
      console.error("Import failed:", paths[i], err);
    }
  }
  setProgress(0, 0);
  if (failed > 0) {
    setStatus(`Imported ${success} slide(s). ${failed} failed.`, true);
  } else {
    setStatus(`Imported ${success} slide(s) successfully.`);
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
