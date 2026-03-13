const folderPathInput = document.getElementById("training-folder-path");
const selectFolderBtn = document.getElementById("btn-select-training-folder");
const startTrainingBtn = document.getElementById("btn-start-training");
const trainingStatus = document.getElementById("training-status");

let pollInterval = null;

function setTrainingStatus(html) {
  if (trainingStatus) trainingStatus.innerHTML = html;
}

async function pollTrainingStatus() {
  try {
    const s = await window.slidesApi.getTrainingStatus();
    if (s.status === "running") {
      const epoch = s.epoch ?? 0;
      const trainInfo = s.train_dice != null
        ? `Epoch ${epoch} — loss: ${(s.train_loss ?? 0).toFixed(4)}  dice: ${(s.train_dice ?? 0).toFixed(4)}`
        : `Starting...`;
      const valInfo = s.val_dice != null ? `  Val dice: ${s.val_dice.toFixed(4)}` : "";
      setTrainingStatus(`<span class="training-running">Training: ${trainInfo}${valInfo}</span>`);
    } else if (s.status === "succeeded") {
      if (pollInterval) clearInterval(pollInterval);
      pollInterval = null;
      startTrainingBtn.disabled = false;
      const best = s.best_dice != null ? (s.best_dice * 100).toFixed(1) : "?";
      setTrainingStatus(
        `<span class="training-success">Training complete. Best dice: ${best}%</span>`
      );
    } else if (s.status === "failed") {
      if (pollInterval) clearInterval(pollInterval);
      pollInterval = null;
      startTrainingBtn.disabled = false;
      const err = s.error_message || "Unknown error";
      setTrainingStatus(`<span class="training-error">Failed: ${err}</span>`);
    } else if (s.status === "idle") {
      setTrainingStatus("");
    }
  } catch (e) {
    setTrainingStatus(`<span class="training-error">Error: ${e.message}</span>`);
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = null;
    startTrainingBtn.disabled = false;
  }
}

async function handleStartTraining() {
  const path = folderPathInput?.value?.trim();
  if (!path) {
    setTrainingStatus('<span class="training-error">Select a folder first.</span>');
    return;
  }
  startTrainingBtn.disabled = true;
  setTrainingStatus('<span class="training-running">Starting...</span>');

  try {
    await window.slidesApi.startTraining(path);
    pollInterval = setInterval(pollTrainingStatus, 2000);
    pollTrainingStatus();
  } catch (err) {
    setTrainingStatus(`<span class="training-error">Error: ${err.message}</span>`);
    startTrainingBtn.disabled = false;
  }
}

function initTraining() {
  selectFolderBtn?.addEventListener("click", async () => {
    const path = await window.electronAPI.selectDirectory();
    if (path && folderPathInput) {
      folderPathInput.value = path;
      folderPathInput.title = path;
    }
  });

  startTrainingBtn?.addEventListener("click", handleStartTraining);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initTraining);
} else {
  initTraining();
}
