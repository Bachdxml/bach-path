const folderPathInput = document.getElementById("training-folder-path");
const selectFolderBtn = document.getElementById("btn-select-training-folder");
const startTrainingBtn = document.getElementById("btn-start-training");
const stopTrainingBtn = document.getElementById("btn-stop-training");
const trainingStatus = document.getElementById("training-status");
const trainingLogOutput = document.getElementById("training-log-output");

let pollInterval = null;
let logPollInterval = null;

function setTrainingStatus(html) {
  if (trainingStatus) trainingStatus.innerHTML = html;
}

async function fetchTrainingLog() {
  if (!trainingLogOutput) return;
  try {
    const { lines } = await window.slidesApi.getTrainingLog(500);
    trainingLogOutput.textContent = (lines || []).join("\n");
    trainingLogOutput.scrollTop = trainingLogOutput.scrollHeight;
  } catch (_) {}
}

function startLogPolling() {
  if (logPollInterval) clearInterval(logPollInterval);
  logPollInterval = setInterval(fetchTrainingLog, 4000);
  fetchTrainingLog();
}

function stopLogPolling() {
  if (logPollInterval) {
    clearInterval(logPollInterval);
    logPollInterval = null;
  }
}

async function pollTrainingStatus() {
  try {
    const s = await window.slidesApi.getTrainingStatus();
    if (s.status === "running") {
      stopTrainingBtn.disabled = false;
      if (!logPollInterval) startLogPolling();
      const epoch = s.epoch ?? 0;
      const trainInfo =
        s.train_dice != null
          ? `Epoch ${epoch} — loss: ${(s.train_loss ?? 0).toFixed(4)}  dice: ${(s.train_dice ?? 0).toFixed(4)}`
          : `Starting...`;
      const valInfo = s.val_dice != null ? `  Val dice: ${s.val_dice.toFixed(4)}` : "";
      setTrainingStatus(`<span class="training-running">Training: ${trainInfo}${valInfo}</span>`);
      if (s.error_message) {
        setTrainingStatus(`<span class="training-running">${s.error_message}</span>`);
      }
    } else if (s.status === "stopped") {
      if (pollInterval) clearInterval(pollInterval);
      pollInterval = null;
      stopLogPolling();
      await fetchTrainingLog();
      startTrainingBtn.disabled = false;
      stopTrainingBtn.disabled = true;
      const ckpt = s.checkpoint_path ? `<br/>Saved: ${s.checkpoint_path}` : "";
      setTrainingStatus(`<span class="training-success">Training stopped by user.${ckpt}</span>`);
    } else if (s.status === "succeeded") {
      if (pollInterval) clearInterval(pollInterval);
      pollInterval = null;
      stopLogPolling();
      await fetchTrainingLog();
      startTrainingBtn.disabled = false;
      stopTrainingBtn.disabled = true;
      const best = s.best_dice != null ? (s.best_dice * 100).toFixed(1) : "?";
      setTrainingStatus(
        `<span class="training-success">Training complete. Best dice: ${best}%</span>`
      );
    } else if (s.status === "failed") {
      if (pollInterval) clearInterval(pollInterval);
      pollInterval = null;
      stopLogPolling();
      await fetchTrainingLog();
      startTrainingBtn.disabled = false;
      stopTrainingBtn.disabled = true;
      const err = s.error_message || "Unknown error";
      setTrainingStatus(`<span class="training-error">Failed: ${err}</span>`);
    } else if (s.status === "idle") {
      stopLogPolling();
      startTrainingBtn.disabled = false;
      stopTrainingBtn.disabled = true;
      setTrainingStatus("");
    }
  } catch (e) {
    setTrainingStatus(`<span class="training-error">Error: ${e.message}</span>`);
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = null;
    stopLogPolling();
    startTrainingBtn.disabled = false;
    stopTrainingBtn.disabled = true;
  }
}

async function handleStartTraining() {
  const path = folderPathInput?.value?.trim();
  if (!path) {
    setTrainingStatus('<span class="training-error">Select a folder first.</span>');
    return;
  }
  if (trainingLogOutput) trainingLogOutput.textContent = "";
  startTrainingBtn.disabled = true;
  stopTrainingBtn.disabled = false;
  setTrainingStatus('<span class="training-running">Starting...</span>');

  try {
    await window.slidesApi.startTraining(path);
    startLogPolling();
    pollInterval = setInterval(pollTrainingStatus, 2000);
    pollTrainingStatus();
  } catch (err) {
    setTrainingStatus(`<span class="training-error">Error: ${err.message}</span>`);
    startTrainingBtn.disabled = false;
    stopTrainingBtn.disabled = true;
    stopLogPolling();
  }
}

async function handleStopTraining() {
  stopTrainingBtn.disabled = true;
  setTrainingStatus('<span class="training-running">Stopping and saving checkpoint...</span>');
  try {
    await window.slidesApi.stopTraining();
    if (!pollInterval) {
      pollInterval = setInterval(pollTrainingStatus, 2000);
    }
  } catch (err) {
    setTrainingStatus(`<span class="training-error">Error: ${err.message}</span>`);
    stopTrainingBtn.disabled = false;
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
  stopTrainingBtn?.addEventListener("click", handleStopTraining);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initTraining);
} else {
  initTraining();
}
