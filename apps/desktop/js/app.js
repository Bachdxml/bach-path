document.addEventListener("DOMContentLoaded", () => {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const modelSelect = document.getElementById("inference-model-select");
  const modelStatus = document.getElementById("inference-model-status");
  const refreshModelsBtn = document.getElementById("btn-refresh-models");

  function setModelStatus(text, isError = false) {
    if (!modelStatus) return;
    modelStatus.textContent = text;
    modelStatus.style.color = isError ? "var(--error)" : "var(--text-secondary)";
  }

  async function loadModelOptions() {
    if (!modelSelect) return;
    try {
      setModelStatus("Loading models...");
      const data = await window.slidesApi.listInferenceModels();
      const models = data.models || [];
      modelSelect.innerHTML = "";
      if (models.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No model files found";
        modelSelect.appendChild(opt);
        modelSelect.disabled = true;
        setModelStatus("No models found in wsi-fungal-segmentation/models", true);
        return;
      }
      modelSelect.disabled = false;
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label;
        modelSelect.appendChild(opt);
      }

      const saved = localStorage.getItem("selectedInferenceModel");
      const defaultId = data.default_model_id || models[0].id;
      const selected = models.some((m) => m.id === saved) ? saved : defaultId;
      modelSelect.value = selected;
      localStorage.setItem("selectedInferenceModel", selected);
      setModelStatus(`Using model: ${selected}`);
    } catch (err) {
      setModelStatus(`Failed to load models: ${err.message}`, true);
    }
  }

  window.getSelectedInferenceModel = () => {
    if (!modelSelect || modelSelect.disabled) return null;
    return modelSelect.value || null;
  };

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const pane = document.getElementById("tab-" + tab);
      if (pane) pane.classList.add("active");
      if (tab === "gallery" && typeof window.galleryRefresh === "function") {
        window.galleryRefresh();
      }
    });
  });

  window.electronAPI.onApiReady(({ port, host }) => {
    window.slidesApi.setApiBase(`http://${host}:${port}`);
    if (typeof window.galleryRefresh === "function") {
      window.galleryRefresh();
    }
    loadModelOptions();
  });

  window.electronAPI.getConfig().then((config) => {
    const portInput = document.getElementById("api-port");
    if (portInput) portInput.value = config.apiPort ?? 8765;
  });

  document.getElementById("btn-save-settings")?.addEventListener("click", async () => {
    const portInput = document.getElementById("api-port");
    const port = parseInt(portInput?.value || "8765", 10);
    if (port < 1024 || port > 65535) {
      alert("Port must be between 1024 and 65535");
      return;
    }
    await window.electronAPI.setConfig({ apiPort: port });
    alert("Settings saved. Restart the app for changes to take effect.");
  });

  modelSelect?.addEventListener("change", () => {
    if (!modelSelect.value) return;
    localStorage.setItem("selectedInferenceModel", modelSelect.value);
    setModelStatus(`Using model: ${modelSelect.value}`);
  });

  refreshModelsBtn?.addEventListener("click", loadModelOptions);
});
