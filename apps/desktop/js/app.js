document.addEventListener("DOMContentLoaded", () => {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const modelSelect = document.getElementById("inference-model-select");
  const modelStatus = document.getElementById("inference-model-status");
  const refreshModelsBtn = document.getElementById("btn-refresh-models");
  const apiStatusEl = document.getElementById("api-status");
  const apiStatusMeta = document.getElementById("api-status-meta");
  const themeSelect = document.getElementById("ui-theme");
  const helpModal = document.getElementById("help-modal");
  const btnOpenHelp = document.getElementById("btn-open-help");
  const btnHelpDismiss = document.getElementById("btn-help-dismiss");
  const helpNeverShow = document.getElementById("help-never-show");
  const toastHost = document.getElementById("toast-host");

  const THEME_KEY = "uiTheme";
  const HELP_DISMISS_KEY = "helpWelcomeDismissed";

  function toast(message, type = "info", duration = 4200) {
    if (!toastHost) return;
    const el = document.createElement("div");
    el.className = "toast" + (type === "error" ? " toast--error" : type === "success" ? " toast--success" : "");
    el.textContent = message;
    toastHost.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity 0.25s ease";
      setTimeout(() => el.remove(), 250);
    }, duration);
  }
  window.appToast = toast;

  function resolveTheme() {
    const pref = localStorage.getItem(THEME_KEY) || "system";
    if (pref === "light") return "light";
    if (pref === "dark") return "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function applyTheme() {
    const resolved = resolveTheme();
    document.documentElement.setAttribute("data-theme", resolved);
    if (themeSelect) themeSelect.value = localStorage.getItem(THEME_KEY) || "system";
  }

  applyTheme();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if ((localStorage.getItem(THEME_KEY) || "system") === "system") applyTheme();
  });

  themeSelect?.addEventListener("change", () => {
    localStorage.setItem(THEME_KEY, themeSelect.value);
    applyTheme();
    toast("Theme updated", "success", 2000);
  });

  function setApiStatus(state, meta = "") {
    if (!apiStatusEl) return;
    apiStatusEl.classList.remove("api-status--ok", "api-status--error", "api-status--unknown");
    if (state === "ok") {
      apiStatusEl.classList.add("api-status--ok");
      apiStatusEl.textContent = "API connected";
    } else if (state === "error") {
      apiStatusEl.classList.add("api-status--error");
      apiStatusEl.textContent = "API unavailable";
    } else {
      apiStatusEl.classList.add("api-status--unknown");
      apiStatusEl.textContent = "Checking API…";
    }
    if (apiStatusMeta) apiStatusMeta.textContent = meta;
  }

  let healthTimer = null;
  async function pingApi() {
    try {
      const data = await window.slidesApi.healthCheck();
      const up = typeof data?.uptime_seconds === "number" ? data.uptime_seconds : null;
      const meta =
        up != null
          ? `uptime ${Math.floor(up / 60)}m`
          : window.slidesApi.getApiBase().replace(/^https?:\/\//, "");
      setApiStatus("ok", meta);
    } catch {
      setApiStatus("error", "Check that the local API started");
    }
  }

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
        setModelStatus("No models found in wsi-fungal-segmentation/models or checkpoints", true);
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

  function activateTab(tab) {
    tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    tabPanes.forEach((p) => p.classList.toggle("active", p.id === "tab-" + tab));
    if (tab === "gallery" && typeof window.galleryRefresh === "function") {
      window.galleryRefresh();
    }
  }

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  const offApiReady = window.electronAPI.onApiReady(({ port, host }) => {
    const hostForUrl = host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
    window.slidesApi.setApiBase(`http://${hostForUrl}:${port}`);
    if (healthTimer) clearInterval(healthTimer);
    pingApi();
    healthTimer = setInterval(pingApi, 10000);
    if (typeof window.galleryRefresh === "function") {
      window.galleryRefresh();
    }
    loadModelOptions();
  });
  window.addEventListener("beforeunload", () => {
    if (typeof offApiReady === "function") offApiReady();
  });

  window.electronAPI
    .getConfig()
    .then((config) => {
      const portInput = document.getElementById("api-port");
      if (portInput) portInput.value = config.apiPort ?? 8765;
    })
    .catch((err) => {
      toast(`Failed to load settings: ${err?.message || err}`, "error");
    });

  document.getElementById("btn-save-settings")?.addEventListener("click", async () => {
    const portInput = document.getElementById("api-port");
    const raw = (portInput?.value || "8765").trim();
    if (!/^\d+$/.test(raw)) {
      toast("Port must be a whole number between 1024 and 65535", "error");
      return;
    }
    const port = Number(raw);
    if (port < 1024 || port > 65535) {
      toast("Port must be between 1024 and 65535", "error");
      return;
    }
    await window.electronAPI.setConfig({ apiPort: port });
    toast("Port saved. Restart the app for it to take effect.", "success", 5000);
  });

  modelSelect?.addEventListener("change", () => {
    if (!modelSelect.value) return;
    localStorage.setItem("selectedInferenceModel", modelSelect.value);
    setModelStatus(`Using model: ${modelSelect.value}`);
  });

  refreshModelsBtn?.addEventListener("click", loadModelOptions);

  function openHelp() {
    if (helpModal) helpModal.hidden = false;
  }

  function closeHelp() {
    if (helpModal) helpModal.hidden = true;
  }

  btnOpenHelp?.addEventListener("click", openHelp);
  btnHelpDismiss?.addEventListener("click", () => {
    if (helpNeverShow?.checked) {
      localStorage.setItem(HELP_DISMISS_KEY, "1");
    }
    closeHelp();
  });
  helpModal?.querySelectorAll("[data-close-modal]").forEach((el) => {
    el.addEventListener("click", closeHelp);
  });

  if (!localStorage.getItem(HELP_DISMISS_KEY)) {
    setTimeout(openHelp, 400);
  }

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const typing =
      t &&
      (t.tagName === "INPUT" ||
        t.tagName === "TEXTAREA" ||
        t.tagName === "SELECT" ||
        t.isContentEditable);
    if (typing && e.key !== "Escape") return;

    if (e.key === "?" && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      openHelp();
      return;
    }
    if (e.key === "Escape" && helpModal && !helpModal.hidden) {
      closeHelp();
      return;
    }

    const map = { 1: "import", 2: "gallery", 3: "training", 4: "settings" };
    if (map[e.key] && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      activateTab(map[e.key]);
      return;
    }
    if (!typing) {
      const k = e.key.toLowerCase();
      if (k === "g") {
        e.preventDefault();
        activateTab("gallery");
      } else if (k === "i") {
        e.preventDefault();
        activateTab("import");
      }
    }
  });
});
