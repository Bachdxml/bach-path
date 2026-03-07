document.addEventListener("DOMContentLoaded", () => {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

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
});
