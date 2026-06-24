const { contextBridge, ipcRenderer, webUtils } = require("electron");

function sanitizeConfigPayload(config) {
  if (!config || typeof config !== "object" || Array.isArray(config)) return {};
  const next = {};
  if (Object.prototype.hasOwnProperty.call(config, "apiPort")) next.apiPort = config.apiPort;
  if (Object.prototype.hasOwnProperty.call(config, "apiHost")) next.apiHost = config.apiHost;
  if (Object.prototype.hasOwnProperty.call(config, "apiKey")) next.apiKey = config.apiKey;
  return next;
}

function sanitizeApiReadyPayload(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) return null;
  const port = Number.isInteger(data.port) ? data.port : null;
  const host = typeof data.host === "string" ? data.host : null;
  if (!port || !host) return null;

  const payload = { port, host };
  if (typeof data.apiKey === "string" && data.apiKey.trim() !== "") {
    payload.apiKey = data.apiKey.trim();
  }
  return payload;
}

function sanitizePathStrings(pathStrings) {
  if (!Array.isArray(pathStrings)) return [];
  return pathStrings.filter((value) => typeof value === "string");
}

function sanitizeCapturePayload(rect, defaultFilename) {
  return {
    rect: rect && typeof rect === "object" ? rect : {},
    defaultFilename:
      typeof defaultFilename === "string" ? defaultFilename : "slide-view.png",
  };
}

contextBridge.exposeInMainWorld("electronAPI", {
  getConfig: () => ipcRenderer.invoke("get-config"),
  setConfig: (config) => ipcRenderer.invoke("set-config", sanitizeConfigPayload(config)),
  onApiReady: (callback) => {
    const listener = (_, data) => {
      const payload = sanitizeApiReadyPayload(data);
      if (payload) callback(payload);
    };
    ipcRenderer.on("api-ready", listener);
    return () => ipcRenderer.removeListener("api-ready", listener);
  },
  getDroppedFilePaths: (pathStrings) =>
    ipcRenderer.invoke("get-dropped-file-paths", sanitizePathStrings(pathStrings)),
  getPathsForFiles: (files) => {
    if (!Array.isArray(files)) return [];
    const paths = [];
    for (const file of files) {
      try {
        const filePath = webUtils.getPathForFile(file);
        if (typeof filePath === "string" && filePath !== "") paths.push(filePath);
      } catch {
        // Skip files the OS cannot resolve to a path.
      }
    }
    return paths;
  },
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  selectFiles: () => ipcRenderer.invoke("select-files"),
  selectDirectory: () => ipcRenderer.invoke("select-directory"),
  saveViewerCapture: (rect, defaultFilename) =>
    ipcRenderer.invoke(
      "save-viewer-capture",
      sanitizeCapturePayload(rect, defaultFilename)
    ),
  importModel: () => ipcRenderer.invoke("import-model"),
});
