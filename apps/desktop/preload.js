const { contextBridge, ipcRenderer } = require("electron");

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

function sanitizeUploadRequest(payload) {
  const safe = {};
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    if (typeof payload.filePath === "string") safe.filePath = payload.filePath;
    if (Number.isInteger(payload.collectionId)) safe.collectionId = payload.collectionId;
    if (payload.allowTileLikeImport === true) safe.allowTileLikeImport = true;
  }
  return safe;
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
  // Picked file paths are read and uploaded by the Electron main process so the
  // renderer never loads multi-GB slides into memory.
  uploadSlideFile: (payload) =>
    ipcRenderer.invoke("upload-slide-file", sanitizeUploadRequest(payload)),
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  selectFiles: () => ipcRenderer.invoke("select-files"),
  selectDirectory: () => ipcRenderer.invoke("select-directory"),
  saveViewerCapture: (rect, defaultFilename) =>
    ipcRenderer.invoke(
      "save-viewer-capture",
      sanitizeCapturePayload(rect, defaultFilename)
    ),
});
