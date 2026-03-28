const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  getConfig: () => ipcRenderer.invoke("get-config"),
  setConfig: (config) => ipcRenderer.invoke("set-config", config),
  onApiReady: (callback) => {
    ipcRenderer.on("api-ready", (_, data) => callback(data));
  },
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  selectFiles: () => ipcRenderer.invoke("select-files"),
  selectDirectory: () => ipcRenderer.invoke("select-directory"),
  saveViewerCapture: (rect, defaultFilename) =>
    ipcRenderer.invoke("save-viewer-capture", { rect, defaultFilename }),
});
