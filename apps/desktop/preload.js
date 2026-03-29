const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  getConfig: () => ipcRenderer.invoke("get-config"),
  setConfig: (config) => ipcRenderer.invoke("set-config", config),
  onApiReady: (callback) => {
    const listener = (_, data) => callback(data);
    ipcRenderer.on("api-ready", listener);
    return () => ipcRenderer.removeListener("api-ready", listener);
  },
  getDroppedFilePaths: (pathStrings) => ipcRenderer.invoke("get-dropped-file-paths", pathStrings),
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  selectFiles: () => ipcRenderer.invoke("select-files"),
  selectDirectory: () => ipcRenderer.invoke("select-directory"),
  saveViewerCapture: (rect, defaultFilename) =>
    ipcRenderer.invoke("save-viewer-capture", { rect, defaultFilename }),
});
