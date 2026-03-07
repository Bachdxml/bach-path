const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  getConfig: () => ipcRenderer.invoke("get-config"),
  setConfig: (config) => ipcRenderer.invoke("set-config", config),
  onApiReady: (callback) => {
    ipcRenderer.on("api-ready", (_, data) => callback(data));
  },
  selectFolder: () => ipcRenderer.invoke("select-folder"),
});
