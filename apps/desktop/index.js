const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const http = require("http");

const DEFAULT_PORT = 8765;
const WSI_EXTENSIONS = new Set([".svs", ".tif", ".tiff", ".png"]);

let apiProcess = null;
let mainWindow = null;

function getConfigPath() {
  return path.join(app.getPath("userData"), "config.json");
}

function loadConfig() {
  try {
    const data = fs.readFileSync(getConfigPath(), "utf-8");
    return JSON.parse(data);
  } catch {
    return { apiPort: DEFAULT_PORT, apiHost: "127.0.0.1" };
  }
}

function saveConfig(config) {
  fs.writeFileSync(getConfigPath(), JSON.stringify(config, null, 2));
}

function getApiDataDir() {
  return path.join(app.getPath("userData"), "api-data");
}

function getApiLogDir() {
  return path.join(app.getPath("userData"), "api-logs");
}

function getApiScriptPath() {
  return path.join(__dirname, "..", "..", "services", "local-api", "run_api.py");
}

function getPythonPath() {
  const venvPython = path.join(__dirname, "..", "..", "services", "local-api", ".venv", "bin", "python");
  const venvPythonWin = path.join(__dirname, "..", "..", "services", "local-api", ".venv", "Scripts", "python.exe");
  if (fs.existsSync(venvPython)) return venvPython;
  if (fs.existsSync(venvPythonWin)) return venvPythonWin;
  return "python";
}

function waitForHealth(port, host = "127.0.0.1", maxAttempts = 30) {
  return new Promise((resolve, reject) => {
    let attempts = 0;

    const tryFetch = () => {
      const req = http.get(`http://${host}:${port}/health`, (res) => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve();
        } else if (attempts < maxAttempts) {
          attempts++;
          setTimeout(tryFetch, 500);
        } else {
          reject(new Error("Health check failed"));
        }
      });
      req.on("error", () => {
        if (attempts < maxAttempts) {
          attempts++;
          setTimeout(tryFetch, 500);
        } else {
          reject(new Error("API did not start"));
        }
      });
      req.setTimeout(3000, () => {
        req.destroy();
        if (attempts < maxAttempts) {
          attempts++;
          setTimeout(tryFetch, 500);
        } else {
          reject(new Error("API did not start"));
        }
      });
    };

    setTimeout(tryFetch, 300);
  });
}

function startApi() {
  const config = loadConfig();
  const port = config.apiPort ?? DEFAULT_PORT;
  const host = config.apiHost ?? "127.0.0.1";

  const dataDir = getApiDataDir();
  const logDir = getApiLogDir();
  fs.mkdirSync(dataDir, { recursive: true });
  fs.mkdirSync(logDir, { recursive: true });

  const scriptPath = getApiScriptPath();
  if (!fs.existsSync(scriptPath)) {
    console.error("API script not found:", scriptPath);
    return Promise.reject(new Error("API script not found"));
  }

  const pythonPath = getPythonPath();
  // Homebrew OpenSlide: /opt/homebrew/opt/openslide/lib (Apple Silicon) or /usr/local/opt/openslide/lib (Intel)
  const homebrewLibPaths = [
    "/opt/homebrew/opt/openslide/lib",
    "/usr/local/opt/openslide/lib",
    "/opt/homebrew/lib",
    "/usr/local/lib",
  ].filter((p) => fs.existsSync(p));
  const dyldPath = [...homebrewLibPaths, process.env.DYLD_LIBRARY_PATH].filter(Boolean).join(path.delimiter);
  const spawnEnv = { ...process.env };
  if (dyldPath) spawnEnv.DYLD_LIBRARY_PATH = dyldPath;

  apiProcess = spawn(pythonPath, [scriptPath, "--port", String(port), "--data-dir", dataDir, "--log-dir", logDir], {
    cwd: path.dirname(scriptPath),
    stdio: "pipe",
    env: spawnEnv,
  });

  apiProcess.on("error", (err) => {
    console.error("API process error:", err);
  });

  apiProcess.stderr?.on("data", (data) => {
    console.error("API stderr:", data.toString());
  });

  return waitForHealth(port, host).then(() => {
    return { port, host };
  });
}

function stopApi() {
  if (apiProcess) {
    apiProcess.kill();
    apiProcess = null;
  }
}

function createWindow(apiReady) {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadFile("index.html");
  mainWindow.webContents.on("did-finish-load", () => {
    mainWindow.webContents.send("api-ready", apiReady);
  });
}

function recursivelyFindWsiFiles(dir) {
  const results = [];
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return results;

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...recursivelyFindWsiFiles(fullPath));
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase();
      if (WSI_EXTENSIONS.has(ext)) {
        results.push(fullPath);
      }
    }
  }
  return results;
}

app.whenReady().then(async () => {
  let apiReady = { port: DEFAULT_PORT, host: "127.0.0.1" };
  try {
    apiReady = await startApi();
  } catch (err) {
    console.error("Failed to start API:", err);
  }
  createWindow(apiReady);
});

app.on("window-all-closed", () => {
  stopApi();
  app.quit();
});

app.on("before-quit", () => {
  stopApi();
});

ipcMain.handle("get-config", () => loadConfig());
ipcMain.handle("set-config", (_, config) => {
  saveConfig(config);
  return loadConfig();
});

ipcMain.handle("get-dropped-file-paths", (_, pathStrings) => {
  if (!Array.isArray(pathStrings)) return [];
  return pathStrings.filter((p) => {
    if (typeof p !== "string") return false;
    return WSI_EXTENSIONS.has(path.extname(p).toLowerCase());
  });
});

ipcMain.handle("select-folder", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  });
  if (result.canceled || !result.filePaths.length) return [];
  return recursivelyFindWsiFiles(result.filePaths[0]);
});

ipcMain.handle("select-directory", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  });
  if (result.canceled || !result.filePaths.length) return null;
  return result.filePaths[0];
});

ipcMain.handle("select-files", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile", "multiSelections"],
    filters: [
      { name: "Slides & images", extensions: ["svs", "tif", "tiff", "png"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (result.canceled || !result.filePaths.length) return [];
  return result.filePaths.filter((p) =>
    WSI_EXTENSIONS.has(path.extname(p).toLowerCase())
  );
});
