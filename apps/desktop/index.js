const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawnSync } = require("child_process");
const http = require("http");
const crypto = require("crypto");
const { fileURLToPath } = require("url");
const net = require("net");

const APP_DISPLAY_NAME = "Bach Path";
const DEFAULT_PORT = 8765;

// Backend runs as a Docker container instead of a native Python subprocess.
// Pin a specific, tested tag — never :latest — so app and backend ship as a
// known-good pair (M13, "pinning over floating").
const BACKEND_IMAGE = "bachpath/bach-path-api:0.1.0";
const BACKEND_CONTAINER_NAME = "bach-path-backend";
const BACKEND_CONTAINER_PORT = 8765;
const WSI_EXTENSIONS = new Set([".svs", ".tif", ".tiff", ".png"]);
const GENERATED_TILE_DIR_NAMES = new Set(["mastertile", "tiles", "patches"]);
const GENERATED_TILE_CLASS_NAMES = new Set(["high", "medium", "low", "negative", "unclassified", "images", "masks"]);
const TILE_FILENAME_RE = /(?:^|[_-])tile[_-]?x?\d+[_-]y?\d+(?:[_-]|$)/i;
const MAX_DROPPED_PATHS = 2048;
const MAX_FILENAME_LENGTH = 255;

let mainWindow = null;
let apiReadyState = { port: DEFAULT_PORT, host: "127.0.0.1" };

// Raised by startApi() when Docker is missing/not running or the image cannot
// be obtained, so app startup can surface an actionable message (M14).
class BackendUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = "BackendUnavailableError";
    this.userMessage = message;
  }
}

app.setName(APP_DISPLAY_NAME);

function getConfigPath() {
  return path.join(app.getPath("userData"), "config.json");
}

function getAppIndexPath() {
  return path.join(__dirname, "index.html");
}

function isPlainObject(value) {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim() !== "";
}

function parseTrustedFileUrl(urlString) {
  if (!isNonEmptyString(urlString)) return null;
  try {
    const parsed = new URL(urlString);
    if (parsed.protocol !== "file:") return null;
    return path.resolve(fileURLToPath(parsed));
  } catch {
    return null;
  }
}

function isAllowedAppUrl(urlString) {
  const filePath = parseTrustedFileUrl(urlString);
  return filePath === path.resolve(getAppIndexPath());
}

function isTrustedSenderFrame(frame) {
  if (!frame) return false;
  const frameUrl = frame.url || "";
  const frameOrigin = frame.origin || "";
  if (!isAllowedAppUrl(frameUrl)) return false;
  return frameOrigin === "" || frameOrigin === "null" || frameOrigin === "file://";
}

function assertTrustedIpcSender(event) {
  const frame = event.senderFrame;
  const frameUrl = frame?.url || event.sender?.getURL?.() || "";
  const senderIsMainWindow = event.sender === mainWindow?.webContents;
  if (!senderIsMainWindow || !isTrustedSenderFrame({ url: frameUrl, origin: frame?.origin || "" })) {
    throw new Error("Blocked untrusted IPC sender");
  }
}

function handleTrusted(channel, handler) {
  ipcMain.handle(channel, (event, ...args) => {
    assertTrustedIpcSender(event);
    return handler(event, ...args);
  });
}

function normalizeApiHost(value) {
  if (!isNonEmptyString(value)) return "127.0.0.1";
  const raw = value.trim();
  const host = raw.replace(/^\[(.*)\]$/, "$1");
  if (host === "localhost" || host === "127.0.0.1" || host === "::1") {
    return host;
  }
  if (net.isIP(host) && (host.startsWith("127.") || host === "::1")) {
    return host;
  }
  throw new Error("apiHost must be loopback-only (localhost, 127.0.0.1, or ::1)");
}

function normalizeApiKey(value) {
  if (!isNonEmptyString(value)) return undefined;
  return value.trim();
}

function buildApiReadyState(config) {
  const state = {
    port: config.apiPort ?? DEFAULT_PORT,
    host: normalizeApiHost(config.apiHost),
  };
  const apiKey = normalizeApiKey(config.apiKey);
  if (apiKey) state.apiKey = apiKey;
  return state;
}

function normalizeApiReadyPayload(data) {
  if (!isPlainObject(data)) return buildApiReadyState(loadConfig());

  const port =
    Number.isInteger(data.port) ? data.port : Number.isInteger(data.apiPort) ? data.apiPort : DEFAULT_PORT;
  const host = normalizeApiHost(data.host ?? data.apiHost);
  const apiKey = normalizeApiKey(data.apiKey);
  const payload = { port, host };
  if (apiKey) payload.apiKey = apiKey;
  return payload;
}

function validateConfigPayload(config) {
  if (!isPlainObject(config)) {
    throw new Error("Invalid config payload");
  }

  const nextConfig = loadConfig();

  if (Object.prototype.hasOwnProperty.call(config, "apiPort")) {
    const port = config.apiPort;
    if (!Number.isInteger(port) || port < 1024 || port > 65535) {
      throw new Error("apiPort must be an integer between 1024 and 65535");
    }
    nextConfig.apiPort = port;
  }

  if (Object.prototype.hasOwnProperty.call(config, "apiHost")) {
    if (!isNonEmptyString(config.apiHost)) {
      throw new Error("apiHost must be a non-empty string");
    }
    nextConfig.apiHost = normalizeApiHost(config.apiHost);
  }

  if (Object.prototype.hasOwnProperty.call(config, "apiKey")) {
    if (config.apiKey == null || config.apiKey === "") {
      delete nextConfig.apiKey;
    } else if (!isNonEmptyString(config.apiKey)) {
      throw new Error("apiKey must be a non-empty string");
    } else {
      nextConfig.apiKey = config.apiKey.trim();
    }
  }

  return nextConfig;
}

function validateDroppedPaths(pathStrings) {
  if (!Array.isArray(pathStrings)) return [];
  const validPaths = [];

  for (const value of pathStrings.slice(0, MAX_DROPPED_PATHS)) {
    if (!isNonEmptyString(value)) continue;
    if (!path.isAbsolute(value)) continue;
    const resolvedPath = path.resolve(value);
    if (!WSI_EXTENSIONS.has(path.extname(resolvedPath).toLowerCase())) continue;
    if (looksLikeGeneratedTilePath(resolvedPath)) continue;
    if (!fs.existsSync(resolvedPath)) continue;
    validPaths.push(resolvedPath);
  }

  return validPaths;
}

function looksLikeGeneratedTilePath(filePath) {
  const parts = path.resolve(filePath).split(path.sep).map((part) => part.toLowerCase());
  const base = path.basename(filePath, path.extname(filePath)).toLowerCase();
  if (TILE_FILENAME_RE.test(base)) return true;
  return (
    parts.some((part) => GENERATED_TILE_DIR_NAMES.has(part)) &&
    parts.some((part) => GENERATED_TILE_CLASS_NAMES.has(part))
  );
}

function validateCapturePayload(payload) {
  if (!isPlainObject(payload) || !isPlainObject(payload.rect)) {
    return { ok: false, error: "invalid_payload" };
  }

  const { rect } = payload;
  const keys = ["x", "y", "width", "height"];
  if (!keys.every((key) => Number.isFinite(rect[key]) && Number.isInteger(rect[key]))) {
    return { ok: false, error: "invalid_rect" };
  }

  if (rect.x < 0 || rect.y < 0 || rect.width < 1 || rect.height < 1) {
    return { ok: false, error: "invalid_rect" };
  }

  const bounds = mainWindow?.getContentBounds?.();
  if (
    bounds &&
    (rect.x + rect.width > bounds.width || rect.y + rect.height > bounds.height)
  ) {
    return { ok: false, error: "invalid_rect" };
  }

  let defaultFilename = "slide-view.png";
  if (isNonEmptyString(payload.defaultFilename)) {
    const baseName = path.basename(payload.defaultFilename.trim()).slice(0, MAX_FILENAME_LENGTH);
    defaultFilename = baseName.toLowerCase().endsWith(".png") ? baseName : `${baseName}.png`;
  }

  return {
    ok: true,
    rect: {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    },
    defaultFilename,
  };
}

function loadConfig() {
  try {
    const data = fs.readFileSync(getConfigPath(), "utf-8");
    const parsed = JSON.parse(data);
    if (!isPlainObject(parsed)) {
      return { apiPort: DEFAULT_PORT, apiHost: "127.0.0.1" };
    }

    const config = { apiPort: DEFAULT_PORT, apiHost: "127.0.0.1" };
    if (Number.isInteger(parsed.apiPort) && parsed.apiPort >= 1024 && parsed.apiPort <= 65535) {
      config.apiPort = parsed.apiPort;
    }
    try {
      config.apiHost = normalizeApiHost(parsed.apiHost);
    } catch {
      config.apiHost = "127.0.0.1";
    }
    if (typeof parsed.apiKey === "string" && parsed.apiKey.trim() !== "") {
      config.apiKey = parsed.apiKey.trim();
    }
    return config;
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

// Read-only slides inbox bind-mounted into the container at /import (M7).
function getApiImportDir() {
  return path.join(app.getPath("userData"), "import-inbox");
}

// Optional host models folder; when it holds a valid deploy weight it is mounted
// over the baked-in model path (M8).
function getApiModelsDir() {
  return path.join(app.getPath("userData"), "models");
}

function hasValidDeployWeight(dir) {
  try {
    return fs.readdirSync(dir).some((name) => /\.(pth|pt)\.gz$/i.test(name));
  } catch {
    return false;
  }
}

// Collapse subprocess output to a short single line for user-facing messages.
function oneLine(text) {
  if (!text) return "";
  return String(text).replace(/\s+/g, " ").trim().slice(0, 500);
}

function runDocker(args, { timeout = 120000 } = {}) {
  return spawnSync("docker", args, {
    encoding: "utf-8",
    timeout,
    windowsHide: true,
  });
}

// True only when the Docker CLI exists AND the daemon answers (M14).
function dockerAvailable() {
  const result = runDocker(["info", "--format", "{{.ServerVersion}}"], { timeout: 15000 });
  return !result.error && result.status === 0;
}

// Ensure the pinned image is present locally, pulling it once if needed (M13).
function ensureBackendImage() {
  const inspect = runDocker(["image", "inspect", BACKEND_IMAGE], { timeout: 30000 });
  if (!inspect.error && inspect.status === 0) return;
  const pull = runDocker(["pull", BACKEND_IMAGE], { timeout: 600000 });
  if (pull.error || pull.status !== 0) {
    throw new BackendUnavailableError(
      `Could not pull the backend image ${BACKEND_IMAGE}. ` +
        `Check your internet connection and Docker Hub access. ${oneLine(pull.stderr)}`
    );
  }
}

// Remove any stale/previous backend container so a fixed name/port can be
// reused without a conflict (M15).
function removeExistingContainer() {
  runDocker(["rm", "-f", BACKEND_CONTAINER_NAME], { timeout: 30000 });
}

function buildDockerRunArgs({ withGpu, port, host, apiKey, dataDir, logDir, importDir, modelsDir }) {
  const publishHost = host === "::1" ? "[::1]" : host;
  const args = ["run", "-d", "--name", BACKEND_CONTAINER_NAME];
  if (withGpu) args.push("--gpus", "all");
  args.push(
    // Publish on loopback only (M9/DoD #14).
    "-p", `${publishHost}:${port}:${BACKEND_CONTAINER_PORT}`,
    // API key injected at runtime; never baked into the image (M11).
    "-e", `APP_API_KEY=${apiKey}`,
    "-e", "APP_ALLOW_QUERY_API_KEY=true",
    "-e", "APP_CORS_ALLOW_FILE_ORIGIN=true",
    "-e", "APP_INFERENCE_DEVICE=auto",
    // Persistent app-data + logs from the existing host paths (M6).
    "-v", `${dataDir}:/data`,
    "-v", `${logDir}:/logs`,
    // Read-only slides inbox (M7).
    "-v", `${importDir}:/import:ro`
  );
  // Optional models override mount (M8).
  if (modelsDir) {
    args.push("-v", `${modelsDir}:/app/wsi-fungal-segmentation/models`);
  }
  args.push(BACKEND_IMAGE);
  return args;
}

function waitForHealth(port, host = "127.0.0.1", maxAttempts = 30) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const hostForUrl = host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;

    const tryFetch = () => {
      const req = http.get(`http://${hostForUrl}:${port}/health`, (res) => {
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
  const host = normalizeApiHost(config.apiHost);
  const apiKey = normalizeApiKey(config.apiKey) || crypto.randomBytes(32).toString("hex");

  // Docker must be installed and the daemon running (M14).
  if (!dockerAvailable()) {
    return Promise.reject(
      new BackendUnavailableError(
        "Docker Desktop is required and must be running to start the Bach Path backend. " +
          "Start Docker Desktop and relaunch Bach Path."
      )
    );
  }

  const dataDir = getApiDataDir();
  const logDir = getApiLogDir();
  const importDir = getApiImportDir();
  const modelsDir = getApiModelsDir();
  for (const dir of [dataDir, logDir, importDir]) {
    fs.mkdirSync(dir, { recursive: true });
  }
  // Only mount the host models folder when it actually holds a deploy weight,
  // otherwise the empty mount would hide the baked-in model (M8 edge cases).
  const modelsMount = hasValidDeployWeight(modelsDir) ? modelsDir : null;

  try {
    ensureBackendImage();
  } catch (err) {
    return Promise.reject(err);
  }

  // Replace any stale container with the same fixed name (M15).
  removeExistingContainer();

  const baseOpts = { port, host, apiKey, dataDir, logDir, importDir, modelsDir: modelsMount };
  let run = runDocker(buildDockerRunArgs({ ...baseOpts, withGpu: true }));
  if (run.error || run.status !== 0) {
    // No NVIDIA GPU / toolkit missing: retry on CPU so the app still works (M10).
    console.warn(
      "docker run with --gpus all failed; retrying on CPU:",
      oneLine(run.stderr) || oneLine(run.error && run.error.message)
    );
    removeExistingContainer();
    run = runDocker(buildDockerRunArgs({ ...baseOpts, withGpu: false }));
  }
  if (run.error || run.status !== 0) {
    return Promise.reject(
      new BackendUnavailableError(
        "Failed to start the Bach Path backend container. " +
          (oneLine(run.stderr) || oneLine(run.error && run.error.message) || "")
      )
    );
  }

  return waitForHealth(port, host)
    .then(() => buildApiReadyState({ ...config, apiPort: port, apiHost: host, apiKey }))
    .catch((err) => {
      stopApi();
      throw err;
    });
}

function stopApi() {
  // Best-effort stop + remove; never throw from lifecycle hooks (M12).
  try {
    runDocker(["rm", "-f", BACKEND_CONTAINER_NAME], { timeout: 30000 });
  } catch (err) {
    console.error("Failed to stop backend container:", err);
  }
}

function createWindow(apiReady) {
  const appIconPath = path.join(__dirname, "assets", "icons", "icon.png");
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: APP_DISPLAY_NAME,
    icon: fs.existsSync(appIconPath) ? appIconPath : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (!isAllowedAppUrl(url)) {
      event.preventDefault();
    }
  });

  mainWindow.loadFile(path.join(__dirname, "index.html"));
  mainWindow.webContents.on("did-finish-load", () => {
    mainWindow.webContents.send("api-ready", normalizeApiReadyPayload(apiReady));
  });
}

async function recursivelyFindWsiFiles(dir) {
  const results = [];
  const stack = [dir];
  while (stack.length) {
    const nextDir = stack.pop();
    if (!nextDir) continue;
    let entries = [];
    try {
      entries = await fs.promises.readdir(nextDir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const fullPath = path.join(nextDir, entry.name);
      if (entry.isDirectory()) {
        if (GENERATED_TILE_DIR_NAMES.has(entry.name.toLowerCase())) continue;
        stack.push(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (WSI_EXTENSIONS.has(ext) && !looksLikeGeneratedTilePath(fullPath)) {
          results.push(fullPath);
        }
      }
    }
  }
  return results;
}

app.whenReady().then(async () => {
  const appIconPath = path.join(__dirname, "assets", "icons", "icon.png");
  if (process.platform === "darwin" && fs.existsSync(appIconPath) && app.dock?.setIcon) {
    app.dock.setIcon(appIconPath);
  }
  try {
    apiReadyState = await startApi();
  } catch (err) {
    console.error("Failed to start API:", err);
    // Surface a clear, actionable error instead of failing silently; the app
    // still opens rather than crashing (M14/DoD #10).
    const message = (err && err.userMessage) || (err && err.message) || String(err);
    try {
      dialog.showErrorBox(
        `${APP_DISPLAY_NAME} — backend unavailable`,
        `${message}\n\n` +
          "The app will open, but importing slides and running inference will not " +
          "work until the backend is available."
      );
    } catch (dialogErr) {
      console.error("Failed to show backend error dialog:", dialogErr);
    }
  }
  createWindow(apiReadyState);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    stopApi();
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow(apiReadyState);
  }
});

app.on("before-quit", () => {
  stopApi();
});

handleTrusted("get-config", () => loadConfig());
handleTrusted("set-config", (_, config) => {
  const nextConfig = validateConfigPayload(config);
  saveConfig(nextConfig);
  // apiReadyState is owned by startApi(); persisted config changes take effect
  // after a restart (the UI already tells users this for port changes).
  return loadConfig();
});

handleTrusted("get-dropped-file-paths", (_, pathStrings) => validateDroppedPaths(pathStrings));

handleTrusted("select-folder", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  });
  if (result.canceled || !result.filePaths.length) return [];
  return recursivelyFindWsiFiles(result.filePaths[0]);
});

handleTrusted("select-directory", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  });
  if (result.canceled || !result.filePaths.length) return null;
  return result.filePaths[0];
});

handleTrusted("select-files", async () => {
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

handleTrusted("save-viewer-capture", async (_, payload) => {
  if (!mainWindow?.webContents) return { ok: false, error: "no_window" };
  const validation = validateCapturePayload(payload);
  if (!validation.ok) return validation;

  const { rect, defaultFilename } = validation;
  try {
    const image = await mainWindow.webContents.capturePage(rect);
    const result = await dialog.showSaveDialog(mainWindow, {
      defaultPath: defaultFilename,
      filters: [{ name: "PNG", extensions: ["png"] }],
    });
    if (result.canceled || !result.filePath) return { ok: true, canceled: true };
    fs.writeFileSync(result.filePath, image.toPNG());
    return { ok: true, canceled: false, path: result.filePath };
  } catch (e) {
    return { ok: false, canceled: false, error: String(e?.message || e) };
  }
});
