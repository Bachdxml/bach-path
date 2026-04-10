let apiBase = "http://127.0.0.1:8765";
let apiKey = null;

async function parseErrorResponse(res) {
  const text = await res.text();
  try {
    const data = JSON.parse(text);
    if (data?.error?.message) return data.error.message;
  } catch (_) {}
  return text;
}

function setApiBase(base) {
  apiBase = base.replace(/\/$/, "");
}

function setApiKey(key) {
  apiKey = typeof key === "string" && key.trim() ? key.trim() : null;
}

function getApiBase() {
  return apiBase;
}

function buildApiUrl(path) {
  return new URL(path, `${apiBase}/`);
}

function withApiKeyQuery(url) {
  const nextUrl = typeof url === "string" ? new URL(url) : new URL(url.toString());
  if (apiKey) {
    nextUrl.searchParams.set("api_key", apiKey);
  }
  return nextUrl.toString();
}

function withApiHeaders(headers = {}) {
  const nextHeaders = { ...headers };
  if (apiKey) {
    nextHeaders["x-api-key"] = apiKey;
  }
  return nextHeaders;
}

async function apiFetch(path, options = {}) {
  const { headers, signal, ...rest } = options;
  return fetch(`${apiBase}${path}`, {
    ...rest,
    signal,
    headers: withApiHeaders(headers),
  });
}

async function apiFetchWithFallback(paths, options = {}) {
  let lastResponse = null;
  for (const path of paths) {
    const res = await apiFetch(path, options);
    if (res.ok || res.status !== 404) {
      return res;
    }
    lastResponse = res;
  }
  return lastResponse;
}

async function listSlides() {
  const res = await apiFetch("/slides");
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function importSlide(filePath) {
  const res = await apiFetch("/slides/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_path: filePath }),
  });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function importCollection(filePaths, title = null, sourceType = null, signal = null) {
  const payload = {
    file_paths: filePaths || [],
  };
  if (typeof title === "string" && title.trim()) payload.title = title.trim();
  if (typeof sourceType === "string" && sourceType.trim()) payload.source_type = sourceType.trim();
  const res = await apiFetchWithFallback(["/slides/import-collection", "/slides/import-collections"], {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function renameImportCollection(collectionId, title) {
  const payload = JSON.stringify({ title });
  const attempts = [
    { method: "POST", path: `/import-collections/${collectionId}/rename` },
    { method: "POST", path: `/slides/import-collections/${collectionId}/rename` },
    { method: "PATCH", path: `/import-collections/${collectionId}` },
    { method: "PATCH", path: `/slides/import-collections/${collectionId}` },
    { method: "PATCH", path: `/slides/import-collection/${collectionId}` },
  ];
  let res = null;
  for (const attempt of attempts) {
    try {
      const nextRes = await apiFetch(attempt.path, {
        method: attempt.method,
        headers: { "Content-Type": "application/json" },
        body: payload,
      });
      if (nextRes.ok || nextRes.status !== 404) {
        res = nextRes;
        break;
      }
      res = nextRes;
    } catch (_) {
      // Continue to next route/method combo if this attempt fails at transport/CORS level.
    }
  }
  if (!res) throw new Error("Rename request could not reach the API");
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

function getThumbnailUrl(slideId, size = 200) {
  const url = buildApiUrl(`/slides/${slideId}/thumbnail`);
  url.searchParams.set("size", String(size));
  return withApiKeyQuery(url);
}

async function getSlideMetadata(slideId) {
  const res = await apiFetch(`/slides/${slideId}/metadata`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

function getTileUrl(slideId, level, x, y) {
  return withApiKeyQuery(buildApiUrl(`/slides/${slideId}/tiles/${level}/${x}/${y}.jpg`));
}

function getDeepZoomDziUrl(slideId) {
  return withApiKeyQuery(buildApiUrl(`/slides/${slideId}/deepzoom.dzi`));
}

function getDeepZoomTileUrl(slideId, level, x, y, format = "jpg") {
  return withApiKeyQuery(buildApiUrl(`/slides/${slideId}/slide_files/${level}/${x}_${y}.${format}`));
}

async function runInference(slideId, modelFile = null, threshold = null) {
  const payload = {};
  if (modelFile) payload.model_file = modelFile;
  if (Number.isFinite(threshold)) payload.threshold = threshold;
  const res = await apiFetch(`/inference/slides/${slideId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function runBatchInference(slideIds, modelFile = null, threshold = null) {
  const payload = { slide_ids: slideIds || [] };
  if (modelFile) payload.model_file = modelFile;
  if (Number.isFinite(threshold)) payload.threshold = threshold;
  const res = await apiFetch("/inference/slides/batch-run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function runFolderInference(folderKey, modelFile = null, threshold = null) {
  const payload = { folder_key: folderKey };
  if (modelFile) payload.model_file = modelFile;
  if (Number.isFinite(threshold)) payload.threshold = threshold;
  const res = await apiFetch("/inference/folders/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function getInferenceRun(runId) {
  const res = await apiFetch(`/inference/runs/${runId}`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function getInferenceRegions(runId) {
  const res = await apiFetch(`/inference/runs/${runId}/regions`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function getSlideInferenceRuns(slideId) {
  const res = await apiFetch(`/inference/slides/${slideId}/runs`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function listInferenceModels() {
  const res = await apiFetch("/inference/models");
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function getTrainingInfo() {
  const res = await apiFetch("/training/info");
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function healthCheck() {
  const res = await apiFetch("/health", { method: "GET" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function deleteSlide(slideId) {
  const res = await apiFetch(`/slides/${slideId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

const slidesApi = {
  setApiBase,
  setApiKey,
  getApiBase,
  healthCheck,
  listSlides,
  importSlide,
  importCollection,
  renameImportCollection,
  deleteSlide,
  getThumbnailUrl,
  getSlideMetadata,
  getTileUrl,
  getDeepZoomDziUrl,
  getDeepZoomTileUrl,
  runInference,
  runBatchInference,
  runFolderInference,
  getInferenceRun,
  getInferenceRegions,
  getSlideInferenceRuns,
  listInferenceModels,
  getTrainingInfo,
};

if (window.BachPath?.registerService) {
  window.BachPath.registerService("slidesApi", slidesApi);
}

// Backward-compatible alias used by existing feature modules.
window.slidesApi = slidesApi;
