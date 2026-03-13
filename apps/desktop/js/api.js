let apiBase = "http://127.0.0.1:8765";

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

function getApiBase() {
  return apiBase;
}

async function listSlides() {
  const res = await fetch(`${apiBase}/slides`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function importSlide(filePath) {
  const res = await fetch(`${apiBase}/slides/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_path: filePath }),
  });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

function getThumbnailUrl(slideId, size = 200) {
  return `${apiBase}/slides/${slideId}/thumbnail?size=${size}`;
}

async function getSlideMetadata(slideId) {
  const res = await fetch(`${apiBase}/slides/${slideId}/metadata`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

function getTileUrl(slideId, level, x, y) {
  return `${apiBase}/slides/${slideId}/tiles/${level}/${x}/${y}.jpg`;
}

async function runInference(slideId) {
  const res = await fetch(`${apiBase}/inference/slides/${slideId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function getInferenceRun(runId) {
  const res = await fetch(`${apiBase}/inference/runs/${runId}`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function getInferenceRegions(runId) {
  const res = await fetch(`${apiBase}/inference/runs/${runId}/regions`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function getSlideInferenceRuns(slideId) {
  const res = await fetch(`${apiBase}/inference/slides/${slideId}/runs`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function startTraining(exportRoot) {
  const res = await fetch(`${apiBase}/training/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ export_root: exportRoot }),
  });
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

async function getTrainingStatus() {
  const res = await fetch(`${apiBase}/training/status`);
  if (!res.ok) throw new Error(await parseErrorResponse(res));
  return res.json();
}

window.slidesApi = {
  setApiBase,
  getApiBase,
  listSlides,
  importSlide,
  getThumbnailUrl,
  getSlideMetadata,
  getTileUrl,
  runInference,
  getInferenceRun,
  getInferenceRegions,
  getSlideInferenceRuns,
  startTraining,
  getTrainingStatus,
};
