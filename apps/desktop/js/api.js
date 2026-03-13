let apiBase = "http://127.0.0.1:8765";

function setApiBase(base) {
  apiBase = base.replace(/\/$/, "");
}

function getApiBase() {
  return apiBase;
}

async function listSlides() {
  const res = await fetch(`${apiBase}/slides`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function importSlide(filePath) {
  const res = await fetch(`${apiBase}/slides/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_path: filePath }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }
  return res.json();
}

function getThumbnailUrl(slideId, size = 200) {
  return `${apiBase}/slides/${slideId}/thumbnail?size=${size}`;
}

async function getSlideMetadata(slideId) {
  const res = await fetch(`${apiBase}/slides/${slideId}/metadata`);
  if (!res.ok) throw new Error(await res.text());
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
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function getInferenceRun(runId) {
  const res = await fetch(`${apiBase}/inference/runs/${runId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function getInferenceRegions(runId) {
  const res = await fetch(`${apiBase}/inference/runs/${runId}/regions`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function getSlideInferenceRuns(slideId) {
  const res = await fetch(`${apiBase}/inference/slides/${slideId}/runs`);
  if (!res.ok) throw new Error(await res.text());
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
};
