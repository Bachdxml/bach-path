const modelsDeployPathsEl = document.getElementById("models-deploy-paths");
const modelsApp = window.BachPath || null;
const modelsSlidesApi = modelsApp?.services?.slidesApi || window.slidesApi;

function waitForModelsApiReady() {
  if (typeof modelsApp?.whenApiReady === "function") {
    return modelsApp.whenApiReady();
  }
  return Promise.resolve();
}

async function loadDeployInfo() {
  if (!modelsDeployPathsEl) return;
  await waitForModelsApiReady();
  try {
    const info = await modelsSlidesApi.getTrainingInfo();
    modelsDeployPathsEl.textContent =
      info.models_dir?.resolved || info.models_dir?.repo_relative || "wsi-fungal-segmentation/models";
  } catch (e) {
    modelsDeployPathsEl.textContent = "Could not load models folder";
  }
}

window.loadDeployInfo = loadDeployInfo;
if (modelsApp?.registerFeature) {
  modelsApp.registerFeature("models", {
    loadDeployInfo,
  });
}

window.addEventListener("bach-path-authenticated", loadDeployInfo);
