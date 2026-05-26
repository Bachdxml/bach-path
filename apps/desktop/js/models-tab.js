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
    const lines = [];
    lines.push("Training runs outside this application. Use the repo paths below on your training machine.");
    lines.push("");
    for (const note of info.notes || []) {
      lines.push(note);
    }
    lines.push("");
    lines.push("Canonical paths (repo-relative):");
    for (const key of [
      "train_script",
      "config_default",
      "export_deploy_script",
      "models_dir",
      "qupath_export_script",
    ]) {
      const e = info[key];
      if (e?.repo_relative) lines.push(`  ${e.repo_relative}`);
    }
    lines.push("");
    lines.push("Resolved on this machine (API server):");
    for (const key of [
      "train_script",
      "config_default",
      "export_deploy_script",
      "models_dir",
      "qupath_export_script",
    ]) {
      const e = info[key];
      if (e?.resolved) lines.push(`  ${e.resolved}`);
    }
    modelsDeployPathsEl.textContent = lines.join("\n");
  } catch (e) {
    modelsDeployPathsEl.textContent =
      "Could not load deployment paths from the API. " + (e && e.message ? e.message : String(e));
  }
}

window.loadDeployInfo = loadDeployInfo;
if (modelsApp?.registerFeature) {
  modelsApp.registerFeature("models", {
    loadDeployInfo,
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", loadDeployInfo);
} else {
  loadDeployInfo();
}
