// Reusable confirm/prompt modals backed by the native <dialog> element.
// These replace window.confirm()/window.prompt(), which block the renderer and
// render the "The page at null says:" chrome inside Electron.
//
// Public API (also exposed on BachPath.features.modals):
//   showConfirm(message, options?) -> Promise<boolean>
//   showPrompt(message, defaultValue?, options?) -> Promise<string|null>
(function setupModals(global) {
  const dialog = document.getElementById("app-modal");
  const titleEl = document.getElementById("app-modal-title");
  const messageEl = document.getElementById("app-modal-message");
  const fieldWrap = document.getElementById("app-modal-field-wrap");
  const inputEl = document.getElementById("app-modal-input");
  const cancelBtn = document.getElementById("app-modal-cancel");
  const confirmBtn = document.getElementById("app-modal-confirm");

  // If the markup or <dialog> support is missing, fall back to native dialogs so
  // callers still function. (showModal is unsupported in very old runtimes.)
  const hasDialog =
    dialog &&
    titleEl &&
    messageEl &&
    fieldWrap &&
    inputEl &&
    cancelBtn &&
    confirmBtn &&
    typeof dialog.showModal === "function";

  let activeResolve = null;
  let activeMode = "confirm"; // "confirm" | "prompt"

  function settle(value) {
    const resolve = activeResolve;
    activeResolve = null;
    if (dialog.open) dialog.close();
    if (resolve) resolve(value);
  }

  function onConfirm() {
    if (!activeResolve) return;
    if (activeMode === "prompt") settle(inputEl.value);
    else settle(true);
  }

  function onCancel() {
    if (!activeResolve) return;
    if (activeMode === "prompt") settle(null);
    else settle(false);
  }

  if (hasDialog) {
    confirmBtn.addEventListener("click", onConfirm);
    cancelBtn.addEventListener("click", onCancel);
    // Esc key / programmatic close => treat as cancel.
    dialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      onCancel();
    });
    // Clicking the backdrop (outside the inner panel) cancels.
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) onCancel();
    });
    inputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        onConfirm();
      }
    });
  }

  function open({ mode, message, title, defaultValue, confirmLabel, cancelLabel }) {
    return new Promise((resolve) => {
      if (!hasDialog) {
        // Graceful degradation to native dialogs.
        if (mode === "prompt") {
          resolve(global.prompt(message, defaultValue ?? ""));
        } else {
          resolve(global.confirm(message));
        }
        return;
      }

      // If a modal is already open, resolve the previous one as cancelled.
      if (activeResolve) settle(activeMode === "prompt" ? null : false);

      activeResolve = resolve;
      activeMode = mode;
      titleEl.textContent = title || (mode === "prompt" ? "Enter a value" : "Confirm");
      messageEl.textContent = message || "";
      confirmBtn.textContent = confirmLabel || "OK";
      cancelBtn.textContent = cancelLabel || "Cancel";

      if (mode === "prompt") {
        fieldWrap.hidden = false;
        inputEl.value = defaultValue == null ? "" : String(defaultValue);
      } else {
        fieldWrap.hidden = true;
        inputEl.value = "";
      }

      dialog.showModal();
      if (mode === "prompt") {
        inputEl.focus();
        inputEl.select();
      } else {
        confirmBtn.focus();
      }
    });
  }

  function showConfirm(message, options = {}) {
    return open({
      mode: "confirm",
      message,
      title: options.title,
      confirmLabel: options.confirmLabel,
      cancelLabel: options.cancelLabel,
    });
  }

  function showPrompt(message, defaultValue = "", options = {}) {
    return open({
      mode: "prompt",
      message,
      defaultValue,
      title: options.title,
      confirmLabel: options.confirmLabel,
      cancelLabel: options.cancelLabel,
    });
  }

  const app = global.BachPath || null;
  if (app?.registerFeature) {
    app.registerFeature("modals", { showConfirm, showPrompt });
  }
  // Expose globally so any module can call without the registry handle.
  global.showConfirm = showConfirm;
  global.showPrompt = showPrompt;
})(window);
