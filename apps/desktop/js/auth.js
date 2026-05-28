document.addEventListener("DOMContentLoaded", () => {
  const app = window.BachPath || null;
  const slidesApi = app?.services?.slidesApi || window.slidesApi;
  const loginScreen = document.getElementById("login-screen");
  const appShell = document.getElementById("app-shell");
  const form = document.getElementById("login-form");
  const usernameInput = document.getElementById("login-username");
  const passwordInput = document.getElementById("login-password");
  const loginButton = document.getElementById("btn-login");
  const status = document.getElementById("login-status");
  const accountUser = document.getElementById("account-current-user");

  function setStatus(message, isError = false) {
    if (!status) return;
    status.textContent = message;
    status.classList.toggle("login-status--error", isError);
  }

  function revealApp(user) {
    loginScreen.hidden = true;
    appShell.hidden = false;
    if (accountUser && user) {
      accountUser.textContent = `${user.username} (${user.role})`;
    }
    if (app?.registerService) {
      app.registerService("currentUser", user);
    }
    if (app?.emit) {
      app.emit("bach-path-authenticated", { user });
    } else {
      window.dispatchEvent(new CustomEvent("bach-path-authenticated", { detail: { user } }));
    }
  }

  function revealLogin() {
    slidesApi.clearSession?.();
    appShell.hidden = true;
    loginScreen.hidden = false;
    form?.reset();
    setStatus("");
    usernameInput?.focus();
    window.dispatchEvent(new CustomEvent("bach-path-signed-out"));
  }

  async function waitForApi() {
    if (typeof app?.whenApiReady === "function") {
      await app.whenApiReady();
    }
  }

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const username = usernameInput?.value.trim() || "";
    const password = passwordInput?.value || "";
    if (!username || !password) {
      setStatus("Enter your username and password.", true);
      return;
    }

    loginButton.disabled = true;
    setStatus("Signing in...");
    try {
      await waitForApi();
      const result = await slidesApi.login(username, password);
      passwordInput.value = "";
      setStatus("");
      revealApp(result.user);
    } catch (err) {
      setStatus(err?.message || "Sign in failed.", true);
      passwordInput?.focus();
    } finally {
      loginButton.disabled = false;
    }
  });

  window.addEventListener("bach-path-sign-out", revealLogin);
});
