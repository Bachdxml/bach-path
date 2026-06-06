(function bootstrapBachPath(global) {
  const existing = global.BachPath || {};
  const eventBus = existing.events instanceof EventTarget ? existing.events : new EventTarget();
  let apiReadyState = existing.apiReadyState || null;
  let resolveApiReady = null;
  const apiReadyPromise =
    existing.apiReady ||
    new Promise((resolve) => {
      resolveApiReady = resolve;
    });
  if (apiReadyState && resolveApiReady) {
    resolveApiReady(apiReadyState);
  }

  const BachPath = {
    ...existing,
    events: eventBus,
    apiReady: apiReadyPromise,
    apiReadyState,
    services: existing.services || {},
    features: existing.features || {},
    constants: {
      ...(existing.constants || {}),
      events: {
        ...(existing.constants?.events || {}),
        apiReady: "bach-path-api-ready",
        inferenceModelChanged: "inference-model-changed",
        positiveOutlineColorChanged: "positive-outline-color-changed",
      },
    },
    on(eventName, listener) {
      eventBus.addEventListener(eventName, listener);
      return () => eventBus.removeEventListener(eventName, listener);
    },
    emit(eventName, detail) {
      const event = new CustomEvent(eventName, { detail });
      eventBus.dispatchEvent(event);
      global.dispatchEvent(event);
    },
  };

  BachPath.registerService = function registerService(name, service) {
    if (!name) return;
    BachPath.services[name] = service;
  };

  BachPath.registerFeature = function registerFeature(name, feature) {
    if (!name) return;
    BachPath.features[name] = feature;
  };

  BachPath.markApiReady = function markApiReady(detail) {
    if (apiReadyState) return apiReadyState;
    apiReadyState = detail || {};
    BachPath.apiReadyState = apiReadyState;
    if (resolveApiReady) resolveApiReady(apiReadyState);
    BachPath.emit(BachPath.constants.events.apiReady, apiReadyState);
    return apiReadyState;
  };

  BachPath.whenApiReady = function whenApiReady() {
    return apiReadyState ? Promise.resolve(apiReadyState) : apiReadyPromise;
  };

  BachPath.isApiReady = function isApiReady() {
    return Boolean(apiReadyState);
  };

  global.BachPath = BachPath;

  if (!existing.apiReadyListenerRegistered && global.electronAPI?.onApiReady) {
    BachPath.apiReadyListenerRegistered = true;
    global.electronAPI.onApiReady(({ port, host, apiKey }) => {
      const slidesApi = BachPath.services?.slidesApi || global.slidesApi;
      if (slidesApi && port && host) {
        const hostForUrl = host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
        if (typeof slidesApi.setApiBase === "function") {
          slidesApi.setApiBase(`http://${hostForUrl}:${port}`);
        }
        if (typeof slidesApi.setApiKey === "function") {
          slidesApi.setApiKey(apiKey || null);
        }
      }
      BachPath.markApiReady({ port, host, apiKey });
    });
  }
})(window);
