(function bootstrapBachPath(global) {
  const existing = global.BachPath || {};
  const eventBus = existing.events instanceof EventTarget ? existing.events : new EventTarget();

  const BachPath = {
    ...existing,
    events: eventBus,
    services: existing.services || {},
    features: existing.features || {},
    constants: existing.constants || {
      events: {
        inferenceModelChanged: "inference-model-changed",
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

  global.BachPath = BachPath;
})(window);
