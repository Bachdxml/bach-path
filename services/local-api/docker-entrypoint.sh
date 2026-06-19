#!/usr/bin/env bash
set -euo pipefail
# (entrypoint copied verbatim into the image; see services/local-api/Dockerfile)

# Start the local API inside the container.
#
# Bind to all interfaces *inside* the container so the published port reaches
# it. The host only publishes this port on 127.0.0.1 (docker -p
# 127.0.0.1:8765:8765 / compose), so the API is never reachable on a
# non-loopback host interface by default (M9 / DoD #14). app.cli only accepts a
# 0.0.0.0 bind when APP_CONTAINER is set (baked into this image).
exec python -m app.cli \
  --host "${APP_BIND_HOST:-0.0.0.0}" \
  --port "${APP_PORT:-8765}" \
  --data-dir "${APP_DATA_DIR:-/data}" \
  --log-dir "${APP_LOG_DIR:-/logs}" \
  --log-level "${APP_CLI_LOG_LEVEL:-info}"
