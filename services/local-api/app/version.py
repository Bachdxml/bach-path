# app/version.py
from __future__ import annotations

import os
from pathlib import Path

# Single source of truth for the version is the repo-root VERSION file, which is
# COPYd into the image at build time (see services/local-api/Dockerfile). The
# backend reports the baked value, so it always reflects the source the image was
# built from rather than anything supplied at container-run time (spec M10/M11).

_FALLBACK_VERSION = "unknown"


def _candidate_version_paths() -> list[Path]:
    paths: list[Path] = []
    # In the container APP_PROJECT_ROOT is /app and VERSION lands at /app/VERSION.
    project_root = os.environ.get("APP_PROJECT_ROOT")
    if project_root:
        paths.append(Path(project_root) / "VERSION")
    # Dev/test on the host: walk up from this file to the repo root.
    here = Path(__file__).resolve()
    for parent in here.parents:
        paths.append(parent / "VERSION")
    return paths


def read_backend_version() -> str:
    for candidate in _candidate_version_paths():
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return _FALLBACK_VERSION


def read_git_sha() -> str:
    # Diagnostic only; baked into the image as APP_GIT_SHA at build time and
    # degrades to "unknown" when the SHA was not available (spec M11).
    sha = os.environ.get("APP_GIT_SHA", "").strip()
    return sha or "unknown"
