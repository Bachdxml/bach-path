# syntax=docker/dockerfile:1
#
# Bach Path local API + WSI fungal-segmentation inference, packaged as a single
# Linux image so it runs identically on any Docker host (M1-M5).
#
# Build from the repo root so both source trees are in the build context:
#   docker build -f services/local-api/Dockerfile -t bachpath/bach-path-api:0.1.0 .

# CUDA 12.4 runtime base, compatible with the cu124 PyTorch wheels (M2).
# The image is Linux/amd64; GPU support is NVIDIA-only (see spec constraints).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# --- Native runtime libraries via the OS package manager (M1) ---
# OpenSlide (openslide-python) and libvips (pyvips) plus the system Python that
# the API and the inference subprocess share.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      libopenslide0 \
      libvips42 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Single shared environment; `python` resolves to the system interpreter so the
# inference subprocess fallbacks find it. No .venv directories in the image (M3).
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# --- Python dependencies (M1) ---
# Install the API deps first, then the inference deps. The inference
# requirements file declares the PyTorch cu124 wheel index, so torch/torchvision
# come from the CUDA wheel index (a non-+cpu build, M1).
COPY services/local-api/requirements.txt /tmp/api-requirements.txt
COPY wsi-fungal-segmentation/requirements.txt /tmp/wsi-requirements.txt
# openslide-python has no prebuilt wheel for this platform, so pip compiles its
# small `openslide._convert` C extension from source. Install the build toolchain
# (gcc + Python headers) for the compile, then purge it in the same layer so the
# runtime image keeps only libopenslide0/libvips42 (installed above).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3-dev \
 && python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /tmp/api-requirements.txt \
 && python -m pip install -r /tmp/wsi-requirements.txt \
 && apt-get purge -y build-essential python3-dev \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# --- Application code ---
WORKDIR /app
COPY services/local-api /app/services/local-api
COPY wsi-fungal-segmentation /app/wsi-fungal-segmentation
# The build context excludes everything under wsi-fungal-segmentation/models
# except deploy-fungus.pth.gz (see .dockerignore), so the baked-in deploy model
# lands at the model-discovery path /app/wsi-fungal-segmentation/models (M4).
# A models bind mount over that directory overrides it without a rebuild (M8).

# --- Container runtime configuration ---
# APP_CONTAINER lets app.cli bind 0.0.0.0 inside the container; the host only
# publishes the port on loopback (M9), so the API stays loopback-only.
# APP_PROJECT_ROOT pins model/script discovery to /app.
# INFERENCE_PYTHON defaults to the container interpreter so the API spawns the
# inference subprocess in this same environment (M3).
ENV APP_CONTAINER=1 \
    APP_PROJECT_ROOT=/app \
    APP_DEPLOYMENT_MODE=local \
    APP_DATA_DIR=/data \
    APP_LOG_DIR=/logs \
    APP_INFERENCE_DEVICE=auto \
    INFERENCE_PYTHON=/usr/bin/python3 \
    PYTHONPATH=/app/wsi-fungal-segmentation:/app/services/local-api

WORKDIR /app/services/local-api

EXPOSE 8765

# Liveness probe against the unauthenticated /health endpoint (DoD #4).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3).status < 300 else 1)" || exit 1

COPY services/local-api/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
