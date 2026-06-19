# Bach Path

Bach Path is a local pathology workflow app with three main capabilities:

1. Import whole-slide images (`.svs`, `.tif`, `.tiff`, `.png`)
2. Run fungal inference on imported slides
3. Train/update segmentation models outside the desktop UI and deploy them for inference

The desktop UI lives in `apps/desktop`, and it starts a local FastAPI backend from `services/local-api`.

## Repository Layout

- `apps/desktop`: Electron desktop application
- `services/local-api`: local FastAPI service used by the desktop app
- `wsi-fungal-segmentation`: model training, evaluation, export, and inference scripts
- `training_data`: gitignored training-data dropzone (recommended dataset root: `training_data/MASTERTILE`)
- `docs`: architecture decisions and roadmap docs (see `docs/roadmap/phase1-roadmap.md`)

## Run with Docker (primary path)

The backend (the `services/local-api` API plus the `wsi-fungal-segmentation`
inference pipeline) ships as a single Linux Docker image, so it runs identically
on any Docker host with no Python, venv, OpenSlide, libvips, or CUDA setup. The
Electron desktop UI still runs natively on the host and talks to the
containerized backend over `http://127.0.0.1:8765`.

### Prerequisites (Docker)

- **Docker Desktop** (Windows 11 + the **WSL2** backend on the lab machine).
- For GPU inference: an **NVIDIA GPU**, a driver new enough for **CUDA 12.4**, and
  the **NVIDIA Container Toolkit** (Docker Desktop's WSL2 GPU support). Without a
  GPU the container still starts and inference falls back to CPU (slower).
- The image is **Linux/amd64**; GPU support is **NVIDIA-only**. Any future
  arm64/Mac image would be CPU-only.

### Run via the Electron desktop app

```powershell
cd apps/desktop
npm install
npm start
```

On launch the desktop app starts the backend container (pulling the pinned image
on first run), waits for `/health`, then loads the window. Quitting the app
stops and removes the container. If Docker Desktop is not installed or not
running, the app shows an actionable error and still opens (the backend just
won't be reachable until Docker is up).

Host folders used by the desktop launch (under the Electron `userData` dir):

- `api-data/` -> container `/data` (slides, `app.db`, `inference_runs`,
  `training_runs`, `tiles_cache`). Point this at a prior native install's
  `api-data` to keep existing slides/runs/database — no migration needed.
- `api-logs/` -> container `/logs`.
- `import-inbox/` -> container `/import` (read-only **slides inbox**; place
  slides here to import them).
- `models/` -> mounted over the baked-in model **only if** it contains a valid
  deploy weight (`.pth.gz` / `.pt.gz`); otherwise the baked-in model is used.

### Run headless via Docker Compose

For non-Electron/server users, the committed `docker-compose.yml` runs the same
image. Provide an API key at runtime (never baked into the image):

```powershell
$env:APP_API_KEY = "<a-strong-secret>"
docker compose up
```

Then reach the API at `http://127.0.0.1:8765/health`. Compose uses
`./docker-data/{app-data,app-logs,import}` on the host; point `app-data` at a
prior native install's `api-data` to reuse existing data. The slides inbox is
`./docker-data/import` (read-only at `/import`). To override the baked-in model,
uncomment the `models` volume in `docker-compose.yml` and drop a `.pth.gz` into
`./docker-data/models`. On a CPU-only host, comment out the `deploy.resources`
GPU block.

### Mount conventions

| Purpose            | Host (Electron)            | Host (compose)            | Container path                          | Mode |
| ------------------ | -------------------------- | ------------------------- | --------------------------------------- | ---- |
| App data           | `userData/api-data`        | `./docker-data/app-data`  | `/data`                                 | rw   |
| Logs               | `userData/api-logs`        | `./docker-data/app-logs`  | `/logs`                                 | rw   |
| Slides inbox       | `userData/import-inbox`    | `./docker-data/import`    | `/import`                               | ro   |
| Models override    | `userData/models`          | `./docker-data/models`    | `/app/wsi-fungal-segmentation/models`   | rw   |

The API publishes on the host **loopback only** (`127.0.0.1:8765`); it is not
reachable on the host's LAN IP by default.

### Build and publish the image

Build from the repo root (the build context needs both source trees):

```powershell
docker build -f services/local-api/Dockerfile -t bachpath/bach-path-api:0.1.0 .
```

Verify the native libs and a CUDA (non-`+cpu`) torch build inside the image:

```powershell
docker run --rm bachpath/bach-path-api:0.1.0 `
  python -c "import openslide, pyvips, torch; print(torch.__version__, torch.version.cuda)"
```

Confirm the GPU is visible when run with `--gpus all` on the lab machine:

```powershell
docker run --rm --gpus all bachpath/bach-path-api:0.1.0 `
  python -c "import torch; print(torch.cuda.is_available())"
```

Tag a semantic version plus `latest` and push to the public Docker Hub repo
(confirm the namespace, e.g. `bachpath/bach-path-api`, before the first push,
and confirm the baked-in model's license permits public redistribution):

```powershell
docker tag bachpath/bach-path-api:0.1.0 bachpath/bach-path-api:latest
docker push bachpath/bach-path-api:0.1.0
docker push bachpath/bach-path-api:latest
```

The desktop app references a **pinned tag** (`bachpath/bach-path-api:0.1.0` in
`apps/desktop/index.js`), never `latest`, so the app and backend ship as a
known-good pair. Bump that constant and the `docker-compose.yml` image when you
publish a new version.

---

## Native (development) setup — fallback

The native Python setup below is no longer the primary path; use it for backend
development without Docker. The desktop app launches the **containerized**
backend, not this native one.

## Prerequisites

- macOS, Linux, or Windows
- Node.js 18+ and npm
- Python 3.12 for the local API dependencies

Optional but recommended:

- `libvips` for faster Deep Zoom pre-caching (macOS: `brew install vips`)
- OpenSlide runtime dependencies (if not already available in your environment)
- An NVIDIA GPU with CUDA for inference. Inference auto-selects CUDA when a
  GPU-enabled PyTorch is installed; on a CPU-only torch build it falls back to
  CPU and whole-slide runs can be very slow (minutes-to-hours). The
  `wsi-fungal-segmentation/requirements.txt` pins CUDA (cu124) wheels by default.
  If `pip` installs a `+cpu` torch build, force the CUDA wheels into the
  inference venv:

  ```powershell
  wsi-fungal-segmentation\.venv\Scripts\python.exe -m pip install --force-reinstall `
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  # then restore the pinned Pillow that --force-reinstall may bump:
  wsi-fungal-segmentation\.venv\Scripts\python.exe -m pip install "Pillow==10.4.0"
  ```

  Verify with:

  ```powershell
  wsi-fungal-segmentation\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

## 1) Set Up Desktop App

From repo root:

```bash
cd apps/desktop
npm install
```

## 2) Set Up Local API Environment

From repo root, use Python 3.12 and follow your OS-specific commands.

The local API currently pins several packages with native wheels (`pydantic-core`,
`orjson`, `openslide-python`, `Pillow`, SQLAlchemy). Use Python 3.12 for this
environment; Python 3.13 may try to build some pinned packages locally or install
incompatible compiled extensions.

macOS/Linux:

```bash
cd services/local-api
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd services/local-api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## 3) Prepare Training Data

Default training dataset root is `training_data/MASTERTILE` (relative to repo root).
If `data.export_root` is unset, training auto-detects in this order:
`training_data/MASTERTILE` -> `training_data` -> `MASTERTILE` (legacy).

Accepted dataset layouts:

- Density format (default): `<slide>/<density>/images/*.png` and `<slide>/<density>/masks/*_mask.png`
- Flat format: `<slide>/images/*.png` and `<slide>/masks/*_mask.png` (use `--flat-format`)

Important validation behavior:

- Training indexing runs in strict mode.
- Every image tile must have a corresponding mask tile.
- If masks are missing or mismatched, training fails early.

### QuPath Export

Use the export script at:

- `wsi-fungal-segmentation/qu-path-scripts/export_tiles.groovy`

Then classify unclassified exports into density folders with:

```bash
cd wsi-fungal-segmentation
python utils.py/Classifytiles.py --export_dir /path/to/exported/slide --apply
```

## 4) Train, Evaluate, and Export Deploy Weights

From repo root.

macOS/Linux:

```bash
cd wsi-fungal-segmentation
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
python scripts/export_deploy_weights.py --checkpoint checkpoints/best_model.pth --output models/deploy-fungus.pth.gz
```

Windows (PowerShell):

```powershell
cd wsi-fungal-segmentation
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
python scripts/export_deploy_weights.py --checkpoint checkpoints/best_model.pth --output models/deploy-fungus.pth.gz
```

Notes:

- Override dataset location with `--export-root /path/to/dataset`.
- Inference model discovery uses deployed gzip checkpoints in `wsi-fungal-segmentation/models`.
- Supported deploy formats for app/API model selection: `.pth.gz`, `.pt.gz`.

### Mask output size and adaptive coarsening

Inference output size is dominated by per-tile segmentation masks, so dense
slides used to fail with `Inference output exceeded server limits`. The pipeline
now coarsens masks at the source to stay within a size budget instead of failing:

- **Degradation ladder.** The inference subprocess walks a fixed ladder and stops
  at the first step that fits the budget: (1) **full** native-resolution masks;
  (2) **downsampled** — masks are progressively halved (factor 2, 4, 8, …) to the
  highest resolution that fits; (3) **dropped** — as a last resort no masks are
  emitted, but every detection box and score is kept. Detection geometry is never
  dropped for size.
- **Degradation status.** Each run records a `mask_degradation_status` of `full`,
  `downsampled` (with a `mask_downsample_factor`), or `dropped`, exposed on the run
  via the API. The viewer shows a reduced-fidelity notice for `downsampled` and
  `dropped` runs; `full` (and legacy runs without the field) show no notice.
- **`APP_MAX_INFERENCE_OUTPUT_BYTES`** (default **10,000,000** = 10 MB, overridable
  via the environment variable) now tunes **how much mask detail is retained
  before coarsening begins, not whether a run succeeds**. The subprocess targets a
  budget of ~90% of this limit as a safety margin; the API keeps the hard-limit
  file-size gate as a safety net. Raising it retains more detail at the cost of
  in-browser mask decode/render time; the 10 MB default balances the two.

## 5) Run the App (Development)

From repo root:

```bash
cd apps/desktop
npm start
```

On startup, Electron launches the local API automatically.

## 6) Build Desktop Distributions

From `apps/desktop`:

- Windows unpacked app:

```bash
npm run dist:win:exe
```

- Windows installer + portable:

```bash
npm run dist:win
```

- Cross-platform package directory:

```bash
npm run pack
```

Build artifacts are written under `apps/desktop/dist/`.

## 7) Use the App

1. Open **Import** and add slides (single files or recursive folder import)
2. Open **Models** and confirm a deploy model is available
3. Open **Gallery** and select a slide
4. Run inference and review overlays in the viewer

## Troubleshooting

- `No deploy model found`
  - Add `.pth.gz` or `.pt.gz` to `wsi-fungal-segmentation/models`, or set `INFERENCE_CHECKPOINT`.
- Repeated import failures
  - Check API logs under `~/Library/Application Support/Bach Path/api-logs` on macOS.
- DeepZoom `libvips` warning
  - Install `vips`; fallback on-demand tiles still work.
- `ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'` or Pillow/OpenSlide native import errors
  - Recreate `services/local-api/.venv` with Python 3.12 and reinstall `services/local-api/requirements.txt`.
