# Bach Path

Bach Path is a local desktop application for pathology slide workflows:

1. Import whole-slide images (`.svs`, `.tif`, `.tiff`, `.png`)
2. Run fungal inference on imported slides
3. Review results in the built-in slide viewer

The app includes an Electron desktop UI (`apps/desktop`) and a local FastAPI backend (`services/local-api`).

## Repository Layout

- `apps/desktop`: Electron desktop application
- `services/local-api`: Local FastAPI service used by the desktop app
- `wsi-fungal-segmentation`: Training and inference model code
- `MASTERTILE`: Example/exported tile data used for training workflows

## Prerequisites

- macOS, Linux, or Windows
- Node.js 18+ and npm
- Python 3.10+ (project currently uses Python 3.13 in local venvs)

Optional but recommended:

- `libvips` for faster Deep Zoom pre-caching (macOS: `brew install vips`)
- OpenSlide runtime dependencies (if not already present through your environment)

## 1) Set Up the Desktop App

```bash
cd apps/desktop
npm install
```

## 2) Set Up the Local API Environment

From repo root:

```bash
cd services/local-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Train a Model and Export Deploy Weights

From repo root:

```bash
cd wsi-fungal-segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
python scripts/export_deploy_weights.py --checkpoint checkpoints/best_model.pth --output checkpoints/deploy.pth.gz
cp checkpoints/deploy.pth.gz models/deploy.pth.gz
```

Notes:

- The desktop app/API loads deployable model files from `wsi-fungal-segmentation/models`.
- Supported deploy weight suffixes: `.pth.gz` and `.pt.gz`.
- You can keep multiple models in `wsi-fungal-segmentation/models` and select them in the UI.

## 4) Run the Application (Development)

From repo root:

```bash
cd apps/desktop
npm start
```

On startup, Electron launches the local API automatically.

## 5) Use the App

1. Open **Import** tab and add slides (file picker or folder recursive import)
2. Open **Models** tab and confirm your deploy model appears
3. Open **Gallery** and select a slide
4. Run inference and review overlay results in the viewer

## Build Desktop Distributions

From `apps/desktop`:

- Windows unpacked executable:
  - `npm run dist:win:exe`
- Windows installer + portable:
  - `npm run dist:win`
- Cross-platform package dir:
  - `npm run pack`

See `apps/desktop/README.md` for platform-specific packaging notes.

## Troubleshooting

- `No deploy model found`
  - Put a `.pth.gz` or `.pt.gz` file in `wsi-fungal-segmentation/models`.
- Import failures after repeated runs
  - Check API logs under `~/Library/Application Support/Bach Path/api-logs` (macOS).
- DeepZoom warning about `libvips`
  - Install `vips` (optional); app still works with on-demand tiles.

## QA Smoke Test

Before merging:

```bash
python3 scripts/qa_smoke.py
```

This validates syntax/compile checks and a bulk import collision scenario.
It also validates Phase 1 board compliance for completed tickets when the local board file is present.

## Delivery Tracking

- Phase 1 execution board (local-only, gitignored): `PHASE1_SPRINT_BOARD.md`
- Rule: whenever a task is completed, update the board in the same PR (status, date, and PR/commit reference).
