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

## Prerequisites

- macOS, Linux, or Windows
- Node.js 18+ and npm
- Python 3.13 (required for local API dependencies)

Optional but recommended:

- `libvips` for faster Deep Zoom pre-caching (macOS: `brew install vips`)
- OpenSlide runtime dependencies (if not already available in your environment)

## 1) Set Up Desktop App

From repo root:

```bash
cd apps/desktop
npm install
```

## 2) Set Up Local API Environment

From repo root:

```bash
cd services/local-api
python3.13 -m venv .venv
source .venv/bin/activate
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

From repo root:

```bash
cd wsi-fungal-segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
python scripts/export_deploy_weights.py --checkpoint checkpoints/best_model.pth --output models/deploy-fungus.pth.gz
```

Notes:

- Override dataset location with `--export-root /path/to/dataset`.
- Inference model discovery uses deployed gzip checkpoints in `wsi-fungal-segmentation/models`.
- Supported deploy formats for app/API model selection: `.pth.gz`, `.pt.gz`.

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

## QA Smoke Test

From repo root:

```bash
python3 scripts/qa_smoke.py
```

This runs syntax/compile checks and validates import collision behavior.

## Delivery Tracking

- Local-only board file: `PHASE1_SPRINT_BOARD.md` (gitignored)
- When a task is completed, update status/date/reference in the same PR.
