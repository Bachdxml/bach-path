# Pathology Viewer (Desktop)

## Build a Windows `.exe`

From this directory:

```bash
npm run dist:win:exe
```

This generates a directly runnable Windows executable at:

- `dist/win-unpacked/Pathology Viewer.exe`

Use that `.exe` to start the app without running `npm start`.

## Optional: Build installer + portable `.exe`

```bash
npm run dist:win
```

This generates Windows executables in `dist/`, including:

- `Pathology Viewer Setup <version>.exe` (installer)
- `Pathology Viewer <version>-portable.exe` (direct runnable app)
