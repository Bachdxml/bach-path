# QA Smoke Test

Run this before merging `application` into `main`:

```bash
python3 scripts/qa_smoke.py
```

What it validates:

1. JavaScript syntax checks for Electron entry and renderer files.
2. Python compile checks for local API entrypoints and import route.
3. End-to-end API bulk import with a forced filename-collision scenario.
