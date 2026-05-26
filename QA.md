# QA Smoke Test

Run this before merging `application` into `main`:

```bash
python3.13 scripts/qa_smoke.py
```

What it validates:

1. JavaScript syntax checks for Electron entry and renderer files.
2. Python compile checks for local API entrypoints and import route.
3. Phase 1 sprint board compliance (`done` tickets must include date and PR/commit) when local board exists.
4. End-to-end API grouped import plus collision robustness checks, including collection rename verification.
