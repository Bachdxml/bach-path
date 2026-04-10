# Raw Findings

- F1: `services/local-api/app/settings.py` accepted malformed or credential-embedded remote URLs for hybrid/cloud profile fields.
- F2: `services/local-api/app/api/routes/inference.py` accepted `INFERENCE_CHECKPOINT` env path without existence/type validation.
- F3: No issue confirmed from queue abstraction contract after test validation (false positive concern about lifecycle semantics; tests pass for current intent).
