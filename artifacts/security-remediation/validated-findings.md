# Validated Findings

## F1
- id: F1
- category: security
- severity: medium
- confidence: high
- evidence: `services/local-api/app/settings.py` lacked URL parsing/validation for `APP_REMOTE_API_BASE_URL`, `APP_REMOTE_AUTH_PROVIDER_URL`, `APP_REMOTE_STORAGE_URL`.
- decision: confirmed

## F2
- id: F2
- category: bug
- severity: medium
- confidence: high
- evidence: `services/local-api/app/api/routes/inference.py` returned `Path(env_path).resolve()` for `INFERENCE_CHECKPOINT` without checking file existence/type.
- decision: confirmed

## F3
- id: F3
- category: bug
- severity: low
- confidence: medium
- evidence: Initial suspicion about queue lifecycle correctness was tested via `services/local-api/tests/test_queue_abstraction.py`.
- decision: discarded

## F4
- id: F4
- category: bug
- severity: high
- confidence: high
- evidence: `queued`/`running` inference runs could remain orphaned after process restart because in-memory queue state is lost while DB statuses remained non-terminal.
- decision: confirmed
