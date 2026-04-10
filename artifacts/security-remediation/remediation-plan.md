# Remediation Plan

## Priority Order
1. Config boundary hardening (`settings.py`) for remote deployment URLs.
2. Inference env safety hardening (`inference.py`) for checkpoint path validation.
3. Add negative tests for both paths.

## Subsystem Grouping
- Config subsystem:
  - `services/local-api/app/settings.py`
  - `services/local-api/tests/test_settings_profiles.py`
- Inference subsystem:
  - `services/local-api/app/api/routes/inference.py`
  - `services/local-api/tests/test_inference_limits.py`

## Rollback Notes
- If remote URL validation blocks existing deployments unexpectedly, rollback by removing `_validate_remote_url` calls while keeping tests for documentation.
- If checkpoint validation breaks startup automation, rollback env-check branch in `_resolve_model_checkpoint` and restore prior behavior.
