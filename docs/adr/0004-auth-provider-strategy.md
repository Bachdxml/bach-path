# ADR 0004: Auth Provider Strategy by Deployment Mode

**Status:** Accepted

## Context

Bach Path supports three deployment modes through `APP_DEPLOYMENT_MODE`:

- `local`
- `hybrid`
- `cloud`

The current settings and request-auth behavior already vary by mode:

- Local mode is optimized for a single workstation and can run without an API key.
- Hybrid mode requires `APP_API_KEY`, `APP_REMOTE_API_BASE_URL`, and `APP_REMOTE_AUTH_PROVIDER_URL`.
- Cloud mode requires all hybrid settings plus `APP_REMOTE_STORAGE_URL`.
- The API only enforces authentication when `APP_API_KEY` is configured.
- When query-string authentication is enabled, it is only intended for the local desktop-launched API process; hybrid and cloud explicitly disable it.

The system also treats the auth provider as part of the remote trust boundary in hybrid and cloud deployments. That means auth provider selection must be driven by deployment mode, not inferred implicitly by client behavior.

## Decision

We will use a deployment-mode-driven auth provider strategy:

1. **Local mode**
   - Do not require a remote auth provider.
   - Allow the API to run without `APP_API_KEY`.
   - Preserve optional local API-key protection for cases where an operator wants to enable it.
   - Keep query-string API-key support limited to the local desktop-launched API process.

2. **Hybrid mode**
   - Require a remote auth provider URL via `APP_REMOTE_AUTH_PROVIDER_URL`.
   - Require `APP_API_KEY` so the API remains protected at the service boundary.
   - Disallow query-string API-key use.
   - Treat the auth provider as a trusted internal dependency reachable over a controlled network boundary.

3. **Cloud mode**
   - Require a remote auth provider URL via `APP_REMOTE_AUTH_PROVIDER_URL`.
   - Require `APP_API_KEY`.
   - Disallow query-string API-key use.
   - Pair auth-provider usage with remote storage and the rest of the cloud trust boundary.

In all modes, the auth provider configuration must be explicit and validated at startup through the deployment profile checks.

## Consequences

### Positive

- Local development and offline use stay simple and low-friction.
- Hybrid and cloud deployments get explicit identity and access-control configuration.
- The deployment profile makes security expectations visible in configuration instead of hidden in code paths.
- Startup validation fails fast when a remote deployment is missing required auth settings.

### Tradeoffs

- Operators must supply more configuration in hybrid and cloud modes.
- Local mode remains more permissive, which is appropriate for a single trusted workstation but less strict than remote environments.
- The API-key model remains a coarse service gate rather than a full identity system on its own.

### Security impact

- Remote deployments avoid accidental anonymous access.
- Query-string API keys remain confined to the local launcher scenario and are not permitted in hybrid or cloud.
- Auth-provider behavior stays aligned with the deployment trust boundary.

## Alternatives Considered

### 1. Require a remote auth provider in all modes

Rejected because it would make local development and offline operation unnecessarily complex and would not match the current local-first product model.

### 2. Use only API keys everywhere

Rejected because API keys alone do not express the stronger identity and trust requirements needed for hybrid and cloud deployments.

### 3. Infer auth-provider behavior dynamically from network reachability

Rejected because deployment intent should be explicit. `APP_DEPLOYMENT_MODE` is the source of truth for whether the system is running locally, in a managed hybrid environment, or in the cloud.

### 4. Allow query-string API keys in all modes

Rejected because query-string credentials are harder to protect and are only acceptable in the tightly controlled local launcher path.

## Notes

This ADR documents the current strategy and supports future expansion if the auth provider becomes a first-class user identity integration rather than a deployment-bound dependency.
