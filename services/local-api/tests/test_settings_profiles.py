from __future__ import annotations

from pathlib import Path

import pytest

from app.settings import load_settings


def _set_base_env(monkeypatch: pytest.MonkeyPatch, app_data_dir: Path) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(app_data_dir))
    monkeypatch.setenv("APP_LOG_DIR", str(app_data_dir / "logs"))
    monkeypatch.setenv("APP_IMPORT_ALLOWED_ROOTS", str(app_data_dir.parent / "source-slides"))
    monkeypatch.delenv("APP_API_KEY", raising=False)
    monkeypatch.delenv("APP_ALLOW_QUERY_API_KEY", raising=False)


def test_load_settings_defaults_to_local_profile(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])

    settings = load_settings()

    assert settings.deployment_mode.value == "local"
    assert settings.app_data_dir == app_paths["app_data_dir"].resolve()
    assert settings.log_dir == (app_paths["app_data_dir"] / "logs").resolve()
    assert settings.log_level == "INFO"
    assert settings.slides_dir == app_paths["app_data_dir"].resolve() / "slides"
    assert settings.inference_runs_dir == app_paths["app_data_dir"].resolve() / "inference_runs"
    assert settings.training_runs_dir == app_paths["app_data_dir"].resolve() / "training_runs"
    assert settings.tiles_cache_dir == app_paths["app_data_dir"].resolve() / "tiles_cache"
    assert settings.sqlite_path == app_paths["app_data_dir"].resolve() / "app.db"
    assert settings.inference_tile_batch_size == 4
    assert settings.inference_device == "auto"
    assert settings.inference_level == "auto"
    assert settings.inference_target_tiles == 1500


def test_load_settings_accepts_inference_tile_batch_size_override(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_INFERENCE_TILE_BATCH_SIZE", "4")

    settings = load_settings()

    assert settings.inference_tile_batch_size == 4


def test_load_settings_accepts_inference_device_override(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_INFERENCE_DEVICE", "mps")

    settings = load_settings()

    assert settings.inference_device == "mps"


def test_load_settings_rejects_invalid_inference_device(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_INFERENCE_DEVICE", "quantum")

    with pytest.raises(RuntimeError, match="APP_INFERENCE_DEVICE"):
        load_settings()


def test_load_settings_accepts_inference_level_and_target_tiles(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_INFERENCE_LEVEL", "2")
    monkeypatch.setenv("APP_INFERENCE_TARGET_TILES", "900")

    settings = load_settings()

    assert settings.inference_level == "2"
    assert settings.inference_target_tiles == 900


def test_load_settings_rejects_invalid_inference_level(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_INFERENCE_LEVEL", "-1")

    with pytest.raises(RuntimeError, match="APP_INFERENCE_LEVEL"):
        load_settings()


def test_load_settings_requires_app_data_dir(monkeypatch):
    monkeypatch.delenv("APP_DATA_DIR", raising=False)

    with pytest.raises(RuntimeError, match="APP_DATA_DIR is required"):
        load_settings()


def test_load_settings_rejects_relative_allowed_roots(app_paths, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(app_paths["app_data_dir"]))
    monkeypatch.setenv("APP_IMPORT_ALLOWED_ROOTS", "relative/path")

    with pytest.raises(RuntimeError, match="APP_IMPORT_ALLOWED_ROOTS entries must be absolute paths"):
        load_settings()


def test_invalid_deployment_mode_is_rejected(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "moon")

    with pytest.raises(RuntimeError, match="Invalid APP_DEPLOYMENT_MODE"):
        load_settings()


@pytest.mark.parametrize(
    ("deployment_mode", "missing_env"),
    [
        ("hybrid", "APP_REMOTE_API_BASE_URL"),
        ("cloud", "APP_API_KEY"),
    ],
)
def test_hybrid_and_cloud_profiles_reject_missing_required_env(
    app_paths,
    monkeypatch,
    deployment_mode: str,
    missing_env: str,
):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", deployment_mode)
    monkeypatch.delenv(missing_env, raising=False)

    with pytest.raises(RuntimeError, match=missing_env):
        load_settings()


@pytest.mark.parametrize(
    ("deployment_mode", "extra_env"),
    [
        (
            "hybrid",
            {
                "APP_API_KEY": "hybrid-key",
                "APP_REMOTE_API_BASE_URL": "https://api.example.test",
                "APP_REMOTE_AUTH_PROVIDER_URL": "https://auth.example.test",
            },
        ),
        (
            "cloud",
            {
                "APP_API_KEY": "cloud-key",
                "APP_DATABASE_URL": "postgresql+psycopg://db.example.test/bach_path",
                "APP_COGNITO_ISSUER": "https://cognito-idp.us-east-1.amazonaws.com/pool",
                "APP_COGNITO_AUDIENCE": "client-id",
                "APP_S3_BUCKET": "bach-path-slides",
                "APP_S3_REGION": "us-east-1",
                "APP_REMOTE_API_BASE_URL": "https://api.example.test",
                "APP_REMOTE_AUTH_PROVIDER_URL": "https://auth.example.test",
                "APP_REMOTE_STORAGE_URL": "s3://bucket/path",
                "STRIPE_SECRET_KEY": "sk_test_key",
                "STRIPE_WEBHOOK_SECRET": "whsec_test",
            },
        ),
    ],
)
def test_valid_env_loads_settings_for_each_profile(
    app_paths,
    monkeypatch,
    deployment_mode: str,
    extra_env: dict[str, str],
):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", deployment_mode)
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)

    settings = load_settings()

    assert settings.deployment_mode.value == deployment_mode
    assert settings.app_data_dir == app_paths["app_data_dir"].resolve()


def test_hybrid_rejects_non_http_remote_api_base_url(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "hybrid")
    monkeypatch.setenv("APP_API_KEY", "hybrid-key")
    monkeypatch.setenv("APP_REMOTE_API_BASE_URL", "file:///tmp/private")
    monkeypatch.setenv("APP_REMOTE_AUTH_PROVIDER_URL", "https://auth.example.test")

    with pytest.raises(RuntimeError, match="APP_REMOTE_API_BASE_URL must be an absolute URL with scheme https\\|http"):
        load_settings()


def test_cloud_rejects_remote_url_with_embedded_credentials(app_paths, monkeypatch):
    _set_base_env(monkeypatch, app_paths["app_data_dir"])
    monkeypatch.setenv("APP_DEPLOYMENT_MODE", "cloud")
    monkeypatch.setenv("APP_API_KEY", "cloud-key")
    monkeypatch.setenv("APP_DATABASE_URL", "postgresql+psycopg://db.example.test/bach_path")
    monkeypatch.setenv("APP_COGNITO_ISSUER", "https://cognito-idp.us-east-1.amazonaws.com/pool")
    monkeypatch.setenv("APP_COGNITO_AUDIENCE", "client-id")
    monkeypatch.setenv("APP_S3_BUCKET", "bach-path-slides")
    monkeypatch.setenv("APP_S3_REGION", "us-east-1")
    monkeypatch.setenv("APP_REMOTE_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("APP_REMOTE_AUTH_PROVIDER_URL", "https://user:pass@auth.example.test")
    monkeypatch.setenv("APP_REMOTE_STORAGE_URL", "https://storage.example.test")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_key")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")

    with pytest.raises(RuntimeError, match="APP_REMOTE_AUTH_PROVIDER_URL must not include embedded credentials"):
        load_settings()
