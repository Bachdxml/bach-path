from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def app_paths(tmp_path, monkeypatch):
    app_data_dir = tmp_path / "app-data"
    source_dir = tmp_path / "source-slides"
    app_data_dir.mkdir()
    source_dir.mkdir()

    monkeypatch.setenv("APP_DATA_DIR", str(app_data_dir))
    monkeypatch.setenv("APP_IMPORT_ALLOWED_ROOTS", str(source_dir.resolve()))
    monkeypatch.delenv("APP_API_KEY", raising=False)

    return {
        "app_data_dir": app_data_dir,
        "source_dir": source_dir,
    }
