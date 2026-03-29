from __future__ import annotations
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

def configure_logging(log_dir: Path, log_level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    root = logging.getLogger()
    resolved_level = getattr(logging, str(log_level).upper(), logging.INFO)
    root.setLevel(resolved_level)
    if getattr(root, "_app_logging_configured", False):
        return

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    fh = RotatingFileHandler(str(log_path), maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    root._app_logging_configured = True  # type: ignore[attr-defined]
