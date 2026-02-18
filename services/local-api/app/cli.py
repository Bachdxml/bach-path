import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from app.main import app as fastapi_app

# Argument Parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Pathology API Service")

    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to bind to (localhost only)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory for persistent application data"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory for logs"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log verbosity level"
    )

    return parser.parse_args()

# Logging Setup
def setup_logging(log_dir: Path, log_level: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "local-api.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
    )

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Graceful Shutdown
def install_signal_handlers():
    def handle_signal(sig, frame):
        logging.info(f"Received signal {sig}. Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

# Main Entrypoint
def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    log_dir = Path(args.log_dir).resolve()

    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Validate write access (must be INSIDE main)
    if not os.access(data_dir, os.W_OK):
        print("Data directory is not writable.")
        sys.exit(1)

    if not os.access(log_dir, os.W_OK):
        print("Log directory is not writable.")
        sys.exit(1)

    os.environ["APP_DATA_DIR"] = str(data_dir)
    os.environ["APP_LOG_DIR"] = str(log_dir)

    setup_logging(log_dir, args.log_level)
    install_signal_handlers()

    uvicorn.run(
        fastapi_app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )

if __name__ == "__main__":
    main()
