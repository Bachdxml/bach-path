import argparse
import getpass
import logging
import os
import signal
import sys
from pathlib import Path

USER_ROLE_CHOICES = ("admin", "pathologist", "technician", "viewer")

# Argument Parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Pathology API Service")
    subparsers = parser.add_subparsers(dest="command")

    create_user = subparsers.add_parser(
        "create-user",
        help="Create a provisioned user account",
    )
    create_user.add_argument("--data-dir", type=str, required=True, help="Directory for persistent application data")
    create_user.add_argument("--username", type=str, required=True, help="Username for the new account")
    create_user.add_argument(
        "--role",
        type=str,
        default="viewer",
        choices=USER_ROLE_CHOICES,
        help="Role for the new account",
    )
    create_user.add_argument(
        "--inactive",
        action="store_true",
        help="Create the account disabled",
    )

    serve = subparsers.add_parser("serve", help="Run the local API service")

    serve.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to bind to (localhost only)"
    )

    serve.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    serve.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory for persistent application data"
    )

    serve.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory for logs"
    )

    serve.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log verbosity level"
    )

    if len(sys.argv) > 1 and sys.argv[1] not in {"create-user", "serve", "-h", "--help"}:
        sys.argv.insert(1, "serve")
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


def configure_data_environment(data_dir: Path, log_dir: Path | None = None, log_level: str = "INFO") -> None:
    os.environ["APP_DATA_DIR"] = str(data_dir)
    if log_dir is not None:
        os.environ["APP_LOG_DIR"] = str(log_dir)
    os.environ["APP_LOG_LEVEL"] = log_level.upper()


def prompt_password() -> str:
    password = getpass.getpass("Password: ")
    if not password:
        print("Password cannot be empty.")
        sys.exit(1)
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords do not match.")
        sys.exit(1)
    return password


def create_user_command(args: argparse.Namespace) -> None:
    try:
        from sqlalchemy.exc import IntegrityError

        from app.auth.passwords import hash_password
        from app.db.init import init_database
        from app.db.session import make_engine, make_session_factory
        from app.models.user import User
        from app.settings import load_settings
    except ModuleNotFoundError as exc:
        print(
            f"Missing local API dependency: {exc.name}. "
            "Activate services/local-api/.venv and run `pip install -r requirements.txt`, "
            "then retry create-user."
        )
        sys.exit(1)

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(data_dir, os.W_OK):
        print("Data directory is not writable.")
        sys.exit(1)

    configure_data_environment(data_dir)
    settings = load_settings()
    init_database(settings)

    username = args.username.strip()
    if not username:
        print("Username cannot be empty.")
        sys.exit(1)

    password = prompt_password()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    with session_factory() as db:
        if db.query(User).filter(User.username == username).one_or_none():
            print(f"User {username!r} already exists.")
            sys.exit(1)
        user = User(
            username=username,
            password_hash=hash_password(password),
            role=args.role,
            is_active=not args.inactive,
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            print(f"User {username!r} already exists.")
            sys.exit(1)

    state = "inactive" if args.inactive else "active"
    print(f"Created {state} {args.role} account for {username!r}.")


# Main Entrypoint
def main() -> None:
    args = parse_args()
    if args.command == "create-user":
        create_user_command(args)
        return
    if args.command not in {None, "serve"}:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

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

    configure_data_environment(data_dir, log_dir, args.log_level)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Host must be loopback (127.0.0.1, localhost, or ::1).")
        sys.exit(1)

    setup_logging(log_dir, args.log_level)
    install_signal_handlers()

    import uvicorn
    from app.main import app as fastapi_app

    uvicorn.run(
        fastapi_app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )

if __name__ == "__main__":
    main()
