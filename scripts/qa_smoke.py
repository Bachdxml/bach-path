#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "services" / "local-api"
API_VENV_PY = API_DIR / ".venv" / "bin" / "python"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or ROOT, env=env, check=True)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(port: int, timeout_sec: float = 20.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return
        except URLError:
            time.sleep(0.2)
    raise RuntimeError(f"API health check timed out on {url}")


def post_json(url: str, payload: dict[str, object]) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url=url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=body,
    )
    try:
        with urlopen(req, timeout=20.0) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        code = getattr(exc, "code", 0)
        raw = getattr(exc, "read", lambda: b"")()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(exc)
        return int(code or 0), text


def js_syntax_checks() -> None:
    files = [
        ROOT / "apps" / "desktop" / "index.js",
        ROOT / "apps" / "desktop" / "preload.js",
        ROOT / "apps" / "desktop" / "js" / "api.js",
        ROOT / "apps" / "desktop" / "js" / "app.js",
        ROOT / "apps" / "desktop" / "js" / "gallery.js",
        ROOT / "apps" / "desktop" / "js" / "import.js",
        ROOT / "apps" / "desktop" / "js" / "models-tab.js",
        ROOT / "apps" / "desktop" / "js" / "viewer.js",
    ]
    for file in files:
        run(["node", "--check", str(file)])


def python_compile_checks() -> None:
    files = [
        API_DIR / "run_api.py",
        API_DIR / "app" / "api" / "routes" / "slides.py",
        API_DIR / "app" / "main.py",
        API_DIR / "app" / "cli.py",
    ]
    run([str(API_VENV_PY), "-m", "py_compile", *[str(f) for f in files]])


def import_collision_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="bp-qa-") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        data_dir = root / "appdata"
        log_dir = root / "logs"
        slides_dir = data_dir / "slides"
        input_dir.mkdir(parents=True, exist_ok=True)
        slides_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        for i in range(30):
            p = input_dir / f"mask_{i:03d}.png"
            Image.new("RGB", (32, 32), (i % 255, 10, 10)).save(p)

        # Simulate stale/orphan files that previously caused collision-based import failures.
        for i in range(11, 31):
            Image.new("RGB", (8, 8), (0, 0, 0)).save(slides_dir / f"{i}.png")

        port = free_port()
        env = os.environ.copy()
        env["APP_DATA_DIR"] = str(data_dir)
        env["APP_LOG_DIR"] = str(log_dir)
        env["APP_IMPORT_ALLOWED_ROOTS"] = str(input_dir)
        env["PYTHONPATH"] = f"{API_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

        proc = subprocess.Popen(
            [
                str(API_VENV_PY),
                "-m",
                "app.cli",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--data-dir",
                str(data_dir),
                "--log-dir",
                str(log_dir),
            ],
            cwd=str(API_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_for_health(port)
            failures: list[tuple[str, int, str]] = []
            for file in sorted(input_dir.glob("*.png")):
                status, text = post_json(
                    f"http://127.0.0.1:{port}/slides/import",
                    {"file_path": str(file.resolve())},
                )
                if status != 200:
                    failures.append((file.name, status, text))

            if failures:
                print("Import failures detected:")
                for name, status, text in failures[:10]:
                    print(f"- {name}: status={status} body={text}")
                raise RuntimeError(f"Bulk import smoke failed with {len(failures)} failure(s)")

            print("Import collision smoke passed (30/30 imported).")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def main() -> int:
    if not API_VENV_PY.exists():
        print(f"Missing API venv python at {API_VENV_PY}", file=sys.stderr)
        return 1

    print("== QA Smoke: JS syntax ==")
    js_syntax_checks()
    print("== QA Smoke: Python compile ==")
    python_compile_checks()
    print("== QA Smoke: Bulk import collision scenario ==")
    import_collision_smoke()
    print("All QA smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
