#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
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
PHASE1_BOARD = ROOT / "PHASE1_SPRINT_BOARD.md"


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


def request_json(method: str, url: str, payload: dict[str, object] | None = None) -> tuple[int, object | str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = Request(url=url, method=method, headers=headers, data=data)
    try:
        with urlopen(req, timeout=20.0) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
            try:
                return int(resp.status), json.loads(raw_text)
            except json.JSONDecodeError:
                return int(resp.status), raw_text
    except Exception as exc:
        code = getattr(exc, "code", 0)
        raw = getattr(exc, "read", lambda: b"")()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(exc)
        try:
            return int(code or 0), json.loads(text)
        except json.JSONDecodeError:
            return int(code or 0), text


def get_json(url: str) -> object:
    status, payload = request_json("GET", url)
    if status < 200 or status >= 300:
        raise RuntimeError(f"GET {url} failed with status {status}: {payload}")
    return payload


def _text_haystack(path: str, operation: dict[str, object]) -> str:
    parts: list[str] = [path]
    for key in ("summary", "operationId", "description"):
        value = operation.get(key)
        if isinstance(value, str):
            parts.append(value)
    tags = operation.get("tags")
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags)
    return " ".join(parts).lower()


def _find_openapi_operations(
    openapi: object,
    method: str,
    required_tokens: tuple[str, ...],
    optional_tokens: tuple[str, ...] = (),
) -> list[tuple[str, dict[str, object]]]:
    if not isinstance(openapi, dict):
        raise RuntimeError("OpenAPI document is not a JSON object")
    paths = openapi.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError("OpenAPI document does not contain paths")

    matches: list[tuple[int, str, dict[str, object]]] = []
    for path, operations in paths.items():
        if not isinstance(path, str) or not isinstance(operations, dict):
            continue
        operation = operations.get(method.lower())
        if not isinstance(operation, dict):
            continue
        haystack = _text_haystack(path, operation)
        if any(token not in haystack for token in required_tokens):
            continue
        if optional_tokens and not any(token in haystack for token in optional_tokens):
            continue
        score = sum(token in haystack for token in (*required_tokens, *optional_tokens))
        matches.append((score, path, operation))

    matches.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return [(path, operation) for _, path, operation in matches]


def _replace_path_param(path: str, value: object) -> str:
    return re.sub(r"\{[^/}]+\}", str(value), path, count=1)


def _extract_value(payload: object, keys: tuple[str, ...]) -> object | None:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                value = payload[key]
                if value is not None:
                    return value
        for value in payload.values():
            nested = _extract_value(value, keys)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _extract_value(item, keys)
            if nested is not None:
                return nested
    return None


def _extract_collection_id(payload: object) -> int:
    value = _extract_value(payload, ("collection_id", "collectionId", "id"))
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise RuntimeError(f"Could not find collection id in payload: {payload}")


def _extract_name(payload: object) -> str:
    value = _extract_value(payload, ("collection_name", "name", "title"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise RuntimeError(f"Could not find collection name in payload: {payload}")


def _normalize_slide_rows(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected slides payload to be a JSON object, got {type(payload).__name__}")
    slides = payload.get("slides")
    if not isinstance(slides, list):
        raise RuntimeError(f"Expected slides payload to contain a slides list, got: {payload}")
    rows: list[dict[str, object]] = []
    for item in slides:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _slide_collection_id(row: dict[str, object]) -> int | None:
    value = _extract_value(row, ("collection_id", "collectionId"))
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _import_collection_smoke(port: int, input_dir: Path, slides_dir: Path) -> None:
    batch_files = []
    batch_dir = input_dir / "grouped"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        p = batch_dir / f"group_{i:03d}.png"
        Image.new("RGB", (32, 32), (40 + i, 80, 160)).save(p)
        batch_files.append(p)

    openapi = get_json(f"http://127.0.0.1:{port}/openapi.json")
    import_ops = _find_openapi_operations(openapi, "post", ("import",), ("collection", "batch"))
    if not import_ops:
        raise RuntimeError("Could not find a collection batch import endpoint in OpenAPI")

    collection_name = "QA grouped import smoke"
    import_result = None
    import_path = None
    for path, _operation in import_ops:
        url = f"http://127.0.0.1:{port}{path}"
        payloads = [
            {"file_paths": [str(p.resolve()) for p in batch_files], "collection_name": collection_name},
            {"file_paths": [str(p.resolve()) for p in batch_files], "name": collection_name},
            {"paths": [str(p.resolve()) for p in batch_files], "collection_name": collection_name},
            {"paths": [str(p.resolve()) for p in batch_files], "name": collection_name},
            {"files": [str(p.resolve()) for p in batch_files], "collection_name": collection_name},
            {"files": [str(p.resolve()) for p in batch_files], "name": collection_name},
        ]
        for payload in payloads:
            status, response = request_json("POST", url, payload)
            if status == 200:
                import_result = response
                import_path = path
                break
        if import_result is not None:
            break

    if import_result is None or import_path is None:
        raise RuntimeError(f"Collection batch import smoke failed via endpoints: {import_ops}")

    collection_id = _extract_collection_id(import_result)
    if isinstance(import_result, dict):
        slide_ids = import_result.get("slide_ids")
        if isinstance(slide_ids, list) and len(slide_ids) != 20:
            raise RuntimeError(f"Expected 20 imported slide ids, got {len(slide_ids)}")
    print(f"Batch import created collection id {collection_id} via {import_path}")

    slides_payload = get_json(f"http://127.0.0.1:{port}/slides")
    slide_rows = _normalize_slide_rows(slides_payload)
    batch_slide_rows = [row for row in slide_rows if row.get("original_path") in {p.name for p in batch_files}]
    if len(batch_slide_rows) != 20:
        raise RuntimeError(f"Expected 20 imported slides in GET /slides, found {len(batch_slide_rows)}")

    collection_ids = {cid for cid in (_slide_collection_id(row) for row in batch_slide_rows) if cid is not None}
    if collection_ids != {collection_id}:
        raise RuntimeError(f"Imported slides did not all reference collection id {collection_id}: {sorted(collection_ids)}")

    rename_ops = _find_openapi_operations(openapi, "patch", ("collection", "rename"))
    rename_ops += _find_openapi_operations(openapi, "put", ("collection", "rename"))
    rename_ops += _find_openapi_operations(openapi, "post", ("collection", "rename"))
    if not rename_ops:
        rename_ops = _find_openapi_operations(openapi, "patch", ("collection",))
        rename_ops += _find_openapi_operations(openapi, "put", ("collection",))
        rename_ops += _find_openapi_operations(openapi, "post", ("collection",))
    if not rename_ops:
        raise RuntimeError("Could not find a collection rename endpoint in OpenAPI")

    renamed_name = "QA grouped import smoke renamed"
    rename_result = None
    rename_path = None
    for path, _operation in rename_ops:
        url = f"http://127.0.0.1:{port}{_replace_path_param(path, collection_id)}"
        payloads = [
            {"name": renamed_name},
            {"collection_name": renamed_name},
            {"title": renamed_name},
        ]
        for payload in payloads:
            status, response = request_json("PATCH", url, payload)
            if status == 405:
                status, response = request_json("PUT", url, payload)
            if status == 405:
                status, response = request_json("POST", url, payload)
            if status == 200:
                rename_result = response
                rename_path = path
                break
        if rename_result is not None:
            break

    if rename_result is None or rename_path is None:
        raise RuntimeError(f"Collection rename smoke failed via endpoints: {rename_ops}")

    try:
        returned_name = _extract_name(rename_result)
    except RuntimeError:
        returned_name = renamed_name
    if returned_name != renamed_name:
        raise RuntimeError(f"Rename endpoint did not return the new collection name: {rename_result}")

    get_ops = _find_openapi_operations(openapi, "get", ("collection",))
    rename_verified = False
    for path, _operation in get_ops:
        if "{" not in path:
            continue
        url = f"http://127.0.0.1:{port}{_replace_path_param(path, collection_id)}"
        status, response = request_json("GET", url)
        if status != 200:
            continue
        try:
            if _extract_name(response) == renamed_name:
                rename_verified = True
                break
        except RuntimeError:
            continue
    if not rename_verified:
        raise RuntimeError(
            f"Collection rename was not confirmed via GET /collection after {rename_path}: {rename_result}"
        )


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
            _import_collection_smoke(port, input_dir, slides_dir)
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


def phase1_board_compliance_check() -> None:
    if not PHASE1_BOARD.exists():
        print(f"Skipping Phase 1 board compliance check (local-only file missing): {PHASE1_BOARD}")
        return

    lines = PHASE1_BOARD.read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith("| P1-")]
    if not rows:
        raise RuntimeError("No Phase 1 ticket rows found in PHASE1_SPRINT_BOARD.md")

    allowed_statuses = {"todo", "in_progress", "blocked", "done"}
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    errors: list[str] = []

    for row in rows:
        parts = [cell.strip() for cell in row.strip().split("|")[1:-1]]
        if len(parts) != 11:
            errors.append(f"Malformed board row (expected 11 columns): {row}")
            continue

        ticket_id, _title, _stream, _priority, _estimate, status, _owner, _deps, _sprint, done_date, pr_ref = parts

        if status not in allowed_statuses:
            errors.append(f"{ticket_id}: invalid status '{status}'")
            continue

        if status == "done":
            if not done_date or not date_pattern.match(done_date):
                errors.append(f"{ticket_id}: status=done requires Done Date in YYYY-MM-DD")
            if not pr_ref:
                errors.append(f"{ticket_id}: status=done requires PR/Commit reference")
        else:
            if done_date:
                errors.append(f"{ticket_id}: Done Date must be empty unless status=done")

    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise RuntimeError(f"Phase 1 board compliance failed:\n{joined}")


def main() -> int:
    if not API_VENV_PY.exists():
        print(f"Missing API venv python at {API_VENV_PY}", file=sys.stderr)
        return 1

    print("== QA Smoke: JS syntax ==")
    js_syntax_checks()
    print("== QA Smoke: Python compile ==")
    python_compile_checks()
    print("== QA Smoke: Phase 1 board compliance ==")
    phase1_board_compliance_check()
    print("== QA Smoke: Bulk import collision scenario ==")
    import_collision_smoke()
    print("All QA smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
