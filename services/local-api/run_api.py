from __future__ import annotations

import sys
from pathlib import Path

# Ensure the local API root is importable regardless of the caller's cwd.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.cli import main

if __name__ == "__main__":
    main()
