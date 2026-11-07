#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.28,<3",
# ]
# ///
"""Zero-install entrypoint for running directly from a git checkout (`uv run python/kinderpedia.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kinderpedia_backup.cli import cli  # noqa: E402

if __name__ == "__main__":
    cli()
