"""Dynamic loader for pluggable VLM backends."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

from .base import VlmProvider
from .null_provider import NullVlmProvider

logger = logging.getLogger("kinderpedia.faces.vlm")

_CACHED_PROVIDER: VlmProvider | None = None


def get_vlm_provider(force_reload: bool = False) -> VlmProvider:
    """Discover and instantiate the active VLM provider.

    Checks:
    1. Environment variable KINDERPEDIA_VLM_BACKEND_DIR.
    2. Adjacent workspace directory (e.g. ../vlm_backend or ../../vlm_backend).
    3. Untracked .local/ directory.
    Falls back to NullVlmProvider if no external backend is present.
    """
    global _CACHED_PROVIDER
    if _CACHED_PROVIDER is not None and not force_reload:
        return _CACHED_PROVIDER

    backend_candidates: list[Path] = []

    # 1. Explicit env var
    env_dir = os.environ.get("KINDERPEDIA_VLM_BACKEND_DIR")
    if env_dir:
        backend_candidates.append(Path(env_dir).expanduser().resolve())

    # 2. Adjacent directories relative to this file
    # file is at kinderpedia-backup/python/src/kinderpedia_backup/faces/vlm/loader.py
    try:
        faces_dir = Path(__file__).resolve().parent.parent  # faces
        repo_root = faces_dir.parent.parent.parent.parent  # kinderpedia-backup
        parent_workspace = repo_root.parent
        backend_candidates.append(parent_workspace / "vlm_backend")
        backend_candidates.append(repo_root / ".local" / "vlm_backend")
    except Exception as exc:
        logger.debug("Failed to inspect relative parent directories: %s", exc)

    # 3. Known home workspace
    backend_candidates.append(Path.home() / "kinderpedia_bk" / "vlm_backend")

    for candidate in backend_candidates:
        if candidate.is_dir() and (candidate / "provider.py").is_file():
            cand_str = str(candidate.parent)
            if cand_str not in sys.path:
                sys.path.insert(0, cand_str)
            try:
                mod = importlib.import_module(f"{candidate.name}.provider")
                factory = getattr(mod, "create_vlm_provider", None)
                if callable(factory):
                    provider: VlmProvider = factory()
                    if provider.is_available():
                        logger.info("Loaded VLM provider from %s", candidate)
                        _CACHED_PROVIDER = provider
                        return provider
            except Exception as exc:
                logger.warning("Failed to initialize VLM backend at %s: %s", candidate, exc)

    _CACHED_PROVIDER = NullVlmProvider()
    return _CACHED_PROVIDER
