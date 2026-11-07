"""Filesystem discovery and zero-copy photo linking for face-based organization."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

from ..security import ensure_within_directory
from ..sync.naming import sanitize_name
from .models import LinkMode

_PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic"})
_IGNORED_DIR_NAMES = frozenset({
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "Metadata",
    "References",
    "Videos",
    "Documents",
    "ByChild",
    "Organized",
})


def discover_backup_photos(backup_dir: Path, *, exclude_dir: Path | None = None) -> list[Path]:
    """
    Locate all downloaded album photos inside a Kinderpedia backup folder.

    Works identically on backups created by either the Browser Bookmarklet or the Python `sync` CLI.
    """
    if not backup_dir.exists():
        return []

    excl_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    found: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(backup_dir):
        current = Path(dirpath).resolve()
        if excl_resolved is not None and (current == excl_resolved or excl_resolved in current.parents):
            dirnames[:] = []
            continue

        dirnames[:] = sorted(
            d for d in dirnames if d not in _IGNORED_DIR_NAMES and not d.startswith(".")
        )
        for fn in sorted(filenames):
            if fn.startswith(".") or fn == "avatar.png" or fn.endswith(".part"):
                continue
            p = Path(dirpath) / fn
            if p.suffix.lower() in _PHOTO_SUFFIXES:
                found.append(p)
    return found


def discover_reference_photos(refs_dir: Path) -> dict[str, list[Path]]:
    """
    Load reference portrait folders organized as `<refs_dir>/<Person Name>/*.jpg`.
    """
    profiles: dict[str, list[Path]] = {}
    if not refs_dir.is_dir():
        return profiles

    for entry in sorted(refs_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        person = sanitize_name(entry.name, max_len=60)
        photos = [
            p
            for p in sorted(entry.iterdir())
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in _PHOTO_SUFFIXES
        ]
        if photos:
            profiles[person] = photos
    return profiles


def link_or_copy_photo(src: Path, dst: Path, *, root_dir: Path, mode: LinkMode = "hardlink") -> bool:
    """
    Materialize `src` at `dst` inside `root_dir` without duplicating disk usage when possible.

    Returns True if created, False if `dst` already exists.
    """
    safe_dst = ensure_within_directory(root_dir, dst)
    if safe_dst.exists() or safe_dst.is_symlink():
        return False

    safe_dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, safe_dst)
            return True
        except OSError as exc:
            # Cross-device link (e.g. external FAT/exFAT or separate volume) -> fall back to copy2
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EACCES):
                raise
            shutil.copy2(src, safe_dst)
            return True
    if mode == "symlink":
        rel_target = os.path.relpath(src.resolve(), start=safe_dst.parent.resolve())
        os.symlink(rel_target, safe_dst)
        return True
    shutil.copy2(src, safe_dst)
    return True
