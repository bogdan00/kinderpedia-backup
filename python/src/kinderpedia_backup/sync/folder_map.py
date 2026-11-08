"""Persisted `album_id -> folder` / `video_id -> (folder, file)` identity map (shared with the Bookmarklet).

Folder names are derived from the live listing (`naming.assign_album_folder_names`), which numbers
same-day/same-title siblings by position. Without persistence, a teacher deleting, renaming or
re-dating one album (or a new album landing on a day that previously held a standalone video folder)
would silently rename siblings and trigger re-downloads under new names. `Metadata/<Child>/folder_map.json`
pins every folder ever assigned so only unknown IDs get new names.

File format (`version` 1)::

    {
      "version": 1,
      "albums": {"<album_id>": "<Year>/<Month>/<YYYY-MM-DD Title>"},
      "videos": {"<video_id>": {"folder": "<Year>/<Month>/<leaf>", "file": "<Title>.mp4"}}
    }
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..security import write_json_atomic
from .naming import split_year_month

FOLDER_MAP_FILE = "folder_map.json"
FOLDER_MAP_VERSION = 1
_LEAF_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) ")


@dataclass
class FolderMap:
    albums: dict[int, str] = field(default_factory=dict)
    videos: dict[int, tuple[str, str]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": FOLDER_MAP_VERSION,
            "albums": {str(k): v for k, v in sorted(self.albums.items())},
            "videos": {str(k): {"folder": f, "file": n} for k, (f, n) in sorted(self.videos.items())},
        }


def is_safe_rel_folder(rel: object) -> bool:
    """Reject absolute paths, backslashes, empty or `.`/`..` segments from an untrusted map file."""
    if not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel:
        return False
    if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def _is_safe_file_name(name: object) -> bool:
    return isinstance(name, str) and is_safe_rel_folder(name) and "/" not in name


def parse_folder_map(data: object) -> FolderMap | None:
    """Validate a decoded `folder_map.json` payload; unsafe/malformed entries are dropped."""
    if not isinstance(data, dict) or not isinstance(data.get("albums"), dict):
        return None
    fm = FolderMap()
    for k, v in data["albums"].items():
        with contextlib.suppress(ValueError, TypeError):
            if is_safe_rel_folder(v):
                fm.albums[int(k)] = v
    raw_videos = data.get("videos")
    if isinstance(raw_videos, dict):
        for k, v in raw_videos.items():
            if not isinstance(v, dict):
                continue
            folder, fname = v.get("folder"), v.get("file")
            with contextlib.suppress(ValueError, TypeError):
                if is_safe_rel_folder(folder) and _is_safe_file_name(fname):
                    fm.videos[int(k)] = (str(folder), str(fname))
    return fm


def load_folder_map(path: Path) -> FolderMap | None:
    if not path.is_file():
        return None
    try:
        return parse_folder_map(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def save_folder_map(path: Path, fm: FolderMap) -> None:
    write_json_atomic(path, fm.to_json())


def album_folder_from_leaf(leaf: str) -> str:
    """`2025-06-12 Title (2)` -> `2025/06/2025-06-12 Title (2)`; undated leaves go under `Unknown/00`."""
    m = _LEAF_DATE_RE.match(leaf)
    year_s, month_s = split_year_month(m.group(1) if m else "")
    return f"{year_s}/{month_s}/{leaf}"


def rebuild_folder_map(meta_albums_dir: Path, albums_dir: Path | None = None) -> FolderMap:
    """
    Recover the album map of an existing archive from `Metadata/<Child>/Albums/<leaf>.json`
    (`album_listing.id`), written by both this CLI and the Bookmarklet. When one album id owns several
    meta files (an earlier sibling rename), the leaf whose media folder exists on disk wins, then the
    lexicographically smallest.
    """
    fm = FolderMap()
    if not meta_albums_dir.is_dir():
        return fm
    candidates: dict[int, list[str]] = {}
    with contextlib.suppress(OSError):
        for entry in os.scandir(meta_albums_dir):
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            leaf = entry.name[: -len(".json")]
            if not _is_safe_file_name(leaf):
                continue
            try:
                payload = json.loads(Path(entry.path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            listing = payload.get("album_listing") if isinstance(payload, dict) else None
            if not isinstance(listing, dict) or listing.get("id") is None:
                continue
            with contextlib.suppress(ValueError, TypeError):
                candidates.setdefault(int(listing["id"]), []).append(leaf)
    for aid, leaves in candidates.items():
        def _rank(leaf: str) -> tuple[int, str]:
            on_disk = albums_dir is not None and (albums_dir / album_folder_from_leaf(leaf)).is_dir()
            return (0 if on_disk else 1, leaf)

        fm.albums[aid] = album_folder_from_leaf(min(leaves, key=_rank))
    return fm
