"""Kinderpedia archive provenance & live CDN URL resolution from `Metadata/<Child>/Albums/*.json`."""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

from .models import PhotoProvenance

_SEQ_ID_FILENAME_RE = re.compile(r"^(?P<seq>\d{1,5})_(?:.+_)?(?P<img_id>\d+)\.[A-Za-z0-9]+$")
_SEQ_ONLY_FILENAME_RE = re.compile(r"^(?P<seq>\d{1,5})_[^.]+\.[A-Za-z0-9]+$")
_ONLY_ID_FILENAME_RE = re.compile(r"^(?P<img_id>\d{5,})\.[A-Za-z0-9]+$")
_DIM_URL_RE = re.compile(r"/(?P<w>\d{2,5})x(?P<h>\d{2,5})/")
_YEAR_PREFIX_RE = re.compile(r"^(?P<y>20\d{2})-(?P<m>\d{2})-(?P<d>\d{2})\b")


def parse_photo_filename_ids(filename: str) -> tuple[int | None, int | None]:
    """Extract `(seq_in_album, kinderpedia_image_id)` from `<001>_<50753307>.jpg` (or `<001>_title_<id>.jpg`)."""
    m = _SEQ_ID_FILENAME_RE.match(filename)
    if m:
        return int(m.group("seq")), int(m.group("img_id"))
    m2 = _ONLY_ID_FILENAME_RE.match(filename)
    if m2:
        return None, int(m2.group("img_id"))
    m3 = _SEQ_ONLY_FILENAME_RE.match(filename)
    if m3:
        return int(m3.group("seq")), None
    return None, None


def resolve_best_live_url(
    *,
    download_url: str,
    fullsize_url: str,
    thumb_url: str,
    width: int = 0,
    height: int = 0,
) -> str:
    """
    Pick the known-working Kinderpedia CDN URL without making network requests.

    On pre-2024 migrated Kinderpedia albums, `download_url` returns HTTP 404 and `sync`
    fell back to `fullsize` (e.g. `https://images.kinderpedia.co/900x1200/files/photos/...`).
    When the downloaded JPEG's `(width, height)` matches the `/<W>x<H>/` prefix in `fullsize_url`
    (and `download_url` lacks that prefix), `fullsize_url` is the live working URL.
    """
    dl = download_url.strip()
    fs = fullsize_url.strip()
    th = thumb_url.strip()

    if fs and width > 0 and height > 0:
        m = _DIM_URL_RE.search(fs)
        if m:
            fw, fh = int(m.group("w")), int(m.group("h"))
            if (width, height) == (fw, fh) and (not dl or _DIM_URL_RE.search(dl) is None):
                return fs
    return dl or fs or th


def discover_child_avatars(backup_dir: Path) -> dict[str, Path]:
    """Discover `<Child>/avatar.png` files at the top level of `backup_dir`."""
    avatars: dict[str, Path] = {}
    if not backup_dir.is_dir():
        return avatars
    for entry in sorted(backup_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        for cand_name in ("avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"):
            cand = entry / cand_name
            if cand.is_file():
                avatars[entry.name] = cand
                break
    return avatars


class ProvenanceResolver:
    """Caches parsed `Metadata/<Child>/Albums/<Leaf>.json` manifests per album directory."""

    def __init__(self, backup_dir: Path) -> None:
        self.backup_root = backup_dir.resolve()
        self._album_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _load_album_meta(self, archive_child: str, album_leaf: str) -> dict[str, Any]:
        key = (archive_child, album_leaf)
        if key in self._album_cache:
            return self._album_cache[key]

        meta_path = self.backup_root / "Metadata" / archive_child / "Albums" / f"{album_leaf}.json"
        parsed: dict[str, Any] = {"listing": {}, "by_id": {}, "by_idx": []}
        if meta_path.is_file():
            with contextlib.suppress(OSError, ValueError):
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    listing = raw.get("album_listing")
                    if isinstance(listing, dict):
                        parsed["listing"] = listing
                    images = raw.get("images")
                    if isinstance(images, list):
                        by_id: dict[int, dict[str, Any]] = {}
                        by_idx: list[dict[str, Any]] = []
                        for item in images:
                            if isinstance(item, dict):
                                by_idx.append(item)
                                raw_id = item.get("id")
                                if isinstance(raw_id, int) or (isinstance(raw_id, str) and raw_id.isdigit()):
                                    by_id[int(raw_id)] = item
                        parsed["by_id"] = by_id
                        parsed["by_idx"] = by_idx

        self._album_cache[key] = parsed
        return parsed

    def resolve(self, photo_path: Path, *, width: int = 0, height: int = 0) -> PhotoProvenance:
        """Build `PhotoProvenance` for `photo_path` inside `backup_root`."""
        resolved = photo_path.resolve()
        rel_parts = resolved.relative_to(self.backup_root).parts
        rel_path = "/".join(rel_parts)

        archive_child = rel_parts[0] if len(rel_parts) >= 2 else ""
        album_leaf = resolved.parent.name
        filename = resolved.name

        # Expected layout: <Child>/Albums/<Year>/<Month>/<YYYY-MM-DD Title>/<file>
        year = ""
        month = ""
        if len(rel_parts) >= 5 and rel_parts[1] == "Albums":
            year = rel_parts[2]
            month = rel_parts[3]
        m_ymd = _YEAR_PREFIX_RE.match(album_leaf)
        if m_ymd:
            if not year:
                year = m_ymd.group("y")
            if not month:
                month = m_ymd.group("m")
        if not year:
            year = "unknown"

        seq_num, img_id = parse_photo_filename_ids(filename)
        album_meta = self._load_album_meta(archive_child, album_leaf) if archive_child else {}
        listing: dict[str, Any] = album_meta.get("listing") or {}
        by_id: dict[int, dict[str, Any]] = album_meta.get("by_id") or {}
        by_idx: list[dict[str, Any]] = album_meta.get("by_idx") or []

        img_entry: dict[str, Any] | None = None
        if img_id is not None and img_id in by_id:
            img_entry = by_id[img_id]
        elif seq_num is not None and 1 <= seq_num <= len(by_idx):
            img_entry = by_idx[seq_num - 1]
            if img_id is None and img_entry is not None:
                raw_id = img_entry.get("id")
                if isinstance(raw_id, int) or (isinstance(raw_id, str) and raw_id.isdigit()):
                    img_id = int(raw_id)

        raw_aid = listing.get("id")
        album_id = (
            int(raw_aid)
            if isinstance(raw_aid, int) or (isinstance(raw_aid, str) and raw_aid.isdigit())
            else None
        )
        album_title = str(listing.get("name") or album_leaf)
        album_date = str(listing.get("date") or (m_ymd.group(0) if m_ymd else ""))

        groups_raw = listing.get("groups")
        groups_str = ""
        if isinstance(groups_raw, list):
            names = [
                str(g.get("name") or g.get("group_name") or "") if isinstance(g, dict) else str(g)
                for g in groups_raw
                if isinstance(g, (dict, str))
            ]
            groups_str = ", ".join(n for n in names if n)
        elif isinstance(groups_raw, str):
            groups_str = groups_raw

        dl_url = str((img_entry or {}).get("download_url") or "")
        fs_url = str((img_entry or {}).get("fullsize") or "")
        th_url = str((img_entry or {}).get("thumb") or "")
        dateadd = str((img_entry or {}).get("dateadd") or "")

        best_url = resolve_best_live_url(
            download_url=dl_url,
            fullsize_url=fs_url,
            thumb_url=th_url,
            width=width,
            height=height,
        )

        return PhotoProvenance(
            rel_path=rel_path,
            archive_child=archive_child,
            year=year,
            month=month,
            album_leaf=album_leaf,
            filename=filename,
            kinderpedia_album_id=album_id,
            album_title=album_title,
            album_date=album_date,
            classroom_groups=groups_str,
            kinderpedia_image_id=img_id,
            seq_in_album=seq_num,
            image_dateadd=dateadd,
            best_live_url=best_url,
            download_url=dl_url,
            fullsize_url=fs_url,
            thumb_url=th_url,
        )
