"""Portable JSON index, per-person URL exports and flat `ByChild/` folders."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from ...security import ensure_within_directory, write_json_atomic
from ...sync.naming import sanitize_name
from ..models import (
    LinkMode,
    OrganizeStats,
)
from ..storage import link_or_copy_photo
from .codec import (
    _now_iso,
)
from .queries_ui import QueriesUIMixin


def _prune_orphan_person_folders(out_root: Path, backup_root: Path, live_folders: set[str]) -> int:
    """
    Remove links from `ByChild/<folder>` directories that no longer belong to any materialized
    person (renamed / merged / deleted). Only entries we could have created are removed — symlinks
    and regular files that are hardlinked elsewhere (`st_nlink > 1`, i.e. the archive still has the
    photo); plain copies and user directories are left alone. Empty folders are removed. Never runs
    when `out_root` is (or contains) the backup root. Returns the number of entries removed.
    """
    if out_root == backup_root or out_root in backup_root.parents:
        return 0
    removed = 0
    for folder in list(out_root.iterdir()):
        if folder.name.startswith(".") or folder.is_symlink() or not folder.is_dir():
            continue
        if folder.name in live_folders:
            continue
        for root, _dirs, files in os.walk(folder, topdown=False):
            rpath = Path(root)
            for fname in files:
                entry = rpath / fname
                try:
                    if entry.is_symlink() or (
                        not entry.name.startswith(".") and entry.is_file() and entry.stat().st_nlink > 1
                    ):
                        ensure_within_directory(out_root, entry.parent)
                        entry.unlink()
                        removed += 1
                    elif entry.name == ".DS_Store":
                        entry.unlink()
                except OSError:
                    continue
            with contextlib.suppress(OSError):
                rpath.rmdir()
        with contextlib.suppress(OSError):
            folder.rmdir()
    return removed


class ExportsMixin(QueriesUIMixin):
    """Portable JSON index, per-person URL exports and flat `ByChild/` folders."""

    def export_portable_index(self, out_json_path: Path) -> dict[str, Any]:
        """
        Write `Metadata/categorization_index.json` containing all named people and every categorized
        photo with full Kinderpedia provenance & CDN URLs (omitting bulky float16 blobs).
        """
        with self._lock:
            people = self.list_people()
            rows = self._conn.execute(
                """
                SELECT
                    p.photo_id, p.rel_path, p.archive_child, p.year, p.month,
                    p.album_leaf, p.kinderpedia_album_id, p.album_title, p.album_date,
                    p.classroom_groups, p.kinderpedia_image_id, p.seq_in_album, p.image_dateadd,
                    p.best_live_url, p.download_url, p.fullsize_url, p.thumb_url,
                    p.width, p.height,
                    a.person_id, pe.display_name, pe.role, a.status, a.source, a.similarity,
                    f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left
                FROM assignments a
                JOIN photos p ON p.photo_id = a.photo_id
                JOIN people pe ON pe.person_id = a.person_id
                LEFT JOIN faces f ON f.face_id = a.face_id
                WHERE pe.role != 'ignore' AND a.status IN ('confirmed', 'auto_high')
                ORDER BY p.album_date ASC, p.album_leaf ASC, p.filename ASC
                """
            ).fetchall()

        photos_map: dict[int, dict[str, Any]] = {}
        for r in rows:
            pid = int(r["photo_id"])
            entry = photos_map.get(pid)
            if entry is None:
                entry = {
                    "photo_id": pid,
                    "rel_path": str(r["rel_path"]),
                    "archive_child": str(r["archive_child"]),
                    "year": str(r["year"]),
                    "month": str(r["month"]),
                    "album_leaf": str(r["album_leaf"]),
                    "kinderpedia_album_id": r["kinderpedia_album_id"],
                    "album_title": str(r["album_title"]),
                    "album_date": str(r["album_date"]),
                    "classroom_groups": str(r["classroom_groups"]),
                    "kinderpedia_image_id": r["kinderpedia_image_id"],
                    "seq_in_album": r["seq_in_album"],
                    "image_dateadd": str(r["image_dateadd"]),
                    "best_live_url": str(r["best_live_url"]),
                    "download_url": str(r["download_url"]),
                    "fullsize_url": str(r["fullsize_url"]),
                    "thumb_url": str(r["thumb_url"]),
                    "width": int(r["width"]),
                    "height": int(r["height"]),
                    "people": [],
                }
                photos_map[pid] = entry
            entry["people"].append(
                {
                    "person_id": int(r["person_id"]),
                    "name": str(r["display_name"]),
                    "role": str(r["role"]),
                    "status": str(r["status"]),
                    "source": str(r["source"]),
                    "similarity": round(float(r["similarity"]), 4),
                    "box": (
                        {
                            "top": int(r["bbox_top"]),
                            "right": int(r["bbox_right"]),
                            "bottom": int(r["bbox_bottom"]),
                            "left": int(r["bbox_left"]),
                        }
                        if r["bbox_top"] is not None
                        else None
                    ),
                }
            )

        payload = {
            "schema_version": 1,
            "updated_at": _now_iso(),
            "people": [p for p in people if p["role"] != "ignore"],
            "categorized_photo_count": len(photos_map),
            "photos": list(photos_map.values()),
        }
        write_json_atomic(out_json_path, payload)
        return payload

    def format_person_urls(
        self,
        person_query: str,
        *,
        fmt: str = "txt",
        include_candidates: bool = False,
    ) -> tuple[str, str, int]:
        """
        Build a URL & provenance export (`txt`, `csv`, or `json`) for a single named child/classmate.
        Returns `(resolved_display_name, formatted_content, url_count)`.
        """
        s_tuple = (
            ("confirmed", "auto_high", "candidate")
            if include_candidates
            else ("confirmed", "auto_high", "auto_high")
        )

        with self._lock:
            p_row = self._conn.execute(
                "SELECT person_id, display_name, role FROM people WHERE display_name = ? COLLATE NOCASE",
                (person_query.strip(),),
            ).fetchone()
            if p_row is None:
                raise KeyError(f"No categorized person named {person_query!r} found in index.")
            person_id = int(p_row["person_id"])
            display_name = str(p_row["display_name"])

            rows = self._conn.execute(
                """
                SELECT
                    p.photo_id, p.rel_path, p.archive_child, p.year, p.month, p.album_leaf,
                    p.kinderpedia_album_id, p.album_title, p.album_date, p.classroom_groups,
                    p.kinderpedia_image_id, p.seq_in_album, p.filename,
                    p.best_live_url, p.download_url, p.fullsize_url, p.thumb_url,
                    p.width, p.height,
                    a.status, a.similarity
                FROM assignments a
                JOIN photos p ON p.photo_id = a.photo_id
                WHERE a.person_id = ? AND a.status IN (?, ?, ?)
                ORDER BY p.album_date ASC, p.album_leaf ASC, p.filename ASC
                """,
                (person_id, *s_tuple),
            ).fetchall()

        seen_urls: set[str] = set()
        records: list[dict[str, Any]] = []
        for r in rows:
            url = str(r["best_live_url"] or r["download_url"] or r["fullsize_url"] or "").strip()
            dedup_key = url if url else f"local:{r['rel_path']}"
            if dedup_key in seen_urls:
                continue
            seen_urls.add(dedup_key)
            records.append(
                {
                    "album_date": str(r["album_date"]),
                    "year": str(r["year"]),
                    "month": str(r["month"]),
                    "album_leaf": str(r["album_leaf"]),
                    "album_title": str(r["album_title"]),
                    "kinderpedia_album_id": r["kinderpedia_album_id"],
                    "kinderpedia_image_id": r["kinderpedia_image_id"],
                    "seq_in_album": r["seq_in_album"],
                    "filename": str(r["filename"]),
                    "archive_child": str(r["archive_child"]),
                    "classroom_groups": str(r["classroom_groups"]),
                    "status": str(r["status"]),
                    "similarity": round(float(r["similarity"]), 4),
                    "url": url,
                    "download_url": str(r["download_url"]),
                    "fullsize_url": str(r["fullsize_url"]),
                    "thumb_url": str(r["thumb_url"]),
                    "width": int(r["width"]),
                    "height": int(r["height"]),
                    "rel_path": str(r["rel_path"]),
                }
            )

        norm_fmt = fmt.strip().lower()
        if norm_fmt == "txt":
            lines = [rec["url"] for rec in records if rec["url"]]
            return display_name, ("\n".join(lines) + ("\n" if lines else "")), len(lines)

        if norm_fmt == "csv":
            buf = io.StringIO()
            writer = csv.DictWriter(
                buf,
                fieldnames=[
                    "album_date",
                    "album_leaf",
                    "filename",
                    "kinderpedia_album_id",
                    "kinderpedia_image_id",
                    "status",
                    "similarity",
                    "url",
                    "fullsize_url",
                    "archive_child",
                    "rel_path",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(records)
            return display_name, buf.getvalue(), len(records)

        if norm_fmt == "json":
            doc = {
                "person": display_name,
                "exported_at": _now_iso(),
                "count": len(records),
                "items": records,
            }
            return display_name, json.dumps(doc, indent=2, ensure_ascii=False) + "\n", len(records)

        raise ValueError(f"Unsupported export format {fmt!r}; choose txt, csv, or json.")

    def materialize_flat_folders(
        self,
        *,
        backup_dir: Path,
        output_dir: Path,
        link_mode: LinkMode = "hardlink",
        person_filter: str | None = None,
        include_candidates: bool = False,
        prune: bool = True,
        layout: str = "split",
        max_curated_per_album: int = 3,
        min_curated_face_px: int = 70,
    ) -> OrganizeStats:
        """
        Populate organized per-person folders `ByChild/<Person>/...` using zero-space hardlinks
        (or symlinks/copies), pruning stale links when disproved.

        Layout options:
        - "split" (default): Creates dual branches:
            `ByChild/<Person>/curated/<YYYY>/<MM>/<AlbumPrefix>_<filename>` (monthly highlights)
            `ByChild/<Person>/full/<YYYY>/<MM>/<AlbumTitle>/<filename>` (exhaustive event archive)
        - "curated": Only the curated highlights branch.
        - "full": Only the full event-folder hierarchy.
        - "flat": Legacy single flat folder `ByChild/<Person>/<AlbumPrefix>_<filename>`.
        """
        backup_root = backup_dir.resolve()
        out_root = output_dir.resolve()

        s_tuple = (
            ("confirmed", "auto_high", "candidate")
            if include_candidates
            else ("confirmed", "auto_high", "auto_high")
        )

        plan: list[tuple[str, list[dict[str, Any]]]] = []
        with self._lock:
            if person_filter:
                people_rows = self._conn.execute(
                    "SELECT person_id, display_name FROM people WHERE display_name = ? COLLATE NOCASE AND role != 'ignore'",
                    (person_filter.strip(),),
                ).fetchall()
            else:
                people_rows = self._conn.execute(
                    "SELECT person_id, display_name FROM people WHERE role IN ('family', 'classmate')"
                ).fetchall()
            for p_row in people_rows:
                rows = self._conn.execute(
                    """
                    SELECT p.photo_id, p.rel_path, p.year, p.month, p.album_leaf, p.filename,
                           a.status, a.similarity, f.bbox_top, f.bbox_bottom
                    FROM assignments a
                    JOIN photos p ON p.photo_id = a.photo_id
                    LEFT JOIN faces f ON f.face_id = a.face_id
                    WHERE a.person_id = ? AND a.status IN (?, ?, ?)
                    ORDER BY p.album_date ASC, p.album_leaf ASC, p.filename ASC
                    """,
                    (int(p_row["person_id"]), *s_tuple),
                ).fetchall()

                # Deduplicate by photo_id per person, keeping largest face / confirmed status
                p_map: dict[int, dict[str, Any]] = {}
                for r in rows:
                    pid = int(r["photo_id"])
                    top = r["bbox_top"]
                    bot = r["bbox_bottom"]
                    h = int(bot - top) if (top is not None and bot is not None) else 0
                    existing = p_map.get(pid)
                    if existing is None:
                        p_map[pid] = {
                            "photo_id": pid,
                            "rel_path": str(r["rel_path"]),
                            "year": str(r["year"] or "unknown"),
                            "month": str(r["month"] or "00"),
                            "album_leaf": str(r["album_leaf"] or "Unknown"),
                            "filename": str(r["filename"]),
                            "status": str(r["status"]),
                            "similarity": float(r["similarity"] or 0.0),
                            "face_h": h,
                        }
                    else:
                        if r["status"] == "confirmed" and existing["status"] != "confirmed":
                            existing["status"] = "confirmed"
                        if h > existing["face_h"]:
                            existing["face_h"] = h
                plan.append((str(p_row["display_name"]), list(p_map.values())))

        # ---- filesystem pass (no DB lock held) ----
        out_root.mkdir(parents=True, exist_ok=True)
        stats = OrganizeStats(reference_profiles=len(plan))
        matched_photo_ids: set[int] = set()
        live_folders: set[str] = set()

        for display_name, photos_list in plan:
            folder_name = sanitize_name(display_name, max_len=60)
            live_folders.add(folder_name)
            person_dir = ensure_within_directory(out_root, out_root / folder_name)
            person_dir.mkdir(parents=True, exist_ok=True)

            expected_rel_paths: set[Path] = set()

            if layout == "flat":
                expected_flat_names: set[str] = set()
                for p in photos_list:
                    src = ensure_within_directory(backup_root, backup_root / p["rel_path"])
                    if not src.is_file():
                        continue
                    matched_photo_ids.add(p["photo_id"])
                    album_prefix = sanitize_name(p["album_leaf"], max_len=80)
                    base_target = f"{album_prefix}_{p['filename']}"
                    target_name = base_target
                    if target_name in expected_flat_names:
                        stem, dot, ext = base_target.rpartition(".")
                        target_name = f"{stem}_p{p['photo_id']}{dot}{ext}" if dot else f"{base_target}_p{p['photo_id']}"
                    expected_flat_names.add(target_name)

                    dst = ensure_within_directory(person_dir, person_dir / target_name)
                    if dst.is_symlink() and not dst.exists():
                        dst.unlink()
                    if link_or_copy_photo(src, dst, root_dir=out_root, mode=link_mode):
                        stats.links_created += 1
                    else:
                        stats.links_skipped_existing += 1

                if prune and person_dir.is_dir():
                    for existing_entry in list(person_dir.iterdir()):
                        if existing_entry.name.startswith("."):
                            continue
                        is_link = existing_entry.is_symlink()
                        if (
                            is_link or existing_entry.is_file()
                        ) and existing_entry.name not in expected_flat_names:
                            existing_entry.unlink()
                            stats.links_pruned += 1

            else:
                # Hierarchical layouts: "split", "curated", or "full"
                # 1. Curated selection: group by album and pick top photos
                if layout in ("split", "curated"):
                    albums: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
                    for p in photos_list:
                        albums[(p["year"], p["month"], p["album_leaf"])].append(p)

                    curated_selection: list[dict[str, Any]] = []
                    for _key, a_photos in albums.items():
                        # Sort: confirmed first, largest face first, highest similarity first
                        a_photos.sort(key=lambda x: (
                            0 if x["status"] == "confirmed" else 1,
                            -x["face_h"],
                            -x["similarity"],
                            x["photo_id"],
                        ))
                        # Pick #1 always (album representative)
                        curated_selection.append(a_photos[0])
                        # Additional up to max_curated_per_album if high quality / confirmed
                        for extra in a_photos[1:max_curated_per_album]:
                            if extra["face_h"] >= min_curated_face_px or extra["status"] == "confirmed":
                                curated_selection.append(extra)

                    for p in curated_selection:
                        src = ensure_within_directory(backup_root, backup_root / p["rel_path"])
                        if not src.is_file():
                            continue
                        matched_photo_ids.add(p["photo_id"])
                        album_prefix = sanitize_name(p["album_leaf"], max_len=80)
                        base_target = f"{album_prefix}_{p['filename']}"
                        rel_target = Path("curated") / p["year"] / p["month"] / base_target
                        if rel_target in expected_rel_paths:
                            stem, dot, ext = base_target.rpartition(".")
                            tname = f"{stem}_p{p['photo_id']}{dot}{ext}" if dot else f"{base_target}_p{p['photo_id']}"
                            rel_target = Path("curated") / p["year"] / p["month"] / tname
                        expected_rel_paths.add(rel_target)

                        dst = ensure_within_directory(person_dir, person_dir / rel_target)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if dst.is_symlink() and not dst.exists():
                            dst.unlink()
                        if link_or_copy_photo(src, dst, root_dir=out_root, mode=link_mode):
                            stats.links_created += 1
                        else:
                            stats.links_skipped_existing += 1

                # 2. Full archive: organized into year/month/album_leaf folders
                if layout in ("split", "full"):
                    for p in photos_list:
                        src = ensure_within_directory(backup_root, backup_root / p["rel_path"])
                        if not src.is_file():
                            continue
                        matched_photo_ids.add(p["photo_id"])
                        album_dir_name = sanitize_name(p["album_leaf"], max_len=80)
                        base_target = p["filename"]
                        rel_target = Path("full") / p["year"] / p["month"] / album_dir_name / base_target
                        if rel_target in expected_rel_paths:
                            stem, dot, ext = base_target.rpartition(".")
                            tname = f"{stem}_p{p['photo_id']}{dot}{ext}" if dot else f"{base_target}_p{p['photo_id']}"
                            rel_target = Path("full") / p["year"] / p["month"] / album_dir_name / tname
                        expected_rel_paths.add(rel_target)

                        dst = ensure_within_directory(person_dir, person_dir / rel_target)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if dst.is_symlink() and not dst.exists():
                            dst.unlink()
                        if link_or_copy_photo(src, dst, root_dir=out_root, mode=link_mode):
                            stats.links_created += 1
                        else:
                            stats.links_skipped_existing += 1

                # 3. Recursive prune pass for hierarchical layout
                if prune and person_dir.is_dir():
                    for root, _dirs, files in os.walk(person_dir, topdown=False):
                        rpath = Path(root)
                        for fname in files:
                            if fname.startswith("."):
                                continue
                            entry = rpath / fname
                            rel = entry.relative_to(person_dir)
                            if rel not in expected_rel_paths:
                                try:
                                    if entry.is_symlink() or (entry.is_file() and entry.stat().st_nlink > 1):
                                        entry.unlink()
                                        stats.links_pruned += 1
                                except OSError:
                                    pass
                        if rpath != person_dir:
                            with contextlib.suppress(OSError):
                                rpath.rmdir()

        if prune and not person_filter:
            stats.links_pruned += _prune_orphan_person_folders(out_root, backup_root, live_folders)

        stats.photos_matched = len(matched_photo_ids)
        return stats
