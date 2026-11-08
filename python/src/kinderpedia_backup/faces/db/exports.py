"""Portable JSON index, per-person URL exports and flat `ByChild/` folders."""

from __future__ import annotations

import contextlib
import csv
import io
import json
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
    photo); plain copies and sub-directories are left alone. Empty folders are removed. Never runs
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
        for entry in list(folder.iterdir()):
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
    ) -> OrganizeStats:
        """
        Populate flat per-person folders `ByChild/<Person>/<YYYY-MM-DD Title>_001_<id>.jpg`
        using zero-space hardlinks (or symlinks/copies), pruning stale links when disproved.

        Rows are read under `_lock`; all filesystem work then runs WITHOUT the DB lock so the UI
        keeps serving requests (this may copy gigabytes with `link_mode="copy"`). With `prune=True`
        stale entries (incl. dangling symlinks) are removed from each person folder, and on a full run
        (no `person_filter`) folders of people that no longer exist (renamed / merged / deleted) are
        emptied of links we could have created (symlinks, or files that are hardlinked elsewhere) and
        removed when empty — always strictly inside `output_dir`, never in the backup itself.
        """
        backup_root = backup_dir.resolve()
        out_root = output_dir.resolve()

        s_tuple = (
            ("confirmed", "auto_high", "candidate")
            if include_candidates
            else ("confirmed", "auto_high", "auto_high")
        )

        plan: list[tuple[str, list[tuple[int, str, str, str]]]] = []
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
                    SELECT p.photo_id, p.rel_path, p.album_leaf, p.filename
                    FROM assignments a
                    JOIN photos p ON p.photo_id = a.photo_id
                    WHERE a.person_id = ? AND a.status IN (?, ?, ?)
                    ORDER BY p.album_date ASC, p.album_leaf ASC, p.filename ASC
                    """,
                    (int(p_row["person_id"]), *s_tuple),
                ).fetchall()
                plan.append(
                    (
                        str(p_row["display_name"]),
                        [
                            (int(r["photo_id"]), str(r["rel_path"]), str(r["album_leaf"]), str(r["filename"]))
                            for r in rows
                        ],
                    )
                )

        # ---- filesystem pass (no DB lock held) ----
        out_root.mkdir(parents=True, exist_ok=True)
        stats = OrganizeStats(reference_profiles=len(plan))
        matched_photo_ids: set[int] = set()
        live_folders: set[str] = set()
        for display_name, rows_t in plan:
            folder_name = sanitize_name(display_name, max_len=60)
            live_folders.add(folder_name)
            person_dir = ensure_within_directory(out_root, out_root / folder_name)
            person_dir.mkdir(parents=True, exist_ok=True)

            expected_filenames: set[str] = set()
            for photo_id, rel_path, album_leaf, filename in rows_t:
                src = ensure_within_directory(backup_root, backup_root / rel_path)
                if not src.is_file():
                    continue
                matched_photo_ids.add(photo_id)
                album_prefix = sanitize_name(album_leaf, max_len=80)
                base_target = f"{album_prefix}_{filename}"
                target_name = base_target
                if target_name in expected_filenames:
                    stem, dot, ext = base_target.rpartition(".")
                    target_name = f"{stem}_p{photo_id}{dot}{ext}" if dot else f"{base_target}_p{photo_id}"
                expected_filenames.add(target_name)

                dst = ensure_within_directory(person_dir, person_dir / target_name)
                if dst.is_symlink() and not dst.exists():
                    dst.unlink()  # dangling symlink (photo moved): recreate below
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
                    ) and existing_entry.name not in expected_filenames:
                        existing_entry.unlink()
                        stats.links_pruned += 1

        if prune and not person_filter:
            stats.links_pruned += _prune_orphan_person_folders(out_root, backup_root, live_folders)

        stats.photos_matched = len(matched_photo_ids)
        return stats
