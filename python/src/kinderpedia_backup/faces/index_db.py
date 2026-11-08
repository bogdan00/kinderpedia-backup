"""Persistent SQLite WAL categorization & provenance index (`Metadata/faces_index.sqlite`)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..security import ensure_within_directory, write_json_atomic
from ..sync.naming import sanitize_name
from .db.relink import StoredFace, match_faces
from .db.schema import apply_migrations, read_schema_version
from .models import (
    AssignmentSource,
    AssignmentStatus,
    BoundingBox,
    DetectedFace,
    LinkMode,
    OrganizeStats,
    PersonRole,
    PhotoProvenance,
    PhotoScanState,
    RelinkReport,
)
from .storage import link_or_copy_photo


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


UNKNOWN_DATE_ORD = 99991231


def normalize_iso_date(value: str | None) -> str:
    """Validate a `YYYY-MM-DD` date string (or empty) and return it normalized; raises `ValueError`."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return date.fromisoformat(raw[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid date {raw!r}; expected YYYY-MM-DD.") from exc


def date_to_ord(iso: str) -> int:
    """`"2025-03-07"` -> `20250307` (0 when empty/invalid)."""
    s = iso[:10]
    if len(s) == 10 and s[4] == "-" and s[7] == "-" and (s[:4] + s[5:7] + s[8:10]).isdigit():
        return int(s[:4] + s[5:7] + s[8:10])
    return 0


def date_ord_to_day_num(o: int) -> int:
    """`20250307` -> proleptic Gregorian day number (`-1` when unknown); clamps day-of-month."""
    if o <= 0 or o >= UNKNOWN_DATE_ORD:
        return -1
    y, m, d = o // 10000, (o // 100) % 100, o % 100
    if not (1 <= m <= 12 and 1 <= y <= 9998):
        return -1
    first = date(y, m, 1)
    next_first = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    last_day = (next_first - first).days
    return first.toordinal() + min(max(d, 1), last_day) - 1


def photo_date_ord(album_date: str, image_dateadd: str, year: str, month: str) -> int:
    """
    Latest possible capture date of a photo as `YYYYMMDD`: the earliest of the album / upload
    timestamps (a photo is always taken on or before it is posted), falling back to the last day of
    the `<Year>/<Month>` folder. Unknown dates return `UNKNOWN_DATE_ORD` (never excluded).
    """
    ords = [o for o in (date_to_ord(album_date or ""), date_to_ord(image_dateadd or "")) if o > 0]
    if ords:
        return min(ords)
    y = str(year or "")
    m = str(month or "")
    if len(y) == 4 and y.isdigit():
        mm = int(m) if m.isdigit() and 1 <= int(m) <= 12 else 12
        return int(y) * 10000 + mm * 100 + 31
    return UNKNOWN_DATE_ORD


def pack_embedding_f16(vec: tuple[float, ...] | NDArray[Any]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr.astype(np.float16).tobytes()


def unpack_embeddings_f32(blobs: list[bytes], dim: int = 512) -> NDArray[np.float32]:
    if not blobs:
        return np.zeros((0, dim), dtype=np.float32)
    joined = b"".join(blobs)
    f16 = np.frombuffer(joined, dtype=np.float16).reshape(len(blobs), dim)
    f32 = f16.astype(np.float32)
    norms = np.maximum(np.linalg.norm(f32, axis=1, keepdims=True), np.float32(1e-8))
    return np.asarray(f32 / norms, dtype=np.float32)


def _pack_person_cluster(
    year_items: list[dict[str, Any]],
    sims: NDArray[np.float32],
    leader_loc: int,
    member_locs: list[int],
    is_loose: bool,
    yr_key: str,
) -> dict[str, Any]:
    """Serialize one per-child visual sub-cluster (faces sorted by similarity to the leader)."""
    l_sims = sims[leader_loc]
    face_list: list[dict[str, Any]] = []
    for m in member_locs:
        it = dict(year_items[m])
        it.pop("det_score", None)
        it["sim_to_leader"] = round(float(l_sims[m]), 4)
        face_list.append(it)
    face_list.sort(key=lambda f_it: float(f_it["sim_to_leader"]), reverse=True)
    counts = {"confirmed": 0, "auto_high": 0, "candidate": 0, "none": 0}
    boost_n = 0
    pen_n = 0
    for f_it in face_list:
        counts[str(f_it["status"])] = counts.get(str(f_it["status"]), 0) + 1
        if int(f_it["pressure"]) > 0:
            boost_n += 1
        elif int(f_it["pressure"]) < 0:
            pen_n += 1
    n_m = len(member_locs)
    if is_loose:
        cohesion = 0.0
        if n_m > 1:
            sub_s = sims[np.ix_(member_locs, member_locs)]
            cohesion = float((float(sub_s.sum()) - n_m) / (n_m * (n_m - 1)))
    else:
        cohesion = float(np.mean(l_sims[member_locs]))
    leader_fid = int(year_items[leader_loc]["face_id"])
    return {
        "cluster_key": f"{yr_key}-{'loose' if is_loose else leader_fid}",
        "year": yr_key,
        "size": len(face_list),
        "is_loose": is_loose,
        "leader_face_id": leader_fid,
        "cohesion": round(cohesion, 4),
        "mean_similarity": round(float(np.mean([float(f_it["similarity"]) for f_it in face_list])), 4),
        "counts": counts,
        "pressure": {"boost": boost_n, "penalize": pen_n},
        "faces": face_list,
    }


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


class FacesIndexDB:
    """Thread-safe SQLite WAL database backing `faces scan`, `faces ui`, and URL/folder exports.

    Concurrency rules:
    - One shared connection; every cursor touch happens under `_lock` (an `RLock`), and rows are
      materialized before it is released.
    - Lock order: `_faces_matrix_lock` is the OUTER lock and `_lock` the INNER one. Code holding
      `_faces_matrix_lock` may take `_lock` (e.g. the cold matrix load); code holding `_lock` must
      NEVER take `_faces_matrix_lock` — so never call `load_all_faces_matrix()` (or anything that
      reaches it, e.g. `_select_impactful_rows`) while holding `_lock`; release it first.
    - Long CPU/filesystem work (SGEMMs, `materialize_flat_folders` I/O) runs without `_lock`.
    - Cross-process: `faces scan` holds `Metadata/faces_scan.lock` and bumps `get_data_version()`
      when it finishes; a long-lived reader (the Web UI) polls that value to drop its caches.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._faces_matrix_lock = threading.Lock()
        self._faces_matrix_cache: dict[str, Any] | None = None
        self._photo_summary_cache: dict[str, Any] | None = None
        self._knn_neighbor_cache: dict[str, Any] | None = None
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.commit()
            apply_migrations(self._conn)

    def close(self) -> None:
        with self._faces_matrix_lock:
            self._faces_matrix_cache = None
        with self._lock:
            self._photo_summary_cache = None
            self._conn.close()

    def invalidate_caches(self) -> None:
        """Clear in-memory face matrix and static photo overview caches (e.g. after a new scan)."""
        with self._faces_matrix_lock:
            self._faces_matrix_cache = None
        with self._lock:
            self._photo_summary_cache = None

    def __enter__(self) -> FacesIndexDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ------------------------------------------------------------------
    # Scan & Cache Operations
    # ------------------------------------------------------------------

    def get_cached_photo_states(self) -> dict[str, tuple[int, int, int, str]]:
        """Return `{rel_path: (photo_id, file_size, mtime_ns, detector_id)}` for fast incremental diffs."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT photo_id, rel_path, file_size, mtime_ns, detector_id FROM photos"
            ).fetchall()
        return {
            str(r["rel_path"]): (
                int(r["photo_id"]),
                int(r["file_size"]),
                int(r["mtime_ns"]),
                str(r["detector_id"]),
            )
            for r in rows
        }

    def get_photo_scan_states(self) -> dict[str, PhotoScanState]:
        """Return `{rel_path: PhotoScanState}` (size, mtime, detector, content hash, Kinderpedia id)."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT photo_id, rel_path, file_size, mtime_ns, detector_id, content_hash,
                       kinderpedia_image_id, filename
                FROM photos
                """
            ).fetchall()
        return {
            str(r["rel_path"]): PhotoScanState(
                photo_id=int(r["photo_id"]),
                file_size=int(r["file_size"]),
                mtime_ns=int(r["mtime_ns"]),
                detector_id=str(r["detector_id"]),
                content_hash=str(r["content_hash"] or ""),
                kinderpedia_image_id=(
                    int(r["kinderpedia_image_id"]) if r["kinderpedia_image_id"] is not None else None
                ),
                filename=str(r["filename"]),
            )
            for r in rows
        }

    def set_photo_fingerprints(self, items: list[tuple[int, str, int]]) -> int:
        """Store `(photo_id, content_hash, mtime_ns)` for photos verified unchanged (no rescan)."""
        if not items:
            return 0
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE photos SET content_hash = ?, mtime_ns = ? WHERE photo_id = ?",
                [(h, int(m), int(pid)) for pid, h, m in items],
            )
        return len(items)

    def rekey_photo(
        self,
        photo_id: int,
        prov: PhotoProvenance,
        *,
        mtime_ns: int,
        content_hash: str,
    ) -> None:
        """Move an indexed photo to a new `rel_path` (album renamed / file moved) keeping `photo_id`,
        its faces and every label. Provenance columns are refreshed from `prov`."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE photos SET
                    rel_path = ?, archive_child = ?, year = ?, month = ?, album_leaf = ?, filename = ?,
                    kinderpedia_album_id = ?, album_title = ?, album_date = ?, classroom_groups = ?,
                    kinderpedia_image_id = ?, seq_in_album = ?, image_dateadd = ?,
                    best_live_url = ?, download_url = ?, fullsize_url = ?, thumb_url = ?,
                    mtime_ns = ?, content_hash = ?
                WHERE photo_id = ?
                """,
                (
                    prov.rel_path,
                    prov.archive_child,
                    prov.year,
                    prov.month,
                    prov.album_leaf,
                    prov.filename,
                    prov.kinderpedia_album_id,
                    prov.album_title,
                    prov.album_date,
                    prov.classroom_groups,
                    prov.kinderpedia_image_id,
                    prov.seq_in_album,
                    prov.image_dateadd,
                    prov.best_live_url,
                    prov.download_url,
                    prov.fullsize_url,
                    prov.thumb_url,
                    int(mtime_ns),
                    content_hash,
                    int(photo_id),
                ),
            )
            self._photo_summary_cache = None
        with self._faces_matrix_lock:
            self._faces_matrix_cache = None

    def delete_photos(self, photo_ids: list[int]) -> dict[str, Any]:
        """Remove photos that no longer exist on disk, with all their faces, labels and pressure.

        Rows are deleted explicitly child-first (instead of relying on FK cascades) so the count
        triggers stay exact; `trg_faces_ad_refs` clears covers / medoids pointing at removed faces.
        Returns `{photos, faces, human_labels, face_ids}`.
        """
        out: dict[str, Any] = {"photos": 0, "faces": 0, "human_labels": 0, "face_ids": []}
        if not photo_ids:
            return out
        ids_json = json.dumps(sorted({int(p) for p in photo_ids}))
        with self._lock, self._conn:
            face_ids = [
                int(r["face_id"])
                for r in self._conn.execute(
                    "SELECT face_id FROM faces WHERE photo_id IN (SELECT value FROM json_each(?))",
                    (ids_json,),
                ).fetchall()
            ]
            human = self._conn.execute(
                """
                SELECT COUNT(*) AS n FROM assignments
                WHERE photo_id IN (SELECT value FROM json_each(?))
                  AND status IN ('confirmed', 'rejected', 'unsure')
                """,
                (ids_json,),
            ).fetchone()
            touched = [
                (int(r["person_id"]), int(r["face_id"]))
                for r in self._conn.execute(
                    """
                    SELECT person_id, face_id FROM assignments
                    WHERE photo_id IN (SELECT value FROM json_each(?)) AND face_id IS NOT NULL
                    """,
                    (ids_json,),
                ).fetchall()
            ]
            self._conn.execute(
                "DELETE FROM assignments WHERE photo_id IN (SELECT value FROM json_each(?))", (ids_json,)
            )
            for person_id, fid in touched:
                self._refresh_cover_face_if_needed(person_id, fid)
            self._conn.execute(
                """
                DELETE FROM person_face_pressure
                WHERE face_id IN (SELECT face_id FROM faces WHERE photo_id IN (SELECT value FROM json_each(?)))
                """,
                (ids_json,),
            )
            self._conn.execute(
                "DELETE FROM faces WHERE photo_id IN (SELECT value FROM json_each(?))", (ids_json,)
            )
            cur = self._conn.execute(
                "DELETE FROM photos WHERE photo_id IN (SELECT value FROM json_each(?))", (ids_json,)
            )
            out = {
                "photos": int(cur.rowcount),
                "faces": len(face_ids),
                "human_labels": int(human["n"]) if human is not None else 0,
                "face_ids": face_ids,
            }
            self._photo_summary_cache = None
        with self._faces_matrix_lock:
            self._faces_matrix_cache = None
        return out

    def _relink_photo_faces_locked(
        self,
        photo_id: int,
        faces: list[DetectedFace],
        report: RelinkReport | None,
    ) -> list[int]:
        """Replace the auto-detected faces of an already indexed photo while preserving `face_id`s.

        Old faces are matched to the new detections (bbox IoU, then embedding cosine); matched rows
        are updated in place, unmatched new detections inserted, unmatched old rows deleted (their
        human labels survive as `(photo, person)` rows with `face_id = NULL` and are counted in
        `meta.orphaned_face_labels_total`). Returns face ids in the order of `faces`.
        """
        old_rows = self._conn.execute(
            """
            SELECT face_id, bbox_top, bbox_right, bbox_bottom, bbox_left, embedding_f16
            FROM faces WHERE photo_id = ? AND is_manual = 0
            ORDER BY face_idx ASC
            """,
            (photo_id,),
        ).fetchall()
        old_faces: list[StoredFace] = []
        old_boxes: dict[int, tuple[int, int, int, int]] = {}
        for r in old_rows:
            blob = bytes(r["embedding_f16"]) if r["embedding_f16"] is not None else b""
            emb = unpack_embeddings_f32([blob], dim=len(blob) // 2)[0] if len(blob) >= 2 else None
            fid = int(r["face_id"])
            box = (int(r["bbox_top"]), int(r["bbox_right"]), int(r["bbox_bottom"]), int(r["bbox_left"]))
            old_boxes[fid] = box
            old_faces.append(StoredFace(fid, box[0], box[1], box[2], box[3], emb))

        new_boxes = [(f.box.top, f.box.right, f.box.bottom, f.box.left) for f in faces]
        new_embs: NDArray[np.float32] | None = None
        if faces:
            raw = np.asarray([f.embedding for f in faces], dtype=np.float32)
            if raw.ndim == 2 and raw.shape[1] > 0:
                raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), np.float32(1e-8))
                new_embs = raw
        mapping = match_faces(old_faces, new_boxes, new_embs) if old_faces else {}

        # Park old auto faces on unique negative indexes so in-place updates never collide on
        # UNIQUE(photo_id, face_idx); manual boxes live at >= 1000 and are untouched.
        self._conn.execute(
            "UPDATE faces SET face_idx = -1 - face_id WHERE photo_id = ? AND is_manual = 0", (photo_id,)
        )
        kept = set(mapping.values())
        gone = [f.face_id for f in old_faces if f.face_id not in kept]
        if gone:
            gone_json = json.dumps(gone)
            lab = self._conn.execute(
                """
                SELECT COUNT(*) AS n FROM assignments
                WHERE face_id IN (SELECT value FROM json_each(?))
                  AND status IN ('confirmed', 'rejected', 'unsure')
                """,
                (gone_json,),
            ).fetchone()
            orphaned = int(lab["n"]) if lab is not None else 0
            holders = self._conn.execute(
                "SELECT DISTINCT person_id, face_id FROM assignments WHERE face_id IN (SELECT value FROM json_each(?))",
                (gone_json,),
            ).fetchall()
            self._conn.execute(
                """
                DELETE FROM assignments
                WHERE face_id IN (SELECT value FROM json_each(?)) AND status IN ('auto_high', 'candidate')
                """,
                (gone_json,),
            )
            for h in holders:
                self._refresh_cover_face_if_needed(int(h["person_id"]), int(h["face_id"]))
            self._conn.execute(
                "UPDATE assignments SET is_anchor = 0 WHERE face_id IN (SELECT value FROM json_each(?))",
                (gone_json,),
            )
            self._conn.execute(
                "DELETE FROM faces WHERE face_id IN (SELECT value FROM json_each(?))", (gone_json,)
            )
            if orphaned:
                self._conn.execute(
                    """
                    INSERT INTO meta (key, value) VALUES ('orphaned_face_labels_total', ?)
                    ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT)
                    """,
                    (str(orphaned), orphaned),
                )
            if report is not None:
                report.faces_removed += len(gone)
                report.labels_orphaned += orphaned
                report.removed_face_ids.extend(gone)

        face_ids: list[int] = []
        for f_idx, face in enumerate(faces):
            old_fid = mapping.get(f_idx)
            if old_fid is not None:
                self._conn.execute(
                    """
                    UPDATE faces SET face_idx = ?, det_score = ?,
                        bbox_top = ?, bbox_right = ?, bbox_bottom = ?, bbox_left = ?, embedding_f16 = ?
                    WHERE face_id = ?
                    """,
                    (
                        f_idx,
                        face.confidence,
                        face.box.top,
                        face.box.right,
                        face.box.bottom,
                        face.box.left,
                        pack_embedding_f16(face.embedding),
                        old_fid,
                    ),
                )
                face_ids.append(old_fid)
                if report is not None:
                    report.faces_relinked += 1
                    if old_boxes.get(old_fid) != new_boxes[f_idx]:
                        report.moved_face_ids.append(old_fid)
                continue
            face_ids.append(self._insert_face_locked(photo_id, f_idx, face))
            if report is not None:
                report.faces_new += 1
        return face_ids

    def _insert_face_locked(self, photo_id: int, f_idx: int, face: DetectedFace) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO faces (
                photo_id, face_idx, det_score,
                bbox_top, bbox_right, bbox_bottom, bbox_left, embedding_f16
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                photo_id,
                f_idx,
                face.confidence,
                face.box.top,
                face.box.right,
                face.box.bottom,
                face.box.left,
                pack_embedding_f16(face.embedding),
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("SQLite did not return lastrowid for inserted face.")
        return int(cur.lastrowid)

    def upsert_scanned_batch(
        self,
        items: list[tuple[PhotoProvenance, int, int, int, int, str, list[DetectedFace]]],
        *,
        content_hashes: list[str] | None = None,
        report: RelinkReport | None = None,
    ) -> list[tuple[int, list[int]]]:
        """
        Atomically persist a batch of scanned photos and their detected faces.
        Each item is `(prov, width, height, file_size, mtime_ns, detector_id, faces)`; optional
        `content_hashes[i]` stores the head/tail fingerprint of item `i`.
        Re-scanned photos keep their `face_id`s via `_relink_photo_faces_locked` (label-preserving).
        Returns `[(photo_id, [face_id, ...]), ...]` in matching order so callers can save crop thumbs.
        """
        if content_hashes is not None and len(content_hashes) != len(items):
            raise ValueError("content_hashes must align with items.")
        now = _now_iso()
        out_ids: list[tuple[int, list[int]]] = []
        with self._lock, self._conn:
            for item_i, (prov, width, height, fsize, mtime_ns, det_id, faces) in enumerate(items):
                c_hash = content_hashes[item_i] if content_hashes is not None else ""
                existed = (
                    self._conn.execute(
                        "SELECT photo_id FROM photos WHERE rel_path = ?", (prov.rel_path,)
                    ).fetchone()
                    is not None
                )
                self._conn.execute(
                    """
                    INSERT INTO photos (
                        rel_path, archive_child, year, month, album_leaf, filename,
                        kinderpedia_album_id, album_title, album_date, classroom_groups,
                        kinderpedia_image_id, seq_in_album, image_dateadd,
                        best_live_url, download_url, fullsize_url, thumb_url,
                        width, height, file_size, mtime_ns, detector_id, face_count, scanned_at,
                        content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(rel_path) DO UPDATE SET
                        archive_child = excluded.archive_child,
                        year = excluded.year,
                        month = excluded.month,
                        album_leaf = excluded.album_leaf,
                        filename = excluded.filename,
                        kinderpedia_album_id = excluded.kinderpedia_album_id,
                        album_title = excluded.album_title,
                        album_date = excluded.album_date,
                        classroom_groups = excluded.classroom_groups,
                        kinderpedia_image_id = excluded.kinderpedia_image_id,
                        seq_in_album = excluded.seq_in_album,
                        image_dateadd = excluded.image_dateadd,
                        best_live_url = excluded.best_live_url,
                        download_url = excluded.download_url,
                        fullsize_url = excluded.fullsize_url,
                        thumb_url = excluded.thumb_url,
                        width = excluded.width,
                        height = excluded.height,
                        file_size = excluded.file_size,
                        mtime_ns = excluded.mtime_ns,
                        detector_id = excluded.detector_id,
                        face_count = excluded.face_count,
                        scanned_at = excluded.scanned_at,
                        content_hash = excluded.content_hash
                    """,
                    (
                        prov.rel_path,
                        prov.archive_child,
                        prov.year,
                        prov.month,
                        prov.album_leaf,
                        prov.filename,
                        prov.kinderpedia_album_id,
                        prov.album_title,
                        prov.album_date,
                        prov.classroom_groups,
                        prov.kinderpedia_image_id,
                        prov.seq_in_album,
                        prov.image_dateadd,
                        prov.best_live_url,
                        prov.download_url,
                        prov.fullsize_url,
                        prov.thumb_url,
                        width,
                        height,
                        fsize,
                        mtime_ns,
                        det_id,
                        len(faces),
                        now,
                        c_hash,
                    ),
                )
                row = self._conn.execute(
                    "SELECT photo_id FROM photos WHERE rel_path = ?",
                    (prov.rel_path,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"Failed to read photo_id after upsert: {prov.rel_path}")
                photo_id = int(row["photo_id"])

                if existed:
                    face_ids = self._relink_photo_faces_locked(photo_id, faces, report)
                else:
                    face_ids = [self._insert_face_locked(photo_id, i, f) for i, f in enumerate(faces)]
                    if report is not None:
                        report.faces_new += len(faces)
                manual_row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM faces WHERE photo_id = ? AND is_manual != 0",
                    (photo_id,),
                ).fetchone()
                manual_cnt = int(manual_row["n"]) if manual_row is not None else 0
                if manual_cnt > 0:
                    self._conn.execute(
                        "UPDATE photos SET face_count = ? WHERE photo_id = ?",
                        (len(faces) + manual_cnt, photo_id),
                    )
                out_ids.append((photo_id, face_ids))
            self._photo_summary_cache = None
        with self._faces_matrix_lock:
            self._faces_matrix_cache = None
        return out_ids

    # ------------------------------------------------------------------
    # Cross-process change signal (`faces scan` -> running `faces ui`)
    # ------------------------------------------------------------------

    def get_data_version(self) -> int:
        """Monotonic counter bumped by `faces scan` (and photo reconciliation) after it commits.

        A long-running process (the Web UI) polls this cheaply and drops its in-memory caches
        (`invalidate_caches()`) when the value changes, since the scan runs in another process.
        """
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'data_version'").fetchone()
        try:
            return int(row["value"]) if row is not None else 0
        except (TypeError, ValueError):
            return 0

    def bump_data_version(self, *, scan_finished: bool = False) -> int:
        """Atomically increment `meta.data_version` (and stamp `meta.last_scan_at` for scans)."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO meta (key, value) VALUES ('data_version', '1')
                ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                """
            )
            if scan_finished:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('last_scan_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (_now_iso(),),
                )
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'data_version'").fetchone()
        return int(row["value"])

    def get_schema_version(self) -> int:
        with self._lock:
            return read_schema_version(self._conn)

    # ------------------------------------------------------------------
    # People & Reference Seed Management
    # ------------------------------------------------------------------

    def ensure_person(
        self,
        display_name: str,
        *,
        role: PersonRole = "classmate",
        archive_hint: str = "",
    ) -> int:
        clean = display_name.strip()
        if not clean:
            raise ValueError("Person display_name must not be empty.")
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT person_id, role FROM people WHERE display_name = ? COLLATE NOCASE",
                (clean,),
            ).fetchone()
            if existing is not None:
                pid = int(existing["person_id"])
                if role == "family" and str(existing["role"]) != "family":
                    self._conn.execute(
                        "UPDATE people SET role = 'family', archive_hint = COALESCE(NULLIF(archive_hint, ''), ?) WHERE person_id = ?",
                        (archive_hint, pid),
                    )
                return pid
            cur = self._conn.execute(
                """
                INSERT INTO people (display_name, role, archive_hint, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean, role, archive_hint, _now_iso()),
            )
            if cur.lastrowid is None:
                raise RuntimeError("SQLite did not return lastrowid for inserted person.")
            return int(cur.lastrowid)

    def upsert_reference_seed(
        self,
        *,
        person_id: int,
        source_key: str,
        embedding: tuple[float, ...] | NDArray[Any],
        anchor_year: int | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO reference_seeds (person_id, source_key, anchor_year, embedding_f16, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    person_id = excluded.person_id,
                    anchor_year = excluded.anchor_year,
                    embedding_f16 = excluded.embedding_f16,
                    created_at = excluded.created_at
                """,
                (person_id, source_key, anchor_year, pack_embedding_f16(embedding), _now_iso()),
            )

    def update_person(
        self,
        person_id: int,
        *,
        display_name: str | None = None,
        role: PersonRole | None = None,
        cover_face_id: int | None = None,
        birth_date: str | None = None,
    ) -> dict[str, Any]:
        birthday_removed = 0
        with self._lock, self._conn:
            if self._conn.execute(
                "SELECT 1 FROM people WHERE person_id = ?", (person_id,)
            ).fetchone() is None:
                raise KeyError(f"Person {person_id} not found.")
            if birth_date is not None:
                clean_bd = normalize_iso_date(birth_date)
                self._conn.execute(
                    "UPDATE people SET birth_date = ? WHERE person_id = ?",
                    (clean_bd, person_id),
                )
                if clean_bd:
                    birthday_removed = self._enforce_birth_date_locked(person_id, clean_bd)
            if display_name is not None:
                clean = display_name.strip()
                if not clean:
                    raise ValueError("Display name cannot be empty.")
                self._conn.execute(
                    "UPDATE people SET display_name = ? WHERE person_id = ?",
                    (clean, person_id),
                )
            if role is not None:
                self._conn.execute("UPDATE people SET role = ? WHERE person_id = ?", (role, person_id))
                if role == "ignore":
                    self._conn.execute(
                        "DELETE FROM assignments WHERE person_id = ? AND status IN ('auto_high', 'candidate')",
                        (person_id,),
                    )
                    self._conn.execute(
                        "UPDATE clusters SET suggested_person_id = NULL, suggested_similarity = NULL WHERE suggested_person_id = ?",
                        (person_id,),
                    )
            if cover_face_id is not None:
                self._conn.execute(
                    "UPDATE people SET cover_face_id = ? WHERE person_id = ?",
                    (cover_face_id, person_id),
                )
        people = {p["person_id"]: p for p in self.list_people()}
        if person_id not in people:
            raise KeyError(f"Person {person_id} not found.")
        return {**people[person_id], "birthday_removed": birthday_removed}

    # ------------------------------------------------------------------
    # Birthdays, Classes & Eligibility (who can appear on which photo)
    # ------------------------------------------------------------------

    def _delete_manual_faces_locked(self, face_ids: list[int]) -> list[int]:
        """Delete user-drawn manual face rows that no longer carry any assignment."""
        deleted: list[int] = []
        for fid in face_ids:
            row = self._conn.execute(
                "SELECT photo_id, is_manual FROM faces WHERE face_id = ?", (fid,)
            ).fetchone()
            if row is None or not bool(row["is_manual"]):
                continue
            still = self._conn.execute(
                "SELECT 1 FROM assignments WHERE face_id = ? AND status != 'rejected' LIMIT 1",
                (fid,),
            ).fetchone()
            if still is not None:
                continue
            self._conn.execute("DELETE FROM assignments WHERE face_id = ?", (fid,))
            self._conn.execute("DELETE FROM faces WHERE face_id = ?", (fid,))
            self._conn.execute(
                "UPDATE photos SET face_count = MAX(0, face_count - 1) WHERE photo_id = ?",
                (int(row["photo_id"]),),
            )
            if self._photo_summary_cache is not None:
                self._photo_summary_cache["faces_total"] = max(
                    0, int(self._photo_summary_cache["faces_total"]) - 1
                )
            deleted.append(fid)
        return deleted

    def _enforce_birth_date_locked(self, person_id: int, birth_date: str) -> int:
        """Remove every non-rejected tag of `person_id` on photos dated before `birth_date`."""
        bd_ord = date_to_ord(birth_date)
        if bd_ord <= 0:
            return 0
        rows = self._conn.execute(
            """
            SELECT a.photo_id, a.face_id, p.album_date, p.image_dateadd, p.year, p.month
            FROM assignments a
            JOIN photos p ON p.photo_id = a.photo_id
            WHERE a.person_id = ? AND a.status != 'rejected'
            """,
            (person_id,),
        ).fetchall()
        doomed = [
            r
            for r in rows
            if photo_date_ord(
                str(r["album_date"] or ""),
                str(r["image_dateadd"] or ""),
                str(r["year"] or ""),
                str(r["month"] or ""),
            )
            < bd_ord
        ]
        if not doomed:
            return 0
        self._conn.executemany(
            "DELETE FROM assignments WHERE photo_id = ? AND person_id = ? AND status != 'rejected'",
            [(int(r["photo_id"]), person_id) for r in doomed],
        )
        lost_fids = [int(r["face_id"]) for r in doomed if r["face_id"] is not None]
        for fid in lost_fids:
            self._refresh_cover_face_if_needed(person_id, fid)
        self._delete_manual_faces_locked(lost_fids)
        return len(doomed)

    def _photo_ord_and_feed_locked(self, photo_id: int) -> tuple[int, str]:
        row = self._conn.execute(
            "SELECT album_date, image_dateadd, year, month, archive_child FROM photos WHERE photo_id = ?",
            (photo_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Photo {photo_id} not found.")
        return (
            photo_date_ord(
                str(row["album_date"] or ""),
                str(row["image_dateadd"] or ""),
                str(row["year"] or ""),
                str(row["month"] or ""),
            ),
            str(row["archive_child"] or ""),
        )

    def _person_rules_locked(self, person_id: int) -> tuple[int, list[tuple[str, int, int]]]:
        """Return `(birth_ord, [(feed, start_ord, end_ord), ...])` for `person_id`."""
        p_row = self._conn.execute(
            "SELECT birth_date FROM people WHERE person_id = ?", (person_id,)
        ).fetchone()
        birth_ord = date_to_ord(str(p_row["birth_date"] or "")) if p_row is not None else 0
        cls_rows = self._conn.execute(
            """
            SELECT c.archive_child, c.start_date, c.end_date
            FROM class_members m
            JOIN school_classes c ON c.class_id = m.class_id
            WHERE m.person_id = ?
            """,
            (person_id,),
        ).fetchall()
        windows = [
            (
                str(r["archive_child"] or ""),
                date_to_ord(str(r["start_date"] or "")),
                date_to_ord(str(r["end_date"] or "")) or UNKNOWN_DATE_ORD,
            )
            for r in cls_rows
        ]
        return birth_ord, windows

    @staticmethod
    def _rules_allow(
        rules: tuple[int, list[tuple[str, int, int]]],
        photo_ord: int,
        feed: str,
        *,
        check_classes: bool,
    ) -> bool:
        birth_ord, windows = rules
        if birth_ord > 0 and photo_ord < birth_ord:
            return False
        if check_classes and windows:
            return any(
                (not w_feed or w_feed == feed) and w_start <= photo_ord <= w_end
                for w_feed, w_start, w_end in windows
            )
        return True

    def assert_can_tag(self, photo_id: int, person_id: int) -> None:
        """Raise `ValueError` when `photo_id` predates `person_id`'s birthday."""
        with self._lock:
            p_ord, feed = self._photo_ord_and_feed_locked(photo_id)
            rules = self._person_rules_locked(person_id)
            name_row = self._conn.execute(
                "SELECT display_name, birth_date FROM people WHERE person_id = ?", (person_id,)
            ).fetchone()
        if not self._rules_allow(rules, p_ord, feed, check_classes=False):
            nm = str(name_row["display_name"]) if name_row is not None else str(person_id)
            bd = str(name_row["birth_date"]) if name_row is not None else ""
            raise ValueError(f"This photo is dated before {nm}'s birthday ({bd}).")

    def load_eligibility_rules(self) -> dict[int, tuple[int, list[tuple[str, int, int]]]]:
        """Return `{person_id: (birth_ord, class_windows)}` for people with any restriction."""
        with self._lock:
            pids = [
                int(r["person_id"])
                for r in self._conn.execute(
                    """
                    SELECT person_id FROM people WHERE birth_date != ''
                    UNION
                    SELECT DISTINCT person_id FROM class_members
                    """
                ).fetchall()
            ]
            return {pid: self._person_rules_locked(pid) for pid in pids}

    def delete_person(self, person_id: int) -> dict[str, Any]:
        """Delete a profile entirely; its faces become uncategorized and its manual boxes are removed."""
        with self._lock, self._conn:
            if self._conn.execute(
                "SELECT 1 FROM people WHERE person_id = ?", (person_id,)
            ).fetchone() is None:
                raise KeyError(f"Person {person_id} not found.")
            fids = [
                int(r["face_id"])
                for r in self._conn.execute(
                    "SELECT face_id FROM assignments WHERE person_id = ? AND face_id IS NOT NULL",
                    (person_id,),
                ).fetchall()
            ]
            removed = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM assignments WHERE person_id = ? AND status != 'rejected'",
                    (person_id,),
                ).fetchone()["n"]
            )
            # Explicit delete (instead of relying on FK cascade) keeps count triggers exact:
            self._conn.execute("DELETE FROM assignments WHERE person_id = ?", (person_id,))
            deleted_manual = self._delete_manual_faces_locked(fids)
            self._conn.execute(
                "UPDATE clusters SET person_id = NULL WHERE person_id = ?", (person_id,)
            )
            self._conn.execute(
                "UPDATE clusters SET suggested_person_id = NULL, suggested_similarity = NULL WHERE suggested_person_id = ?",
                (person_id,),
            )
            self._conn.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
            self._conn.execute(
                "DELETE FROM person_assignment_counts WHERE person_id = ?", (person_id,)
            )
        return {
            "deleted_person_id": person_id,
            "removed_assignments": removed,
            "deleted_manual_face_ids": deleted_manual,
        }

    def _class_dict_locked(self, class_id: int) -> dict[str, Any]:
        r = self._conn.execute(
            "SELECT class_id, name, archive_child, start_date, end_date FROM school_classes WHERE class_id = ?",
            (class_id,),
        ).fetchone()
        if r is None:
            raise KeyError(f"Class {class_id} not found.")
        members = [
            int(m["person_id"])
            for m in self._conn.execute(
                "SELECT person_id FROM class_members WHERE class_id = ? ORDER BY person_id",
                (class_id,),
            ).fetchall()
        ]
        return {
            "class_id": int(r["class_id"]),
            "name": str(r["name"]),
            "archive_child": str(r["archive_child"] or ""),
            "start_date": str(r["start_date"] or ""),
            "end_date": str(r["end_date"] or ""),
            "member_ids": members,
        }

    def list_classes(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = [
                int(r["class_id"])
                for r in self._conn.execute(
                    "SELECT class_id FROM school_classes ORDER BY archive_child COLLATE NOCASE, start_date, name COLLATE NOCASE"
                ).fetchall()
            ]
            return [self._class_dict_locked(cid) for cid in ids]

    def _validated_class_fields(
        self, *, name: str | None, archive_child: str | None, start_date: str | None, end_date: str | None
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        if name is not None:
            clean = name.strip()
            if not clean:
                raise ValueError("Class name cannot be empty.")
            out["name"] = clean
        if archive_child is not None:
            out["archive_child"] = archive_child.strip()
        if start_date is not None:
            out["start_date"] = normalize_iso_date(start_date)
        if end_date is not None:
            out["end_date"] = normalize_iso_date(end_date)
        return out

    def create_class(
        self, *, name: str, archive_child: str = "", start_date: str = "", end_date: str = ""
    ) -> dict[str, Any]:
        f = self._validated_class_fields(
            name=name, archive_child=archive_child, start_date=start_date, end_date=end_date
        )
        if f["start_date"] and f["end_date"] and f["end_date"] < f["start_date"]:
            raise ValueError("Class end date must be on or after its start date.")
        with self._lock, self._conn:
            if self._conn.execute(
                "SELECT 1 FROM school_classes WHERE name = ? COLLATE NOCASE", (f["name"],)
            ).fetchone() is not None:
                raise ValueError(f"A class named {f['name']!r} already exists.")
            cur = self._conn.execute(
                "INSERT INTO school_classes (name, archive_child, start_date, end_date, created_at) VALUES (?, ?, ?, ?, ?)",
                (f["name"], f["archive_child"], f["start_date"], f["end_date"], _now_iso()),
            )
            if cur.lastrowid is None:
                raise RuntimeError("SQLite did not return lastrowid for inserted class.")
            return self._class_dict_locked(int(cur.lastrowid))

    def update_class(
        self,
        class_id: int,
        *,
        name: str | None = None,
        archive_child: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        f = self._validated_class_fields(
            name=name, archive_child=archive_child, start_date=start_date, end_date=end_date
        )
        with self._lock, self._conn:
            cur = self._class_dict_locked(class_id)
            merged = {**cur, **f}
            if merged["start_date"] and merged["end_date"] and merged["end_date"] < merged["start_date"]:
                raise ValueError("Class end date must be on or after its start date.")
            for col, val in f.items():
                self._conn.execute(
                    f"UPDATE school_classes SET {col} = ? WHERE class_id = ?",  # noqa: S608 - fixed column names
                    (val, class_id),
                )
            return self._class_dict_locked(class_id)

    def delete_class(self, class_id: int) -> None:
        with self._lock, self._conn:
            self._class_dict_locked(class_id)
            self._conn.execute("DELETE FROM school_classes WHERE class_id = ?", (class_id,))

    def set_class_members(self, class_id: int, person_ids: list[int]) -> dict[str, Any]:
        with self._lock, self._conn:
            self._class_dict_locked(class_id)
            valid = {
                int(r["person_id"])
                for r in self._conn.execute("SELECT person_id FROM people").fetchall()
            }
            self._conn.execute("DELETE FROM class_members WHERE class_id = ?", (class_id,))
            self._conn.executemany(
                "INSERT OR IGNORE INTO class_members (class_id, person_id) VALUES (?, ?)",
                [(class_id, pid) for pid in sorted(set(person_ids)) if pid in valid],
            )
            return self._class_dict_locked(class_id)

    def auto_create_classes(self) -> int:
        """Create one class per feed x school year (Sep 1 - Aug 31) that contains photos."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT archive_child, album_date, image_dateadd, year, month FROM photos WHERE archive_child != ''"
            ).fetchall()
            feeds = {
                str(r["archive_child"])
                for r in self._conn.execute(
                    "SELECT DISTINCT archive_child FROM photos WHERE archive_child != ''"
                ).fetchall()
            }
        counts: dict[tuple[str, int], int] = {}
        for r in rows:
            o = photo_date_ord(
                str(r["album_date"] or ""),
                str(r["image_dateadd"] or ""),
                str(r["year"] or ""),
                str(r["month"] or ""),
            )
            if o >= UNKNOWN_DATE_ORD:
                continue
            y, m = o // 10000, (o // 100) % 100
            sy = y if m >= 9 else y - 1
            key = (str(r["archive_child"]), sy)
            counts[key] = counts.get(key, 0) + 1
        created = 0
        for (feed, sy), n in sorted(counts.items()):
            if feed not in feeds or n < 5:
                continue
            name = f"{feed} {sy}\u2013{str(sy + 1)[-2:]}"
            try:
                self.create_class(
                    name=name,
                    archive_child=feed,
                    start_date=f"{sy}-09-01",
                    end_date=f"{sy + 1}-08-31",
                )
                created += 1
            except ValueError:
                continue
        return created

    def get_class_matrix(self) -> dict[str, Any]:
        classes = self.list_classes()
        with self._lock:
            people = [
                {"person_id": int(r["person_id"]), "display_name": str(r["display_name"]), "role": str(r["role"])}
                for r in self._conn.execute(
                    "SELECT person_id, display_name, role FROM people ORDER BY display_name COLLATE NOCASE"
                ).fetchall()
            ]
            photo_rows = self._conn.execute(
                "SELECT photo_id, archive_child, album_date, image_dateadd, year, month FROM photos"
            ).fetchall()
            tag_rows = self._conn.execute(
                "SELECT photo_id, person_id FROM assignments WHERE status IN ('confirmed', 'auto_high')"
            ).fetchall()
        photo_info = {
            int(r["photo_id"]): (
                str(r["archive_child"] or ""),
                photo_date_ord(
                    str(r["album_date"] or ""),
                    str(r["image_dateadd"] or ""),
                    str(r["year"] or ""),
                    str(r["month"] or ""),
                ),
            )
            for r in photo_rows
        }
        windows = [
            (
                c["class_id"],
                c["archive_child"],
                date_to_ord(c["start_date"]),
                date_to_ord(c["end_date"]) or UNKNOWN_DATE_ORD,
            )
            for c in classes
        ]
        photo_counts: dict[int, int] = {c["class_id"]: 0 for c in classes}
        photo_classes: dict[int, list[int]] = {}
        for ph_id, (feed, o) in photo_info.items():
            hit = [
                cid
                for cid, w_feed, w_s, w_e in windows
                if (not w_feed or w_feed == feed) and w_s <= o <= w_e
            ]
            if hit:
                photo_classes[ph_id] = hit
                for cid in hit:
                    photo_counts[cid] += 1
        counts: dict[str, dict[str, int]] = {str(c["class_id"]): {} for c in classes}
        for r in tag_rows:
            for cid in photo_classes.get(int(r["photo_id"]), []):
                bucket = counts[str(cid)]
                key = str(int(r["person_id"]))
                bucket[key] = bucket.get(key, 0) + 1
        return {
            "classes": [{**c, "photo_count": photo_counts[c["class_id"]]} for c in classes],
            "people": people,
            "counts": counts,
        }

    def merge_people(self, source_person_id: int, target_person_id: int) -> None:
        """Merge `source_person_id` into `target_person_id`, keeping human-confirmed ground truth."""
        if source_person_id == target_person_id:
            return
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE clusters SET person_id = ? WHERE person_id = ?",
                (target_person_id, source_person_id),
            )
            self._conn.execute(
                "UPDATE reference_seeds SET person_id = ? WHERE person_id = ?",
                (target_person_id, source_person_id),
            )
            src_rows = self._conn.execute(
                "SELECT photo_id, face_id, status, source, similarity, is_anchor, updated_at FROM assignments WHERE person_id = ?",
                (source_person_id,),
            ).fetchall()
            for r in src_rows:
                existing = self._conn.execute(
                    "SELECT status FROM assignments WHERE photo_id = ? AND person_id = ?",
                    (int(r["photo_id"]), target_person_id),
                ).fetchone()
                if existing is None or (
                    str(r["status"]) == "confirmed" and str(existing["status"]) != "confirmed"
                ):
                    self._conn.execute(
                        """
                        INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(photo_id, person_id) DO UPDATE SET
                            face_id = excluded.face_id,
                            status = excluded.status,
                            source = excluded.source,
                            similarity = excluded.similarity,
                            is_anchor = excluded.is_anchor,
                            updated_at = excluded.updated_at
                        """,
                        (
                            int(r["photo_id"]),
                            target_person_id,
                            r["face_id"],
                            str(r["status"]),
                            str(r["source"]),
                            float(r["similarity"]),
                            int(r["is_anchor"]),
                            str(r["updated_at"]),
                        ),
                    )
            self._conn.execute(
                "INSERT OR IGNORE INTO class_members (class_id, person_id) SELECT class_id, ? FROM class_members WHERE person_id = ?",
                (target_person_id, source_person_id),
            )
            self._conn.execute(
                """
                INSERT OR IGNORE INTO person_face_pressure (person_id, face_id, weight, updated_at)
                SELECT ?, face_id, weight, updated_at FROM person_face_pressure WHERE person_id = ?
                """,
                (target_person_id, source_person_id),
            )
            self._conn.execute("DELETE FROM assignments WHERE person_id = ?", (source_person_id,))
            self._conn.execute("DELETE FROM people WHERE person_id = ?", (source_person_id,))
            self._conn.execute(
                "DELETE FROM person_assignment_counts WHERE person_id = ?", (source_person_id,)
            )
            tgt_bd = self._conn.execute(
                "SELECT birth_date FROM people WHERE person_id = ?", (target_person_id,)
            ).fetchone()
            if tgt_bd is not None and str(tgt_bd["birth_date"] or ""):
                self._enforce_birth_date_locked(target_person_id, str(tgt_bd["birth_date"]))

    def list_people(self) -> list[dict[str, Any]]:
        """List all enrolled people with live counts of confirmed, auto_high, candidate, and rejected photos."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    p.person_id,
                    p.display_name,
                    p.role,
                    p.archive_hint,
                    p.cover_face_id,
                    p.birth_date,
                    COALESCE(c.confirmed_count, 0) AS confirmed_count,
                    COALESCE(c.auto_high_count, 0) AS auto_high_count,
                    COALESCE(c.album_cohesion_count, 0) AS album_cohesion_count,
                    COALESCE(c.candidate_count, 0) AS candidate_count,
                    COALESCE(c.unsure_count, 0) AS unsure_count,
                    COALESCE(c.rejected_count, 0) AS rejected_count
                FROM people p
                LEFT JOIN person_assignment_counts c ON c.person_id = p.person_id
                ORDER BY
                    CASE p.role
                        WHEN 'family' THEN 0
                        WHEN 'classmate' THEN 1
                        WHEN 'educator' THEN 2
                        ELSE 3
                    END,
                    (COALESCE(c.confirmed_count, 0) + COALESCE(c.auto_high_count, 0)) DESC,
                    p.display_name COLLATE NOCASE ASC
                """
            ).fetchall()

            class_ids_by_pid: dict[int, list[int]] = {}
            for m in self._conn.execute(
                "SELECT person_id, class_id FROM class_members ORDER BY class_id"
            ).fetchall():
                class_ids_by_pid.setdefault(int(m["person_id"]), []).append(int(m["class_id"]))

            result: list[dict[str, Any]] = []
            for r in rows:
                pid = int(r["person_id"])
                confirmed_n = int(r["confirmed_count"] or 0)
                auto_n = int(r["auto_high_count"] or 0)
                cohesion_n = int(r["album_cohesion_count"] or 0)
                cand_n = int(r["candidate_count"] or 0)
                cover_fid = int(r["cover_face_id"]) if r["cover_face_id"] is not None else None
                if cover_fid is None and (confirmed_n + auto_n) > 0:
                    fb = self._conn.execute(
                        """
                        SELECT a.face_id
                        FROM assignments a
                        JOIN faces f ON f.face_id = a.face_id
                        WHERE a.person_id = ? AND a.status IN ('confirmed', 'auto_high')
                        ORDER BY
                            f.is_manual ASC,
                            (CASE WHEN a.status = 'confirmed' THEN 1 ELSE 0 END) DESC,
                            f.det_score DESC
                        LIMIT 1
                        """,
                        (pid,),
                    ).fetchone()
                    if fb is not None and fb["face_id"] is not None:
                        cover_fid = int(fb["face_id"])
                result.append(
                    {
                        "person_id": pid,
                        "display_name": str(r["display_name"]),
                        "role": str(r["role"]),
                        "archive_hint": str(r["archive_hint"] or ""),
                        "cover_face_id": cover_fid,
                        "confirmed_count": confirmed_n,
                        "auto_high_count": auto_n,
                        "album_cohesion_count": cohesion_n,
                        "candidate_count": cand_n,
                        "review_ready_count": max(0, auto_n - cohesion_n) + cand_n,
                        "total_included_count": confirmed_n + auto_n,
                        "unsure_count": int(r["unsure_count"] or 0),
                        "rejected_count": int(r["rejected_count"] or 0),
                        "birth_date": str(r["birth_date"] or ""),
                        "class_ids": class_ids_by_pid.get(pid, []),
                    }
                )
            return result

    # ------------------------------------------------------------------
    # Vector Matrix Loaders for Fast Numpy Clustering & Matching
    # ------------------------------------------------------------------

    def invalidate_faces_matrix_cache(self) -> None:
        """Explicitly clear the in-memory face embedding matrix cache (e.g. after external schema changes)."""
        with self._faces_matrix_lock:
            self._faces_matrix_cache = None

    def _query_suppressed_mask(self, face_ids: NDArray[np.int64]) -> NDArray[np.bool_]:
        n = int(face_ids.shape[0])
        suppressed = np.zeros((n,), dtype=bool)
        if n == 0:
            return suppressed
        with self._lock:
            sup_rows = self._conn.execute(
                "SELECT face_id FROM faces WHERE suppress_auto != 0 AND is_manual = 0"
            ).fetchall()
        if sup_rows:
            sup_fids = np.fromiter(
                (int(r["face_id"]) for r in sup_rows),
                dtype=np.int64,
                count=len(sup_rows),
            )
            pos = np.searchsorted(face_ids, sup_fids)
            valid = (pos < n) & (face_ids[np.minimum(pos, n - 1)] == sup_fids)
            if np.any(valid):
                suppressed[pos[valid]] = True
        return suppressed

    def load_all_faces_matrix(self, *, dim: int = 512) -> dict[str, Any]:
        """
        Return all face vectors and metadata as contiguous numpy arrays.
        Cached in memory across active-learning sweeps (`0.03ms` warm vs `~1s` cold SQLite BLOB read)
        while reading live `suppress_auto` flags from SQLite on every call.

        The cold load streams rows with `fetchmany` straight into a preallocated `(N, dim)` float32
        matrix and L2-normalizes it in place, so peak memory stays close to the final matrix size
        (no list of 120k `sqlite3.Row`s + joined blob + float32 copy + normalized copy).
        Lock order: takes `_faces_matrix_lock` then `_lock` (never call while holding `_lock`).
        """
        with self._faces_matrix_lock:
            cached = self._faces_matrix_cache
            if cached is not None and cached.get("dim") == dim:
                face_ids_c: NDArray[np.int64] = cached["face_ids"]
                return {
                    **cached,
                    "suppressed_auto": self._query_suppressed_mask(face_ids_c),
                }

            chunk_rows = 4096
            with self._lock:
                cnt_row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM faces WHERE is_manual = 0"
                ).fetchone()
                cap = max(1, int(cnt_row["n"]) if cnt_row is not None else 0)
                embs_buf = np.empty((cap, dim), dtype=np.float32)
                face_ids = np.empty((cap,), dtype=np.int64)
                photo_ids = np.empty((cap,), dtype=np.int64)
                det_scores = np.empty((cap,), dtype=np.float32)
                min_dims = np.empty((cap,), dtype=np.int32)
                suppressed_auto = np.empty((cap,), dtype=bool)
                date_ords = np.empty((cap,), dtype=np.int32)
                archive_children: list[str] = []
                years: list[str] = []
                album_keys: list[str] = []
                cur = self._conn.execute(
                    """
                    SELECT
                        f.face_id,
                        f.photo_id,
                        f.det_score,
                        (f.bbox_right - f.bbox_left) AS bw,
                        (f.bbox_bottom - f.bbox_top) AS bh,
                        f.embedding_f16,
                        f.suppress_auto,
                        p.archive_child,
                        p.year,
                        p.month,
                        p.album_date,
                        p.image_dateadd,
                        p.album_leaf
                    FROM faces f
                    JOIN photos p ON p.photo_id = f.photo_id
                    WHERE f.is_manual = 0
                    ORDER BY f.face_id ASC
                    """
                )
                n = 0
                while True:
                    chunk = cur.fetchmany(chunk_rows)
                    if not chunk:
                        break
                    k = len(chunk)
                    if n + k > cap:  # rows were added between COUNT and SELECT (other process)
                        cap = max(cap * 2, n + k)
                        embs_buf = np.resize(embs_buf, (cap, dim))
                        face_ids = np.resize(face_ids, (cap,))
                        photo_ids = np.resize(photo_ids, (cap,))
                        det_scores = np.resize(det_scores, (cap,))
                        min_dims = np.resize(min_dims, (cap,))
                        suppressed_auto = np.resize(suppressed_auto, (cap,))
                        date_ords = np.resize(date_ords, (cap,))
                    embs_buf[n : n + k] = np.frombuffer(
                        b"".join(r["embedding_f16"] for r in chunk), dtype=np.float16
                    ).reshape(k, dim)
                    for j, r in enumerate(chunk):
                        i = n + j
                        face_ids[i] = int(r["face_id"])
                        photo_ids[i] = int(r["photo_id"])
                        det_scores[i] = float(r["det_score"])
                        min_dims[i] = min(int(r["bw"]), int(r["bh"]))
                        suppressed_auto[i] = bool(r["suppress_auto"])
                        ach = str(r["archive_child"])
                        yr = str(r["year"])
                        archive_children.append(ach)
                        years.append(yr)
                        album_keys.append(f"{ach}/{r['album_leaf']}")
                        date_ords[i] = photo_date_ord(
                            str(r["album_date"] or ""),
                            str(r["image_dateadd"] or ""),
                            yr,
                            str(r["month"] or ""),
                        )
                    n += k
                    del chunk

            if n == 0:
                empty_payload: dict[str, Any] = {
                    "dim": dim,
                    "face_ids": np.zeros((0,), dtype=np.int64),
                    "photo_ids": np.zeros((0,), dtype=np.int64),
                    "det_scores": np.zeros((0,), dtype=np.float32),
                    "min_dims": np.zeros((0,), dtype=np.int32),
                    "suppressed_auto": np.zeros((0,), dtype=bool),
                    "archive_children": [],
                    "years": [],
                    "album_keys": [],
                    "rows_by_album": {},
                    "fc_by_photo": {},
                    "rows_by_photo": {},
                    "year_ints": np.zeros((0,), dtype=np.int32),
                    "date_ords": np.zeros((0,), dtype=np.int32),
                    "day_nums": np.zeros((0,), dtype=np.int32),
                    "embeddings": np.zeros((0, dim), dtype=np.float32),
                }
                self._faces_matrix_cache = empty_payload
                return dict(empty_payload)

            embs = embs_buf[:n] if n < cap else embs_buf
            face_ids = face_ids[:n]
            photo_ids = photo_ids[:n]
            det_scores = det_scores[:n]
            min_dims = min_dims[:n]
            suppressed_auto = suppressed_auto[:n]
            date_ords = date_ords[:n]
            norm_block = 16384
            for s in range(0, n, norm_block):
                blk = embs[s : s + norm_block]
                norms = np.linalg.norm(blk, axis=1, keepdims=True)
                np.maximum(norms, np.float32(1e-8), out=norms)
                blk /= norms
            embs.setflags(write=False)

            year_ints = np.fromiter(
                (int(y) if y.isdigit() and len(y) == 4 else 0 for y in years),
                dtype=np.int32,
                count=n,
            )
            tmp_alb: dict[str, list[int]] = {}
            tmp_photo: dict[int, list[int]] = {}
            photo_list = photo_ids.tolist()
            for idx_i in range(n):
                akey = album_keys[idx_i]
                alb_list = tmp_alb.get(akey)
                if alb_list is None:
                    tmp_alb[akey] = [idx_i]
                else:
                    alb_list.append(idx_i)
                pid_i = int(photo_list[idx_i])
                ph_list = tmp_photo.get(pid_i)
                if ph_list is None:
                    tmp_photo[pid_i] = [idx_i]
                else:
                    ph_list.append(idx_i)

            rows_by_album = {
                akey: np.asarray(r_list, dtype=np.int64) for akey, r_list in tmp_alb.items()
            }
            fc_by_photo = {pid_i: len(r_list) for pid_i, r_list in tmp_photo.items()}

            base_cache: dict[str, Any] = {
                "dim": dim,
                "face_ids": face_ids,
                "photo_ids": photo_ids,
                "det_scores": det_scores,
                "min_dims": min_dims,
                "archive_children": archive_children,
                "years": years,
                "album_keys": album_keys,
                "rows_by_album": rows_by_album,
                "fc_by_photo": fc_by_photo,
                "rows_by_photo": tmp_photo,
                "year_ints": year_ints,
                "date_ords": date_ords,
                "day_nums": np.fromiter(
                    (date_ord_to_day_num(int(o)) for o in date_ords.tolist()),
                    dtype=np.int32,
                    count=n,
                ),
                "embeddings": embs,
            }
            self._faces_matrix_cache = base_cache
            return {
                **base_cache,
                "suppressed_auto": suppressed_auto,
            }

    def load_anchors_and_negatives(self, *, dim: int = 512) -> dict[str, Any]:
        """Load active people, reference seeds, human-confirmed faces, and hard-negative rejected faces."""
        with self._faces_matrix_lock:
            cached_mat = (
                self._faces_matrix_cache
                if self._faces_matrix_cache is not None and self._faces_matrix_cache.get("dim") == dim
                else None
            )

        with self._lock:
            people_rows = self._conn.execute(
                """
                SELECT person_id, display_name, role, archive_hint
                FROM people
                WHERE role IN ('family', 'classmate', 'educator')
                """
            ).fetchall()
            seed_rows = self._conn.execute(
                "SELECT person_id, anchor_year, embedding_f16 FROM reference_seeds"
            ).fetchall()
            pressure_rows = self._conn.execute(
                "SELECT person_id, face_id, weight FROM person_face_pressure WHERE weight != 0"
            ).fetchall()
            ignore_ids = [
                int(r["person_id"])
                for r in self._conn.execute("SELECT person_id FROM people WHERE role = 'ignore'").fetchall()
            ]
            if cached_mat is not None:
                human_rows = self._conn.execute(
                    """
                    SELECT
                        a.photo_id, a.person_id, a.face_id, a.status, a.source,
                        p.year, p.archive_child, p.album_leaf, p.face_count
                    FROM assignments a
                    JOIN photos p ON p.photo_id = a.photo_id
                    WHERE a.status IN ('confirmed', 'rejected', 'unsure')
                    ORDER BY a.updated_at ASC
                    """
                ).fetchall()
            else:
                human_rows = self._conn.execute(
                    """
                    SELECT
                        a.photo_id, a.person_id, a.face_id, a.status, a.source,
                        f.det_score, f.embedding_f16,
                        p.year, p.archive_child, p.album_leaf, p.face_count
                    FROM assignments a
                    LEFT JOIN faces f ON f.face_id = a.face_id AND f.is_manual = 0
                    JOIN photos p ON p.photo_id = a.photo_id
                    WHERE a.status IN ('confirmed', 'rejected', 'unsure')
                    ORDER BY a.updated_at ASC
                    """
                ).fetchall()

        seed_blobs = [bytes(r["embedding_f16"]) for r in seed_rows]
        seed_embs = unpack_embeddings_f32(seed_blobs, dim=dim) if seed_blobs else None

        human_items: list[dict[str, Any]] = []
        if cached_mat is not None:
            c_fids: NDArray[np.int64] = cached_mat["face_ids"]
            c_scores: NDArray[np.float32] = cached_mat["det_scores"]
            c_embs: NDArray[np.float32] = cached_mat["embeddings"]
            n_cached = int(c_fids.shape[0])
            n_hr = len(human_rows)
            if n_hr > 0 and n_cached > 0:
                raw_fids = np.fromiter(
                    (int(r["face_id"]) if r["face_id"] is not None else -1 for r in human_rows),
                    dtype=np.int64,
                    count=n_hr,
                )
                pos_arr = np.searchsorted(c_fids, np.maximum(raw_fids, 0))
                safe_pos = np.minimum(pos_arr, n_cached - 1)
                hit_arr = (raw_fids >= 0) & (pos_arr < n_cached) & (c_fids[safe_pos] == raw_fids)
                pos_list = pos_arr.tolist()
                hit_list = hit_arr.tolist()
            else:
                pos_list = [0] * n_hr
                hit_list = [False] * n_hr

            for idx, r in enumerate(human_rows):
                fid_val = int(r["face_id"]) if r["face_id"] is not None else None
                det_sc = 0.0
                emb_vec: NDArray[np.float32] | None = None
                if hit_list[idx]:
                    pos = pos_list[idx]
                    det_sc = float(c_scores[pos])
                    emb_vec = c_embs[pos]
                yr_str = str(r["year"])
                human_items.append(
                    {
                        "photo_id": int(r["photo_id"]),
                        "person_id": int(r["person_id"]),
                        "face_id": fid_val,
                        "status": str(r["status"]),
                        "source": str(r["source"] or ""),
                        "det_score": det_sc,
                        "year": int(yr_str) if yr_str.isdigit() and len(yr_str) == 4 else 0,
                        "archive_child": str(r["archive_child"] or ""),
                        "album_key": f"{r['archive_child'] or ''}/{r['album_leaf'] or ''}",
                        "face_count": int(r["face_count"] or 0),
                        "emb": emb_vec,
                    }
                )
        else:
            for r in human_rows:
                yr_str = str(r["year"])
                human_items.append(
                    {
                        "photo_id": int(r["photo_id"]),
                        "person_id": int(r["person_id"]),
                        "face_id": int(r["face_id"]) if r["face_id"] is not None else None,
                        "status": str(r["status"]),
                        "source": str(r["source"] or ""),
                        "det_score": float(r["det_score"] or 0.0),
                        "year": int(yr_str) if yr_str.isdigit() and len(yr_str) == 4 else 0,
                        "archive_child": str(r["archive_child"] or ""),
                        "album_key": f"{r['archive_child'] or ''}/{r['album_leaf'] or ''}",
                        "face_count": int(r["face_count"] or 0),
                        "emb": (
                            unpack_embeddings_f32([bytes(r["embedding_f16"])], dim=dim)[0]
                            if r["embedding_f16"] is not None
                            else None
                        ),
                    }
                )

        return {
            "people": [
                {
                    "person_id": int(r["person_id"]),
                    "role": str(r["role"]),
                    "archive_hint": str(r["archive_hint"] or ""),
                }
                for r in people_rows
            ],
            "seeds": [
                {
                    "person_id": int(r["person_id"]),
                    "year": int(r["anchor_year"]) if r["anchor_year"] is not None else 0,
                    "emb": seed_embs[idx],
                }
                for idx, r in enumerate(seed_rows)
                if seed_embs is not None
            ],
            "human": human_items,
            "ignore_person_ids": ignore_ids,
            "pressure": [
                {
                    "person_id": int(r["person_id"]),
                    "face_id": int(r["face_id"]),
                    "weight": float(r["weight"]),
                }
                for r in pressure_rows
            ],
        }

    # ------------------------------------------------------------------
    # Cluster Persistence & Assignment Updates
    # ------------------------------------------------------------------

    def save_clusters(self, cluster_specs: list[dict[str, Any]]) -> int:
        """Replace unsupervised per-year clusters while keeping any cluster already assigned to a person."""
        with self._lock, self._conn:
            self._conn.execute("UPDATE faces SET cluster_id = NULL")
            self._conn.execute("DELETE FROM clusters")
            created = 0
            for spec in cluster_specs:
                fids: list[int] = [int(x) for x in spec["face_ids"]]
                if not fids:
                    continue
                cur = self._conn.execute(
                    """
                    INSERT INTO clusters (
                        archive_child, year, person_id, medoid_face_id,
                        sample_face_ids_json, face_count, photo_count,
                        suggested_person_id, suggested_similarity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(spec["archive_child"]),
                        str(spec["year"]),
                        spec.get("person_id"),
                        int(spec["medoid_face_id"]),
                        json.dumps([int(x) for x in spec["sample_face_ids"]]),
                        len(fids),
                        int(spec["photo_count"]),
                        spec.get("suggested_person_id"),
                        spec.get("suggested_similarity"),
                    ),
                )
                if cur.lastrowid is None:
                    raise RuntimeError("SQLite did not return lastrowid for inserted cluster.")
                cid = int(cur.lastrowid)
                self._conn.executemany(
                    "UPDATE faces SET cluster_id = ? WHERE face_id = ?",
                    [(cid, fid) for fid in fids],
                )
                created += 1
        return created

    def list_clusters(
        self,
        *,
        year: str | None = None,
        archive_child: str | None = None,
        include_assigned: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        inc_flag = 1 if include_assigned else 0
        yr_param = year if year else None
        arch_param = archive_child if archive_child else None

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    c.cluster_id, c.archive_child, c.year, c.person_id,
                    p.display_name AS person_name,
                    c.medoid_face_id, c.sample_face_ids_json,
                    c.face_count, c.photo_count,
                    sp.person_id AS eff_suggested_person_id,
                    sp.display_name AS suggested_person_name,
                    CASE WHEN sp.person_id IS NOT NULL THEN c.suggested_similarity ELSE NULL END AS eff_suggested_similarity
                FROM clusters c
                LEFT JOIN people p ON p.person_id = c.person_id
                LEFT JOIN people sp ON sp.person_id = c.suggested_person_id AND sp.role != 'ignore'
                WHERE (? = 1 OR c.person_id IS NULL)
                  AND (? IS NULL OR c.year = ?)
                  AND (? IS NULL OR c.archive_child = ?)
                ORDER BY c.photo_count DESC, c.face_count DESC
                LIMIT ?
                """,
                (inc_flag, yr_param, yr_param, arch_param, arch_param, limit),
            ).fetchall()

        return [
            {
                "cluster_id": int(r["cluster_id"]),
                "archive_child": str(r["archive_child"]),
                "year": str(r["year"]),
                "person_id": int(r["person_id"]) if r["person_id"] is not None else None,
                "person_name": str(r["person_name"]) if r["person_name"] is not None else None,
                "medoid_face_id": int(r["medoid_face_id"]) if r["medoid_face_id"] is not None else None,
                "sample_face_ids": json.loads(str(r["sample_face_ids_json"] or "[]")),
                "face_count": int(r["face_count"]),
                "photo_count": int(r["photo_count"]),
                "suggested_person_id": (
                    int(r["eff_suggested_person_id"])
                    if r["eff_suggested_person_id"] is not None
                    else None
                ),
                "suggested_person_name": (
                    str(r["suggested_person_name"]) if r["suggested_person_name"] is not None else None
                ),
                "suggested_similarity": (
                    round(float(r["eff_suggested_similarity"]), 4)
                    if r["eff_suggested_similarity"] is not None
                    else None
                ),
            }
            for r in rows
        ]

    def assign_cluster_to_person(self, cluster_id: int, person_id: int) -> int:
        """
        Confirm all faces in `cluster_id` for `person_id` (unless already human-rejected for that photo,
        outside the person's birthday/class window, or human-confirmed for someone else — a bulk
        cluster confirm never overrides another person's confirmed tag), set cover face if unset, and
        return the number of newly confirmed photos.
        """
        now = _now_iso()
        with self._lock, self._conn:
            c_row = self._conn.execute(
                "SELECT medoid_face_id FROM clusters WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
            if c_row is None:
                raise KeyError(f"Cluster {cluster_id} does not exist.")

            self._conn.execute(
                "UPDATE clusters SET person_id = ? WHERE cluster_id = ?",
                (person_id, cluster_id),
            )
            if c_row["medoid_face_id"] is not None:
                self._conn.execute(
                    "UPDATE people SET cover_face_id = COALESCE(cover_face_id, ?) WHERE person_id = ?",
                    (int(c_row["medoid_face_id"]), person_id),
                )

            face_rows = self._conn.execute(
                """
                SELECT face_id, photo_id, det_score
                FROM faces
                WHERE cluster_id = ?
                ORDER BY det_score DESC
                """,
                (cluster_id,),
            ).fetchall()

            seen_photos: set[int] = set()
            confirmed = 0
            cluster_rules = self._person_rules_locked(person_id)
            for r in face_rows:
                pid = int(r["photo_id"])
                fid = int(r["face_id"])
                if pid in seen_photos:
                    continue
                seen_photos.add(pid)
                ph_ord, ph_feed = self._photo_ord_and_feed_locked(pid)
                if not self._rules_allow(cluster_rules, ph_ord, ph_feed, check_classes=True):
                    continue
                existing = self._conn.execute(
                    "SELECT status FROM assignments WHERE photo_id = ? AND person_id = ?",
                    (pid, person_id),
                ).fetchone()
                if existing is not None and str(existing["status"]) == "rejected":
                    continue
                res = self._claim_face_locked(
                    fid, person_id, source="cluster_confirm", now=now, explicit=False
                )
                if res["claimed"]:
                    confirmed += 1
            return confirmed

    def remove_face_from_cluster(
        self,
        *,
        cluster_id: int,
        face_id: int,
    ) -> dict[str, Any] | None:
        """
        Detach `face_id` from `cluster_id` so confirming the cluster leaves this face untouched.
        Returns updated `get_cluster_faces(cluster_id)`, or `None` if the cluster became empty.
        """
        with self._lock, self._conn:
            c_row = self._conn.execute(
                "SELECT medoid_face_id, sample_face_ids_json FROM clusters WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
            if c_row is None:
                raise KeyError(f"Cluster {cluster_id} does not exist.")

            self._conn.execute(
                "UPDATE faces SET cluster_id = NULL WHERE face_id = ? AND cluster_id = ?",
                (face_id, cluster_id),
            )
            rem_rows = self._conn.execute(
                """
                SELECT face_id, photo_id, det_score
                FROM faces
                WHERE cluster_id = ?
                ORDER BY det_score DESC, face_id ASC
                """,
                (cluster_id,),
            ).fetchall()
            if not rem_rows:
                self._conn.execute("DELETE FROM clusters WHERE cluster_id = ?", (cluster_id,))
                return None

            rem_fids = [int(r["face_id"]) for r in rem_rows]
            rem_set = set(rem_fids)
            photo_count = len({int(r["photo_id"]) for r in rem_rows})
            old_medoid = int(c_row["medoid_face_id"]) if c_row["medoid_face_id"] is not None else None
            new_medoid = old_medoid if old_medoid in rem_set else rem_fids[0]

            old_samples = [
                int(x) for x in json.loads(str(c_row["sample_face_ids_json"] or "[]"))
            ]
            new_samples = [fid for fid in old_samples if fid in rem_set]
            for fid in rem_fids:
                if len(new_samples) >= 6:
                    break
                if fid not in new_samples:
                    new_samples.append(fid)

            self._conn.execute(
                """
                UPDATE clusters
                SET medoid_face_id = ?,
                    sample_face_ids_json = ?,
                    face_count = ?,
                    photo_count = ?
                WHERE cluster_id = ?
                """,
                (new_medoid, json.dumps(new_samples), len(rem_fids), photo_count, cluster_id),
            )
        return self.get_cluster_faces(cluster_id)

    def replace_model_assignments(
        self,
        auto_rows: list[tuple[int, int, int, AssignmentStatus, float, AssignmentSource]],
        anchor_face_ids: set[int],
    ) -> tuple[int, int]:
        """
        Replace non-human (`model_match` / `album_cohesion`) assignments while keeping all `confirmed`,
        `unsure`, and `rejected` human decisions completely untouched. Uses an in-memory diff against
        existing model rows so incremental rematches only mutate changed rows in SQLite.
        Returns `(auto_high_count, candidate_count)`.
        """
        now = _now_iso()
        auto_high = 0
        candidates = 0
        with self._lock, self._conn:
            prev_anchor_rows = self._conn.execute(
                "SELECT photo_id, person_id, face_id, status FROM assignments WHERE is_anchor = 1"
            ).fetchall()
            prev_valid_anchors: set[int] = set()
            stale_anchor_keys: list[tuple[int, int]] = []
            for ar in prev_anchor_rows:
                afid = int(ar["face_id"]) if ar["face_id"] is not None else None
                if str(ar["status"]) == "confirmed" and afid is not None and afid in anchor_face_ids:
                    prev_valid_anchors.add(afid)
                else:
                    stale_anchor_keys.append((int(ar["photo_id"]), int(ar["person_id"])))

            if stale_anchor_keys:
                self._conn.executemany(
                    "UPDATE assignments SET is_anchor = 0 WHERE photo_id = ? AND person_id = ?",
                    stale_anchor_keys,
                )
            added_anchors = anchor_face_ids - prev_valid_anchors
            if added_anchors:
                self._conn.executemany(
                    "UPDATE assignments SET is_anchor = 1 WHERE face_id = ? AND status = 'confirmed'",
                    [(fid,) for fid in added_anchors],
                )

            existing_human = {
                (int(r["photo_id"]), int(r["person_id"]))
                for r in self._conn.execute(
                    """
                    SELECT photo_id, person_id
                    FROM assignments
                    WHERE status IN ('confirmed', 'unsure')
                       OR (status = 'rejected' AND source != 'reassigned')
                    """
                ).fetchall()
            }
            reassigned_faces = {
                (int(r["photo_id"]), int(r["person_id"])): (
                    int(r["face_id"]) if r["face_id"] is not None else -1
                )
                for r in self._conn.execute(
                    "SELECT photo_id, person_id, face_id FROM assignments WHERE status = 'rejected' AND source = 'reassigned'"
                ).fetchall()
            }
            prev_model = {
                (int(r["photo_id"]), int(r["person_id"])): (
                    int(r["face_id"]) if r["face_id"] is not None else -1,
                    str(r["status"]),
                    str(r["source"]),
                    round(float(r["similarity"]), 5),
                )
                for r in self._conn.execute(
                    """
                    SELECT photo_id, person_id, face_id, status, source, similarity
                    FROM assignments
                    WHERE status IN ('auto_high', 'candidate')
                    """
                ).fetchall()
            }

            desired_model: dict[tuple[int, int], tuple[int, str, str, float]] = {}
            for photo_id, person_id, face_id, status, sim, src_kind in auto_rows:
                if (photo_id, person_id) in existing_human:
                    continue
                if reassigned_faces.get((photo_id, person_id)) == face_id:
                    continue
                if status == "auto_high":
                    auto_high += 1
                else:
                    candidates += 1
                desired_model[(photo_id, person_id)] = (
                    face_id,
                    status,
                    src_kind,
                    round(sim, 5),
                )

            to_delete = [
                key for key in prev_model if key not in desired_model
            ]
            if to_delete:
                self._conn.executemany(
                    "DELETE FROM assignments WHERE photo_id = ? AND person_id = ? AND status IN ('auto_high', 'candidate')",
                    to_delete,
                )

            to_upsert = [
                (p_id, per_id, fid, st, src, r_sim, 0, now)
                for (p_id, per_id), (fid, st, src, r_sim) in desired_model.items()
                if prev_model.get((p_id, per_id)) != (fid, st, src, r_sim)
            ]
            if to_upsert:
                self._conn.executemany(
                    """
                    INSERT INTO assignments (
                        photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(photo_id, person_id) DO UPDATE SET
                        face_id = excluded.face_id,
                        status = excluded.status,
                        source = excluded.source,
                        similarity = excluded.similarity,
                        is_anchor = 0,
                        updated_at = excluded.updated_at
                    WHERE assignments.status IN ('auto_high', 'candidate')
                       OR (
                           assignments.status = 'rejected'
                           AND assignments.source = 'reassigned'
                           AND assignments.face_id IS NOT excluded.face_id
                       )
                    """,
                    to_upsert,
                )
        return auto_high, candidates

    def apply_batch_decisions(
        self,
        *,
        person_id: int,
        approved: list[tuple[int, int | None]],
        disproved: list[tuple[int, int | None]],
        unsure: list[tuple[int, int | None]] | None = None,
        reassigned: list[tuple[int, int, int]] | None = None,
    ) -> dict[str, int]:
        """
        Persist an active-learning batch submission:
        - `approved`: `[(photo_id, face_id)]` -> `confirmed` for `person_id`. A face that meanwhile
          became human-confirmed for someone else (stale proposal) is skipped and counted in
          `skipped_conflicts` — another person's confirmed tag is only changed by an explicit action.
        - `disproved`: `[(photo_id, face_id)]` -> `rejected` (hard negative!) for `person_id`
        - `unsure`: `[(photo_id, face_id)]` -> `unsure` (skipped without teaching positive or negative)
        - `reassigned`: `[(photo_id, face_id, target_person_id)]` -> explicit claim of the face for
          `target_person_id` + face-scoped negative (`rejected` / `reassigned`) for `person_id`
        """
        now = _now_iso()
        unsure_list = unsure or []
        reassigned_list = reassigned or []
        skipped_conflicts = 0
        with self._lock, self._conn:
            rules_cache: dict[int, tuple[int, list[tuple[str, int, int]]]] = {}

            def _birth_ok(ph_id: int, per_id: int) -> bool:
                if per_id not in rules_cache:
                    rules_cache[per_id] = self._person_rules_locked(per_id)
                ph_ord, ph_feed = self._photo_ord_and_feed_locked(ph_id)
                return self._rules_allow(rules_cache[per_id], ph_ord, ph_feed, check_classes=False)

            approved = [(ph, fc) for ph, fc in approved if _birth_ok(ph, person_id)]
            reassigned_list = [(ph, fc, tp) for ph, fc, tp in reassigned_list if _birth_ok(ph, tp)]
            applied_approved = 0
            for photo_id, face_id in approved:
                if face_id is None:
                    self._conn.execute(
                        """
                        INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
                        VALUES (?, ?, NULL, 'confirmed', 'batch_approve', 1.0, 0, ?)
                        ON CONFLICT(photo_id, person_id) DO UPDATE SET
                            status = 'confirmed',
                            source = 'batch_approve',
                            similarity = 1.0,
                            updated_at = excluded.updated_at
                        """,
                        (photo_id, person_id, now),
                    )
                    applied_approved += 1
                    continue
                res = self._claim_face_locked(
                    face_id, person_id, source="batch_approve", now=now, explicit=False, photo_id=photo_id
                )
                if res["claimed"]:
                    applied_approved += 1
                else:
                    skipped_conflicts += 1

            for photo_id, face_id in disproved:
                self._conn.execute(
                    """
                    INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
                    VALUES (?, ?, ?, 'rejected', 'batch_approve', 0.0, 0, ?)
                    ON CONFLICT(photo_id, person_id) DO UPDATE SET
                        face_id = COALESCE(excluded.face_id, assignments.face_id),
                        status = 'rejected',
                        source = 'batch_approve',
                        similarity = 0.0,
                        is_anchor = 0,
                        updated_at = excluded.updated_at
                    """,
                    (photo_id, person_id, face_id, now),
                )
                if face_id is not None:
                    self._refresh_cover_face_if_needed(person_id, face_id)

            for photo_id, face_id in unsure_list:
                self._conn.execute(
                    """
                    INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
                    VALUES (?, ?, ?, 'unsure', 'batch_approve', 0.0, 0, ?)
                    ON CONFLICT(photo_id, person_id) DO UPDATE SET
                        face_id = COALESCE(excluded.face_id, assignments.face_id),
                        status = 'unsure',
                        source = 'batch_approve',
                        similarity = 0.0,
                        is_anchor = 0,
                        updated_at = excluded.updated_at
                    """,
                    (photo_id, person_id, face_id, now),
                )

            for photo_id, face_id, target_pid in reassigned_list:
                self._claim_face_locked(
                    face_id, target_pid, source="manual_tag", now=now, explicit=True, photo_id=photo_id
                )
                if target_pid != person_id:
                    self._record_face_negative_locked(photo_id, person_id, face_id, now=now)

        return {
            "approved": applied_approved,
            "disproved": len(disproved),
            "unsure": len(unsure_list),
            "reassigned": len(reassigned_list),
            "skipped_conflicts": skipped_conflicts,
        }

    def _record_face_negative_locked(self, photo_id: int, person_id: int, face_id: int, *, now: str) -> None:
        """Face-scoped hard negative (`rejected` / `reassigned`) for `person_id` on `face_id`, without
        touching any row `person_id` holds on a DIFFERENT face of the same photo."""
        self._conn.execute(
            """
            INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
            VALUES (?, ?, ?, 'rejected', 'reassigned', 0.0, 0, ?)
            ON CONFLICT(photo_id, person_id) DO UPDATE SET
                face_id = excluded.face_id,
                status = 'rejected',
                source = 'reassigned',
                similarity = 0.0,
                is_anchor = 0,
                updated_at = excluded.updated_at
            WHERE assignments.face_id IS NULL
               OR assignments.face_id = excluded.face_id
            """,
            (photo_id, person_id, face_id, now),
        )
        self._refresh_cover_face_if_needed(person_id, face_id)

    def _claim_face_locked(
        self,
        face_id: int,
        winner_person_id: int,
        *,
        source: AssignmentSource,
        now: str,
        explicit: bool,
        photo_id: int | None = None,
        similarity: float = 1.0,
    ) -> dict[str, Any]:
        """
        The single "this face box is `winner_person_id`" primitive used by every confirm path
        (batch approve, cluster confirm, batch reassign, `tag_single_face`, manual boxes).

        Policy (must be called with `_lock` held, inside a transaction):
        - A human `confirmed` tag of another (non-`ignore`) person is never deleted. Bulk/implicit
          callers (`explicit=False`: batch approve, cluster confirm) skip such faces and get
          `{"claimed": False, "blocked_by": [...]}`; explicit single-face actions demote the holder
          to a face-scoped negative (`rejected` / `reassigned`).
        - Other holders' `unsure` rows become face-scoped negatives; model guesses (`auto_high` /
          `candidate`) become negatives on explicit claims and are dropped on bulk ones.
        - Implicit claims also never move the winner's own confirmed tag off another face of the photo.
        - Clears `suppress_auto` (manual boxes stay suppressed), detaches the face from a cluster
          assigned to someone else, refreshes displaced covers and sets the winner's cover if unset.
        """
        f_row = self._conn.execute(
            "SELECT photo_id, cluster_id FROM faces WHERE face_id = ?", (face_id,)
        ).fetchone()
        if f_row is None or (photo_id is not None and int(f_row["photo_id"]) != int(photo_id)):
            raise KeyError(f"Face {face_id} not found" + (f" on photo {photo_id}." if photo_id else "."))
        ph = int(f_row["photo_id"])
        holders = self._conn.execute(
            """
            SELECT a.person_id, a.status, pe.role
            FROM assignments a
            JOIN people pe ON pe.person_id = a.person_id
            WHERE a.face_id = ? AND a.person_id != ? AND a.status != 'rejected'
            """,
            (face_id, winner_person_id),
        ).fetchall()
        blocked_by = [
            int(h["person_id"])
            for h in holders
            if str(h["status"]) == "confirmed" and str(h["role"]) != "ignore"
        ]
        if blocked_by and not explicit:
            return {"claimed": False, "blocked_by": blocked_by, "displaced": []}
        own = self._conn.execute(
            "SELECT face_id, status FROM assignments WHERE photo_id = ? AND person_id = ?",
            (ph, winner_person_id),
        ).fetchone()
        own_other_fid = (
            int(own["face_id"])
            if own is not None and own["face_id"] is not None and int(own["face_id"]) != face_id
            else None
        )
        if not explicit and own_other_fid is not None and str(own["status"]) == "confirmed":
            return {"claimed": False, "blocked_by": [winner_person_id], "displaced": []}

        displaced: list[int] = []
        for h in holders:
            old_pid = int(h["person_id"])
            if not explicit and str(h["status"]) in ("auto_high", "candidate"):
                self._conn.execute(
                    "DELETE FROM assignments WHERE photo_id = ? AND person_id = ? AND face_id = ?",
                    (ph, old_pid, face_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE assignments
                    SET status = 'rejected', source = 'reassigned', similarity = 0.0, is_anchor = 0, updated_at = ?
                    WHERE photo_id = ? AND person_id = ? AND face_id = ?
                    """,
                    (now, ph, old_pid, face_id),
                )
            displaced.append(old_pid)

        self._conn.execute(
            """
            INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
            VALUES (?, ?, ?, 'confirmed', ?, ?, 0, ?)
            ON CONFLICT(photo_id, person_id) DO UPDATE SET
                face_id = excluded.face_id,
                status = 'confirmed',
                source = excluded.source,
                similarity = excluded.similarity,
                is_anchor = CASE
                    WHEN assignments.status = 'confirmed' AND assignments.face_id IS excluded.face_id
                    THEN assignments.is_anchor ELSE 0 END,
                updated_at = excluded.updated_at
            """,
            (ph, winner_person_id, face_id, source, float(similarity), now),
        )
        self._conn.execute(
            "UPDATE faces SET suppress_auto = CASE WHEN is_manual != 0 THEN 1 ELSE 0 END WHERE face_id = ?",
            (face_id,),
        )
        if f_row["cluster_id"] is not None:
            c_owner = self._conn.execute(
                "SELECT person_id FROM clusters WHERE cluster_id = ?", (int(f_row["cluster_id"]),)
            ).fetchone()
            if (
                c_owner is not None
                and c_owner["person_id"] is not None
                and int(c_owner["person_id"]) != winner_person_id
            ):
                self._conn.execute("UPDATE faces SET cluster_id = NULL WHERE face_id = ?", (face_id,))
        for old_pid in displaced:
            self._refresh_cover_face_if_needed(old_pid, face_id)
        if own_other_fid is not None:
            self._refresh_cover_face_if_needed(winner_person_id, own_other_fid)
        self._conn.execute(
            "UPDATE people SET cover_face_id = COALESCE(cover_face_id, ?) WHERE person_id = ?",
            (face_id, winner_person_id),
        )
        return {"claimed": True, "blocked_by": [], "displaced": displaced}

    def _refresh_cover_face_if_needed(self, person_id: int, lost_face_id: int) -> None:
        self._conn.execute(
            """
            UPDATE people
            SET cover_face_id = (
                SELECT a.face_id
                FROM assignments a
                LEFT JOIN faces f ON f.face_id = a.face_id
                WHERE a.person_id = ?
                  AND a.status = 'confirmed'
                  AND a.face_id IS NOT NULL
                  AND a.face_id != ?
                ORDER BY COALESCE(f.is_manual, 1) ASC, a.is_anchor DESC, a.updated_at DESC
                LIMIT 1
            )
            WHERE person_id = ? AND cover_face_id = ?
            """,
            (person_id, lost_face_id, person_id, lost_face_id),
        )

    def create_manual_face_and_tag(
        self,
        *,
        photo_id: int,
        box: BoundingBox,
        person_id: int,
        dim: int = 512,
    ) -> int:
        """
        Insert a user-drawn manual bounding box (`is_manual = 1`, zero embedding vector,
        `suppress_auto = 1`) on `photo_id` and confirm it for `person_id` (`is_anchor = 0`).
        Reuses all `/media/face/<id>.jpg` crop and `ByChild/` export logic while remaining
        100% invisible to ArcFace biometric matching and clustering.
        """
        now = _now_iso()
        zero_blob = bytes(dim * 2)
        self.assert_can_tag(photo_id, person_id)
        with self._lock, self._conn:
            p_row = self._conn.execute(
                "SELECT width, height FROM photos WHERE photo_id = ?",
                (photo_id,),
            ).fetchone()
            if p_row is None:
                raise KeyError(f"Photo {photo_id} not found.")
            pw = int(p_row["width"] or 0)
            ph = int(p_row["height"] or 0)

            left = max(0, int(box.left))
            top = max(0, int(box.top))
            right = min(pw, int(box.right)) if pw > 0 else int(box.right)
            bottom = min(ph, int(box.bottom)) if ph > 0 else int(box.bottom)
            if right - left < 4 or bottom - top < 4:
                raise ValueError("Drawn bounding box is too small; drag a larger rectangle.")

            idx_row = self._conn.execute(
                "SELECT COALESCE(MAX(face_idx), 999) AS max_idx FROM faces WHERE photo_id = ?",
                (photo_id,),
            ).fetchone()
            next_idx = max(1000, int(idx_row["max_idx"]) + 1) if idx_row is not None else 1000

            cur = self._conn.execute(
                """
                INSERT INTO faces (
                    photo_id, face_idx, det_score,
                    bbox_top, bbox_right, bbox_bottom, bbox_left,
                    embedding_f16, cluster_id, suppress_auto, is_manual
                ) VALUES (?, ?, 1.0, ?, ?, ?, ?, ?, NULL, 1, 1)
                """,
                (photo_id, next_idx, top, right, bottom, left, zero_blob),
            )
            if cur.lastrowid is None:
                raise RuntimeError("SQLite did not return lastrowid for manual face.")
            new_fid = int(cur.lastrowid)

            self._conn.execute(
                "UPDATE photos SET face_count = face_count + 1 WHERE photo_id = ?",
                (photo_id,),
            )
            if self._photo_summary_cache is not None:
                self._photo_summary_cache["faces_total"] = (
                    int(self._photo_summary_cache["faces_total"]) + 1
                )

            self._claim_face_locked(
                new_fid, person_id, source="manual_tag", now=now, explicit=True, photo_id=photo_id
            )
        return new_fid

    def tag_single_face(
        self,
        *,
        photo_id: int,
        face_id: int | None,
        person_id: int,
        status: AssignmentStatus = "confirmed",
        source: AssignmentSource = "manual_tag",
        replace_person_id: int | None = None,
    ) -> int | None:
        """
        Tag (or reject / mark unsure) a face on `photo_id` for `person_id`.
        When `status == 'confirmed'`:
        - Resolves `face_id` automatically if `face_id is None` (from `replace_person_id`'s row or
          from `faces` when `photo_id` has a single detected face) so the face box itself is tagged.
        - Explicitly claims the face via `_claim_face_locked`: any previous holder of `face_id` (and
          `replace_person_id` if provided) becomes a face-scoped hard negative
          (`status = 'rejected', source = 'reassigned'`) and their `cover_face_id` is refreshed.
        Returns the resolved `face_id` (if any).
        """
        now = _now_iso()
        if status == "confirmed":
            self.assert_can_tag(photo_id, person_id)
        with self._lock, self._conn:
            if face_id is None and replace_person_id is not None:
                prev_f = self._conn.execute(
                    "SELECT face_id FROM assignments WHERE photo_id = ? AND person_id = ? AND face_id IS NOT NULL",
                    (photo_id, replace_person_id),
                ).fetchone()
                if prev_f is not None:
                    face_id = int(prev_f["face_id"])

            if face_id is None and status == "confirmed":
                only_faces = self._conn.execute(
                    "SELECT face_id FROM faces WHERE photo_id = ? AND is_manual = 0",
                    (photo_id,),
                ).fetchall()
                if len(only_faces) == 1:
                    face_id = int(only_faces[0]["face_id"])

            if face_id is not None:
                mf_row = self._conn.execute(
                    "SELECT is_manual FROM faces WHERE face_id = ? AND photo_id = ?",
                    (face_id, photo_id),
                ).fetchone()
                if mf_row is not None and bool(mf_row["is_manual"]) and status == "rejected":
                    self.untag_face(photo_id=photo_id, face_id=face_id)
                    return None

            if status == "confirmed" and face_id is not None:
                self._claim_face_locked(
                    face_id, person_id, source=source, now=now, explicit=True, photo_id=photo_id
                )
                if replace_person_id is not None and replace_person_id != person_id:
                    self._record_face_negative_locked(photo_id, replace_person_id, face_id, now=now)
                return face_id

            if status == "confirmed" and replace_person_id is not None and replace_person_id != person_id:
                self._conn.execute(
                    """
                    INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
                    VALUES (?, ?, NULL, 'rejected', 'reassigned', 0.0, 0, ?)
                    ON CONFLICT(photo_id, person_id) DO UPDATE SET
                        status = 'rejected',
                        source = 'reassigned',
                        similarity = 0.0,
                        is_anchor = 0,
                        updated_at = excluded.updated_at
                    """,
                    (photo_id, replace_person_id, now),
                )

            self._conn.execute(
                """
                INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(photo_id, person_id) DO UPDATE SET
                    face_id = COALESCE(excluded.face_id, assignments.face_id),
                    status = excluded.status,
                    source = excluded.source,
                    similarity = excluded.similarity,
                    is_anchor = CASE WHEN excluded.status = 'confirmed' THEN assignments.is_anchor ELSE 0 END,
                    updated_at = excluded.updated_at
                """,
                (
                    photo_id,
                    person_id,
                    face_id,
                    status,
                    source,
                    1.0 if status == "confirmed" else 0.0,
                    now,
                ),
            )

            if face_id is not None and status != "confirmed":
                self._refresh_cover_face_if_needed(person_id, face_id)

        return face_id

    def untag_face(
        self,
        *,
        photo_id: int,
        face_id: int,
    ) -> list[int]:
        """
        Remove any tag or tentative model guess from `face_id` and leave the face uncategorized:
        - Converts any active holder (`confirmed`, `auto_high`, `candidate`) into a face-scoped
          hard-negative (`status = 'rejected', source = 'reassigned'`) so that person learns this
          face box is not them while still allowing them to match another face in the same photo.
        - Clears any `'unsure'` row on `face_id` and refreshes `people.cover_face_id` if needed.
        - Sets `faces.suppress_auto = 1` so `recompute_active_matching()` does not immediately
          slap a runner-up guess onto this face (until explicitly confirmed later).
        Returns the list of previously assigned `person_id`s removed from `face_id`.
        """
        now = _now_iso()
        removed_pids: list[int] = []
        with self._lock, self._conn:
            f_row = self._conn.execute(
                "SELECT cluster_id, is_manual FROM faces WHERE face_id = ? AND photo_id = ?",
                (face_id, photo_id),
            ).fetchone()
            if f_row is None:
                raise KeyError(f"Face {face_id} not found on photo {photo_id}.")

            holders = self._conn.execute(
                """
                SELECT person_id, status
                FROM assignments
                WHERE face_id = ?
                """,
                (face_id,),
            ).fetchall()

            if bool(f_row["is_manual"]):
                for hr in holders:
                    old_pid = int(hr["person_id"])
                    if str(hr["status"]) in ("confirmed", "auto_high", "candidate"):
                        removed_pids.append(old_pid)
                self._conn.execute("DELETE FROM assignments WHERE face_id = ?", (face_id,))
                self._conn.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
                self._conn.execute(
                    "UPDATE photos SET face_count = MAX(0, face_count - 1) WHERE photo_id = ?",
                    (photo_id,),
                )
                if self._photo_summary_cache is not None:
                    self._photo_summary_cache["faces_total"] = max(
                        0, int(self._photo_summary_cache["faces_total"]) - 1
                    )
                for old_pid in removed_pids:
                    self._refresh_cover_face_if_needed(old_pid, face_id)
                return removed_pids

            for hr in holders:
                old_pid = int(hr["person_id"])
                st = str(hr["status"])
                if st in ("confirmed", "auto_high", "candidate"):
                    removed_pids.append(old_pid)
                    self._conn.execute(
                        """
                        UPDATE assignments
                        SET status = 'rejected',
                            source = 'reassigned',
                            similarity = 0.0,
                            is_anchor = 0,
                            updated_at = ?
                        WHERE photo_id = ? AND person_id = ? AND face_id = ?
                        """,
                        (now, photo_id, old_pid, face_id),
                    )
                    self._refresh_cover_face_if_needed(old_pid, face_id)
                elif st == "unsure":
                    self._conn.execute(
                        "DELETE FROM assignments WHERE photo_id = ? AND person_id = ? AND face_id = ?",
                        (photo_id, old_pid, face_id),
                    )

            if f_row["cluster_id"] is not None:
                cid = int(f_row["cluster_id"])
                c_row = self._conn.execute(
                    "SELECT person_id FROM clusters WHERE cluster_id = ?",
                    (cid,),
                ).fetchone()
                if c_row is not None and c_row["person_id"] is not None:
                    self._conn.execute(
                        "UPDATE faces SET cluster_id = NULL WHERE face_id = ?",
                        (face_id,),
                    )

            self._conn.execute(
                "UPDATE faces SET suppress_auto = 1 WHERE face_id = ?",
                (face_id,),
            )
        return removed_pids

    # ------------------------------------------------------------------
    # Per-Child Clusters & Soft Pressure
    # ------------------------------------------------------------------

    def apply_person_cluster_action(
        self,
        *,
        person_id: int,
        action: str,
        faces: list[tuple[int, int]],
    ) -> int:
        """
        Apply a per-child cluster action to `[(photo_id, face_id)]`:
        - `exclude`: face-scoped hard rejection (`rejected` / `reassigned`) -> hard negatives.
        - `boost` / `penalize`: soft +/- pressure stored in `person_face_pressure` (no status change).
        - `clear`: remove any pressure.
        Returns the number of faces affected.
        """
        if action not in ("exclude", "boost", "penalize", "clear"):
            raise ValueError(f"Unknown cluster action {action!r}.")
        now = _now_iso()
        applied = 0
        with self._lock, self._conn:
            if self._conn.execute(
                "SELECT 1 FROM people WHERE person_id = ?", (person_id,)
            ).fetchone() is None:
                raise KeyError(f"Person {person_id} not found.")
            for photo_id, face_id in faces:
                f_row = self._conn.execute(
                    "SELECT is_manual FROM faces WHERE face_id = ? AND photo_id = ?",
                    (face_id, photo_id),
                ).fetchone()
                if f_row is None:
                    continue
                if action in ("boost", "penalize"):
                    if bool(f_row["is_manual"]):
                        continue
                    self._conn.execute(
                        """
                        INSERT INTO person_face_pressure (person_id, face_id, weight, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(person_id, face_id) DO UPDATE SET
                            weight = excluded.weight, updated_at = excluded.updated_at
                        """,
                        (person_id, face_id, 1.0 if action == "boost" else -1.0, now),
                    )
                    applied += 1
                    continue
                self._conn.execute(
                    "DELETE FROM person_face_pressure WHERE person_id = ? AND face_id = ?",
                    (person_id, face_id),
                )
                if action == "clear":
                    applied += 1
                    continue
                # exclude:
                if bool(f_row["is_manual"]):
                    self._conn.execute(
                        "DELETE FROM assignments WHERE face_id = ? AND person_id = ?",
                        (face_id, person_id),
                    )
                    self._refresh_cover_face_if_needed(person_id, face_id)
                    self._delete_manual_faces_locked([face_id])
                    applied += 1
                    continue
                existing = self._conn.execute(
                    "SELECT face_id, status FROM assignments WHERE photo_id = ? AND person_id = ?",
                    (photo_id, person_id),
                ).fetchone()
                if (
                    existing is not None
                    and existing["face_id"] is not None
                    and int(existing["face_id"]) != face_id
                    and str(existing["status"]) == "confirmed"
                ):
                    # The child is confirmed on a different face in this photo; keep that tag.
                    continue
                self._conn.execute(
                    """
                    INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
                    VALUES (?, ?, ?, 'rejected', 'reassigned', 0.0, 0, ?)
                    ON CONFLICT(photo_id, person_id) DO UPDATE SET
                        face_id = excluded.face_id,
                        status = 'rejected',
                        source = 'reassigned',
                        similarity = 0.0,
                        is_anchor = 0,
                        updated_at = excluded.updated_at
                    """,
                    (photo_id, person_id, face_id, now),
                )
                self._refresh_cover_face_if_needed(person_id, face_id)
                applied += 1
        return applied

    def get_person_face_clusters(
        self,
        person_id: int,
        *,
        threshold: float = 0.50,
        min_size: int = 3,
    ) -> dict[str, Any]:
        """
        Group everything currently attributed to `person_id` (confirmed + auto_high + candidate, plus
        faces carrying boost/penalize pressure) into per-year visual sub-clusters using density-leader
        grouping on the cached ArcFace matrix. Faces that do not form a group of `min_size` land in a
        per-year loose bucket.
        """
        thr = float(min(0.95, max(0.20, threshold)))
        min_sz = max(2, int(min_size))
        with self._lock:
            p_row = self._conn.execute(
                "SELECT display_name FROM people WHERE person_id = ?", (person_id,)
            ).fetchone()
            if p_row is None:
                raise KeyError(f"Person {person_id} not found.")
            rows = self._conn.execute(
                """
                WITH wanted(face_id) AS (
                    SELECT face_id FROM assignments
                    WHERE person_id = ? AND face_id IS NOT NULL
                      AND status IN ('confirmed', 'auto_high', 'candidate')
                    UNION
                    SELECT face_id FROM person_face_pressure WHERE person_id = ?
                )
                SELECT
                    f.face_id, f.photo_id, f.det_score,
                    a.status, a.source, a.similarity,
                    COALESCE(pr.weight, 0) AS pressure,
                    p.year, p.album_date, p.album_title, p.archive_child
                FROM wanted w
                CROSS JOIN faces f ON f.face_id = w.face_id
                JOIN photos p ON p.photo_id = f.photo_id
                LEFT JOIN assignments a
                    ON a.photo_id = f.photo_id AND a.person_id = ? AND a.face_id = f.face_id
                    AND a.status IN ('confirmed', 'auto_high', 'candidate')
                LEFT JOIN person_face_pressure pr
                    ON pr.person_id = ? AND pr.face_id = f.face_id
                WHERE f.is_manual = 0
                """,
                (person_id, person_id, person_id, person_id),
            ).fetchall()

        mat = self.load_all_faces_matrix()
        c_fids: NDArray[np.int64] = mat["face_ids"]
        c_embs: NDArray[np.float32] = mat["embeddings"]
        n_cached = int(c_fids.shape[0])

        items: list[dict[str, Any]] = []
        emb_rows: list[int] = []
        for r in rows:
            fid = int(r["face_id"])
            pos = int(np.searchsorted(c_fids, fid)) if n_cached else 0
            if pos >= n_cached or int(c_fids[pos]) != fid:
                continue
            emb_rows.append(pos)
            items.append(
                {
                    "face_id": fid,
                    "photo_id": int(r["photo_id"]),
                    "det_score": float(r["det_score"] or 0.0),
                    "status": str(r["status"]) if r["status"] is not None else "none",
                    "source": str(r["source"] or ""),
                    "similarity": round(float(r["similarity"] or 0.0), 4),
                    "pressure": round(float(r["pressure"] or 0.0)),
                    "year": str(r["year"] or ""),
                    "album_date": str(r["album_date"] or "")[:10],
                    "album_title": str(r["album_title"] or ""),
                    "archive_child": str(r["archive_child"] or ""),
                }
            )

        by_year: dict[str, list[int]] = {}
        for idx, it in enumerate(items):
            by_year.setdefault(it["year"], []).append(idx)

        clusters: list[dict[str, Any]] = []
        for yr in sorted(by_year.keys(), reverse=True):
            idxs = by_year[yr]
            sub = c_embs[np.asarray([emb_rows[i] for i in idxs], dtype=np.int64)]
            sims = sub @ sub.T
            adj = sims >= thr
            unassigned = np.ones((len(idxs),), dtype=bool)
            det = np.asarray([items[i]["det_score"] for i in idxs], dtype=np.float32)
            year_groups: list[tuple[int, NDArray[np.int64]]] = []
            loose: list[int] = []
            static_deg = adj.sum(axis=1).astype(np.float32) + 0.01 * det
            for leader_raw in np.argsort(-static_deg):
                leader = int(leader_raw)
                if not unassigned[leader]:
                    continue
                members = np.where(unassigned & adj[leader])[0]
                unassigned[members] = False
                unassigned[leader] = False
                if members.size < min_sz:
                    continue
                year_groups.append((leader, members.astype(np.int64)))
            # Merge pose/lighting sub-groups whose centroids clearly look alike (average-linkage style),
            # so one look of the child does not fragment into dozens of tiny cards:
            merge_thr = min(0.95, thr + 0.12)
            if len(year_groups) > 1:
                cents = np.stack([sub[g[1]].mean(axis=0) for g in year_groups], axis=0)
                cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-8)
                alive = list(range(len(year_groups)))
                groups_m = {i: (year_groups[i][0], year_groups[i][1]) for i in alive}
                while len(alive) > 1:
                    c_mat = cents[alive]
                    cs = c_mat @ c_mat.T
                    np.fill_diagonal(cs, -1.0)
                    flat = int(np.argmax(cs))
                    a_pos, b_pos = divmod(flat, len(alive))
                    if float(cs[a_pos, b_pos]) < merge_thr:
                        break
                    ia, ib = alive[a_pos], alive[b_pos]
                    la, ma = groups_m[ia]
                    lb, mb = groups_m[ib]
                    keep_leader = la if ma.size >= mb.size else lb
                    merged_m = np.concatenate([ma, mb])
                    groups_m[ia] = (keep_leader, merged_m)
                    del groups_m[ib]
                    alive.remove(ib)
                    c_new = sub[merged_m].mean(axis=0)
                    cents[ia] = c_new / max(float(np.linalg.norm(c_new)), 1e-8)
                year_groups = [groups_m[i] for i in alive]
            # Second pass: leftover faces from undersized groups join their best group if close enough,
            # otherwise they land in the per-year loose bucket.
            grouped = np.zeros((len(idxs),), dtype=bool)
            for _, members in year_groups:
                grouped[members] = True
            leftovers = np.where(~grouped)[0]
            if year_groups and leftovers.size:
                leader_arr = np.asarray([g[0] for g in year_groups], dtype=np.int64)
                l_sims_all = sims[np.ix_(leftovers, leader_arr)]
                best = np.argmax(l_sims_all, axis=1)
                extra: dict[int, list[int]] = {}
                for pos, loc in enumerate(leftovers.tolist()):
                    b = int(best[pos])
                    if float(l_sims_all[pos, b]) >= thr:
                        extra.setdefault(b, []).append(int(loc))
                    else:
                        loose.append(int(loc))
                for b, locs in extra.items():
                    ld, mem = year_groups[b]
                    year_groups[b] = (ld, np.concatenate([mem, np.asarray(locs, dtype=np.int64)]))
            else:
                loose.extend(int(x) for x in leftovers.tolist())

            year_groups.sort(key=lambda g: int(g[1].size), reverse=True)
            year_items = [items[i] for i in idxs]
            for leader_loc, members in year_groups:
                clusters.append(
                    _pack_person_cluster(year_items, sims, leader_loc, [int(m) for m in members], False, yr)
                )
            if loose:
                loose_sorted = sorted(set(loose))
                clusters.append(
                    _pack_person_cluster(year_items, sims, loose_sorted[0], loose_sorted, True, yr)
                )

        return {
            "person_id": person_id,
            "display_name": str(p_row["display_name"]),
            "threshold": thr,
            "total_faces": len(items),
            "clusters": clusters,
        }

    # ------------------------------------------------------------------
    # Web UI Queries (Batch Proposal, Gallery, Full Photo Inspector, Stats)
    # ------------------------------------------------------------------

    def get_archive_overview(self) -> dict[str, Any]:
        with self._lock:
            photo_summary = self._photo_summary_cache
            if photo_summary is None:
                p_row = self._conn.execute(
                    "SELECT COUNT(*) AS photos_n, COALESCE(SUM(face_count), 0) AS faces_n FROM photos"
                ).fetchone()
                years = [
                    str(r["year"])
                    for r in self._conn.execute(
                        "SELECT DISTINCT year FROM photos ORDER BY year DESC"
                    ).fetchall()
                ]
                children = [
                    str(r["archive_child"])
                    for r in self._conn.execute(
                        "SELECT DISTINCT archive_child FROM photos WHERE archive_child != '' ORDER BY archive_child ASC"
                    ).fetchall()
                ]
                photo_summary = {
                    "photos_total": int(p_row["photos_n"]) if p_row else 0,
                    "faces_total": int(p_row["faces_n"]) if p_row else 0,
                    "years": years,
                    "archive_children": children,
                }
                self._photo_summary_cache = photo_summary

            cat_faces = self._conn.execute(
                """
                SELECT COALESCE(SUM(c.categorized_face_count), 0) AS n
                FROM person_assignment_counts c
                JOIN people pe ON pe.person_id = c.person_id
                WHERE pe.role != 'ignore'
                """
            ).fetchone()
            unassigned_clusters = self._conn.execute(
                "SELECT COUNT(*) AS n FROM clusters WHERE person_id IS NULL"
            ).fetchone()
            people_list = self.list_people()
            classes_list = self.list_classes()

        photos_total = int(photo_summary["photos_total"])
        faces_total = int(photo_summary["faces_total"])
        faces_categorized = int(cat_faces["n"]) if cat_faces else 0
        return {
            "photos_total": photos_total,
            "faces_total": faces_total,
            "faces_categorized": faces_categorized,
            "coverage_pct": round((100.0 * faces_categorized / faces_total), 1) if faces_total > 0 else 0.0,
            "years": list(photo_summary["years"]),
            "archive_children": list(photo_summary["archive_children"]),
            "unassigned_clusters": int(unassigned_clusters["n"]) if unassigned_clusters else 0,
            "people": people_list,
            "classes": classes_list,
        }

    def get_person_peers(self, person_id: int, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return the top co-occurring named peers for `person_id` across small/medium confirmed photos."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    pe.person_id,
                    pe.display_name,
                    COUNT(*) AS shared_photos
                FROM assignments a1
                JOIN assignments a2
                    ON a2.photo_id = a1.photo_id AND a2.person_id != a1.person_id
                JOIN photos p ON p.photo_id = a1.photo_id
                JOIN people pe ON pe.person_id = a2.person_id
                WHERE a1.person_id = ?
                  AND a1.status = 'confirmed'
                  AND a2.status = 'confirmed'
                  AND pe.role != 'ignore'
                  AND p.face_count BETWEEN 2 AND 8
                GROUP BY pe.person_id, pe.display_name
                ORDER BY shared_photos DESC, pe.display_name ASC
                LIMIT ?
                """,
                (person_id, limit),
            ).fetchall()
        return [
            {
                "person_id": int(r["person_id"]),
                "display_name": str(r["display_name"]),
                "shared_photos": int(r["shared_photos"]),
            }
            for r in rows
        ]

    def get_batch_proposal(
        self,
        person_id: int,
        *,
        batch_size: int = 24,
        default_approve_min_sim: float = 0.38,
        max_per_album: int = 2,
        mode: str = "impact",
    ) -> dict[str, Any]:
        """
        Fetch the next active-learning batch of unconfirmed matches (excluding matches already trusted).
        - `mode="impact"` (default): greedy max-coverage selection of the faces whose answer resolves
          the most OTHER open faces of this child (same-album look-alikes and close look-alikes
          anywhere), weighted by how uncertain each face is.
        - `mode="confident"`: highest-similarity first, diversified across albums.
        """
        fetch_pool = 4000 if mode == "impact" else min(600, max(batch_size * 10, 96))
        with self._lock:
            person_row = self._conn.execute(
                "SELECT person_id, display_name, role, (SELECT SUM(CASE WHEN status = 'auto_high' AND source IN ('album_cohesion', 'model_trusted') THEN 1 ELSE 0 END) FROM assignments WHERE person_id = ?) AS cohesion_count FROM people WHERE person_id = ?",
                (person_id, person_id),
            ).fetchone()
            if person_row is None:
                raise KeyError(f"Person {person_id} not found.")

            total_row = self._conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM assignments
                WHERE person_id = ?
                  AND status IN ('auto_high', 'candidate')
                  AND source NOT IN ('album_cohesion', 'model_trusted')
                """,
                (person_id,),
            ).fetchone()
            remaining = int(total_row["n"]) if total_row else 0

            pool_rows = self._conn.execute(
                """
                WITH top_cand AS (
                    SELECT photo_id, face_id, status, similarity
                    FROM assignments
                    WHERE person_id = ?
                      AND status IN ('auto_high', 'candidate')
                      AND source NOT IN ('album_cohesion', 'model_trusted')
                    ORDER BY similarity DESC, photo_id ASC
                    LIMIT ?
                )
                SELECT
                    tc.photo_id,
                    tc.face_id,
                    tc.status,
                    tc.similarity,
                    p.rel_path,
                    p.archive_child,
                    p.year,
                    p.album_leaf,
                    p.album_title,
                    p.album_date,
                    p.filename,
                    p.best_live_url,
                    p.face_count,
                    f.det_score,
                    f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left
                FROM top_cand tc
                JOIN photos p ON p.photo_id = tc.photo_id
                LEFT JOIN faces f ON f.face_id = tc.face_id
                ORDER BY tc.similarity DESC, p.album_date DESC, tc.photo_id ASC
                """,
                (person_id, fetch_pool),
            ).fetchall()

        # Lock order rule: `_faces_matrix_lock` is always taken BEFORE (never inside) `_lock`, so the
        # impact selection (which reads the RAM matrix + runs an SGEMM) must run with `_lock` released.
        impact_by_photo: dict[int, int] = {}
        rows: list[sqlite3.Row] = []
        overflow_rows: list[sqlite3.Row] = []
        seen_per_album: dict[tuple[str, str], int] = {}
        if mode == "impact" and pool_rows:
            rows, impact_by_photo = self._select_impactful_rows(pool_rows, batch_size)
            pool_rows = []
        with self._lock:
            # Pass 1: diversify across distinct albums (up to `max_per_album` per album) so confirming
            # 1-2 shots in an album lets Same-Album Cohesion sweep the rest of that outfit's burst!
            for r_cand in pool_rows:
                akey = (str(r_cand["archive_child"]), str(r_cand["album_leaf"]))
                cur_n = seen_per_album.get(akey, 0)
                if cur_n < max_per_album and len(rows) < batch_size:
                    rows.append(r_cand)
                    seen_per_album[akey] = cur_n + 1
                else:
                    overflow_rows.append(r_cand)
            if len(rows) < batch_size and overflow_rows:
                rows.extend(overflow_rows[: batch_size - len(rows)])

            companions_by_photo: dict[int, list[str]] = {}
            if rows:
                batch_pids_json = json.dumps([int(r["photo_id"]) for r in rows])
                comp_rows = self._conn.execute(
                    """
                    SELECT a.photo_id, pe.display_name
                    FROM assignments a
                    JOIN people pe ON pe.person_id = a.person_id
                    WHERE a.photo_id IN (SELECT value FROM json_each(?))
                      AND a.person_id != ?
                      AND pe.role != 'ignore'
                      AND (a.status = 'confirmed' OR (a.status = 'auto_high' AND a.similarity >= 0.50))
                    ORDER BY CASE WHEN a.status = 'confirmed' THEN 0 ELSE 1 END, a.similarity DESC
                    """,
                    (batch_pids_json, person_id),
                ).fetchall()
                for cr in comp_rows:
                    cpid = int(cr["photo_id"])
                    cname = str(cr["display_name"])
                    clist = companions_by_photo.setdefault(cpid, [])
                    if cname not in clist:
                        clist.append(cname)

        top_peers = self.get_person_peers(person_id)

        cards: list[dict[str, Any]] = []
        for r in rows:
            sim = float(r["similarity"])
            pid_val = int(r["photo_id"])
            cards.append(
                {
                    "photo_id": pid_val,
                    "face_id": int(r["face_id"]) if r["face_id"] is not None else None,
                    "status": str(r["status"]),
                    "similarity": round(sim, 4),
                    "default_approved": sim >= default_approve_min_sim,
                    "archive_child": str(r["archive_child"]),
                    "year": str(r["year"]),
                    "album_leaf": str(r["album_leaf"]),
                    "album_title": str(r["album_title"] or r["album_leaf"]),
                    "album_date": str(r["album_date"] or ""),
                    "filename": str(r["filename"]),
                    "best_live_url": str(r["best_live_url"] or ""),
                    "face_count": int(r["face_count"] or 0),
                    "companions": companions_by_photo.get(pid_val, []),
                    "impact": impact_by_photo.get(pid_val, 0),
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

        return {
            "person_id": int(person_row["person_id"]),
            "display_name": str(person_row["display_name"]),
            "remaining_unreviewed": remaining,
            "album_cohesion_count": int(person_row["cohesion_count"] or 0),
            "batch_count": len(cards),
            "mode": mode,
            "top_peers": top_peers,
            "cards": cards,
        }

    def _select_impactful_rows(
        self,
        pool_rows: list[sqlite3.Row],
        batch_size: int,
        *,
        lookalike_sim: float = 0.55,
        album_sim: float = 0.45,
        uncertain_center: float = 0.45,
    ) -> tuple[list[sqlite3.Row], dict[int, int]]:
        """
        Greedy max-coverage active learning: repeatedly pick the open face whose answer (confirm OR
        reject) would settle the most still-uncovered open faces of the same child — close look-alikes
        anywhere (`>= lookalike_sim`) plus same-album look-alikes (`>= album_sim`), which is exactly
        what album cohesion, temporal cohesion, hard negatives and the label vote propagate. Faces near
        the auto-accept boundary count more (most uncertain). Returns rows + `{photo_id: unlocks}`.
        """
        mat = self.load_all_faces_matrix()
        c_fids: NDArray[np.int64] = mat["face_ids"]
        c_embs: NDArray[np.float32] = mat["embeddings"]
        n_c = int(c_fids.shape[0])
        usable: list[sqlite3.Row] = []
        emb_rows: list[int] = []
        for r in pool_rows:
            if r["face_id"] is None or n_c == 0:
                continue
            fid = int(r["face_id"])
            pos = int(np.searchsorted(c_fids, fid))
            if pos < n_c and int(c_fids[pos]) == fid:
                usable.append(r)
                emb_rows.append(pos)
        if not usable:
            return list(pool_rows[:batch_size]), {}
        e_sub = c_embs[np.asarray(emb_rows, dtype=np.int64)]
        sims = e_sub @ e_sub.T
        album_keys = [f"{r['archive_child']}/{r['album_leaf']}" for r in usable]
        _, alb_codes = np.unique(np.asarray(album_keys, dtype=object), return_inverse=True)
        same_album = alb_codes[:, np.newaxis] == alb_codes[np.newaxis, :]
        adj = (sims >= lookalike_sim) | (same_album & (sims >= album_sim))
        np.fill_diagonal(adj, False)
        sim_vals = np.asarray([float(r["similarity"]) for r in usable], dtype=np.float32)
        uncert = np.clip(1.0 - np.abs(sim_vals - uncertain_center) / 0.15, 0.25, 1.0).astype(np.float32)
        covered = np.zeros((len(usable),), dtype=bool)
        picked: list[int] = []
        impact: dict[int, int] = {}
        adj_f = adj.astype(np.float32)
        for _ in range(min(batch_size, len(usable))):
            open_w = np.where(covered, 0.0, uncert).astype(np.float32)
            gains = np.where(covered, -1.0, uncert * (1.0 + adj_f @ open_w))
            best = int(np.argmax(gains))
            if gains[best] < 0:
                break
            unlocked = adj[best] & ~covered
            picked.append(best)
            impact[int(usable[best]["photo_id"])] = int(unlocked.sum())
            covered[best] = True
            covered |= unlocked
        return [usable[i] for i in picked], impact

    def list_person_photos(
        self,
        person_id: int,
        *,
        status_filter: str = "included",
        year: str | None = None,
        offset: int = 0,
        limit: int = 96,
    ) -> dict[str, Any]:
        s_tuple = (
            ("confirmed", "confirmed", "confirmed")
            if status_filter == "confirmed"
            else ("unsure", "unsure", "unsure")
            if status_filter == "unsure"
            else ("confirmed", "auto_high", "candidate")
            if status_filter == "all"
            else ("confirmed", "auto_high", "auto_high")
        )
        yr_param = year if year else None

        with self._lock:
            cnt_row = self._conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM assignments a
                JOIN photos p ON p.photo_id = a.photo_id
                WHERE a.person_id = ?
                  AND a.status IN (?, ?, ?)
                  AND (? IS NULL OR p.year = ?)
                """,
                (person_id, *s_tuple, yr_param, yr_param),
            ).fetchone()
            total = int(cnt_row["n"]) if cnt_row else 0

            rows = self._conn.execute(
                """
                SELECT
                    a.photo_id, a.face_id, a.status, a.source, a.similarity, a.is_anchor,
                    p.rel_path, p.archive_child, p.year, p.album_leaf, p.album_title,
                    p.album_date, p.filename, p.best_live_url, p.kinderpedia_image_id
                FROM assignments a
                JOIN photos p ON p.photo_id = a.photo_id
                WHERE a.person_id = ?
                  AND a.status IN (?, ?, ?)
                  AND (? IS NULL OR p.year = ?)
                ORDER BY p.album_date DESC, p.album_leaf DESC, p.filename ASC
                LIMIT ? OFFSET ?
                """,
                (person_id, *s_tuple, yr_param, yr_param, limit, offset),
            ).fetchall()

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": [
                {
                    "photo_id": int(r["photo_id"]),
                    "face_id": int(r["face_id"]) if r["face_id"] is not None else None,
                    "status": str(r["status"]),
                    "source": str(r["source"] or ""),
                    "similarity": round(float(r["similarity"]), 4),
                    "is_anchor": bool(r["is_anchor"]),
                    "archive_child": str(r["archive_child"]),
                    "year": str(r["year"]),
                    "album_leaf": str(r["album_leaf"]),
                    "album_title": str(r["album_title"] or r["album_leaf"]),
                    "album_date": str(r["album_date"] or ""),
                    "filename": str(r["filename"]),
                    "best_live_url": str(r["best_live_url"] or ""),
                    "kinderpedia_image_id": r["kinderpedia_image_id"],
                }
                for r in rows
            ],
        }

    def get_photo_detail(self, photo_id: int) -> dict[str, Any]:
        """Return full photo metadata + all detected face boxes and their current person assignments."""
        with self._lock:
            p_row = self._conn.execute(
                "SELECT * FROM photos WHERE photo_id = ?",
                (photo_id,),
            ).fetchone()
            if p_row is None:
                raise KeyError(f"Photo {photo_id} not found.")

            face_rows = self._conn.execute(
                """
                WITH best_face_assign AS (
                    SELECT
                        a.face_id,
                        a.person_id,
                        a.status,
                        a.similarity,
                        pe.display_name,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.face_id
                            ORDER BY
                                CASE a.status
                                    WHEN 'confirmed' THEN 0
                                    WHEN 'auto_high' THEN 1
                                    ELSE 2
                                END ASC,
                                a.updated_at DESC,
                                a.person_id ASC
                        ) AS rn
                    FROM assignments a
                    JOIN people pe ON pe.person_id = a.person_id
                    WHERE a.photo_id = ?
                      AND a.face_id IS NOT NULL
                      AND a.status IN ('confirmed', 'auto_high', 'candidate')
                      AND pe.role != 'ignore'
                )
                SELECT
                    f.face_id, f.face_idx, f.det_score,
                    f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                    f.cluster_id, f.suppress_auto, f.is_manual,
                    bfa.person_id AS eff_person_id,
                    bfa.status AS eff_status,
                    bfa.similarity AS eff_similarity,
                    bfa.display_name AS person_name
                FROM faces f
                LEFT JOIN best_face_assign bfa ON bfa.face_id = f.face_id AND bfa.rn = 1
                WHERE f.photo_id = ?
                ORDER BY f.bbox_left ASC
                """,
                (photo_id, photo_id),
            ).fetchall()

        return {
            "photo_id": int(p_row["photo_id"]),
            "rel_path": str(p_row["rel_path"]),
            "archive_child": str(p_row["archive_child"]),
            "year": str(p_row["year"]),
            "month": str(p_row["month"]),
            "album_leaf": str(p_row["album_leaf"]),
            "album_title": str(p_row["album_title"] or p_row["album_leaf"]),
            "album_date": str(p_row["album_date"] or ""),
            "classroom_groups": str(p_row["classroom_groups"] or ""),
            "filename": str(p_row["filename"]),
            "kinderpedia_album_id": p_row["kinderpedia_album_id"],
            "kinderpedia_image_id": p_row["kinderpedia_image_id"],
            "best_live_url": str(p_row["best_live_url"] or ""),
            "download_url": str(p_row["download_url"] or ""),
            "fullsize_url": str(p_row["fullsize_url"] or ""),
            "width": int(p_row["width"]),
            "height": int(p_row["height"]),
            "faces": [
                {
                    "face_id": int(fr["face_id"]),
                    "face_idx": int(fr["face_idx"]),
                    "det_score": round(float(fr["det_score"]), 4),
                    "box": {
                        "top": int(fr["bbox_top"]),
                        "right": int(fr["bbox_right"]),
                        "bottom": int(fr["bbox_bottom"]),
                        "left": int(fr["bbox_left"]),
                    },
                    "cluster_id": int(fr["cluster_id"]) if fr["cluster_id"] is not None else None,
                    "suppress_auto": bool(fr["suppress_auto"]),
                    "is_manual": bool(fr["is_manual"]),
                    "person_id": (
                        int(fr["eff_person_id"]) if fr["eff_person_id"] is not None else None
                    ),
                    "person_name": str(fr["person_name"]) if fr["person_name"] is not None else None,
                    "status": str(fr["eff_status"]) if fr["eff_status"] is not None else None,
                    "similarity": (
                        round(float(fr["eff_similarity"]), 4)
                        if fr["eff_similarity"] is not None
                        else None
                    ),
                }
                for fr in face_rows
            ],
        }

    def get_photo_detail_by_face(self, face_id: int) -> dict[str, Any]:
        """Look up the parent photo for `face_id` and return its full `get_photo_detail` payload."""
        with self._lock:
            row = self._conn.execute(
                "SELECT photo_id FROM faces WHERE face_id = ?",
                (face_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Face {face_id} not found.")
            photo_id = int(row["photo_id"])
        return self.get_photo_detail(photo_id)

    def get_cluster_faces(self, cluster_id: int, *, limit: int = 120) -> dict[str, Any]:
        """
        Return cluster metadata plus ordered `(face_id, photo_id, album_date, album_title)` items
        for inspecting a cluster's full-size classroom photos in context.
        Sample strip faces appear first in their displayed card order, followed by remaining cluster photos.
        """
        with self._lock:
            c_row = self._conn.execute(
                """
                SELECT
                    c.cluster_id, c.archive_child, c.year, c.person_id,
                    p.display_name AS person_name,
                    c.sample_face_ids_json, c.face_count, c.photo_count
                FROM clusters c
                LEFT JOIN people p ON p.person_id = c.person_id
                WHERE c.cluster_id = ?
                """,
                (cluster_id,),
            ).fetchone()
            if c_row is None:
                raise KeyError(f"Cluster {cluster_id} not found.")

            sample_ids: list[int] = [
                int(x) for x in json.loads(str(c_row["sample_face_ids_json"] or "[]"))
            ]
            rows = self._conn.execute(
                """
                SELECT
                    f.face_id,
                    f.photo_id,
                    f.det_score,
                    ph.album_date,
                    ph.album_title,
                    ph.album_leaf,
                    ph.year
                FROM faces f
                JOIN photos ph ON ph.photo_id = f.photo_id
                WHERE f.cluster_id = ?
                ORDER BY f.det_score DESC, ph.album_date DESC, f.face_id ASC
                """,
                (cluster_id,),
            ).fetchall()

        by_face_id: dict[int, sqlite3.Row] = {int(r["face_id"]): r for r in rows}
        ordered_items: list[dict[str, Any]] = []
        seen_faces: set[int] = set()
        seen_photos: set[int] = set()

        for sfid in sample_ids:
            r = by_face_id.get(sfid)
            if r is None or sfid in seen_faces:
                continue
            seen_faces.add(sfid)
            seen_photos.add(int(r["photo_id"]))
            ordered_items.append(
                {
                    "face_id": sfid,
                    "photo_id": int(r["photo_id"]),
                    "album_date": str(r["album_date"] or r["year"] or ""),
                    "album_title": str(r["album_title"] or r["album_leaf"] or ""),
                }
            )

        for r in rows:
            fid = int(r["face_id"])
            pid = int(r["photo_id"])
            if fid in seen_faces or pid in seen_photos:
                continue
            seen_faces.add(fid)
            seen_photos.add(pid)
            ordered_items.append(
                {
                    "face_id": fid,
                    "photo_id": pid,
                    "album_date": str(r["album_date"] or r["year"] or ""),
                    "album_title": str(r["album_title"] or r["album_leaf"] or ""),
                }
            )
            if len(ordered_items) >= limit:
                break

        return {
            "cluster_id": int(c_row["cluster_id"]),
            "archive_child": str(c_row["archive_child"]),
            "year": str(c_row["year"]),
            "person_id": int(c_row["person_id"]) if c_row["person_id"] is not None else None,
            "person_name": str(c_row["person_name"]) if c_row["person_name"] is not None else None,
            "photo_count": int(c_row["photo_count"]),
            "face_count": int(c_row["face_count"]),
            "faces": ordered_items,
        }

    def get_face_crop_spec(self, face_id: int) -> tuple[str, BoundingBox] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT p.rel_path, f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left
                FROM faces f
                JOIN photos p ON p.photo_id = f.photo_id
                WHERE f.face_id = ?
                """,
                (face_id,),
            ).fetchone()
        if row is None:
            return None
        return (
            str(row["rel_path"]),
            BoundingBox(
                top=int(row["bbox_top"]),
                right=int(row["bbox_right"]),
                bottom=int(row["bbox_bottom"]),
                left=int(row["bbox_left"]),
            ),
        )

    def get_photo_rel_path(self, photo_id: int) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT rel_path FROM photos WHERE photo_id = ?",
                (photo_id,),
            ).fetchone()
        return str(row["rel_path"]) if row is not None else None

    # ------------------------------------------------------------------
    # Portable Index Snapshot, Single-Child URL Export & Flat Hardlinks
    # ------------------------------------------------------------------

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
                    if (is_link or existing_entry.is_file()) and existing_entry.name not in expected_filenames:
                        existing_entry.unlink()
                        stats.links_pruned += 1

        if prune and not person_filter:
            stats.links_pruned += _prune_orphan_person_folders(out_root, backup_root, live_folders)

        stats.photos_matched = len(matched_photo_ids)
        return stats
