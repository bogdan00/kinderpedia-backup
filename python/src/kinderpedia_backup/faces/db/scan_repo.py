"""Scan persistence: change detection state, label-preserving re-link, re-key and photo deletion."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..models import (
    DetectedFace,
    PhotoProvenance,
    PhotoScanState,
    RelinkReport,
)
from .codec import (
    _now_iso,
    pack_embedding_f16,
    unpack_embeddings_f32,
)
from .matrix_cache import MatrixCacheMixin
from .relink import StoredFace, match_faces


class ScanRepoMixin(MatrixCacheMixin):
    """Scan persistence: change detection state, label-preserving re-link, re-key and photo deletion."""

    def get_photo_scan_states(self) -> dict[str, PhotoScanState]:
        """Return `{rel_path: PhotoScanState}` (size, mtime, detector, hash, Kinderpedia id, pixel size)."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT photo_id, rel_path, file_size, mtime_ns, detector_id, content_hash,
                       kinderpedia_image_id, filename, width, height
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
                width=int(r["width"] or 0),
                height=int(r["height"] or 0),
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
