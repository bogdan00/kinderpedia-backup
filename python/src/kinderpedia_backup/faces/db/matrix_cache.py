"""In-RAM embedding matrix cache and the matcher's anchor / negative loaders."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .codec import (
    date_ord_to_day_num,
    photo_date_ord,
    unpack_embeddings_f32,
)
from .people import PeopleMixin


class MatrixCacheMixin(PeopleMixin):
    """In-RAM embedding matrix cache and the matcher's anchor / negative loaders."""

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
                cnt_row = self._conn.execute("SELECT COUNT(*) AS n FROM faces WHERE is_manual = 0").fetchone()
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

            rows_by_album = {akey: np.asarray(r_list, dtype=np.int64) for akey, r_list in tmp_alb.items()}
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
        """Load active people, reference seeds, human decisions (confirmed / rejected / unsure faces),
        background (`ignore`) person ids, cluster pressure and reviewed-OK labels (`label_keeps`)."""
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
            label_keeps = {
                (int(r["face_id"]), int(r["person_id"]))
                for r in self._conn.execute("SELECT face_id, person_id FROM label_keeps").fetchall()
            }
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
            "label_keeps": label_keeps,
            "pressure": [
                {
                    "person_id": int(r["person_id"]),
                    "face_id": int(r["face_id"]),
                    "weight": float(r["weight"]),
                }
                for r in pressure_rows
            ],
        }
