"""Read models for the Web UI: overview, batch proposals, galleries, photo detail, child clusters."""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..label_guard import LABEL_OUTLIER_MIN_FACES, LABEL_OUTLIER_MIN_SIM, label_outlier_mask
from ..models import (
    BoundingBox,
)
from .clusters import ClustersMixin
from .codec import _now_iso


def _pack_person_cluster(
    year_items: list[dict[str, Any]],
    sub: NDArray[np.float32],
    leader_loc: int,
    member_locs: list[int],
    is_loose: bool,
    yr_key: str,
) -> dict[str, Any]:
    """Serialize one per-child visual sub-cluster (faces sorted by similarity to the leader).

    `sub` holds the L2-normalized embeddings of `year_items` (row-aligned); only O(n * dim) work is
    done here (no pairwise matrix), so it is safe for very large per-year groups.
    """
    m_arr = np.asarray(member_locs, dtype=np.int64)
    l_sims = sub[m_arr] @ sub[leader_loc]
    face_list: list[dict[str, Any]] = []
    for pos, m in enumerate(member_locs):
        it = dict(year_items[m])
        it.pop("det_score", None)
        it["sim_to_leader"] = round(float(l_sims[pos]), 4)
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
            # Mean pairwise cosine of unit vectors: (||sum||^2 - n) / (n (n - 1)).
            tot = sub[m_arr].sum(axis=0, dtype=np.float64)
            cohesion = float((float(tot @ tot) - n_m) / (n_m * (n_m - 1)))
    else:
        cohesion = float(np.mean(l_sims))
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


class QueriesUIMixin(ClustersMixin):
    """Read models for the Web UI: overview, batch proposals, galleries, photo detail, child clusters."""

    def get_person_face_clusters(
        self,
        person_id: int,
        *,
        threshold: float = 0.50,
        min_size: int = 3,
        max_pairwise_faces: int = 4096,
    ) -> dict[str, Any]:
        """
        Group everything currently attributed to `person_id` (confirmed + auto_high + candidate, plus
        faces carrying boost/penalize pressure) into per-year visual sub-clusters using density-leader
        grouping on the cached ArcFace matrix. Faces that do not form a group of `min_size` land in a
        per-year loose bucket. Per year, leader discovery uses at most `max_pairwise_faces` faces
        (a `cap x cap` similarity matrix, ~64 MB at the default); the rest are attached to their
        nearest leader in chunks, so memory no longer grows quadratically with the year's size.
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

        cap = max(16, int(max_pairwise_faces))
        clusters: list[dict[str, Any]] = []
        for yr in sorted(by_year.keys(), reverse=True):
            idxs = by_year[yr]
            n_all = len(idxs)
            sub_all = c_embs[np.asarray([emb_rows[i] for i in idxs], dtype=np.int64)]
            det_all = np.asarray([items[i]["det_score"] for i in idxs], dtype=np.float32)
            if n_all > cap:
                # Leader discovery runs on a deterministic sample (best detections first, then
                # face_id) so the pairwise matrix stays <= cap^2; every other face is attached to
                # its nearest leader below in O(n * leaders) chunks.
                fid_all = np.asarray([items[i]["face_id"] for i in idxs], dtype=np.int64)
                sample = np.sort(np.lexsort((fid_all, -det_all))[:cap])
            else:
                sample = np.arange(n_all, dtype=np.int64)
            sub = sub_all[sample]
            sims = sub @ sub.T
            adj = sims >= thr
            unassigned = np.ones((len(sample),), dtype=bool)
            det = det_all[sample]
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
            del sims, adj
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
            # Map sample-local indices back to year-local ones.
            year_groups = [(int(sample[ld]), sample[mem]) for ld, mem in year_groups]
            # Second pass: leftover faces (undersized groups + faces outside the sample) join their
            # best group if close enough, otherwise they land in the per-year loose bucket.
            grouped = np.zeros((n_all,), dtype=bool)
            for _, members in year_groups:
                grouped[members] = True
            leftovers = np.where(~grouped)[0]
            if year_groups and leftovers.size:
                leader_arr = np.asarray([g[0] for g in year_groups], dtype=np.int64)
                lead_embs = sub_all[leader_arr]
                extra: dict[int, list[int]] = {}
                for c0 in range(0, int(leftovers.size), 4096):
                    chunk_locs = leftovers[c0 : c0 + 4096]
                    l_sims_all = sub_all[chunk_locs] @ lead_embs.T
                    best = np.argmax(l_sims_all, axis=1)
                    for pos, loc in enumerate(chunk_locs.tolist()):
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
                    _pack_person_cluster(
                        year_items, sub_all, leader_loc, [int(m) for m in members], False, yr
                    )
                )
            if loose:
                loose_sorted = sorted(set(loose))
                clusters.append(
                    _pack_person_cluster(year_items, sub_all, loose_sorted[0], loose_sorted, True, yr)
                )

        return {
            "person_id": person_id,
            "display_name": str(p_row["display_name"]),
            "threshold": thr,
            "total_faces": len(items),
            "clusters": clusters,
        }

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
            suspicious_by_pid: dict[int, int] = {}
            for _fid, s_pid in self._label_outliers:
                suspicious_by_pid[s_pid] = suspicious_by_pid.get(s_pid, 0) + 1
            for person in people_list:
                person["suspicious_count"] = suspicious_by_pid.get(int(person["person_id"]), 0)

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

    def set_label_outliers(self, flagged: set[tuple[int, int]]) -> None:
        """Remember the `(face_id, person_id)` labels the last rematch flagged as suspicious
        (feeds `people[].suspicious_count` in the overview without recomputing the guard)."""
        self._label_outliers = set(flagged)

    def get_suspicious_labels(
        self,
        person_id: int,
        *,
        min_sim: float = LABEL_OUTLIER_MIN_SIM,
        min_faces: int = LABEL_OUTLIER_MIN_FACES,
    ) -> dict[str, Any]:
        """
        Confirmed labels of `person_id` the label-consistency guard flags (see `faces/label_guard.py`):
        per year with >= `min_faces` confirmed faces, those whose best similarity to the person's
        other confirmed faces that year is < `min_sim`, excluding labels kept via `keep_label()`.
        Sorted by `max_same_person_sim` ascending (most suspicious first). Background (`ignore`)
        profiles are never checked. Computed fresh from the current labels.
        """
        with self._lock:
            p_row = self._conn.execute(
                "SELECT role FROM people WHERE person_id = ?", (person_id,)
            ).fetchone()
            if p_row is None:
                raise KeyError(f"Person {person_id} not found.")
            rows = (
                self._conn.execute(
                    """
                    SELECT
                        a.face_id, a.photo_id, a.source,
                        p.year, p.archive_child, p.album_leaf, p.album_title, p.album_date, p.filename,
                        (k.face_id IS NOT NULL) AS kept
                    FROM assignments a
                    JOIN faces f ON f.face_id = a.face_id AND f.is_manual = 0
                    JOIN photos p ON p.photo_id = a.photo_id
                    LEFT JOIN label_keeps k ON k.face_id = a.face_id AND k.person_id = a.person_id
                    WHERE a.person_id = ? AND a.status = 'confirmed'
                    ORDER BY a.face_id
                    """,
                    (person_id,),
                ).fetchall()
                if str(p_row["role"]) != "ignore"
                else []
            )
        items: list[dict[str, Any]] = []
        if rows:
            mat = self.load_all_faces_matrix()
            m_fids: NDArray[np.int64] = mat["face_ids"]
            m_embs: NDArray[np.float32] = mat["embeddings"]
            q_fids = np.asarray([int(r["face_id"]) for r in rows], dtype=np.int64)
            pos = np.searchsorted(m_fids, q_fids)
            ok = (pos < m_fids.shape[0]) & (m_fids[np.minimum(pos, max(0, m_fids.shape[0] - 1))] == q_fids)
            by_year: dict[int, list[int]] = {}
            for i, r in enumerate(rows):
                if bool(ok[i]):
                    y_str = str(r["year"])
                    by_year.setdefault(int(y_str) if y_str.isdigit() and len(y_str) == 4 else 0, []).append(i)
            for _yr, idxs in sorted(by_year.items()):
                if len(idxs) < max(2, min_faces):
                    continue
                mask, best, nearest = label_outlier_mask(
                    m_embs[pos[np.asarray(idxs, dtype=np.int64)]], min_sim=min_sim, min_faces=min_faces
                )
                for loc, i in enumerate(idxs):
                    r = rows[i]
                    if not bool(mask[loc]) or bool(r["kept"]):
                        continue
                    items.append(
                        {
                            "face_id": int(r["face_id"]),
                            "photo_id": int(r["photo_id"]),
                            "year": str(r["year"]),
                            "source": str(r["source"]),
                            "max_same_person_sim": round(float(best[loc]), 4),
                            "nearest_face_id": int(rows[idxs[int(nearest[loc])]]["face_id"]),
                            "year_confirmed_count": len(idxs),
                            "archive_child": str(r["archive_child"]),
                            "album_leaf": str(r["album_leaf"]),
                            "album_title": str(r["album_title"] or r["album_leaf"]),
                            "album_date": str(r["album_date"] or ""),
                            "filename": str(r["filename"]),
                        }
                    )
        items.sort(key=lambda it: (float(it["max_same_person_sim"]), int(it["face_id"])))
        return {"person_id": person_id, "min_sim": min_sim, "min_faces": min_faces, "items": items}

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
        default_approve_min_sim: float = 0.50,
        max_per_album: int = 2,
        mode: str = "impact",
        audit_per_batch: int | None = None,
        ambiguous_min_sim: float = 0.38,
    ) -> dict[str, Any]:
        """
        Fetch the next active-learning batch of unconfirmed matches (excluding matches already trusted).
        - `mode="impact"` (default): greedy max-coverage selection of the faces whose answer resolves
          the most OTHER open faces of this child (same-album look-alikes and close look-alikes
          anywhere), weighted by how uncertain each face is.
        - `mode="confident"`: highest-similarity first, diversified across albums.

        Card defaults: `default_approved` is only true at `similarity >= default_approve_min_sim`
        (0.50 — below that historic batch decisions show ~20 % errors), and `default_decision` is
        `"approved"` / `"unsure"` (ambiguous band `ambiguous_min_sim..default_approve_min_sim`,
        opt-in) / `"disproved"`.
        Audit sampling: when the person has trusted auto rows (`album_cohesion` / `model_trusted`,
        normally never reviewed), ~2 random ones per 24 cards (`audit_per_batch`, default
        `round(batch_size / 12)`) are inserted at random positions with `"audit": true`, so drift of
        the unreviewed channels is observable; the impact-ordered cards keep their relative order.
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

            n_audit_want = (
                max(0, int(audit_per_batch))
                if audit_per_batch is not None
                else max(0, round(batch_size / 12))
            )
            n_audit_want = min(n_audit_want, max(0, batch_size - 1))
            audit_rows: list[sqlite3.Row] = []
            if n_audit_want > 0 and int(person_row["cohesion_count"] or 0) > 0:
                audit_rows = self._conn.execute(
                    """
                    WITH pick AS (
                        SELECT photo_id, face_id, status, source, similarity
                        FROM assignments
                        WHERE person_id = ?
                          AND status = 'auto_high'
                          AND source IN ('album_cohesion', 'model_trusted')
                        ORDER BY RANDOM()
                        LIMIT ?
                    )
                    SELECT
                        tc.photo_id, tc.face_id, tc.status, tc.source, tc.similarity,
                        p.rel_path, p.archive_child, p.year, p.album_leaf, p.album_title,
                        p.album_date, p.filename, p.best_live_url, p.face_count,
                        f.det_score, f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left
                    FROM pick tc
                    JOIN photos p ON p.photo_id = tc.photo_id
                    LEFT JOIN faces f ON f.face_id = tc.face_id
                    """,
                    (person_id, n_audit_want),
                ).fetchall()
            main_size = max(1, batch_size - len(audit_rows))

            pool_rows = self._conn.execute(
                """
                WITH top_cand AS (
                    SELECT photo_id, face_id, status, source, similarity
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
                    tc.source,
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
            rows, impact_by_photo = self._select_impactful_rows(pool_rows, main_size)
            pool_rows = []
        with self._lock:
            # Pass 1: diversify across distinct albums (up to `max_per_album` per album) so confirming
            # 1-2 shots in an album lets Same-Album Cohesion sweep the rest of that outfit's burst!
            for r_cand in pool_rows:
                akey = (str(r_cand["archive_child"]), str(r_cand["album_leaf"]))
                cur_n = seen_per_album.get(akey, 0)
                if cur_n < max_per_album and len(rows) < main_size:
                    rows.append(r_cand)
                    seen_per_album[akey] = cur_n + 1
                else:
                    overflow_rows.append(r_cand)
            if len(rows) < main_size and overflow_rows:
                rows.extend(overflow_rows[: main_size - len(rows)])
            audit_photo_ids: set[int] = set()
            if audit_rows:
                rng = random.Random()
                merged: list[sqlite3.Row] = list(rows)
                for a_row in audit_rows:
                    merged.insert(rng.randint(0, len(merged)), a_row)
                    audit_photo_ids.add(int(a_row["photo_id"]))
                rows = merged

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
                    "source": str(r["source"] or ""),
                    "similarity": round(sim, 4),
                    "default_approved": sim >= default_approve_min_sim,
                    "default_decision": (
                        "approved"
                        if sim >= default_approve_min_sim
                        else "unsure"
                        if sim >= ambiguous_min_sim
                        else "disproved"
                    ),
                    "audit": pid_val in audit_photo_ids,
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
            "audit_count": sum(1 for c in cards if c["audit"]),
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
                    "person_id": (int(fr["eff_person_id"]) if fr["eff_person_id"] is not None else None),
                    "person_name": str(fr["person_name"]) if fr["person_name"] is not None else None,
                    "status": str(fr["eff_status"]) if fr["eff_status"] is not None else None,
                    "similarity": (
                        round(float(fr["eff_similarity"]), 4) if fr["eff_similarity"] is not None else None
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

    # ------------------------------------------------------------------ Album Browser
    def list_albums(
        self,
        *,
        archive_child: str | None = None,
        year: str | None = None,
        search: str | None = None,
        limit: int = 48,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List albums matching optional filters, with photo count, face count, and label stats."""
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        child_clean = str(archive_child or "").strip() or None
        year_clean = str(year or "").strip() or None
        search_clean = str(search or "").strip() or None

        with self._lock:
            avail_children_rows = self._conn.execute(
                """
                SELECT DISTINCT archive_child
                FROM photos
                WHERE archive_child != '' AND archive_child NOT LIKE 'wt-%'
                  AND archive_child NOT IN ('bookmarklet', 'kinderpedia-backup')
                ORDER BY archive_child ASC
                """
            ).fetchall()
            available_children = [str(r[0]) for r in avail_children_rows]

            avail_years_rows = self._conn.execute(
                """
                SELECT DISTINCT year
                FROM photos
                WHERE year GLOB '[0-9][0-9][0-9][0-9]'
                ORDER BY year DESC
                """
            ).fetchall()
            available_years = [str(r[0]) for r in avail_years_rows]

            search_pat = f"%{search_clean}%" if search_clean else None
            filter_params = (
                child_clean,
                child_clean,
                year_clean,
                year_clean,
                search_pat,
                search_pat,
                search_pat,
            )

            count_sql = """
                SELECT COUNT(DISTINCT p.archive_child || '/' || p.album_leaf)
                FROM photos p
                WHERE p.year GLOB '[0-9][0-9][0-9][0-9]'
                  AND (? IS NULL OR p.archive_child = ?)
                  AND (? IS NULL OR p.year = ?)
                  AND (? IS NULL OR p.album_title LIKE ? OR p.album_leaf LIKE ?)
            """
            total = int(self._conn.execute(count_sql, filter_params).fetchone()[0])

            albums_sql = """
                SELECT
                    p.archive_child,
                    p.year,
                    p.month,
                    p.album_leaf,
                    p.album_title,
                    MIN(p.album_date) AS min_date,
                    COUNT(p.photo_id) AS photo_count,
                    SUM(p.face_count) AS total_faces,
                    MIN(p.photo_id) AS cover_photo_id
                FROM photos p
                WHERE p.year GLOB '[0-9][0-9][0-9][0-9]'
                  AND (? IS NULL OR p.archive_child = ?)
                  AND (? IS NULL OR p.year = ?)
                  AND (? IS NULL OR p.album_title LIKE ? OR p.album_leaf LIKE ?)
                GROUP BY p.archive_child, p.album_leaf
                ORDER BY p.year DESC, p.month DESC, min_date DESC, p.album_leaf ASC
                LIMIT ? OFFSET ?
            """
            paged_params = [*filter_params, limit, offset]
            alb_rows = self._conn.execute(albums_sql, paged_params).fetchall()
            if not alb_rows:
                return {
                    "albums": [],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "available_children": available_children,
                    "available_years": available_years,
                }

            keys = [(str(r["archive_child"]), str(r["album_leaf"])) for r in alb_rows]
            placeholders = ", ".join(["(?, ?)"] * len(keys))
            key_params: list[str] = [val for k in keys for val in k]

            status_sql = f"""
                SELECT
                    p.archive_child,
                    p.album_leaf,
                    COUNT(DISTINCT f.face_id) AS total_faces,
                    COUNT(DISTINCT CASE WHEN a.status = 'confirmed' THEN f.face_id END) AS confirmed_count,
                    COUNT(DISTINCT CASE WHEN a.status = 'auto_high' THEN f.face_id END) AS auto_count,
                    COUNT(DISTINCT CASE WHEN a.status = 'candidate' THEN f.face_id END) AS candidate_count
                FROM photos p
                JOIN faces f ON f.photo_id = p.photo_id
                LEFT JOIN assignments a ON a.face_id = f.face_id AND a.status IN ('confirmed', 'auto_high', 'candidate')
                WHERE (p.archive_child, p.album_leaf) IN ({placeholders})
                GROUP BY p.archive_child, p.album_leaf
            """  # noqa: S608 - safe placeholder tuple expansion
            status_map: dict[tuple[str, str], dict[str, int]] = {}
            for sr in self._conn.execute(status_sql, key_params).fetchall():
                t_faces = int(sr["total_faces"] or 0)
                conf = int(sr["confirmed_count"] or 0)
                auto = int(sr["auto_count"] or 0)
                cand = int(sr["candidate_count"] or 0)
                unass = max(0, t_faces - conf - auto - cand)
                status_map[(str(sr["archive_child"]), str(sr["album_leaf"]))] = {
                    "confirmed": conf,
                    "auto": auto,
                    "candidate": cand,
                    "unassigned": unass,
                }

            people_sql = f"""
                SELECT
                    p.archive_child,
                    p.album_leaf,
                    pe.person_id,
                    pe.display_name,
                    COUNT(DISTINCT a.face_id) AS cnt,
                    COUNT(DISTINCT CASE WHEN a.status = 'confirmed' THEN a.face_id END) AS confirmed_cnt
                FROM photos p
                JOIN assignments a ON a.photo_id = p.photo_id AND a.status IN ('confirmed', 'auto_high') AND a.face_id IS NOT NULL
                JOIN people pe ON pe.person_id = a.person_id
                WHERE (p.archive_child, p.album_leaf) IN ({placeholders})
                  AND pe.role != 'ignore'
                GROUP BY p.archive_child, p.album_leaf, pe.person_id
                ORDER BY p.archive_child, p.album_leaf, cnt DESC
            """  # noqa: S608 - safe placeholder tuple expansion
            people_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for pr in self._conn.execute(people_sql, key_params).fetchall():
                k = (str(pr["archive_child"]), str(pr["album_leaf"]))
                people_map.setdefault(k, []).append(
                    {
                        "person_id": int(pr["person_id"]),
                        "name": str(pr["display_name"]),
                        "count": int(pr["cnt"]),
                        "confirmed_count": int(pr["confirmed_cnt"]),
                    }
                )

            albums = []
            for r in alb_rows:
                ach = str(r["archive_child"])
                leaf = str(r["album_leaf"])
                k = (ach, leaf)
                s = status_map.get(k, {"confirmed": 0, "auto": 0, "candidate": 0, "unassigned": 0})
                albums.append(
                    {
                        "album_key": f"{ach}/{leaf}",
                        "archive_child": ach,
                        "year": str(r["year"]),
                        "month": str(r["month"] or ""),
                        "album_leaf": leaf,
                        "album_title": str(r["album_title"] or leaf),
                        "album_date": str(r["min_date"] or ""),
                        "photo_count": int(r["photo_count"]),
                        "face_count": int(r["total_faces"] or 0),
                        "cover_photo_id": int(r["cover_photo_id"]),
                        "confirmed_count": s["confirmed"],
                        "auto_count": s["auto"],
                        "candidate_count": s["candidate"],
                        "unassigned_count": s["unassigned"],
                        "top_people": people_map.get(k, [])[:6],
                    }
                )

            return {
                "albums": albums,
                "total": total,
                "limit": limit,
                "offset": offset,
                "available_children": available_children,
                "available_years": available_years,
            }

    def get_album_detail(
        self,
        *,
        album_key: str | None = None,
        photo_id: int | None = None,
    ) -> dict[str, Any]:
        """Fetch all photos and faces in an album, with metadata and current assignments."""
        with self._lock:
            if photo_id is not None:
                p_row = self._conn.execute(
                    "SELECT archive_child, album_leaf FROM photos WHERE photo_id = ?",
                    (photo_id,),
                ).fetchone()
                if p_row is None:
                    raise KeyError(f"Photo {photo_id} not found.")
                archive_child = str(p_row["archive_child"])
                album_leaf = str(p_row["album_leaf"])
            elif album_key:
                parts = album_key.split("/", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid album key: {album_key!r}")
                archive_child, album_leaf = parts[0], parts[1]
            else:
                raise ValueError("Must provide either album_key or photo_id.")

            meta_row = self._conn.execute(
                """
                SELECT
                    archive_child, year, month, album_leaf, album_title,
                    MIN(album_date) AS min_date, MAX(album_date) AS max_date,
                    COUNT(photo_id) AS photo_count, SUM(face_count) AS total_faces
                FROM photos
                WHERE archive_child = ? AND album_leaf = ?
                GROUP BY archive_child, album_leaf
                """,
                (archive_child, album_leaf),
            ).fetchone()
            if meta_row is None:
                raise KeyError(f"Album {archive_child}/{album_leaf} not found.")

            photos_rows = self._conn.execute(
                """
                SELECT
                    photo_id, rel_path, filename, seq_in_album, width, height, album_date,
                    face_count, best_live_url, download_url
                FROM photos
                WHERE archive_child = ? AND album_leaf = ?
                ORDER BY seq_in_album ASC, photo_id ASC
                """,
                (archive_child, album_leaf),
            ).fetchall()

            faces_rows = self._conn.execute(
                """
                WITH album_photos AS (
                    SELECT photo_id FROM photos WHERE archive_child = ? AND album_leaf = ?
                ),
                best_face_assign AS (
                    SELECT
                        a.face_id, a.person_id, a.status, a.similarity, pe.display_name,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.face_id
                            ORDER BY CASE a.status WHEN 'confirmed' THEN 0 WHEN 'auto_high' THEN 1 ELSE 2 END ASC,
                                     a.updated_at DESC, a.person_id ASC
                        ) AS rn
                    FROM assignments a
                    JOIN people pe ON pe.person_id = a.person_id
                    WHERE a.photo_id IN (SELECT photo_id FROM album_photos)
                      AND a.face_id IS NOT NULL
                      AND a.status IN ('confirmed', 'auto_high', 'candidate')
                      AND pe.role != 'ignore'
                )
                SELECT
                    f.face_id, f.photo_id, f.face_idx, f.det_score,
                    f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                    f.is_manual,
                    bfa.person_id, bfa.status, bfa.similarity, bfa.display_name AS person_name
                FROM faces f
                JOIN album_photos ap ON ap.photo_id = f.photo_id
                LEFT JOIN best_face_assign bfa ON bfa.face_id = f.face_id AND bfa.rn = 1
                ORDER BY f.photo_id ASC, f.bbox_left ASC
                """,
                (archive_child, album_leaf),
            ).fetchall()

        faces_by_photo: dict[int, list[dict[str, Any]]] = {}
        person_counts: dict[int, dict[str, Any]] = {}
        confirmed_count = 0
        auto_count = 0
        candidate_count = 0

        for fr in faces_rows:
            fid = int(fr["face_id"])
            ph_id = int(fr["photo_id"])
            pid = fr["person_id"]
            pname = fr["person_name"]
            st = fr["status"]
            sim = float(fr["similarity"]) if fr["similarity"] is not None else None

            face_obj = {
                "face_id": fid,
                "photo_id": ph_id,
                "face_idx": int(fr["face_idx"]),
                "det_score": float(fr["det_score"] or 0.0),
                "bbox_top": int(fr["bbox_top"]),
                "bbox_right": int(fr["bbox_right"]),
                "bbox_bottom": int(fr["bbox_bottom"]),
                "bbox_left": int(fr["bbox_left"]),
                "is_manual": bool(fr["is_manual"]),
                "person_id": int(pid) if pid is not None else None,
                "person_name": str(pname) if pname is not None else None,
                "status": str(st) if st is not None else None,
                "similarity": sim,
            }
            faces_by_photo.setdefault(ph_id, []).append(face_obj)

            if st == "confirmed":
                confirmed_count += 1
            elif st == "auto_high":
                auto_count += 1
            elif st == "candidate":
                candidate_count += 1

            if pid is not None and pname is not None:
                pid_int = int(pid)
                if pid_int not in person_counts:
                    person_counts[pid_int] = {
                        "person_id": pid_int,
                        "display_name": str(pname),
                        "total_count": 0,
                        "confirmed_count": 0,
                        "auto_count": 0,
                        "candidate_count": 0,
                    }
                entry = person_counts[pid_int]
                entry["total_count"] += 1
                if st == "confirmed":
                    entry["confirmed_count"] += 1
                elif st == "auto_high":
                    entry["auto_count"] += 1
                elif st == "candidate":
                    entry["candidate_count"] += 1

        total_faces = len(faces_rows)
        unassigned_count = max(0, total_faces - confirmed_count - auto_count - candidate_count)

        photos = []
        for pr in photos_rows:
            ph_id = int(pr["photo_id"])
            photos.append(
                {
                    "photo_id": ph_id,
                    "rel_path": str(pr["rel_path"]),
                    "filename": str(pr["filename"]),
                    "seq_in_album": int(pr["seq_in_album"] or 0),
                    "album_date": str(pr["album_date"] or ""),
                    "width": int(pr["width"] or 0),
                    "height": int(pr["height"] or 0),
                    "face_count": int(pr["face_count"] or 0),
                    "best_live_url": str(pr["best_live_url"] or "") or None,
                    "download_url": str(pr["download_url"] or "") or None,
                    "faces": faces_by_photo.get(ph_id, []),
                }
            )

        sorted_people = sorted(
            person_counts.values(),
            key=lambda x: (x["total_count"], x["confirmed_count"]),
            reverse=True,
        )

        return {
            "album_key": f"{archive_child}/{album_leaf}",
            "archive_child": archive_child,
            "album_leaf": album_leaf,
            "album_title": str(meta_row["album_title"] or album_leaf),
            "year": str(meta_row["year"]),
            "month": str(meta_row["month"] or ""),
            "album_date": str(meta_row["min_date"] or ""),
            "photo_count": len(photos),
            "face_count": total_faces,
            "confirmed_count": confirmed_count,
            "auto_count": auto_count,
            "candidate_count": candidate_count,
            "unassigned_count": unassigned_count,
            "people": sorted_people,
            "appearances": self.get_album_appearances(archive_child, album_leaf),
            "photos": photos,
        }

    def approve_album_matches(
        self,
        archive_child: str,
        album_leaf: str,
        *,
        person_id: int | None = None,
    ) -> dict[str, Any]:
        """Approve all model-matched ('auto_high') faces in an album, claiming them as confirmed."""
        now = _now_iso()
        with self._lock:
            q = """
                SELECT a.photo_id, a.face_id, a.person_id
                FROM assignments a
                JOIN photos p ON p.photo_id = a.photo_id
                WHERE p.archive_child = ? AND p.album_leaf = ?
                  AND a.status = 'auto_high'
                  AND a.face_id IS NOT NULL
                  AND (? IS NULL OR a.person_id = ?)
            """
            rows = self._conn.execute(q, (archive_child, album_leaf, person_id, person_id)).fetchall()
            approved_count = 0
            for r in rows:
                res = self._claim_face_locked(
                    int(r["face_id"]),
                    int(r["person_id"]),
                    source="batch_approve",
                    now=now,
                    explicit=False,
                    photo_id=int(r["photo_id"]),
                )
                if res["claimed"]:
                    approved_count += 1
            if approved_count > 0:
                self._conn.commit()
        return {
            "approved_count": approved_count,
            "archive_child": archive_child,
            "album_leaf": album_leaf,
        }

    def record_album_appearance(
        self,
        archive_child: str,
        album_leaf: str,
        person_id: int,
        sample_photo_id: int,
        sample_face_id: int | None,
        clothing_summary: str,
        top_color: str = "",
        bottom_color: str = "",
        distinctive_features: str = "",
        apparent_gender: str = "unknown",
        verified_count: int = 0,
    ) -> None:
        """Cache the detected clothing & appearance signature for a child in a specific album."""
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO album_appearances (
                    archive_child, album_leaf, person_id, sample_photo_id, sample_face_id,
                    clothing_summary, top_color, bottom_color, distinctive_features,
                    apparent_gender, verified_faces_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(archive_child, album_leaf, person_id) DO UPDATE SET
                    sample_photo_id = excluded.sample_photo_id,
                    sample_face_id = excluded.sample_face_id,
                    clothing_summary = excluded.clothing_summary,
                    top_color = excluded.top_color,
                    bottom_color = excluded.bottom_color,
                    distinctive_features = excluded.distinctive_features,
                    apparent_gender = excluded.apparent_gender,
                    verified_faces_count = excluded.verified_faces_count,
                    created_at = excluded.created_at
                """,
                (
                    archive_child,
                    album_leaf,
                    person_id,
                    sample_photo_id,
                    sample_face_id,
                    clothing_summary,
                    top_color,
                    bottom_color,
                    distinctive_features,
                    apparent_gender,
                    verified_count,
                    now,
                ),
            )
            self._conn.commit()

    def get_album_appearances(
        self,
        archive_child: str,
        album_leaf: str,
    ) -> list[dict[str, Any]]:
        """Fetch cached appearance signatures for an album."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    aa.*, pe.display_name AS person_name
                FROM album_appearances aa
                JOIN people pe ON pe.person_id = aa.person_id
                WHERE aa.archive_child = ? AND aa.album_leaf = ?
                ORDER BY pe.display_name ASC
                """,
                (archive_child, album_leaf),
            ).fetchall()
            return [
                {
                    "archive_child": str(r["archive_child"]),
                    "album_leaf": str(r["album_leaf"]),
                    "person_id": int(r["person_id"]),
                    "person_name": str(r["person_name"]),
                    "sample_photo_id": int(r["sample_photo_id"]),
                    "sample_face_id": int(r["sample_face_id"]) if r["sample_face_id"] is not None else None,
                    "clothing_summary": str(r["clothing_summary"]),
                    "top_color": str(r["top_color"]),
                    "bottom_color": str(r["bottom_color"]),
                    "distinctive_features": str(r["distinctive_features"]),
                    "apparent_gender": str(r["apparent_gender"]),
                    "verified_faces_count": int(r["verified_faces_count"]),
                    "created_at": str(r["created_at"]),
                }
                for r in rows
            ]
