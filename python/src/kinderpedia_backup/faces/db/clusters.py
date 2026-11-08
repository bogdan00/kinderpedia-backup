"""Unsupervised per-year clusters: persistence, listing, inspection and confirmation."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .assignments import AssignmentsMixin
from .codec import (
    _now_iso,
)


class ClustersMixin(AssignmentsMixin):
    """Unsupervised per-year clusters: persistence, listing, inspection and confirmation."""

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
                    int(r["eff_suggested_person_id"]) if r["eff_suggested_person_id"] is not None else None
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

            old_samples = [int(x) for x in json.loads(str(c_row["sample_face_ids_json"] or "[]"))]
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

            sample_ids: list[int] = [int(x) for x in json.loads(str(c_row["sample_face_ids_json"] or "[]"))]
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
