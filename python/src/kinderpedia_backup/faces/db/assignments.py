"""The assignment state machine, built on the single `_claim_face_locked` primitive."""

from __future__ import annotations

from typing import Any

from ..models import (
    AssignmentSource,
    AssignmentStatus,
    BoundingBox,
)
from .codec import (
    _now_iso,
)
from .scan_repo import ScanRepoMixin


class AssignmentsMixin(ScanRepoMixin):
    """The assignment state machine, built on the single `_claim_face_locked` primitive."""

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

            to_delete = [key for key in prev_model if key not in desired_model]
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
                self._photo_summary_cache["faces_total"] = int(self._photo_summary_cache["faces_total"]) + 1

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
            if (
                self._conn.execute("SELECT 1 FROM people WHERE person_id = ?", (person_id,)).fetchone()
                is None
            ):
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
