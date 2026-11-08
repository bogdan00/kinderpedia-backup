"""People, reference seeds, birthdays, school classes and tagging eligibility."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from ..models import (
    PersonRole,
)
from .codec import (
    UNKNOWN_DATE_ORD,
    _now_iso,
    date_to_ord,
    normalize_iso_date,
    pack_embedding_f16,
    photo_date_ord,
)
from .core import DBCore


class PeopleMixin(DBCore):
    """People, reference seeds, birthdays, school classes and tagging eligibility."""

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
            if (
                self._conn.execute("SELECT 1 FROM people WHERE person_id = ?", (person_id,)).fetchone()
                is None
            ):
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
            if (
                self._conn.execute("SELECT 1 FROM people WHERE person_id = ?", (person_id,)).fetchone()
                is None
            ):
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
            self._conn.execute("UPDATE clusters SET person_id = NULL WHERE person_id = ?", (person_id,))
            self._conn.execute(
                "UPDATE clusters SET suggested_person_id = NULL, suggested_similarity = NULL WHERE suggested_person_id = ?",
                (person_id,),
            )
            self._conn.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
            self._conn.execute("DELETE FROM person_assignment_counts WHERE person_id = ?", (person_id,))
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
            if (
                self._conn.execute(
                    "SELECT 1 FROM school_classes WHERE name = ? COLLATE NOCASE", (f["name"],)
                ).fetchone()
                is not None
            ):
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
                int(r["person_id"]) for r in self._conn.execute("SELECT person_id FROM people").fetchall()
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
                {
                    "person_id": int(r["person_id"]),
                    "display_name": str(r["display_name"]),
                    "role": str(r["role"]),
                }
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
                cid for cid, w_feed, w_s, w_e in windows if (not w_feed or w_feed == feed) and w_s <= o <= w_e
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
