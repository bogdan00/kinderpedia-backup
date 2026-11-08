"""Data-layer tests: label-preserving rescans, reconciliation, migrations, claim primitive, locks."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image

from kinderpedia_backup.faces import BoundingBox, DetectedFace, FacesIndexDB, run_archive_face_scan
from kinderpedia_backup.faces.db.relink import StoredFace, bbox_iou, match_faces
from kinderpedia_backup.faces.db.scan_lock import ScanLock, ScanLockBusyError, read_scan_lock
from kinderpedia_backup.faces.db.schema import (
    ASSIGNMENT_SOURCES,
    ASSIGNMENT_STATUSES,
    PERSON_ROLES,
    SCHEMA_VERSION,
)
from kinderpedia_backup.faces.matcher import recompute_active_matching
from kinderpedia_backup.faces.models import PhotoProvenance
from kinderpedia_backup.faces.storage import file_fingerprint

DIM = 512
MATERIALIZE_MODULE = "kinderpedia_backup.faces.index_db"


def _vec(index: int, *, noise: float = 0.0, seed: int = 0) -> tuple[float, ...]:
    arr = np.zeros((DIM,), dtype=np.float32)
    arr[index] = 1.0
    if noise:
        arr += np.random.default_rng(seed).normal(0.0, noise, size=DIM).astype(np.float32)
    arr /= float(np.linalg.norm(arr))
    return tuple(float(x) for x in arr.tolist())


_PEOPLE_BOXES = {
    "leo": (0, BoundingBox(20, 80, 80, 20)),
    "maya": (1, BoundingBox(20, 180, 80, 120)),
    "nina": (2, BoundingBox(110, 80, 170, 20)),
}


class _ScanStub:
    """Filename-keyed recognizer whose `model_id`, box jitter and per-file dropouts are configurable."""

    def __init__(self, model: str = "stub-v1", *, shift: int = 0, drop: tuple[str, ...] = ()) -> None:
        self.model = model
        self.shift = shift
        self.drop = drop
        self.calls = 0
        self.seen: list[str] = []
        self._mu = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.model

    def detect_and_embed(self, image_path: Path) -> list[DetectedFace]:
        with self._mu:
            self.calls += 1
            self.seen.append(image_path.name)
        name = image_path.name.lower()
        out: list[DetectedFace] = []
        for key, (idx, box) in _PEOPLE_BOXES.items():
            if key in name and not any(d in name and key in d for d in self.drop):
                s = self.shift
                out.append(
                    DetectedFace(
                        box=BoundingBox(box.top + s, box.right + s, box.bottom + s, box.left + s),
                        embedding=_vec(idx, noise=0.05, seed=len(name) + idx),
                        confidence=0.9,
                    )
                )
        return out


def _make_jpg(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=70)


def _counts_consistent(db_path: Path) -> bool:
    conn = sqlite3.connect(db_path)
    stored = {
        int(r[0]): tuple(int(x) for x in r[1:])
        for r in conn.execute(
            "SELECT person_id, confirmed_count, auto_high_count, album_cohesion_count, candidate_count, "
            "unsure_count, rejected_count, categorized_face_count FROM person_assignment_counts"
        ).fetchall()
    }
    actual = {
        int(r[0]): tuple(int(x) for x in r[1:])
        for r in conn.execute(
            """
            SELECT person_id, SUM(status = 'confirmed'), SUM(status = 'auto_high'),
                SUM(status = 'auto_high' AND source IN ('album_cohesion', 'model_trusted')),
                SUM(status = 'candidate'), SUM(status = 'unsure'), SUM(status = 'rejected'),
                SUM(face_id IS NOT NULL AND status IN ('confirmed', 'auto_high'))
            FROM assignments GROUP BY person_id
            """
        ).fetchall()
    }
    conn.close()
    stored_nz = {k: v for k, v in stored.items() if any(v)}
    return stored_nz == actual


class _ArchiveCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.backup = self.root / "backup"
        self.album = self.backup / "Alex" / "Albums" / "2025" / "06" / "2025-06-01 Trip"
        self.files = {
            "001_leo_maya.jpg": 1,
            "002_leo.jpg": 2,
            "003_maya_nina.jpg": 3,
            "004_leo_nina.jpg": 4,
            "005_nina.jpg": 5,
        }
        for fn, seed in self.files.items():
            _make_jpg(self.album / fn, seed)
        self.db_path = self.backup / "Metadata" / "faces_index.sqlite"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def scan(self, rec: _ScanStub, **kw: Any) -> Any:
        return run_archive_face_scan(
            backup_dir=self.backup, recognizer=rec, output_dir=self.backup / "ByChild", workers=2, **kw
        )

    def face_of(self, db: FacesIndexDB, filename: str, idx: int) -> tuple[int, int]:
        row = db._conn.execute(
            "SELECT p.photo_id, f.face_id FROM faces f JOIN photos p ON p.photo_id = f.photo_id "
            "WHERE p.filename = ? AND f.face_idx = ?",
            (filename, idx),
        ).fetchone()
        assert row is not None
        return int(row[0]), int(row[1])


class LabelPreservingRescanTests(_ArchiveCase):
    def _label_state(self, db: FacesIndexDB) -> dict[str, Any]:
        c = db._conn
        return {
            "confirmed": sorted(
                tuple(r)
                for r in c.execute(
                    "SELECT photo_id, person_id, face_id, source FROM assignments WHERE status = 'confirmed'"
                ).fetchall()
            ),
            "negatives": sorted(
                tuple(r)
                for r in c.execute(
                    "SELECT photo_id, person_id, face_id FROM assignments WHERE status = 'rejected'"
                ).fetchall()
            ),
            "anchors": sorted(
                int(r[0]) for r in c.execute("SELECT face_id FROM assignments WHERE is_anchor = 1").fetchall()
            ),
            "pressure": sorted(
                tuple(r) for r in c.execute("SELECT person_id, face_id, weight FROM person_face_pressure").fetchall()
            ),
            "suppressed": sorted(
                int(r[0]) for r in c.execute("SELECT face_id FROM faces WHERE suppress_auto != 0").fetchall()
            ),
            "covers": sorted(
                tuple(r) for r in c.execute("SELECT person_id, cover_face_id FROM people").fetchall()
            ),
            "face_ids": sorted(int(r[0]) for r in c.execute("SELECT face_id FROM faces").fetchall()),
            "photo_ids": sorted(int(r[0]) for r in c.execute("SELECT photo_id FROM photos").fetchall()),
        }

    def _seed_labels(self) -> dict[str, Any]:
        with FacesIndexDB(self.db_path) as db:
            leo = db.ensure_person("Leo")
            maya = db.ensure_person("Maya")
            nina = db.ensure_person("Nina")
            for fn, idx, pid in (
                ("001_leo_maya.jpg", 0, leo),
                ("002_leo.jpg", 0, leo),
                ("004_leo_nina.jpg", 0, leo),
                ("001_leo_maya.jpg", 1, maya),
                ("003_maya_nina.jpg", 0, maya),
            ):
                ph, fid = self.face_of(db, fn, idx)
                db.tag_single_face(photo_id=ph, face_id=fid, person_id=pid)
            ph, fid = self.face_of(db, "005_nina.jpg", 0)
            db.untag_face(photo_id=ph, face_id=fid)  # suppress_auto = 1
            ph, fid = self.face_of(db, "003_maya_nina.jpg", 1)
            db.apply_person_cluster_action(person_id=nina, action="boost", faces=[(ph, fid)])
            ph, fid = self.face_of(db, "004_leo_nina.jpg", 1)
            db.apply_person_cluster_action(person_id=maya, action="exclude", faces=[(ph, fid)])
            recompute_active_matching(db)
            state = self._label_state(db)
        self.assertTrue(state["anchors"], "matcher should have chosen anchors")
        self.assertTrue(state["pressure"])
        self.assertTrue(state["suppressed"])
        return state

    def test_model_change_and_det_size_rescan_keep_labels_anchors_pressure_on_same_faces(self) -> None:
        rec1 = _ScanStub("stub-v1")
        self.scan(rec1)
        self.assertEqual(rec1.calls, 5)
        before = self._seed_labels()

        # det-size style change: new model_id, boxes moved by a few pixels.
        rec2 = _ScanStub("stub-v1-det1024", shift=4)
        summary = self.scan(rec2)
        self.assertEqual(rec2.calls, 5)
        self.assertEqual(summary.faces_relinked, 8)
        self.assertEqual(summary.labels_orphaned, 0)
        with FacesIndexDB(self.db_path) as db:
            after = self._label_state(db)
            box_row = db._conn.execute(
                "SELECT bbox_top FROM faces WHERE face_id = ?", (before["confirmed"][0][2],)
            ).fetchone()
        self.assertEqual(int(box_row[0]), 24)  # bbox updated in place
        for key in ("confirmed", "negatives", "anchors", "pressure", "suppressed", "covers", "face_ids", "photo_ids"):
            self.assertEqual(after[key], before[key], key)
        self.assertTrue(_counts_consistent(self.db_path))

        # `--force` with an embedder bump (boxes far away): cosine fallback still re-links.
        rec3 = _ScanStub("stub-v2", shift=60)
        summary3 = self.scan(rec3, force=True)
        self.assertEqual(rec3.calls, 5)
        with FacesIndexDB(self.db_path) as db:
            after3 = self._label_state(db)
        self.assertEqual(after3["confirmed"], before["confirmed"])
        self.assertEqual(after3["face_ids"], before["face_ids"])
        self.assertEqual(summary3.labels_orphaned, 0)

    def test_mtime_only_change_does_not_rescan_but_content_change_does(self) -> None:
        rec = _ScanStub()
        self.scan(rec)
        target = self.album / "002_leo.jpg"
        st = target.stat()
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        rec2 = _ScanStub()
        s2 = self.scan(rec2)
        self.assertEqual(rec2.calls, 0)
        self.assertEqual(s2.photos_verified, 1)
        with FacesIndexDB(self.db_path) as db:
            row = db._conn.execute("SELECT mtime_ns FROM photos WHERE filename = '002_leo.jpg'").fetchone()
        self.assertEqual(int(row[0]), target.stat().st_mtime_ns)

        data = bytearray(target.read_bytes())
        data[-10] ^= 0xFF  # same size, different bytes in the tail window
        target.write_bytes(bytes(data))
        rec3 = _ScanStub()
        self.scan(rec3)
        self.assertEqual(rec3.seen, ["002_leo.jpg"])

    def test_rename_and_move_keep_photo_id_and_labels(self) -> None:
        self.scan(_ScanStub())
        before = self._seed_labels()
        new_album = self.backup / "Alex" / "Albums" / "2025" / "07" / "2025-07-02 Renamed Trip"
        new_album.parent.mkdir(parents=True, exist_ok=True)
        self.album.rename(new_album)
        rec = _ScanStub()
        summary = self.scan(rec)
        self.assertEqual(rec.calls, 0)
        self.assertEqual(summary.photos_rekeyed, 5)
        self.assertEqual(summary.photos_removed, 0)
        with FacesIndexDB(self.db_path) as db:
            after = self._label_state(db)
            rows = db._conn.execute("SELECT rel_path, album_leaf, month FROM photos").fetchall()
        self.assertEqual(after["photo_ids"], before["photo_ids"])
        self.assertEqual(after["confirmed"], before["confirmed"])
        self.assertTrue(all("Renamed Trip" in str(r[0]) for r in rows))
        self.assertTrue(all(str(r[1]) == "2025-07-02 Renamed Trip" and str(r[2]) == "07" for r in rows))

    def test_deleted_photo_rows_are_removed_and_counts_stay_consistent(self) -> None:
        self.scan(_ScanStub())
        self._seed_labels()
        thumbs = self.backup / "Metadata" / "face_thumbs"
        with FacesIndexDB(self.db_path) as db:
            gone_pid, gone_fid = self.face_of(db, "001_leo_maya.jpg", 0)
        (self.album / "001_leo_maya.jpg").unlink()
        summary = self.scan(_ScanStub())
        self.assertEqual(summary.photos_removed, 1)
        self.assertEqual(summary.labels_removed_with_photos, 2)
        with FacesIndexDB(self.db_path) as db:
            c = db._conn
            self.assertIsNone(c.execute("SELECT 1 FROM photos WHERE photo_id = ?", (gone_pid,)).fetchone())
            self.assertIsNone(c.execute("SELECT 1 FROM faces WHERE photo_id = ?", (gone_pid,)).fetchone())
            self.assertIsNone(c.execute("SELECT 1 FROM assignments WHERE photo_id = ?", (gone_pid,)).fetchone())
            covers = [r[0] for r in c.execute("SELECT cover_face_id FROM people").fetchall()]
            self.assertNotIn(gone_fid, covers)
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(), [])
            overview = db.get_archive_overview()
        self.assertEqual(overview["photos_total"], 4)
        self.assertFalse(any(p.name == f"{gone_fid}.jpg" for p in thumbs.rglob("*.jpg")))
        self.assertTrue(_counts_consistent(self.db_path))

    def test_missing_photos_guard_keeps_rows_when_archive_looks_unmounted(self) -> None:
        self.scan(_ScanStub())
        for fn in self.files:
            (self.album / fn).unlink()
        summary = self.scan(_ScanStub())
        self.assertEqual(summary.photos_removed, 0)
        self.assertEqual(summary.photos_missing_kept, 5)
        with FacesIndexDB(self.db_path) as db:
            self.assertEqual(db.get_archive_overview()["photos_total"], 5)

    def test_face_that_disappears_keeps_label_as_photo_level_and_is_counted(self) -> None:
        self.scan(_ScanStub())
        self._seed_labels()
        with FacesIndexDB(self.db_path) as db:
            _, fid = self.face_of(db, "002_leo.jpg", 0)
        summary = self.scan(_ScanStub("stub-v2", drop=("002_leo",)))
        self.assertEqual(summary.labels_orphaned, 1)
        with FacesIndexDB(self.db_path) as db:
            row = db._conn.execute(
                "SELECT a.face_id, a.status FROM assignments a JOIN photos p ON p.photo_id = a.photo_id "
                "WHERE p.filename = '002_leo.jpg' AND a.status = 'confirmed'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertIsNone(row[0])
            self.assertIsNone(db._conn.execute("SELECT 1 FROM faces WHERE face_id = ?", (fid,)).fetchone())
            self.assertEqual(db.get_meta("orphaned_face_labels_total"), "1")
        self.assertTrue(_counts_consistent(self.db_path))

    def test_scan_bumps_data_version_and_last_scan_at(self) -> None:
        self.scan(_ScanStub())
        with FacesIndexDB(self.db_path) as db:
            v1 = db.get_data_version()
            self.assertIsNotNone(db.get_meta("last_scan_at"))
        self.scan(_ScanStub())
        with FacesIndexDB(self.db_path) as db:
            self.assertEqual(db.get_data_version(), v1 + 1)


class RelinkUnitTests(unittest.TestCase):
    def test_iou_then_cosine_greedy_one_to_one(self) -> None:
        e = [np.asarray(_vec(i), dtype=np.float32) for i in range(3)]
        old = [
            StoredFace(10, 0, 50, 50, 0, e[0]),
            StoredFace(11, 0, 150, 50, 100, e[1]),
            StoredFace(12, 100, 50, 150, 0, e[2]),
        ]
        new_boxes = [(2, 152, 52, 102), (300, 350, 350, 300), (1, 51, 51, 1)]
        new_embs = np.stack([e[1], e[2], e[0]])
        self.assertEqual(match_faces(old, new_boxes, new_embs), {0: 11, 2: 10, 1: 12})
        self.assertEqual(match_faces(old, new_boxes, None), {0: 11, 2: 10})
        self.assertAlmostEqual(bbox_iou((0, 10, 10, 0), (0, 10, 10, 0)), 1.0)
        self.assertEqual(bbox_iou((0, 10, 10, 0), (20, 30, 30, 20)), 0.0)

    def test_fingerprint_detects_head_and_tail_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.bin"
            p.write_bytes(bytes(300_000))
            h1 = file_fingerprint(p)
            data = bytearray(p.read_bytes())
            data[-1] = 1
            p.write_bytes(bytes(data))
            self.assertNotEqual(file_fingerprint(p), h1)


class SchemaMigrationTests(unittest.TestCase):
    def test_fresh_db_is_at_latest_version_and_enums_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "f.sqlite") as db:
            self.assertEqual(db.get_schema_version(), SCHEMA_VERSION)
            pid = db.ensure_person("Theo")
            with self.assertRaises(sqlite3.IntegrityError):
                db._conn.execute("UPDATE people SET role = 'villain' WHERE person_id = ?", (pid,))
            db._conn.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                db._conn.execute(
                    "INSERT INTO assignments (photo_id, person_id, face_id, status, source, updated_at) "
                    "VALUES (1, ?, NULL, 'maybe', 'manual_tag', 'x')",
                    (pid,),
                )
            db._conn.rollback()

    def test_validation_triggers_cover_every_enum_literal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "f.sqlite") as db:
            sql = " ".join(
                str(r[0])
                for r in db._conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE '%validate%'"
                ).fetchall()
            )
        for lit in (*ASSIGNMENT_STATUSES, *ASSIGNMENT_SOURCES, *PERSON_ROLES):
            self.assertIn(f"'{lit}'", sql, f"add a migration re-creating validation triggers for {lit!r}")

    def test_legacy_unversioned_db_migrates_and_repair_runs_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.sqlite"
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE photos (
                    photo_id INTEGER PRIMARY KEY AUTOINCREMENT, rel_path TEXT UNIQUE NOT NULL,
                    archive_child TEXT NOT NULL, year TEXT NOT NULL, month TEXT NOT NULL,
                    album_leaf TEXT NOT NULL, filename TEXT NOT NULL, kinderpedia_album_id INTEGER,
                    album_title TEXT NOT NULL DEFAULT '', album_date TEXT NOT NULL DEFAULT '',
                    classroom_groups TEXT NOT NULL DEFAULT '', kinderpedia_image_id INTEGER,
                    seq_in_album INTEGER, image_dateadd TEXT NOT NULL DEFAULT '',
                    best_live_url TEXT NOT NULL DEFAULT '', download_url TEXT NOT NULL DEFAULT '',
                    fullsize_url TEXT NOT NULL DEFAULT '', thumb_url TEXT NOT NULL DEFAULT '',
                    width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
                    file_size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, detector_id TEXT NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0, scanned_at TEXT NOT NULL);
                CREATE TABLE people (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_name TEXT UNIQUE NOT NULL COLLATE NOCASE, role TEXT NOT NULL DEFAULT 'classmate',
                    archive_hint TEXT NOT NULL DEFAULT '', cover_face_id INTEGER, created_at TEXT NOT NULL);
                CREATE TABLE faces (
                    face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo_id INTEGER NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
                    face_idx INTEGER NOT NULL, det_score REAL NOT NULL, bbox_top INTEGER NOT NULL,
                    bbox_right INTEGER NOT NULL, bbox_bottom INTEGER NOT NULL, bbox_left INTEGER NOT NULL,
                    embedding_f16 BLOB NOT NULL, cluster_id INTEGER, UNIQUE(photo_id, face_idx));
                INSERT INTO photos (rel_path, archive_child, year, month, album_leaf, filename,
                    file_size, mtime_ns, detector_id, scanned_at)
                    VALUES ('A/x.jpg', 'A', '2025', '01', 'alb', 'x.jpg', 1, 1, 'd', 't');
                INSERT INTO people (display_name, role, cover_face_id, created_at) VALUES ('Sofia', 'classmate', 999, 't');
                INSERT INTO faces (photo_id, face_idx, det_score, bbox_top, bbox_right, bbox_bottom, bbox_left,
                    embedding_f16) VALUES (1, 0, 0.9, 0, 10, 10, 0, zeroblob(128));
                """
            )
            conn.commit()
            conn.close()
            with FacesIndexDB(path) as db:
                self.assertEqual(db.get_schema_version(), SCHEMA_VERSION)
                cols = {str(r[1]) for r in db._conn.execute("PRAGMA table_info(faces)").fetchall()}
                self.assertTrue({"suppress_auto", "is_manual"} <= cols)
                p_cols = {str(r[1]) for r in db._conn.execute("PRAGMA table_info(photos)").fetchall()}
                self.assertIn("content_hash", p_cols)
                people = db.list_people()
                self.assertIsNone(people[0]["cover_face_id"])  # dangling 999 cleared by migration
                db.update_person(people[0]["person_id"], cover_face_id=1)
                db._conn.execute("DELETE FROM faces WHERE face_id = 1")
                db._conn.commit()
                cover = db._conn.execute("SELECT cover_face_id FROM people").fetchone()[0]
                self.assertIsNone(cover)  # trg_faces_ad_refs acts like ON DELETE SET NULL


class ScanLockTests(unittest.TestCase):
    def test_second_scan_is_refused_and_stale_lock_is_taken_over(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp)
            with ScanLock(meta, heartbeat_s=0.05):
                info = read_scan_lock(meta)
                self.assertIsNotNone(info)
                assert info is not None
                self.assertEqual(info["pid"], os.getpid())
                with self.assertRaises(ScanLockBusyError):
                    ScanLock(meta, timeout_s=0.2).acquire()
            self.assertIsNone(read_scan_lock(meta))
            (meta / "faces_scan.lock").write_text(
                json.dumps({"pid": 999999, "host": "elsewhere", "token": "t", "heartbeat_ts": time.time() - 999}),
                encoding="utf-8",
            )
            with ScanLock(meta):
                self.assertEqual(read_scan_lock(meta)["pid"], os.getpid())  # type: ignore[index]

    def test_run_archive_face_scan_refuses_while_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup = Path(tmp)
            with ScanLock(backup / "Metadata"), self.assertRaises(ScanLockBusyError):
                run_archive_face_scan(backup_dir=backup, recognizer=_ScanStub())


class ClaimPrimitiveTests(unittest.TestCase):
    """Every confirm path goes through `_claim_face_locked` with one policy."""

    def _setup(self, tmp: str) -> tuple[FacesIndexDB, int, int, int, int, int]:
        db = FacesIndexDB(Path(tmp) / "f.sqlite")
        prov = PhotoProvenance(
            rel_path="A/Albums/2025/01/x/a.jpg", archive_child="A", year="2025", month="01",
            album_leaf="x", filename="a.jpg",
        )
        faces = [
            DetectedFace(box=BoundingBox(0, 50, 50, 0), embedding=_vec(0)),
            DetectedFace(box=BoundingBox(0, 150, 50, 100), embedding=_vec(1)),
        ]
        ((photo_id, (f0, f1)),) = db.upsert_scanned_batch([(prov, 200, 200, 1, 1, "d", faces)])
        x = db.ensure_person("Alex")
        y = db.ensure_person("Maya")
        return db, photo_id, f0, f1, x, y

    def _status(self, db: FacesIndexDB, photo_id: int, person_id: int) -> tuple[Any, ...] | None:
        row = db._conn.execute(
            "SELECT status, source, face_id FROM assignments WHERE photo_id = ? AND person_id = ?",
            (photo_id, person_id),
        ).fetchone()
        return tuple(row) if row is not None else None

    def test_stale_batch_approve_never_overrides_cluster_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, ph, f0, _f1, x, y = self._setup(tmp)
            # The model proposed f0 for X (an old batch shows it) ...
            db.replace_model_assignments([(ph, x, f0, "candidate", 0.41, "model_match")], set())
            # ... meanwhile f0 is cluster-confirmed for Y.
            db.save_clusters(
                [{"archive_child": "A", "year": "2025", "medoid_face_id": f0, "sample_face_ids": [f0],
                  "face_ids": [f0], "photo_count": 1}]
            )
            cid = db.list_clusters()[0]["cluster_id"]
            self.assertEqual(db.assign_cluster_to_person(cid, y), 1)
            self.assertIsNone(self._status(db, ph, x))  # stale model guess dropped (bulk claim)
            res = db.apply_batch_decisions(person_id=x, approved=[(ph, f0)], disproved=[])
            self.assertEqual(res["approved"], 0)
            self.assertEqual(res["skipped_conflicts"], 1)
            self.assertEqual(self._status(db, ph, y), ("confirmed", "cluster_confirm", f0))
            self.assertIsNone(self._status(db, ph, x))
            db.close()

    def test_cluster_confirm_skips_face_confirmed_for_someone_else(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, ph, f0, _f1, x, y = self._setup(tmp)
            db.apply_batch_decisions(person_id=x, approved=[(ph, f0)], disproved=[])
            db.save_clusters(
                [{"archive_child": "A", "year": "2025", "medoid_face_id": f0, "sample_face_ids": [f0],
                  "face_ids": [f0], "photo_count": 1}]
            )
            cid = db.list_clusters()[0]["cluster_id"]
            self.assertEqual(db.assign_cluster_to_person(cid, y), 0)
            self.assertEqual(self._status(db, ph, x), ("confirmed", "batch_approve", f0))
            self.assertIsNone(self._status(db, ph, y))
            db.close()

    def test_explicit_tag_demotes_previous_confirmed_holder_to_face_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, ph, f0, f1, x, y = self._setup(tmp)
            db.apply_batch_decisions(person_id=y, approved=[(ph, f0)], disproved=[])
            db.update_person(y, cover_face_id=f0)
            db.untag_face(photo_id=ph, face_id=f1)
            db.tag_single_face(photo_id=ph, face_id=f0, person_id=x)
            self.assertEqual(self._status(db, ph, x), ("confirmed", "manual_tag", f0))
            self.assertEqual(self._status(db, ph, y), ("rejected", "reassigned", f0))
            people = {p["person_id"]: p for p in db.list_people()}
            self.assertNotEqual(people[y]["cover_face_id"], f0)
            self.assertEqual(people[x]["cover_face_id"], f0)
            # Batch reassign is explicit too, and suppress_auto is cleared on the claimed face.
            res = db.apply_batch_decisions(person_id=x, approved=[], disproved=[], reassigned=[(ph, f1, y)])
            self.assertEqual(res["reassigned"], 1)
            self.assertEqual(self._status(db, ph, y), ("confirmed", "manual_tag", f1))
            sup = db._conn.execute("SELECT suppress_auto FROM faces WHERE face_id = ?", (f1,)).fetchone()[0]
            self.assertEqual(sup, 0)
            self.assertEqual(self._status(db, ph, x), ("confirmed", "manual_tag", f0))  # untouched
            db.close()


class MaterializeTests(_ArchiveCase):
    def test_fs_pass_runs_without_db_lock_and_prunes_dangling_and_orphan_folders(self) -> None:
        self.scan(_ScanStub())
        out = self.backup / "ByChild"
        with FacesIndexDB(self.db_path) as db:
            leo = db.ensure_person("Leo")
            for fn in ("001_leo_maya.jpg", "002_leo.jpg"):
                ph, fid = self.face_of(db, fn, 0)
                db.tag_single_face(photo_id=ph, face_id=fid, person_id=leo)
            from kinderpedia_backup.faces.storage import link_or_copy_photo as real_link

            lock_free: list[bool] = []

            def probing_link(*a: Any, **kw: Any) -> bool:
                got: list[bool] = []

                def probe() -> None:
                    ok = db._lock.acquire(timeout=1.0)
                    if ok:
                        db._lock.release()
                    got.append(ok)

                t = threading.Thread(target=probe)
                t.start()
                t.join()
                lock_free.append(bool(got and got[0]))
                return real_link(*a, **kw)

            with mock.patch(f"{MATERIALIZE_MODULE}.link_or_copy_photo", probing_link):
                st = db.materialize_flat_folders(backup_dir=self.backup, output_dir=out, link_mode="symlink")
            self.assertEqual(st.links_created, 2)
            self.assertTrue(lock_free and all(lock_free))

            # A dangling symlink inside a live folder is pruned; a renamed person leaves an orphan folder.
            (out / "Leo" / "stale.jpg").symlink_to(self.backup / "nowhere.jpg")
            keep = out / "Leo Old" / "own-copy.jpg"
            keep.parent.mkdir()
            keep.write_bytes(b"user file")  # a plain copy we did not create: must survive
            (out / "Leo Old" / "link.jpg").symlink_to(self.album / "002_leo.jpg")
            db.update_person(leo, display_name="Leo Renamed")
            st2 = db.materialize_flat_folders(backup_dir=self.backup, output_dir=out, link_mode="symlink")
            self.assertEqual(st2.links_created, 2)
            self.assertFalse((out / "Leo").exists())  # orphan folder emptied (symlinks) and removed
            self.assertTrue(keep.is_file())
            self.assertFalse((out / "Leo Old" / "link.jpg").is_symlink())
            self.assertEqual(sorted(p.name for p in (out / "Leo Renamed").iterdir()), [
                "2025-06-01 Trip_001_leo_maya.jpg", "2025-06-01 Trip_002_leo.jpg"
            ])


class PersonClustersCapTests(unittest.TestCase):
    def test_capped_pairwise_still_places_every_face_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "f.sqlite") as db:
            items = []
            for i in range(60):
                prov = PhotoProvenance(
                    rel_path=f"A/Albums/2025/01/x/{i:03d}.jpg", archive_child="A", year="2025",
                    month="01", album_leaf="x", filename=f"{i:03d}.jpg",
                )
                emb = _vec(i % 3, noise=0.02, seed=i)
                items.append((prov, 100, 100, 1, 1, "d", [DetectedFace(box=BoundingBox(0, 50, 50, 0), embedding=emb)]))
            saved = db.upsert_scanned_batch(items)
            pid = db.ensure_person("Sofia")
            for photo_id, (fid,) in saved:
                db.tag_single_face(photo_id=photo_id, face_id=fid, person_id=pid)
            full = db.get_person_face_clusters(pid)
            capped = db.get_person_face_clusters(pid, max_pairwise_faces=16)
            for res in (full, capped):
                fids = [f["face_id"] for c in res["clusters"] for f in c["faces"]]
                self.assertEqual(len(fids), 60)
                self.assertEqual(len(set(fids)), 60)
            self.assertEqual(len([c for c in capped["clusters"] if not c["is_loose"]]), 3)


class UpsertBatchApiTests(unittest.TestCase):
    def test_upsert_without_hashes_still_works_and_returns_face_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "f.sqlite") as db:
            prov = PhotoProvenance(rel_path="A/a.jpg", archive_child="A", year="2025", month="01", album_leaf="x", filename="a.jpg")
            face = DetectedFace(box=BoundingBox(0, 10, 10, 0), embedding=_vec(0))
            out = db.upsert_scanned_batch([(prov, 10, 10, 1, 1, "d", [face])])
            out2 = db.upsert_scanned_batch([(prov, 10, 10, 1, 2, "d2", [face])])
            self.assertEqual(out[0][1], out2[0][1])


if __name__ == "__main__":
    unittest.main()
