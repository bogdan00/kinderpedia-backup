"""Label-consistency guard (suspicious confirmed labels), its review API and the hidden-label replay tool.

All embeddings are synthetic and ArcFace-like: every identity has a unit centre that shares a common
"child face" component with all other centres (centre cosine ~0.2), and a face is
`alpha * centre + sqrt(1 - alpha^2) * noise`, so same-person cosines land around 0.5-0.8 and
different-person cosines around 0.0-0.3 -- not orthogonal toy bases.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import requests
from numpy.typing import NDArray

from kinderpedia_backup.faces import (
    BoundingBox,
    DetectedFace,
    FacesIndexDB,
    evaluate,
    recompute_active_matching,
)
from kinderpedia_backup.faces.db.schema import SCHEMA_VERSION
from kinderpedia_backup.faces.label_guard import (
    LABEL_OUTLIER_MIN_FACES,
    LABEL_OUTLIER_MIN_SIM,
    label_outlier_mask,
)
from kinderpedia_backup.faces.models import PhotoProvenance
from kinderpedia_backup.faces.server import _AppContext, _FacesHTTPServer

TOKEN = "tok-outlier-tests"
HDR = {"X-KP-Token": TOKEN}


class _Archive:
    """Synthetic ArcFace-like faces, one face per photo, one album per photo."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.common = self._unit(self.rng.standard_normal(512))
        self.centres: dict[str, NDArray[np.float64]] = {}
        self.items: list[tuple[str, NDArray[np.float32], str]] = []  # (identity, embedding, year)

    def _unit(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(v / np.linalg.norm(v))

    def identity(self, name: str) -> None:
        self.centres[name] = self._unit(0.45 * self.common + 0.89 * self._unit(self.rng.standard_normal(512)))

    def vec(self, name: str, *, alpha: tuple[float, float] = (0.74, 0.86)) -> NDArray[np.float32]:
        c = self.centres[name]
        a = float(self.rng.uniform(*alpha))
        noise = self._unit(self.rng.standard_normal(512))
        noise = self._unit(noise - (noise @ c) * c)
        return np.asarray(a * c + np.sqrt(1.0 - a * a) * noise, dtype=np.float32)

    def add(self, name: str, count: int, *, year: str = "2025", alpha: tuple[float, float] = (0.74, 0.86)) -> None:
        base = name.split("-")[0]
        for _ in range(count):
            self.items.append((name, self.vec(base, alpha=alpha), year))

    def add_vec(self, name: str, vec: NDArray[np.float64], *, year: str = "2025") -> None:
        self.items.append((name, self._unit(vec).astype(np.float32), year))

    def save(self, db: FacesIndexDB) -> list[tuple[str, int, int]]:
        """Persist all items; returns `[(identity, photo_id, face_id)]` in insertion order."""
        batch: list[tuple[PhotoProvenance, int, int, int, int, str, list[DetectedFace]]] = []
        for i, (_name, vec, year) in enumerate(self.items):
            month = 1 + (i % 12)
            day = 1 + (i % 27)
            leaf = f"{year}-{month:02d}-{day:02d} Day {i}"
            batch.append(
                (
                    PhotoProvenance(
                        f"Alex/Albums/{year}/{month:02d}/{leaf}/{i:03d}.jpg",
                        "Alex",
                        year,
                        f"{month:02d}",
                        leaf,
                        f"{i:03d}.jpg",
                        album_date=f"{year}-{month:02d}-{day:02d}",
                    ),
                    400,
                    300,
                    1000 + i,
                    2000 + i,
                    "scrfd",
                    [
                        DetectedFace(
                            box=BoundingBox(top=10, right=110, bottom=110, left=10),
                            embedding=tuple(float(x) for x in vec.tolist()),
                            confidence=0.95,
                        )
                    ],
                )
            )
        saved = db.upsert_scanned_batch(batch)
        return [(self.items[i][0], p_id, f_ids[0]) for i, (p_id, f_ids) in enumerate(saved)]


def _row_of(diag: dict[str, Any], face_id: int) -> int:
    fids: NDArray[np.int64] = diag["face_ids"]
    r = int(np.searchsorted(fids, face_id))
    assert int(fids[r]) == face_id
    return r


def _is_anchor(db: FacesIndexDB, face_id: int, person_id: int) -> bool:
    row = db._conn.execute(
        "SELECT is_anchor FROM assignments WHERE face_id = ? AND person_id = ?", (face_id, person_id)
    ).fetchone()
    return bool(row is not None and row["is_anchor"])


def _mislabel_scenario(db: FacesIndexDB, seed: int = 3) -> dict[str, Any]:
    """Maya and Leo with 20 confirmed faces each in 2025; one of Leo's faces is confirmed as Maya
    (a bulk-confirm mistake); six more Leo faces (close to the mislabelled one) are still open."""
    arc = _Archive(seed)
    arc.identity("Maya")
    arc.identity("Leo")
    arc.add("Maya", 20)
    arc.add("Leo", 20)
    mis_vec = arc.vec("Leo")
    arc.add_vec("mislabel", mis_vec.astype(np.float64))
    for _ in range(6):  # open Leo faces near the mislabelled face (same day / outfit)
        noise = arc._unit(arc.rng.standard_normal(512))
        arc.add_vec("Leo-open", 0.92 * mis_vec + 0.39 * noise)
    photos = arc.save(db)
    maya = db.ensure_person("Maya")
    leo = db.ensure_person("Leo")
    for who, p_id, f_id in photos:
        if who == "Maya" or who == "mislabel":
            db.tag_single_face(photo_id=p_id, face_id=f_id, person_id=maya, source="cluster_confirm")
        elif who == "Leo":
            db.tag_single_face(photo_id=p_id, face_id=f_id, person_id=leo, source="cluster_confirm")
    mis = next((p, f) for who, p, f in photos if who == "mislabel")
    return {
        "arc": arc,
        "photos": photos,
        "maya": maya,
        "leo": leo,
        "mis_photo": mis[0],
        "mis_face": mis[1],
        "leo_open": [f for who, _p, f in photos if who == "Leo-open"],
        "maya_faces": [f for who, _p, f in photos if who == "Maya"],
    }


class SyntheticEmbeddingSanityTests(unittest.TestCase):
    def test_generator_is_arcface_like(self) -> None:
        arc = _Archive(1)
        for kid in ("Maya", "Leo", "Sofia"):
            arc.identity(kid)
            arc.add(kid, 12)
        embs = np.stack([v for _, v, _ in arc.items])
        names = [n for n, _, _ in arc.items]
        sims = embs @ embs.T
        same = [sims[i, j] for i in range(len(names)) for j in range(i) if names[i] == names[j]]
        diff = [sims[i, j] for i in range(len(names)) for j in range(i) if names[i] != names[j]]
        self.assertTrue(0.5 <= float(np.mean(same)) <= 0.8, float(np.mean(same)))
        self.assertTrue(0.0 <= float(np.mean(diff)) <= 0.3, float(np.mean(diff)))
        self.assertGreater(float(np.min(same)), float(np.max(diff)) - 0.25)


class LabelGuardTests(unittest.TestCase):
    def test_mask_needs_min_faces_and_flags_only_isolated_faces(self) -> None:
        arc = _Archive(2)
        arc.identity("Maya")
        arc.identity("Leo")
        maya = np.stack([arc.vec("Maya") for _ in range(LABEL_OUTLIER_MIN_FACES)])
        intruder = arc.vec("Leo")[np.newaxis, :]
        mask, best, nearest = label_outlier_mask(np.concatenate([maya, intruder]))
        self.assertEqual(mask.tolist(), [False] * LABEL_OUTLIER_MIN_FACES + [True])
        self.assertLess(float(best[-1]), LABEL_OUTLIER_MIN_SIM)
        self.assertLess(int(nearest[-1]), LABEL_OUTLIER_MIN_FACES)
        # Too few faces in the person-year to judge consistency: nothing is flagged.
        few = np.concatenate([maya[: max(1, LABEL_OUTLIER_MIN_FACES - 2)], intruder])
        self.assertFalse(bool(np.any(label_outlier_mask(few)[0])))

    def test_mislabelled_face_never_becomes_anchor_nor_knn_voter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            sc = _mislabel_scenario(db)
            diag: dict[str, Any] = {}
            recompute_active_matching(db, diagnostics=diag)
            self.assertIn((sc["mis_face"], sc["maya"]), diag["flagged_labels"])
            self.assertEqual(len(diag["flagged_labels"]), 1)
            # Still Maya's confirmed tag (the user decides), but never an anchor ...
            row = db._conn.execute(
                "SELECT status FROM assignments WHERE face_id = ? AND person_id = ?", (sc["mis_face"], sc["maya"])
            ).fetchone()
            self.assertEqual(str(row["status"]), "confirmed")
            self.assertFalse(_is_anchor(db, sc["mis_face"], sc["maya"]))
            self.assertTrue(any(_is_anchor(db, f, sc["maya"]) for f in sc["maya_faces"]))
            # ... nor a kNN voter:
            bank: set[int] = db._knn_neighbor_cache["bank"]  # type: ignore[index]
            self.assertNotIn(_row_of(diag, sc["mis_face"]), bank)
            self.assertIn(_row_of(diag, sc["maya_faces"][0]), bank)
            # So Leo's open faces right next to the mislabelled one are not pulled towards Maya:
            maya_rows = {
                int(r[0])
                for r in db._conn.execute(
                    "SELECT face_id FROM assignments WHERE person_id = ? AND status IN ('auto_high', 'candidate')",
                    (sc["maya"],),
                )
            }
            self.assertFalse(maya_rows & set(sc["leo_open"]))
            for f in sc["leo_open"]:
                self.assertEqual(int(diag["knn_top"][_row_of(diag, f)]), sc["leo"])

            sus = db.get_suspicious_labels(sc["maya"])
            self.assertEqual([it["face_id"] for it in sus["items"]], [sc["mis_face"]])
            item = sus["items"][0]
            self.assertEqual(item["photo_id"], sc["mis_photo"])
            self.assertEqual(item["year"], "2025")
            self.assertLess(item["max_same_person_sim"], LABEL_OUTLIER_MIN_SIM)
            self.assertIn(item["nearest_face_id"], sc["maya_faces"])
            self.assertEqual(item["year_confirmed_count"], 21)
            self.assertEqual(db.get_suspicious_labels(sc["leo"])["items"], [])
            people = {p["person_id"]: p for p in db.get_archive_overview()["people"]}
            self.assertEqual(people[sc["maya"]]["suspicious_count"], 1)
            self.assertEqual(people[sc["leo"]]["suspicious_count"], 0)

    def test_atypical_pose_with_a_close_same_person_neighbour_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            arc = _Archive(4)
            arc.identity("Maya")
            arc.add("Maya", 20)
            pose = arc._unit(arc.rng.standard_normal(512))
            c_w = max(0.0, LABEL_OUTLIER_MIN_SIM - 0.12) / 0.8  # frontal similarity ~ guard threshold - 0.12
            for _ in range(2):  # two profile shots: unlike her frontal faces, but like each other
                noise = arc._unit(arc.rng.standard_normal(512))
                arc.add_vec("Maya-profile", c_w * arc.centres["Maya"] + 0.85 * pose + 0.40 * noise)
            arc.add("Maya-open", 4)  # unlabelled faces, so the kNN vote runs
            photos = arc.save(db)
            maya = db.ensure_person("Maya")
            for who, p_id, f_id in photos:
                if who != "Maya-open":
                    db.tag_single_face(photo_id=p_id, face_id=f_id, person_id=maya, source="cluster_confirm")
            profile = [f for who, _p, f in photos if who == "Maya-profile"]
            embs = np.stack([v for _, v, _ in arc.items[:22]])
            frontal_best = float(np.max(embs[-2:] @ embs[:-2].T))
            self.assertLess(frontal_best, LABEL_OUTLIER_MIN_SIM)  # alone, each would look like a mislabel
            self.assertGreater(float(embs[-1] @ embs[-2]), LABEL_OUTLIER_MIN_SIM + 0.2)
            diag: dict[str, Any] = {}
            recompute_active_matching(db, diagnostics=diag)
            self.assertEqual(diag["flagged_labels"], set())
            self.assertEqual(db.get_suspicious_labels(maya)["items"], [])
            bank: set[int] = db._knn_neighbor_cache["bank"]  # type: ignore[index]
            self.assertTrue(all(_row_of(diag, f) in bank for f in profile))

    def test_keep_unflags_label_and_restores_it_as_voter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            sc = _mislabel_scenario(db)
            recompute_active_matching(db)
            with self.assertRaises(KeyError):
                db.keep_label(person_id=sc["leo"], face_id=sc["mis_face"])  # not Leo's label
            self.assertEqual(db.keep_label(person_id=sc["maya"], face_id=sc["mis_face"])["kept"], True)
            self.assertEqual(db.get_suspicious_labels(sc["maya"])["items"], [])
            people = {p["person_id"]: p for p in db.get_archive_overview()["people"]}
            self.assertEqual(people[sc["maya"]]["suspicious_count"], 0)
            diag: dict[str, Any] = {}
            recompute_active_matching(db, diagnostics=diag)
            self.assertEqual(diag["flagged_labels"], set())
            bank: set[int] = db._knn_neighbor_cache["bank"]  # type: ignore[index]
            self.assertIn(_row_of(diag, sc["mis_face"]), bank)
            # Keeping twice is idempotent; deleting the person drops the keep marks with it.
            db.keep_label(person_id=sc["maya"], face_id=sc["mis_face"])
            db.delete_person(sc["maya"])
            n_keeps = db._conn.execute("SELECT COUNT(*) FROM label_keeps").fetchone()[0]
            self.assertEqual(int(n_keeps), 0)

    def test_untag_removes_suspicious_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            sc = _mislabel_scenario(db)
            recompute_active_matching(db)
            db.untag_face(photo_id=sc["mis_photo"], face_id=sc["mis_face"])
            self.assertEqual(db.get_suspicious_labels(sc["maya"])["items"], [])
            diag: dict[str, Any] = {}
            recompute_active_matching(db, diagnostics=diag)
            self.assertEqual(diag["flagged_labels"], set())

    def test_label_keeps_schema(self) -> None:
        self.assertEqual(SCHEMA_VERSION, 6)
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            self.assertEqual(db.get_schema_version(), 6)
            cols = {str(r[1]) for r in db._conn.execute("PRAGMA table_info(label_keeps)")}
            self.assertEqual(cols, {"face_id", "person_id", "kept_at"})


class KnnCacheExactnessTests(unittest.TestCase):
    def test_warm_rematch_after_removing_voters_equals_cold_rematch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            arc = _Archive(6)
            for kid in ("Maya", "Leo", "Sofia"):
                arc.identity(kid)
                arc.add(kid, 40)
            photos = arc.save(db)
            pid_of = {kid: db.ensure_person(kid) for kid in ("Maya", "Leo", "Sofia")}
            per: dict[str, int] = dict.fromkeys(pid_of, 0)
            tagged: list[tuple[int, int]] = []
            for who, p_id, f_id in photos:
                if per[who] < 20:
                    db.tag_single_face(photo_id=p_id, face_id=f_id, person_id=pid_of[who])
                    per[who] += 1
                    tagged.append((p_id, f_id))
            recompute_active_matching(db)
            for p_id, f_id in tagged[::4]:  # remove a quarter of the voters
                db.untag_face(photo_id=p_id, face_id=f_id)
            warm: dict[str, Any] = {}
            recompute_active_matching(db, diagnostics=warm)
            db._knn_neighbor_cache = None
            cold: dict[str, Any] = {}
            recompute_active_matching(db, diagnostics=cold)
            self.assertEqual(sorted(warm["assignments"]), sorted(cold["assignments"]))
            np.testing.assert_array_equal(warm["knn_top"], cold["knn_top"])
            np.testing.assert_allclose(warm["knn_frac"], cold["knn_frac"], rtol=1e-5, atol=1e-6)


class SuspiciousEndpointTests(unittest.TestCase):
    def test_list_keep_and_overview_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup_dir = Path(tmp) / "bk"
            db = FacesIndexDB(backup_dir / "Metadata" / "faces_index.sqlite")
            sc = _mislabel_scenario(db)
            recompute_active_matching(db)
            ctx = _AppContext(
                backup_dir=backup_dir,
                output_dir=backup_dir / "ByChild",
                db=db,
                csrf_token=TOKEN,
                rematch_debounce_sec=0.05,
                snapshot_debounce_sec=0.2,
                external_poll_sec=0.0,
                warm_cache_on_start=False,
            )
            httpd = _FacesHTTPServer(("127.0.0.1", 0), ctx)
            th = threading.Thread(target=httpd.serve_forever, daemon=True)
            th.start()
            base = f"http://127.0.0.1:{int(httpd.server_address[1])}"
            try:
                maya = sc["maya"]
                r = requests.get(f"{base}/api/people/{maya}/suspicious", timeout=10)
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertEqual([it["face_id"] for it in body["items"]], [sc["mis_face"]])
                self.assertEqual(
                    set(body["items"][0]),
                    {
                        "face_id", "photo_id", "year", "source", "max_same_person_sim", "nearest_face_id",
                        "year_confirmed_count", "archive_child", "album_leaf", "album_title", "album_date", "filename",
                    },
                )
                ov = requests.get(f"{base}/api/overview", timeout=10).json()
                self.assertEqual(next(p for p in ov["people"] if p["person_id"] == maya)["suspicious_count"], 1)
                self.assertEqual(requests.get(f"{base}/api/people/999/suspicious", timeout=10).status_code, 404)
                bad = requests.post(
                    f"{base}/api/people/{sc['leo']}/suspicious/{sc['mis_face']}/keep", headers=HDR, json={}, timeout=10
                )
                self.assertEqual(bad.status_code, 404)
                kept = requests.post(
                    f"{base}/api/people/{maya}/suspicious/{sc['mis_face']}/keep", headers=HDR, json={}, timeout=10
                )
                self.assertEqual(kept.status_code, 200)
                kb = kept.json()
                self.assertEqual((kb["person_id"], kb["face_id"], kb["kept"], kb["rematch_scheduled"]), (maya, sc["mis_face"], True, True))
                self.assertEqual(next(p for p in kb["overview"]["people"] if p["person_id"] == maya)["suspicious_count"], 0)
                self.assertEqual(requests.get(f"{base}/api/people/{maya}/suspicious", timeout=10).json()["items"], [])
            finally:
                httpd.shutdown()
                httpd.server_close()
                db.close()


def _replay_archive(path: Path) -> None:
    """Small archive with every kind of human decision: cluster confirms, batch approvals and
    rejections, a reassign negative, an untag and a background (`ignore`) profile."""
    arc = _Archive(11)
    for kid in ("Maya", "Leo", "Sofia"):
        arc.identity(kid)
        for year in ("2024", "2025"):
            arc.add(kid, 14, year=year)
    for i in range(16):
        arc.identity(f"stranger{i}")
        arc.add(f"stranger{i}", 1, year="2025")
    with FacesIndexDB(path) as db:
        photos = arc.save(db)
        pid_of = {kid: db.ensure_person(kid) for kid in ("Maya", "Leo", "Sofia")}
        bg = db.ensure_person("Background", role="ignore")
        seen: dict[tuple[str, int], int] = {}
        for idx, (who, p_id, f_id) in enumerate(photos):
            if who.startswith("stranger"):
                if idx % 2 == 0:
                    db.tag_single_face(photo_id=p_id, face_id=f_id, person_id=bg, source="cluster_confirm")
                else:
                    db.apply_batch_decisions(person_id=pid_of["Maya"], approved=[], disproved=[(p_id, f_id)])
                continue
            k = seen[(who, idx // 200)] = seen.get((who, idx // 200), 0) + 1
            if k % 3 == 0:
                db.apply_batch_decisions(person_id=pid_of[who], approved=[(p_id, f_id)], disproved=[])
            elif k % 3 == 1:
                db.tag_single_face(photo_id=p_id, face_id=f_id, person_id=pid_of[who], source="cluster_confirm")
        maya_open = [(p, f) for who, p, f in photos if who == "Leo"][-1]
        db.apply_batch_decisions(person_id=pid_of["Maya"], approved=[], disproved=[], reassigned=[(maya_open[0], maya_open[1], pid_of["Leo"])])
        db.untag_face(photo_id=photos[5][1], face_id=photos[5][2])


def _strip_timing(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_timing(v) for k, v in obj.items() if k not in ("elapsed_seconds", "seconds_mean")}
    if isinstance(obj, list):
        return [_strip_timing(v) for v in obj]
    return obj


class EvaluateToolTests(unittest.TestCase):
    def test_replay_runs_end_to_end_deterministically_on_a_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "archive_copy.sqlite"
            _replay_archive(src)
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            outs = []
            for run in range(2):
                js = root / f"out{run}.json"
                md = root / f"out{run}.md"
                rc = evaluate.main(
                    ["--db", str(src), "--folds", "3", "--seed", "7", "--grid", "min_sim=0.3,0.45;min_faces=4",
                     "--json", str(js), "--markdown", str(md), "--workdir", str(root)]
                )
                self.assertEqual(rc, 0)
                outs.append((json.loads(js.read_text()), md.read_text()))
            self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), digest)  # source never written
            self.assertEqual(_strip_timing(outs[0][0]), _strip_timing(outs[1][0]))
            summary = outs[0][0]
            self.assertEqual(len(summary["replicates"]), 3)
            self.assertEqual(sum(r["hidden_faces"] for r in summary["replicates"]), summary["decision_faces"])
            self.assertEqual(summary["cells"][0]["name"], evaluate.BASELINE_NAME)
            self.assertEqual(len(summary["cells"]), 3)
            base = summary["cells"][0]
            self.assertGreater(base["correct"], 0)
            self.assertGreater(base["background_hidden"], 0)
            self.assertGreater(base["rejections_hidden"], 0)
            self.assertIn("| guard-off |", outs[0][1])
            self.assertIn("Threshold check", outs[0][1])
            self.assertEqual(sorted(p.name for p in root.iterdir() if p.name.startswith("faces-eval-")), [])

    def test_refuses_live_archive_layout_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = Path(tmp) / "Metadata" / "faces_index.sqlite"
            live.parent.mkdir()
            _replay_archive(live)
            self.assertTrue(evaluate.is_live_archive_path(live))
            self.assertEqual(evaluate.main(["--db", str(live), "--folds", "2", "--only-folds", "0"]), 2)
            copy = Path(tmp) / "copy.sqlite"
            shutil.copyfile(live, copy)
            self.assertFalse(evaluate.is_live_archive_path(copy))
            md = Path(tmp) / "r.md"
            self.assertEqual(
                evaluate.main(
                    ["--db", str(live), "--folds", "2", "--only-folds", "0", "--i-know-this-is-a-copy",
                     "--markdown", str(md), "--workdir", tmp]
                ),
                0,
            )

    def test_selection_rule_prefers_simpler_cell_on_ties(self) -> None:
        cells = evaluate.parse_grid("min_sim=0.3,0.25;min_faces=8,4")
        self.assertEqual(len(cells), 4)
        keys = sorted(evaluate.simplicity_key(c) for c in cells)
        self.assertEqual(keys[0][:2], (4, 0.25))
        lo, hi = evaluate.wilson_interval(95, 5)
        self.assertTrue(lo < 0.95 < hi)
        with self.assertRaises(ValueError):
            evaluate.parse_grid("bogus=1")


if __name__ == "__main__":
    unittest.main()
