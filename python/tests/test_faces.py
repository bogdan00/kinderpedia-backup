"""Unit & integration tests for facial recognition, active-learning matching, provenance, and Web API."""

from __future__ import annotations

import json
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from kinderpedia_backup.faces import (
    BoundingBox,
    DetectedFace,
    FacesIndexDB,
    ProvenanceResolver,
    cosine_similarity,
    recompute_active_matching,
    resolve_best_live_url,
    run_archive_face_scan,
)
from kinderpedia_backup.faces.alignment import (
    ARCFACE_DST_112,
    greedy_nms,
    letterbox_scrfd_input,
    umeyama_similarity_2d,
)
from kinderpedia_backup.faces.matcher import rebuild_yearly_clusters
from kinderpedia_backup.faces.models import PhotoProvenance
from kinderpedia_backup.faces.server import _AppContext, _FacesHTTPServer


def _basis_vec(index: int, *, dim: int = 512, jitter: float = 0.0, jitter_idx: int = 500) -> tuple[float, ...]:
    arr = np.zeros((dim,), dtype=np.float32)
    arr[index] = 1.0
    if jitter != 0.0:
        arr[jitter_idx] = jitter
    arr = arr / float(np.linalg.norm(arr))
    return tuple(float(x) for x in arr.tolist())


def _await_job_result(base_url: str, resp: requests.Response, timeout: float = 20.0) -> dict[str, Any]:
    """`/api/materialize` & `/api/recluster` answer 202 + job id; poll `/api/jobs/<id>` for the result."""
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = requests.get(f"{base_url}/api/jobs/{job_id}", timeout=5).json()
        if job["state"] != "running":
            assert job["state"] == "done", job
            return dict(job["result"])
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


class _StubRecognizer:
    """Deterministic fake recognizer keyed by filename substring for fast offline tests."""

    def __init__(self, *, dim: int = 3) -> None:
        self.calls: int = 0
        self.dim = dim

    @property
    def model_id(self) -> str:
        return f"stub-d{self.dim}-v1"

    def detect_and_embed(self, image_path: Path) -> list[DetectedFace]:
        self.calls += 1
        name = image_path.name.lower()
        faces: list[DetectedFace] = []
        if "alex" in name or (name == "avatar.png" and "alex" in str(image_path.parent).lower()):
            faces.append(
                DetectedFace(
                    box=BoundingBox(10, 90, 90, 10),
                    embedding=_basis_vec(0, dim=self.dim),
                    confidence=0.88,
                )
            )
        if "maya" in name or (name == "avatar.png" and "maya" in str(image_path.parent).lower()):
            faces.append(
                DetectedFace(
                    box=BoundingBox(10, 190, 90, 110),
                    embedding=_basis_vec(1, dim=self.dim),
                    confidence=0.87,
                )
            )
        if "leo" in name:
            faces.append(
                DetectedFace(
                    box=BoundingBox(100, 90, 180, 10),
                    embedding=_basis_vec(2, dim=self.dim),
                    confidence=0.86,
                )
            )
        if "other" in name:
            faces.append(
                DetectedFace(
                    box=BoundingBox(100, 190, 180, 110),
                    embedding=_basis_vec(min(self.dim - 1, 3), dim=self.dim),
                    confidence=0.85,
                )
            )
        return faces


class FacesPipelineTests(unittest.TestCase):
    def test_cosine_similarity_math(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_alignment_umeyama_and_nms(self) -> None:
        # Rotate + scale canonical landmarks and recover exact inverse mapping:
        theta = 0.25
        scale = 1.6
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        src_pts = ((ARCFACE_DST_112 @ rot.T) * scale + np.array([40.0, 25.0], dtype=np.float32)).astype(np.float32)
        mat = umeyama_similarity_2d(src_pts, ARCFACE_DST_112)
        mapped = (src_pts @ mat[:, :2].T) + mat[:, 2]
        np.testing.assert_allclose(mapped, ARCFACE_DST_112, atol=1e-3)

        boxes = np.array([[0, 0, 100, 100], [5, 5, 102, 102], [200, 200, 280, 280]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(greedy_nms(boxes, scores, iou_thresh=0.4), [0, 2])

    def test_provenance_resolves_legacy_2022_fullsize_vs_2026_download_url(self) -> None:
        # Legacy 2022 migrated photo where download_url 404'd and local JPEG is 900x1200 matching fullsize:
        best_2022 = resolve_best_live_url(
            download_url="https://images.kinderpedia.co/files/photos/k1/100/dead.jpg",
            fullsize_url="https://images.kinderpedia.co/900x1200/files/photos/k1/100/live.jpg",
            thumb_url="https://images.kinderpedia.co/300x300/files/photos/k1/100/live.jpg",
            width=900,
            height=1200,
        )
        self.assertEqual(best_2022, "https://images.kinderpedia.co/900x1200/files/photos/k1/100/live.jpg")

        # Modern 2026 full-resolution photo (1440x1920 != 900x1200) -> prefers download_url:
        best_2026 = resolve_best_live_url(
            download_url="https://images.kinderpedia.co/files/photos/k1/200/orig.jpg",
            fullsize_url="https://images.kinderpedia.co/900x1200/files/photos/k1/200/orig.jpg",
            thumb_url="https://images.kinderpedia.co/300x300/files/photos/k1/200/orig.jpg",
            width=1440,
            height=1920,
        )
        self.assertEqual(best_2026, "https://images.kinderpedia.co/files/photos/k1/200/orig.jpg")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = "2026-05-12 Spring Garden"
            photo_dir = root / "Alex" / "Albums" / "2026" / "05" / leaf
            photo_dir.mkdir(parents=True)
            img_path = photo_dir / "001_998877.jpg"
            img_path.write_bytes(b"jpg")

            meta_dir = root / "Metadata" / "Alex" / "Albums"
            meta_dir.mkdir(parents=True)
            (meta_dir / f"{leaf}.json").write_text(
                json.dumps(
                    {
                        "album_listing": {
                            "id": 4455,
                            "name": "Spring Garden!",
                            "date": "2026-05-12",
                            "groups": [{"name": "Butterflies"}],
                        },
                        "images": [
                            {
                                "id": 998877,
                                "download_url": "https://images.kinderpedia.co/files/photos/k1/0/orig.jpg",
                                "fullsize": "https://images.kinderpedia.co/1200x900/files/photos/k1/0/orig.jpg",
                                "thumb": "https://images.kinderpedia.co/300x300/files/photos/k1/0/orig.jpg",
                                "dateadd": "2026-05-12 10:30:00",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            resolver = ProvenanceResolver(root)
            prov = resolver.resolve(img_path, width=1600, height=1200)
            self.assertEqual(prov.kinderpedia_album_id, 4455)
            self.assertEqual(prov.kinderpedia_image_id, 998877)
            self.assertEqual(prov.seq_in_album, 1)
            self.assertEqual(prov.classroom_groups, "Butterflies")
            self.assertEqual(prov.best_live_url, "https://images.kinderpedia.co/files/photos/k1/0/orig.jpg")

    def test_year_windowed_sibling_guard_and_hard_negative_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "faces_index.sqlite"
            with FacesIndexDB(db_path) as db:
                alex_id = db.ensure_person("Alex", role="family")
                maya_id = db.ensure_person("Maya", role="family")

                # Vector 0 = toddler face morphology (shared by Alex in 2022 and Maya in 2026!)
                # Vector 1 = Alex at age 6 in 2026
                # Vector 2 = Lookalike classmate (partially similar ~0.40 to Vector 1)
                toddler_vec = _basis_vec(0)
                alex_2026_vec = _basis_vec(1)
                lookalike_arr = np.zeros((512,), dtype=np.float32)
                lookalike_arr[1] = 0.40
                lookalike_arr[2] = np.sqrt(1.0 - 0.40**2)
                lookalike_vec = tuple(float(v) for v in lookalike_arr.tolist())

                # Enroll dated anchors: Alex has 2022 (toddler) + 2026 (age 6); Maya has 2026 (toddler slightly noisy)
                maya_seed = np.zeros((512,), dtype=np.float32)
                maya_seed[0] = 0.96
                maya_seed[3] = np.sqrt(1.0 - 0.96**2)
                db.upsert_reference_seed(person_id=alex_id, source_key="a_2022", embedding=toddler_vec, anchor_year=2022)
                db.upsert_reference_seed(person_id=alex_id, source_key="a_2026", embedding=alex_2026_vec, anchor_year=2026)
                db.upsert_reference_seed(
                    person_id=maya_id,
                    source_key="m_2026",
                    embedding=tuple(float(v) for v in maya_seed.tolist()),
                    anchor_year=2026,
                )

                # Insert 3 test photos in 2026:
                # 1) 2026 toddler photo (exact toddler_vec): without sibling guard Alex's 2022 anchor would score 1.0 > Maya's 0.96!
                #    With Year-Windowed Sibling Guard, Alex's 2022 anchor gets penalized (-0.075 -> 0.925) so Maya (0.96) wins!
                # 2) Lookalike photo A in 2026
                # 3) Lookalike photo B in 2026 (nearly identical to Lookalike A)
                box = BoundingBox(10, 90, 90, 10)
                lookalike_b_arr = lookalike_arr.copy()
                lookalike_b_arr[4] = 0.05
                lookalike_b_arr /= float(np.linalg.norm(lookalike_b_arr))

                saved = db.upsert_scanned_batch(
                    [
                        (
                            PhotoProvenance("Maya/Albums/2026/05/2026-05-01 Play/001_1.jpg", "Maya", "2026", "05", "2026-05-01 Play", "001_1.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [DetectedFace(box=box, embedding=toddler_vec, confidence=0.9)],
                        ),
                        (
                            PhotoProvenance("Alex/Albums/2026/05/2026-05-02 Art/001_2.jpg", "Alex", "2026", "05", "2026-05-02 Art", "001_2.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [DetectedFace(box=box, embedding=lookalike_vec, confidence=0.9)],
                        ),
                        (
                            PhotoProvenance("Alex/Albums/2026/05/2026-05-03 Art/001_3.jpg", "Alex", "2026", "05", "2026-05-03 Art", "001_3.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [
                                DetectedFace(
                                    box=box,
                                    embedding=tuple(float(v) for v in lookalike_b_arr.tolist()),
                                    confidence=0.9,
                                )
                            ],
                        ),
                    ]
                )
                p1_id, _ = saved[0]
                p2_id, p2_fids = saved[1]

                recompute_active_matching(db)

                # Verify Maya won the 2026 toddler photo thanks to Year-Windowed Sibling Guard:
                p1_detail = db.get_photo_detail(p1_id)
                self.assertEqual(p1_detail["faces"][0]["person_name"], "Maya")

                # Initially both Lookalike photos (sim ~0.40) appear in Alex's inclusive batch queue:
                batch_before = db.get_batch_proposal(alex_id)
                self.assertEqual(batch_before["remaining_unreviewed"], 2)

                # Disprove Lookalike Photo A once -> Hard-Negative Suppression eliminates Lookalike Photo B too!
                db.apply_batch_decisions(
                    person_id=alex_id,
                    approved=[],
                    disproved=[(p2_id, p2_fids[0])],
                )
                recompute_active_matching(db)
                batch_after = db.get_batch_proposal(alex_id)
                self.assertEqual(batch_after["remaining_unreviewed"], 0)

    def test_social_cooccurrence_boosts_peers_suppresses_small_photo_sibling_and_allows_large_events(
        self,
    ) -> None:
        dim = 512
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "social.sqlite"
            with FacesIndexDB(db_path) as db:
                maya_id = db.ensure_person("Maya", role="family", archive_hint="Maya")
                alex_id = db.ensure_person("Alex", role="family", archive_hint="Alex")
                leo_id = db.ensure_person("Leo", role="classmate")
                sofia_id = db.ensure_person("Sofia", role="classmate")

                def _basis(idx: int) -> tuple[float, ...]:
                    v = np.zeros((dim,), dtype=np.float32)
                    v[idx] = 1.0
                    return tuple(float(x) for x in v.tolist())

                maya_vec = _basis(0)
                alex_vec = _basis(1)
                leo_vec = _basis(2)
                sofia_vec = _basis(3)

                db.upsert_reference_seed(person_id=maya_id, source_key="m", embedding=maya_vec, anchor_year=2026)
                db.upsert_reference_seed(person_id=alex_id, source_key="a", embedding=alex_vec, anchor_year=2026)
                db.upsert_reference_seed(person_id=leo_id, source_key="l", embedding=leo_vec, anchor_year=2026)
                db.upsert_reference_seed(person_id=sofia_id, source_key="s", embedding=sofia_vec, anchor_year=2026)

                box_l = BoundingBox(10, 80, 80, 10)
                box_r = BoundingBox(10, 180, 80, 110)

                # Ambiguous sibling vector: visually closer to Alex (0.39) than Maya (0.37)!
                ambig_arr = np.zeros((dim,), dtype=np.float32)
                ambig_arr[0] = 0.37
                ambig_arr[1] = 0.39
                ambig_arr[10] = float(np.sqrt(1.0 - 0.37**2 - 0.39**2))
                ambig_vec = tuple(float(x) for x in ambig_arr.tolist())

                # Mild Alex-only lookalike (0.37 to Alex, 0.0 to Maya):
                alex_mild_arr = np.zeros((dim,), dtype=np.float32)
                alex_mild_arr[1] = 0.37
                alex_mild_arr[11] = float(np.sqrt(1.0 - 0.37**2))
                alex_mild_vec = tuple(float(x) for x in alex_mild_arr.tolist())

                # Mild Sofia vector (0.36 to Sofia):
                sofia_mild_arr = np.zeros((dim,), dtype=np.float32)
                sofia_mild_arr[3] = 0.36
                sofia_mild_arr[12] = float(np.sqrt(1.0 - 0.36**2))
                sofia_mild_vec = tuple(float(x) for x in sofia_mild_arr.tolist())

                # 8 unrelated faces for a 10-kid large festival photo:
                festival_faces = [
                    DetectedFace(box=box_l, embedding=leo_vec, confidence=0.95),
                    DetectedFace(box=box_r, embedding=alex_mild_vec, confidence=0.92),
                ]
                for k in range(8):
                    festival_faces.append(
                        DetectedFace(
                            box=BoundingBox(100, 60 + k * 20, 150, 10 + k * 20),
                            embedding=_basis(16 + k),
                            confidence=0.90,
                        )
                    )

                saved = db.upsert_scanned_batch(
                    [
                        # Photos 0 & 1: Confirmed small classroom photos of Maya + Leo together
                        (
                            PhotoProvenance("Maya/Albums/2026/05/2026-05-01 Class/001_1.jpg", "Maya", "2026", "05", "2026-05-01 Class", "001_1.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [
                                DetectedFace(box=box_l, embedding=maya_vec, confidence=0.95),
                                DetectedFace(box=box_r, embedding=leo_vec, confidence=0.95),
                            ],
                        ),
                        (
                            PhotoProvenance("Maya/Albums/2026/05/2026-05-02 Class/001_2.jpg", "Maya", "2026", "05", "2026-05-02 Class", "001_2.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [
                                DetectedFace(box=box_l, embedding=maya_vec, confidence=0.95),
                                DetectedFace(box=box_r, embedding=leo_vec, confidence=0.95),
                            ],
                        ),
                        # Photo 2: Confirmed solo photo of Sofia in Maya's classroom
                        (
                            PhotoProvenance("Maya/Albums/2026/05/2026-05-02 Class/002_3.jpg", "Maya", "2026", "05", "2026-05-02 Class", "002_3.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [DetectedFace(box=box_l, embedding=sofia_vec, confidence=0.95)],
                        ),
                        # Photo 3 (2 faces): Unconfirmed Leo + ambiguous sibling face (0.39 Alex vs 0.37 Maya)
                        (
                            PhotoProvenance("Maya/Albums/2026/05/2026-05-03 Play/001_4.jpg", "Maya", "2026", "05", "2026-05-03 Play", "001_4.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [
                                DetectedFace(box=box_l, embedding=leo_vec, confidence=0.95),
                                DetectedFace(box=box_r, embedding=ambig_vec, confidence=0.92),
                            ],
                        ),
                        # Photo 4 (2 faces): Unconfirmed Leo + mild Alex lookalike (0.37 Alex, 0.0 Maya)
                        (
                            PhotoProvenance("Maya/Albums/2026/05/2026-05-04 Play/001_5.jpg", "Maya", "2026", "05", "2026-05-04 Play", "001_5.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [
                                DetectedFace(box=box_l, embedding=leo_vec, confidence=0.95),
                                DetectedFace(box=box_r, embedding=alex_mild_vec, confidence=0.92),
                            ],
                        ),
                        # Photo 5 (10 faces): Large festival event with Leo + Alex (0.37) + 8 kids
                        (
                            PhotoProvenance("Maya/Albums/2026/06/2026-06-01 Fest/001_6.jpg", "Maya", "2026", "06", "2026-06-01 Fest", "001_6.jpg"),
                            1200,
                            800,
                            100,
                            1,
                            "m1",
                            festival_faces,
                        ),
                        # Photo 6 (2 faces): Unconfirmed Leo + Sofia (0.36) in same classroom
                        (
                            PhotoProvenance("Maya/Albums/2026/05/2026-05-05 Art/001_7.jpg", "Maya", "2026", "05", "2026-05-05 Art", "001_7.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [
                                DetectedFace(box=box_l, embedding=leo_vec, confidence=0.95),
                                DetectedFace(box=box_r, embedding=sofia_mild_vec, confidence=0.92),
                            ],
                        ),
                    ]
                )

                # Confirm Maya + Leo in Photos 0 & 1, and Sofia in Photo 2:
                for idx_c in (0, 1):
                    pid_c, fids_c = saved[idx_c]
                    db.tag_single_face(photo_id=pid_c, face_id=fids_c[0], person_id=maya_id, status="confirmed")
                    db.tag_single_face(photo_id=pid_c, face_id=fids_c[1], person_id=leo_id, status="confirmed")
                db.tag_single_face(
                    photo_id=saved[2][0], face_id=saved[2][1][0], person_id=sofia_id, status="confirmed"
                )

                recompute_active_matching(db)

                # 1. Photo 3 (2-face photo with Leo): Maya wins the ambiguous face over Alex thanks to Leo's peer boost + sibling suppression!
                p3_detail = db.get_photo_detail(saved[3][0])
                p3_names = {f["face_idx"]: f["person_name"] for f in p3_detail["faces"]}
                self.assertEqual(p3_names[0], "Leo")
                self.assertEqual(p3_names[1], "Maya")

                # 2. Alex's batch queue: Photo 4 (2-face photo with Leo, raw 0.37) is suppressed below 0.33,
                #    whereas Photo 5 (10-face festival event with Leo, raw 0.37) has 0 penalty and IS included!
                alex_batch = db.get_batch_proposal(alex_id)
                alex_pids = [c["photo_id"] for c in alex_batch["cards"]]
                self.assertNotIn(saved[4][0], alex_pids)
                self.assertIn(saved[5][0], alex_pids)

                # 3. Maya's batch header reports Leo as a learned peer, and Photo 3 card surfaces "Leo" in companions:
                maya_batch = db.get_batch_proposal(maya_id)
                self.assertEqual(maya_batch["top_peers"][0]["display_name"], "Leo")
                self.assertEqual(maya_batch["top_peers"][0]["shared_photos"], 2)
                p3_card = next(c for c in maya_batch["cards"] if c["photo_id"] == saved[3][0])
                self.assertIn("Leo", p3_card["companions"])

                # 4. Sofia (same classroom cohort as Leo) is NOT penalized in Photo 6 (2-face photo with Leo):
                sofia_batch = db.get_batch_proposal(sofia_id)
                sofia_pids = [c["photo_id"] for c in sofia_batch["cards"]]
                self.assertIn(saved[6][0], sofia_pids)

    def test_same_album_cohesion_auto_absorbs_bunnysuit_burst_without_manual_reconfirmation(
        self,
    ) -> None:
        dim = 512
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "cohesion.sqlite"
            with FacesIndexDB(db_path) as db:
                alex_id = db.ensure_person("Alex", role="family", archive_hint="Alex")

                # Older avatar seed (moderately similar 0.39 to the bunnysuit look):
                seed_arr = np.zeros((dim,), dtype=np.float32)
                seed_arr[0] = 1.0
                db.upsert_reference_seed(
                    person_id=alex_id,
                    source_key="avatar",
                    embedding=tuple(float(v) for v in seed_arr.tolist()),
                    anchor_year=2025,
                )

                # Bunnysuit outfit cluster in "2026-04-15 Bunny Party":
                # All 6 shots share the bunnysuit feature axis [1], with slight pose jitter (pairwise sim ~0.68..0.95):
                bunny_base = np.zeros((dim,), dtype=np.float32)
                bunny_base[0] = 0.39
                bunny_base[1] = float(np.sqrt(1.0 - 0.39**2))

                box = BoundingBox(10, 90, 90, 10)
                batch_to_save: list[tuple[PhotoProvenance, int, int, int, int, str, list[DetectedFace]]] = []
                for idx in range(6):
                    vec = bunny_base.copy()
                    vec[2 + idx] = 0.22
                    vec /= float(np.linalg.norm(vec))
                    fname = f"00{idx + 1}_{800 + idx}.jpg"
                    batch_to_save.append(
                        (
                            PhotoProvenance(
                                f"Alex/Albums/2026/04/2026-04-15 Bunny Party/{fname}",
                                "Alex",
                                "2026",
                                "04",
                                "2026-04-15 Bunny Party",
                                fname,
                            ),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [
                                DetectedFace(
                                    box=box,
                                    embedding=tuple(float(v) for v in vec.tolist()),
                                    confidence=0.92,
                                )
                            ],
                        )
                    )

                # Plus 1 unrelated classmate photo in the same Bunny Party album:
                other_arr = np.zeros((dim,), dtype=np.float32)
                other_arr[15] = 1.0
                batch_to_save.append(
                    (
                        PhotoProvenance(
                            "Alex/Albums/2026/04/2026-04-15 Bunny Party/007_899.jpg",
                            "Alex",
                            "2026",
                            "04",
                            "2026-04-15 Bunny Party",
                            "007_899.jpg",
                        ),
                        800,
                        600,
                        100,
                        1,
                        "m1",
                        [
                            DetectedFace(
                                box=box,
                                embedding=tuple(float(v) for v in other_arr.tolist()),
                                confidence=0.92,
                            )
                        ],
                    )
                )

                saved = db.upsert_scanned_batch(batch_to_save)
                recompute_active_matching(db)

                # Confirm ONLY the very first bunnysuit photo in "Bunny Party":
                first_pid, first_fids = saved[0]
                db.apply_batch_decisions(
                    person_id=alex_id,
                    approved=[(first_pid, first_fids[0])],
                    disproved=[],
                )
                recompute_active_matching(db)

                # All 5 remaining bunnysuit photos in "Bunny Party" are auto-absorbed via `album_cohesion`,
                # so 0 bunnysuit duplicates remain in the manual review batch queue!
                next_batch = db.get_batch_proposal(alex_id)
                self.assertEqual(next_batch["remaining_unreviewed"], 0)
                self.assertEqual(next_batch["album_cohesion_count"], 5)

                # All 6 bunnysuit photos (1 confirmed + 5 album_cohesion) are included in Alex's gallery,
                # while the 7th unrelated classmate photo is NOT included:
                gallery = db.list_person_photos(alex_id, status_filter="included")
                self.assertEqual(gallery["total"], 6)
                cohesion_items = [it for it in gallery["items"] if it["source"] == "album_cohesion"]
                self.assertEqual(len(cohesion_items), 5)

    def test_web_api_batch_loop_clusters_url_export_and_flat_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup_dir = root / "bk"
            out_dir = backup_dir / "ByChild"
            album_dir = backup_dir / "Alex" / "Albums" / "2026" / "06" / "2026-06-10 Summer Fest"
            album_dir.mkdir(parents=True)

            # Create real tiny RGB JPEGs so crop thumbnail generation works end-to-end:
            def _make_jpg(path: Path) -> None:
                Image.new("RGB", (240, 240), color=(210, 195, 175)).save(path, format="JPEG")

            _make_jpg(backup_dir / "Alex" / "avatar.png")
            meta_images: list[dict[str, Any]] = []
            for i in range(1, 6):
                img_id = 7000 + i
                fn = f"00{i}_leo_{img_id}.jpg"
                _make_jpg(album_dir / fn)
                meta_images.append(
                    {
                        "id": img_id,
                        "download_url": f"https://images.kinderpedia.co/files/photos/k1/0/{img_id}.jpg",
                        "fullsize": f"https://images.kinderpedia.co/1200x900/files/photos/k1/0/{img_id}.jpg",
                        "thumb": f"https://images.kinderpedia.co/300x300/files/photos/k1/0/{img_id}.jpg",
                    }
                )
            _make_jpg(album_dir / "006_alex_7006.jpg")
            meta_images.append(
                {
                    "id": 7006,
                    "download_url": "https://images.kinderpedia.co/files/photos/k1/0/7006.jpg",
                }
            )

            meta_dir = backup_dir / "Metadata" / "Alex" / "Albums"
            meta_dir.mkdir(parents=True)
            (meta_dir / "2026-06-10 Summer Fest.json").write_text(
                json.dumps(
                    {
                        "album_listing": {"id": 99, "name": "Summer Fest", "date": "2026-06-10"},
                        "images": meta_images,
                    }
                ),
                encoding="utf-8",
            )

            rec = _StubRecognizer(dim=512)
            summary = run_archive_face_scan(
                backup_dir=backup_dir,
                recognizer=rec,
                output_dir=out_dir,
                workers=2,
            )
            self.assertEqual(summary.photos_total, 6)
            self.assertGreaterEqual(summary.clusters_created, 1)
            self.assertEqual(summary.people_seeded, 1)  # Alex from avatar.png

            db_path = backup_dir / "Metadata" / "faces_index.sqlite"
            with FacesIndexDB(db_path) as db:
                ctx = _AppContext(
                    backup_dir=backup_dir,
                    output_dir=out_dir,
                    db=db,
                    csrf_token="test-csrf-secret",
                )
                httpd = _FacesHTTPServer(("127.0.0.1", 0), ctx)
                port = int(httpd.server_address[1])
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{port}"

                try:
                    # 1. Index HTML injects CSRF token:
                    r_html = requests.get(f"{base_url}/", timeout=5)
                    r_html.raise_for_status()
                    self.assertIn("test-csrf-secret", r_html.text)

                    # 2. POST without X-KP-Token must be rejected with 403:
                    r_bad = requests.post(f"{base_url}/api/people", json={"name": "Leo"}, timeout=5)
                    self.assertEqual(r_bad.status_code, 403)

                    # 3. Discover Leo's unassigned 5-photo cluster, inspect full photo context, and assign "Leo":
                    r_cl = requests.get(f"{base_url}/api/clusters", timeout=5)
                    r_cl.raise_for_status()
                    clusters = r_cl.json()["clusters"]
                    self.assertEqual(len(clusters), 1)
                    cid = clusters[0]["cluster_id"]
                    sample_fid = clusters[0]["sample_face_ids"][0]

                    r_cf = requests.get(f"{base_url}/api/clusters/{cid}/faces", timeout=5)
                    r_cf.raise_for_status()
                    cluster_faces = r_cf.json()["faces"]
                    self.assertEqual(len(cluster_faces), 5)
                    self.assertEqual(cluster_faces[0]["face_id"], sample_fid)

                    r_fp = requests.get(f"{base_url}/api/faces/{sample_fid}/photo", timeout=5)
                    r_fp.raise_for_status()
                    fp_json = r_fp.json()
                    self.assertEqual(fp_json["album_title"], "Summer Fest")
                    self.assertTrue(any(f["face_id"] == sample_fid for f in fp_json["faces"]))

                    r_as = requests.post(
                        f"{base_url}/api/clusters/{cid}/assign",
                        json={"new_name": "Leo", "role": "classmate"},
                        headers={"X-KP-Token": "test-csrf-secret"},
                        timeout=5,
                    )
                    r_as.raise_for_status()
                    self.assertEqual(r_as.json()["confirmed_from_cluster"], 5)

                    # 4. Export Leo's Kinderpedia URLs via GET /api/export-urls:
                    r_urls = requests.get(f"{base_url}/api/export-urls?person=Leo&format=txt", timeout=5)
                    r_urls.raise_for_status()
                    urls_txt = r_urls.text.strip().splitlines()
                    self.assertEqual(len(urls_txt), 5)
                    self.assertTrue(all(re.match(r"^https://images\.kinderpedia\.co/", u) for u in urls_txt))

                    # 5. Materialize flat ByChild/Leo/ and ByChild/Alex/ hardlinks:
                    r_mat = requests.post(
                        f"{base_url}/api/materialize",
                        json={},
                        headers={"X-KP-Token": "test-csrf-secret"},
                        timeout=5,
                    )
                    r_mat.raise_for_status()
                    self.assertEqual(_await_job_result(base_url, r_mat)["links_created"], 6)
                    self.assertEqual(len(list((out_dir / "Leo").iterdir())), 5)
                    self.assertEqual(len(list((out_dir / "Alex").iterdir())), 1)

                    # 6. Portable categorization_index.json written with provenance:
                    portable = json.loads(
                        (backup_dir / "Metadata" / "categorization_index.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(portable["categorized_photo_count"], 6)
                finally:
                    httpd.shutdown()
                    httpd.server_close()

    def test_letterbox_pil_vs_ndarray_bit_identical(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        pil_im = Image.fromarray(arr)
        blob_arr, scale_arr = letterbox_scrfd_input(arr, det_size=640)
        blob_pil, scale_pil = letterbox_scrfd_input(pil_im, det_size=640)
        self.assertAlmostEqual(scale_arr, scale_pil)
        np.testing.assert_array_equal(blob_arr, blob_pil)

    def test_unsure_skips_safely_and_modal_reassign_teaches_face_scoped_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "faces_index.sqlite"
            with FacesIndexDB(db_path) as db:
                maya_id = db.ensure_person("Maya", role="family")
                maya_vec = _basis_vec(0)
                db.upsert_reference_seed(
                    person_id=maya_id,
                    source_key="maya_seed",
                    embedding=maya_vec,
                    anchor_year=2026,
                )

                # Two nearly identical ambiguous faces (~0.40 similarity to Maya):
                amb_a = np.zeros((512,), dtype=np.float32)
                amb_a[0] = 0.40
                amb_a[1] = np.sqrt(1.0 - 0.40**2)
                amb_b = amb_a.copy()
                amb_b[2] = 0.03
                amb_b /= float(np.linalg.norm(amb_b))

                # And one 2-face group photo where Face #0 is Nina (0.43 sim) and Face #1 is Maya (0.41 sim):
                nina_face = np.zeros((512,), dtype=np.float32)
                nina_face[0] = 0.43
                nina_face[3] = np.sqrt(1.0 - 0.43**2)
                maya_group_face = np.zeros((512,), dtype=np.float32)
                maya_group_face[0] = 0.41
                maya_group_face[4] = np.sqrt(1.0 - 0.41**2)

                box1 = BoundingBox(10, 90, 90, 10)
                box2 = BoundingBox(10, 210, 90, 130)
                saved = db.upsert_scanned_batch(
                    [
                        (
                            PhotoProvenance("Maya/Albums/2026/05/A/001.jpg", "Maya", "2026", "05", "A", "001.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [DetectedFace(box=box1, embedding=tuple(float(x) for x in amb_a.tolist()), confidence=0.9)],
                        ),
                        (
                            PhotoProvenance("Maya/Albums/2026/05/A/002.jpg", "Maya", "2026", "05", "A", "002.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [DetectedFace(box=box1, embedding=tuple(float(x) for x in amb_b.tolist()), confidence=0.9)],
                        ),
                        (
                            PhotoProvenance("Maya/Albums/2026/05/A/003_group.jpg", "Maya", "2026", "05", "A", "003_group.jpg"),
                            800,
                            600,
                            100,
                            1,
                            "m1",
                            [
                                DetectedFace(box=box1, embedding=tuple(float(x) for x in nina_face.tolist()), confidence=0.9),
                                DetectedFace(box=box2, embedding=tuple(float(x) for x in maya_group_face.tolist()), confidence=0.9),
                            ],
                        ),
                    ]
                )
                p1_id, p1_fids = saved[0]
                _, _ = saved[1]
                p3_id, p3_fids = saved[2]

                recompute_active_matching(db)
                self.assertEqual(db.get_batch_proposal(maya_id)["remaining_unreviewed"], 3)

                # 1. Mark Photo 1 as "I'm not sure" (`unsure`):
                #    Photo 1 leaves the batch queue, appears in status="unsure" gallery,
                #    AND does NOT suppress twin Photo 2!
                db.apply_batch_decisions(
                    person_id=maya_id,
                    approved=[],
                    disproved=[],
                    unsure=[(p1_id, p1_fids[0])],
                )
                recompute_active_matching(db)
                self.assertEqual(db.get_batch_proposal(maya_id)["remaining_unreviewed"], 2)
                unsure_gal = db.list_person_photos(maya_id, status_filter="unsure")
                self.assertEqual(unsure_gal["total"], 1)
                self.assertEqual(unsure_gal["items"][0]["photo_id"], p1_id)

                # 2. In Group Photo 3, Maya originally matched Face #0 (0.43 > 0.41).
                #    User opens Photo Modal and tags Face #0 as "Nina":
                #    - Nina owns Face #0 (`confirmed`)
                #    - Maya gets a face-scoped `rejected` (`reassigned`) on Face #0
                #    - On rematch, Maya now matches her real Face #1 in the SAME group photo!
                nina_id = db.ensure_person("Nina", role="classmate")
                db.tag_single_face(
                    photo_id=p3_id,
                    face_id=p3_fids[0],
                    person_id=nina_id,
                    status="confirmed",
                )
                recompute_active_matching(db)
                detail = db.get_photo_detail(p3_id)
                by_fid = {f["face_id"]: f["person_name"] for f in detail["faces"]}
                self.assertEqual(by_fid[p3_fids[0]], "Nina")
                self.assertEqual(by_fid[p3_fids[1]], "Maya")

    def test_untag_face_and_exclude_cluster_face_via_http_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup_dir = root / "bk"
            out_dir = backup_dir / "ByChild"
            backup_dir.mkdir(parents=True)
            db_path = backup_dir / "Metadata" / "faces_index.sqlite"

            with FacesIndexDB(db_path) as db:
                maya_id = db.ensure_person("Maya", role="family")
                sofia_id = db.ensure_person("Sofia", role="classmate")

                maya_vec = _basis_vec(0)
                sofia_vec = _basis_vec(1)
                db.upsert_reference_seed(person_id=maya_id, source_key="avatar:maya", embedding=maya_vec, anchor_year=2025)
                db.upsert_reference_seed(person_id=sofia_id, source_key="avatar:sofia", embedding=sofia_vec, anchor_year=2025)

                # Face #0 in Group Photo matches BOTH Maya (0.44) and runner-up Sofia (0.40);
                # Face #1 in the same photo is Maya's real face (0.41).
                ambig_arr = np.zeros((512,), dtype=np.float32)
                ambig_arr[0] = 0.44
                ambig_arr[1] = 0.40
                ambig_arr[2] = float(np.sqrt(1.0 - 0.44**2 - 0.40**2))

                maya_real_arr = np.zeros((512,), dtype=np.float32)
                maya_real_arr[0] = 0.41
                maya_real_arr[3] = float(np.sqrt(1.0 - 0.41**2))

                leo_vec = _basis_vec(5)
                stranger_vec = _basis_vec(6)

                saved = db.upsert_scanned_batch(
                    [
                        (
                            PhotoProvenance("Maya/Albums/2025/05/A/001.jpg", "Maya", "2025", "05", "A", "001.jpg"),
                            800,
                            600,
                            1000,
                            1,
                            "scrfd",
                            [
                                DetectedFace(
                                    box=BoundingBox(top=20, right=120, bottom=120, left=20),
                                    embedding=tuple(float(x) for x in ambig_arr),
                                    confidence=0.92,
                                ),
                                DetectedFace(
                                    box=BoundingBox(top=20, right=280, bottom=120, left=180),
                                    embedding=tuple(float(x) for x in maya_real_arr),
                                    confidence=0.90,
                                ),
                            ],
                        ),
                        *(
                            (
                                PhotoProvenance(
                                    f"Maya/Albums/2025/06/B/00{idx}.jpg",
                                    "Maya",
                                    "2025",
                                    "06",
                                    "B",
                                    f"00{idx}.jpg",
                                ),
                                800,
                                600,
                                1000,
                                1,
                                "scrfd",
                                [
                                    DetectedFace(
                                        box=BoundingBox(top=30, right=130, bottom=130, left=30),
                                        embedding=leo_vec if idx < 5 else stranger_vec,
                                        confidence=0.93,
                                    )
                                ],
                            )
                            for idx in range(2, 6)
                        ),
                    ]
                )
                grp_pid, grp_fids = saved[0]
                cluster_fids = [saved[i][1][0] for i in range(1, 5)]
                stray_pid, stray_fids = saved[4]
                stray_fid = stray_fids[0]

                db.save_clusters(
                    [
                        {
                            "archive_child": "Maya",
                            "year": "2025",
                            "person_id": None,
                            "medoid_face_id": cluster_fids[0],
                            "sample_face_ids": cluster_fids,
                            "face_ids": cluster_fids,
                            "photo_count": 4,
                            "suggested_person_id": None,
                            "suggested_similarity": None,
                        }
                    ]
                )
                recompute_active_matching(db)

                # Initially Maya claims Face #0 (0.44 > 0.41), leaving Face #1 untagged or unmatched:
                init_detail = db.get_photo_detail(grp_pid)
                self.assertEqual(init_detail["faces"][0]["person_name"], "Maya")

                ctx = _AppContext(
                    backup_dir=backup_dir,
                    output_dir=out_dir,
                    db=db,
                    csrf_token="test-untag-token",
                )
                httpd = _FacesHTTPServer(("127.0.0.1", 0), ctx)
                port = int(httpd.server_address[1])
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{port}"
                headers = {"X-KP-Token": "test-untag-token"}

                try:
                    # 1. Remove tag on Face #0 -> immediate response (<50ms) leaves Face #0 uncategorized
                    #    with `suppress_auto=True`, and once background rematch finishes, Face #0 stays
                    #    uncategorized (even though runner-up Sofia = 0.40!) while Maya moves onto Face #1!
                    r_untag = requests.post(
                        f"{base_url}/api/faces/untag",
                        json={"photo_id": grp_pid, "face_id": grp_fids[0]},
                        headers=headers,
                        timeout=5,
                    )
                    r_untag.raise_for_status()
                    untag_json = r_untag.json()
                    self.assertTrue(untag_json["rematch_scheduled"])
                    faces_immediate = {f["face_id"]: f for f in untag_json["photo"]["faces"]}
                    self.assertIsNone(faces_immediate[grp_fids[0]]["person_id"])
                    self.assertIsNone(faces_immediate[grp_fids[0]]["person_name"])
                    self.assertTrue(faces_immediate[grp_fids[0]]["suppress_auto"])

                    self.assertTrue(ctx.wait_for_rematch(timeout=5.0))
                    faces_after = {f["face_id"]: f for f in db.get_photo_detail(grp_pid)["faces"]}
                    self.assertIsNone(faces_after[grp_fids[0]]["person_id"])
                    self.assertIsNone(faces_after[grp_fids[0]]["person_name"])
                    self.assertTrue(faces_after[grp_fids[0]]["suppress_auto"])
                    self.assertEqual(faces_after[grp_fids[1]]["person_name"], "Maya")

                    # 2. Explicitly confirming a name later on that uncategorized face clears `suppress_auto` cleanly:
                    r_retag = requests.post(
                        f"{base_url}/api/faces/tag",
                        json={
                            "photo_id": grp_pid,
                            "face_id": grp_fids[0],
                            "new_name": "Alex",
                            "status": "confirmed",
                        },
                        headers=headers,
                        timeout=5,
                    )
                    r_retag.raise_for_status()
                    faces_retag = {f["face_id"]: f for f in r_retag.json()["photo"]["faces"]}
                    self.assertEqual(faces_retag[grp_fids[0]]["person_name"], "Alex")
                    self.assertEqual(faces_retag[grp_fids[0]]["status"], "confirmed")
                    self.assertFalse(faces_retag[grp_fids[0]]["suppress_auto"])

                    # 3. Removing a confirmed tag also leaves the face uncategorized and refreshes cover_face_id:
                    r_untag_conf = requests.post(
                        f"{base_url}/api/faces/untag",
                        json={"photo_id": grp_pid, "face_id": grp_fids[0]},
                        headers=headers,
                        timeout=5,
                    )
                    r_untag_conf.raise_for_status()
                    faces_untag2 = {f["face_id"]: f for f in r_untag_conf.json()["photo"]["faces"]}
                    self.assertIsNone(faces_untag2[grp_fids[0]]["person_name"])

                    # 4. Exclude stray_fid from the 4-photo cluster before confirming the cluster as "Leo":
                    cid = db.list_clusters()[0]["cluster_id"]
                    r_rem = requests.post(
                        f"{base_url}/api/clusters/{cid}/remove-face",
                        json={"face_id": stray_fid},
                        headers=headers,
                        timeout=5,
                    )
                    r_rem.raise_for_status()
                    self.assertEqual(r_rem.json()["cluster"]["photo_count"], 3)

                    r_assign = requests.post(
                        f"{base_url}/api/clusters/{cid}/assign",
                        json={"new_name": "Leo", "role": "classmate"},
                        headers=headers,
                        timeout=5,
                    )
                    r_assign.raise_for_status()
                    self.assertEqual(r_assign.json()["confirmed_from_cluster"], 3)

                    # The excluded stray photo remains completely uncategorized:
                    stray_detail = db.get_photo_detail(stray_pid)
                    self.assertIsNone(stray_detail["faces"][0]["person_name"])
                finally:
                    httpd.shutdown()
                    httpd.server_close()

    def test_ignored_role_never_pollutes_cluster_suggestions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ignored_suggest.sqlite"
            with FacesIndexDB(db_path) as db:
                ignored_id = db.ensure_person("_Ignored_Background", role="ignore")
                alex_id = db.ensure_person("Alex", role="classmate")

                # Cluster vector is 0.65 similar to _Ignored_Background and 0.52 similar to Alex:
                ign_vec = _basis_vec(0)
                alex_vec = _basis_vec(1)
                db.upsert_reference_seed(person_id=ignored_id, source_key="ign:1", embedding=ign_vec, anchor_year=2026)
                db.upsert_reference_seed(person_id=alex_id, source_key="alex:1", embedding=alex_vec, anchor_year=2026)

                cl_arr = np.zeros((512,), dtype=np.float32)
                cl_arr[0] = 0.65
                cl_arr[1] = 0.52
                cl_arr[2] = float(np.sqrt(1.0 - 0.65**2 - 0.52**2))
                cl_vec = tuple(float(x) for x in cl_arr)

                db.upsert_scanned_batch(
                    [
                        (
                            PhotoProvenance(f"Maya/Albums/2026/05/A/00{i}.jpg", "Maya", "2026", "05", "A", f"00{i}.jpg"),
                            800,
                            600,
                            1000,
                            1,
                            "scrfd",
                            [
                                DetectedFace(
                                    box=BoundingBox(top=20, right=120, bottom=120, left=20),
                                    embedding=cl_vec,
                                    confidence=0.95,
                                )
                            ],
                        )
                        for i in range(1, 6)
                    ]
                )

                n_cl = rebuild_yearly_clusters(db)
                self.assertEqual(n_cl, 1)
                clusters = db.list_clusters()
                self.assertEqual(len(clusters), 1)
                self.assertEqual(clusters[0]["suggested_person_name"], "Alex")

    def test_reassign_from_gallery_or_batch_immediately_updates_face_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup_dir = Path(tmp)
            out_dir = backup_dir / "ByChild"
            db_path = backup_dir / "Metadata" / "faces_index.sqlite"
            with FacesIndexDB(db_path) as db:
                alex_id = db.ensure_person("Alex", role="family")
                maya_id = db.ensure_person("Maya", role="family")
                ignored_id = db.ensure_person("_Ignored_Background", role="ignore")

                alex_vec = _basis_vec(0)
                maya_vec = _basis_vec(1)
                saved = db.upsert_scanned_batch(
                    [
                        (
                            PhotoProvenance("Alex/Albums/2026/05/A/001.jpg", "Alex", "2026", "05", "A", "001.jpg"),
                            800,
                            600,
                            1000,
                            1,
                            "scrfd",
                            [
                                DetectedFace(
                                    box=BoundingBox(top=20, right=120, bottom=120, left=20),
                                    embedding=maya_vec,
                                    confidence=0.96,
                                )
                            ],
                        ),
                        (
                            PhotoProvenance("Alex/Albums/2026/05/A/002.jpg", "Alex", "2026", "05", "A", "002.jpg"),
                            800,
                            600,
                            1000,
                            1,
                            "scrfd",
                            [
                                DetectedFace(
                                    box=BoundingBox(top=25, right=125, bottom=125, left=25),
                                    embedding=alex_vec,
                                    confidence=0.97,
                                )
                            ],
                        ),
                    ]
                )
                wrong_pid, wrong_fids = saved[0]
                wrong_fid = wrong_fids[0]
                keep_pid, keep_fids = saved[1]
                keep_fid = keep_fids[0]

                # Suppose wrong_pid was mistakenly confirmed in Alex's gallery (and even set as Alex's cover face),
                # while Maya previously had a rejected row (similarity=0.0) and an ignored profile also held wrong_fid:
                db.tag_single_face(photo_id=wrong_pid, face_id=wrong_fid, person_id=alex_id, status="confirmed")
                db.tag_single_face(photo_id=keep_pid, face_id=keep_fid, person_id=alex_id, status="confirmed")
                db.update_person(alex_id, cover_face_id=wrong_fid)
                db.tag_single_face(photo_id=wrong_pid, face_id=wrong_fid, person_id=maya_id, status="rejected")
                db._conn.execute(
                    """
                    INSERT INTO assignments (photo_id, person_id, face_id, status, source, similarity, is_anchor, updated_at)
                    VALUES (?, ?, ?, 'confirmed', 'cluster_confirm', 1.0, 0, '2026-01-01T00:00:00+00:00')
                    """,
                    (wrong_pid, ignored_id, wrong_fid),
                )
                db._conn.commit()

                # Photo detail must still return strictly 1 face box (never duplicated by ignored profile),
                # initially showing Alex:
                before_detail = db.get_photo_detail(wrong_pid)
                self.assertEqual(len(before_detail["faces"]), 1)
                self.assertEqual(before_detail["faces"][0]["person_name"], "Alex")

                ctx = _AppContext(
                    backup_dir=backup_dir,
                    output_dir=out_dir,
                    db=db,
                    csrf_token="test-reassign-token",
                )
                httpd = _FacesHTTPServer(("127.0.0.1", 0), ctx)
                port = int(httpd.server_address[1])
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{port}"
                headers = {"X-KP-Token": "test-reassign-token"}

                try:
                    # Reassign from Alex's gallery/batch to Maya (even passing face_id=None to verify auto-resolution!):
                    r_re = requests.post(
                        f"{base_url}/api/faces/tag",
                        json={
                            "photo_id": wrong_pid,
                            "face_id": None,
                            "person_id": maya_id,
                            "replace_person_id": alex_id,
                            "status": "confirmed",
                        },
                        headers=headers,
                        timeout=5,
                    )
                    r_re.raise_for_status()
                    payload = r_re.json()
                    self.assertEqual(payload["person_id"], maya_id)
                    self.assertEqual(payload["face_id"], wrong_fid)

                    # 1. Face box tag itself immediately updates from Alex -> Maya with similarity=1.0:
                    self.assertEqual(len(payload["photo"]["faces"]), 1)
                    face_entry = payload["photo"]["faces"][0]
                    self.assertEqual(face_entry["face_id"], wrong_fid)
                    self.assertEqual(face_entry["person_id"], maya_id)
                    self.assertEqual(face_entry["person_name"], "Maya")
                    self.assertEqual(face_entry["status"], "confirmed")
                    self.assertEqual(face_entry["similarity"], 1.0)

                    # 2. Photo leaves Alex's gallery and appears in Maya's gallery:
                    alex_photos = [it["photo_id"] for it in db.list_person_photos(alex_id)["items"]]
                    maya_photos = [it["photo_id"] for it in db.list_person_photos(maya_id)["items"]]
                    self.assertNotIn(wrong_pid, alex_photos)
                    self.assertIn(keep_pid, alex_photos)
                    self.assertIn(wrong_pid, maya_photos)

                    # 3. Alex's cover face refreshes off the reassigned face onto Alex's remaining confirmed face:
                    people_by_id = {p["person_id"]: p for p in payload["overview"]["people"]}
                    self.assertEqual(people_by_id[alex_id]["cover_face_id"], keep_fid)
                    self.assertEqual(people_by_id[maya_id]["cover_face_id"], wrong_fid)
                finally:
                    httpd.shutdown()
                    httpd.server_close()

    def test_async_debounced_rematch_and_sse_event_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup_dir = Path(tmp) / "bk"
            out_dir = backup_dir / "ByChild"
            db_path = backup_dir / "Metadata" / "faces_index.sqlite"

            with FacesIndexDB(db_path) as db:
                leo_vec = _basis_vec(0)
                maya_vec = _basis_vec(1)
                box = BoundingBox(top=15, right=115, bottom=115, left=15)

                saved = db.upsert_scanned_batch(
                    [
                        (
                            PhotoProvenance("Alex/Albums/2026/05/A/001.jpg", "Alex", "2026", "05", "A", "001.jpg"),
                            800,
                            600,
                            1000,
                            1,
                            "scrfd",
                            [DetectedFace(box=box, embedding=leo_vec, confidence=0.95)],
                        ),
                        (
                            PhotoProvenance("Alex/Albums/2026/05/A/002.jpg", "Alex", "2026", "05", "A", "002.jpg"),
                            800,
                            600,
                            1000,
                            1,
                            "scrfd",
                            [DetectedFace(box=box, embedding=leo_vec, confidence=0.94)],
                        ),
                        (
                            PhotoProvenance("Alex/Albums/2026/05/A/003.jpg", "Alex", "2026", "05", "A", "003.jpg"),
                            800,
                            600,
                            1000,
                            1,
                            "scrfd",
                            [DetectedFace(box=box, embedding=maya_vec, confidence=0.96)],
                        ),
                    ]
                )
                p1_id, p1_fids = saved[0]
                p2_id, _ = saved[1]
                p3_id, p3_fids = saved[2]

                ctx = _AppContext(
                    backup_dir=backup_dir,
                    output_dir=out_dir,
                    db=db,
                    csrf_token="test-sse-token",
                    rematch_debounce_sec=0.18,
                    snapshot_debounce_sec=0.35,
                )
                httpd = _FacesHTTPServer(("127.0.0.1", 0), ctx)
                port = int(httpd.server_address[1])
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{port}"
                headers = {"X-KP-Token": "test-sse-token"}

                sse_events: list[tuple[str, dict[str, Any]]] = []
                sse_connected = threading.Event()
                sse_complete = threading.Event()

                def _read_sse() -> None:
                    with requests.get(f"{base_url}/api/events", stream=True, timeout=5) as resp:
                        self.assertEqual(resp.status_code, 200)
                        self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
                        cur_event = "message"
                        for raw_line in resp.iter_lines(chunk_size=1, decode_unicode=True):
                            if raw_line is None:
                                continue
                            line = raw_line.strip()
                            if not line or line.startswith(":"):
                                continue
                            if line.startswith("event:"):
                                cur_event = line[len("event:") :].strip()
                            elif line.startswith("data:"):
                                data_obj = json.loads(line[len("data:") :].strip())
                                sse_events.append((cur_event, data_obj))
                                if cur_event == "connected":
                                    sse_connected.set()
                                elif cur_event == "rematch_complete":
                                    sse_complete.set()
                                    return

                sse_thread = threading.Thread(target=_read_sse, daemon=True)
                sse_thread.start()

                try:
                    self.assertTrue(sse_connected.wait(timeout=3.0))

                    # Two rapid face tags inside the 180ms debounce window return immediately
                    # and coalesce into a single background rematch sweep:
                    r1 = requests.post(
                        f"{base_url}/api/faces/tag",
                        json={"photo_id": p1_id, "face_id": p1_fids[0], "new_name": "Leo", "status": "confirmed"},
                        headers=headers,
                        timeout=5,
                    )
                    r1.raise_for_status()
                    self.assertTrue(r1.json()["rematch_scheduled"])
                    self.assertEqual(r1.json()["photo"]["faces"][0]["person_name"], "Leo")

                    r2 = requests.post(
                        f"{base_url}/api/faces/tag",
                        json={"photo_id": p3_id, "face_id": p3_fids[0], "new_name": "Maya", "status": "confirmed"},
                        headers=headers,
                        timeout=5,
                    )
                    r2.raise_for_status()
                    self.assertTrue(r2.json()["rematch_scheduled"])

                    # Wait for SSE `rematch_complete` push:
                    self.assertTrue(sse_complete.wait(timeout=4.0))
                    complete_payloads = [d for ev, d in sse_events if ev == "rematch_complete"]
                    self.assertEqual(len(complete_payloads), 1)
                    self.assertGreaterEqual(complete_payloads[0]["seq"], 2)
                    self.assertEqual(complete_payloads[0]["overview"]["faces_categorized"], 3)

                    # Photo 2 (untagged twin of Leo) was auto-matched by the background worker:
                    p2_detail = db.get_photo_detail(p2_id)
                    self.assertEqual(p2_detail["faces"][0]["person_name"], "Leo")
                    self.assertEqual(p2_detail["faces"][0]["status"], "auto_high")
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    sse_thread.join(timeout=2.0)

    def test_manual_bounding_box_tag_thumb_export_and_ai_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup_dir = Path(tmp) / "bk"
            out_dir = backup_dir / "ByChild"
            rel_path = "Alex/Albums/2026/05/Park/001.jpg"
            src_img = backup_dir / rel_path
            src_img.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (400, 300), color=(64, 128, 192)).save(src_img, format="JPEG")

            db_path = backup_dir / "Metadata" / "faces_index.sqlite"
            with FacesIndexDB(db_path) as db:
                leo_vec = _basis_vec(0)
                det_box = BoundingBox(top=20, right=120, bottom=120, left=20)
                saved = db.upsert_scanned_batch(
                    [
                        (
                            PhotoProvenance(
                                rel_path,
                                "Alex",
                                "2026",
                                "05",
                                "Park",
                                "001.jpg",
                                album_date="2026-05-10",
                            ),
                            400,
                            300,
                            src_img.stat().st_size,
                            src_img.stat().st_mtime_ns,
                            "scrfd",
                            [DetectedFace(box=det_box, embedding=leo_vec, confidence=0.96)],
                        )
                    ]
                )
                p_id, det_fids = saved[0]
                leo_id = db.ensure_person("Leo", role="classmate")
                db.tag_single_face(photo_id=p_id, face_id=det_fids[0], person_id=leo_id, status="confirmed")

                ctx = _AppContext(
                    backup_dir=backup_dir,
                    output_dir=out_dir,
                    db=db,
                    csrf_token="test-manual-token",
                    rematch_debounce_sec=0.1,
                )
                httpd = _FacesHTTPServer(("127.0.0.1", 0), ctx)
                port = int(httpd.server_address[1])
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{port}"
                headers = {"X-KP-Token": "test-manual-token"}

                try:
                    # 1. Draw a manual box for occluded Maya via POST /api/faces/manual:
                    r_man = requests.post(
                        f"{base_url}/api/faces/manual",
                        json={
                            "photo_id": p_id,
                            "box": {"left": 210, "top": 40, "right": 310, "bottom": 160},
                            "new_name": "Maya",
                        },
                        headers=headers,
                        timeout=5,
                    )
                    r_man.raise_for_status()
                    man_payload = r_man.json()
                    man_fid = int(man_payload["face_id"])
                    maya_id = int(man_payload["person_id"])

                    faces_by_id = {int(f["face_id"]): f for f in man_payload["photo"]["faces"]}
                    self.assertIn(man_fid, faces_by_id)
                    self.assertTrue(faces_by_id[man_fid]["is_manual"])
                    self.assertEqual(faces_by_id[man_fid]["person_name"], "Maya")
                    self.assertEqual(faces_by_id[man_fid]["status"], "confirmed")
                    self.assertEqual(man_payload["overview"]["faces_total"], 2)
                    self.assertEqual(man_payload["overview"]["faces_categorized"], 2)

                    # 2. Thumbnail crop (/media/face/<id>.jpg) works out-of-the-box via PIL:
                    r_thumb = requests.get(f"{base_url}/media/face/{man_fid}.jpg", timeout=5)
                    r_thumb.raise_for_status()
                    self.assertEqual(r_thumb.headers.get("Content-Type"), "image/jpeg")
                    self.assertGreater(len(r_thumb.content), 200)

                    # 3. Hardlink export (ByChild/Maya/) automatically includes the manually tagged photo:
                    r_mat = requests.post(
                        f"{base_url}/api/materialize",
                        json={"person": "Maya"},
                        headers=headers,
                        timeout=5,
                    )
                    r_mat.raise_for_status()
                    self.assertEqual(_await_job_result(base_url, r_mat)["photos_matched"], 1)
                    maya_files = list((out_dir / "Maya").glob("*.jpg"))
                    self.assertEqual(len(maya_files), 1)

                    # 4. AI Biometric Safety: zero-vector manual face is excluded from load_all_faces_matrix()
                    #    and never receives an embedding or `is_anchor = 1` in warm or cold paths:
                    mat = db.load_all_faces_matrix()
                    self.assertEqual(mat["face_ids"].tolist(), [det_fids[0]])

                    anch_warm = db.load_anchors_and_negatives()
                    man_human_warm = [h for h in anch_warm["human"] if h["face_id"] == man_fid]
                    self.assertEqual(len(man_human_warm), 1)
                    self.assertIsNone(man_human_warm[0]["emb"])

                    db.invalidate_faces_matrix_cache()
                    anch_cold = db.load_anchors_and_negatives()
                    man_human_cold = [h for h in anch_cold["human"] if h["face_id"] == man_fid]
                    self.assertEqual(len(man_human_cold), 1)
                    self.assertIsNone(man_human_cold[0]["emb"])

                    recompute_active_matching(db)
                    rebuild_yearly_clusters(db)
                    a_row = db._conn.execute(
                        "SELECT status, is_anchor FROM assignments WHERE photo_id = ? AND person_id = ?",
                        (p_id, maya_id),
                    ).fetchone()
                    self.assertIsNotNone(a_row)
                    self.assertEqual(str(a_row["status"]), "confirmed")
                    self.assertEqual(int(a_row["is_anchor"]), 0)

                    # 5. Untagging a manually drawn box cleanly deletes the manual face row & cached thumb:
                    r_untag = requests.post(
                        f"{base_url}/api/faces/untag",
                        json={"photo_id": p_id, "face_id": man_fid},
                        headers=headers,
                        timeout=5,
                    )
                    r_untag.raise_for_status()
                    untag_data = r_untag.json()
                    rem_fids = [int(f["face_id"]) for f in untag_data["photo"]["faces"]]
                    self.assertEqual(rem_fids, [det_fids[0]])
                    self.assertEqual(untag_data["overview"]["faces_total"], 1)
                    self.assertEqual(untag_data["overview"]["faces_categorized"], 1)
                finally:
                    httpd.shutdown()
                    httpd.server_close()

    def test_birthdays_classes_pressure_child_clusters_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "bk" / "Metadata" / "faces_index.sqlite"
            with FacesIndexDB(db_path) as db:
                specs = [
                    # (feed, year, month, album_date, embedding jitter index)
                    ("Alex", "2026", "01", "2026-01-15", 0),  # confirmed anchor for Leo
                    ("Alex", "2025", "02", "2025-02-10", 1),  # before Leo's birthday
                    ("Alex", "2026", "03", "2026-03-05", 2),  # inside Leo's class window
                    ("Maya", "2026", "03", "2026-03-06", 3),  # other classroom feed
                    ("Alex", "2026", "04", "2026-04-01", 4),
                ]
                batch: list[Any] = []
                for i, (feed, yr, mo, ad, jit) in enumerate(specs):
                    rel = f"{feed}/Albums/{yr}/{mo}/Album{i}/00{i}.jpg"
                    batch.append(
                        (
                            PhotoProvenance(rel, feed, yr, mo, f"Album{i}", f"00{i}.jpg", album_date=ad),
                            400,
                            300,
                            1000 + i,
                            10_000 + i,
                            "scrfd",
                            [
                                DetectedFace(
                                    box=BoundingBox(top=10, right=110, bottom=110, left=10),
                                    embedding=_basis_vec(0, jitter=0.05, jitter_idx=400 + jit),
                                    confidence=0.95,
                                )
                            ],
                        )
                    )
                saved = db.upsert_scanned_batch(batch)
                photo_ids = [p for p, _ in saved]
                face_ids = [f[0] for _, f in saved]
                leo = db.ensure_person("Leo", role="classmate")
                db.tag_single_face(photo_id=photo_ids[0], face_id=face_ids[0], person_id=leo)

                recompute_active_matching(db)

                def leo_photos() -> set[int]:
                    return {
                        int(r["photo_id"])
                        for r in db._conn.execute(
                            "SELECT photo_id FROM assignments WHERE person_id = ? AND status IN ('confirmed', 'auto_high', 'candidate')",
                            (leo,),
                        ).fetchall()
                    }

                self.assertEqual(leo_photos(), set(photo_ids))

                # 1. Birthday removes earlier tags immediately and blocks re-matching / manual tags:
                upd = db.update_person(leo, birth_date="2025-06-01")
                self.assertEqual(upd["birth_date"], "2025-06-01")
                self.assertEqual(upd["birthday_removed"], 1)
                self.assertNotIn(photo_ids[1], leo_photos())
                recompute_active_matching(db)
                self.assertNotIn(photo_ids[1], leo_photos())
                with self.assertRaises(ValueError):
                    db.tag_single_face(photo_id=photo_ids[1], face_id=face_ids[1], person_id=leo)

                # 2. Class membership restricts auto-matching to the class feed & date window:
                cls = db.create_class(
                    name="Sunflowers 2025-26",
                    archive_child="Alex",
                    start_date="2025-09-01",
                    end_date="2026-08-31",
                )
                db.set_class_members(int(cls["class_id"]), [leo])
                ov = db.get_archive_overview()
                self.assertEqual(ov["classes"][0]["member_ids"], [leo])
                leo_ov = next(p for p in ov["people"] if p["person_id"] == leo)
                self.assertEqual(leo_ov["class_ids"], [cls["class_id"]])
                recompute_active_matching(db)
                self.assertEqual(leo_photos(), {photo_ids[0], photo_ids[2], photo_ids[4]})
                matrix = db.get_class_matrix()
                self.assertEqual(matrix["counts"][str(cls["class_id"])][str(leo)], 3)
                self.assertEqual(db.auto_create_classes(), 0)  # < 5 photos per feed-year bucket

                # 3. Per-child clusters + soft penalize pressure + hard exclude:
                cl = db.get_person_face_clusters(leo, threshold=0.5, min_size=2)
                self.assertEqual(cl["total_faces"], 3)
                all_fids = {f["face_id"] for c in cl["clusters"] for f in c["faces"]}
                self.assertEqual(all_fids, {face_ids[0], face_ids[2], face_ids[4]})

                n_pen = db.apply_person_cluster_action(
                    person_id=leo, action="penalize", faces=[(photo_ids[4], face_ids[4])]
                )
                self.assertEqual(n_pen, 1)
                sim_before = float(
                    db._conn.execute(
                        "SELECT similarity FROM assignments WHERE photo_id = ? AND person_id = ?",
                        (photo_ids[4], leo),
                    ).fetchone()["similarity"]
                )
                recompute_active_matching(db)
                row_after = db._conn.execute(
                    "SELECT similarity, status FROM assignments WHERE photo_id = ? AND person_id = ?",
                    (photo_ids[4], leo),
                ).fetchone()
                self.assertIsNotNone(row_after)
                self.assertLess(float(row_after["similarity"]), sim_before - 0.05)
                cl2 = db.get_person_face_clusters(leo, threshold=0.5, min_size=2)
                pen_faces = [f for c in cl2["clusters"] for f in c["faces"] if f["pressure"] < 0]
                self.assertEqual([f["face_id"] for f in pen_faces], [face_ids[4]])

                db.apply_person_cluster_action(
                    person_id=leo, action="exclude", faces=[(photo_ids[2], face_ids[2])]
                )
                recompute_active_matching(db)
                rej = db._conn.execute(
                    "SELECT status, source FROM assignments WHERE photo_id = ? AND person_id = ?",
                    (photo_ids[2], leo),
                ).fetchone()
                self.assertEqual((str(rej["status"]), str(rej["source"])), ("rejected", "reassigned"))

                # 4. Deleting the profile leaves faces uncategorized and clears counters:
                res = db.delete_person(leo)
                self.assertGreaterEqual(res["removed_assignments"], 2)
                self.assertEqual(
                    db._conn.execute("SELECT COUNT(*) AS n FROM assignments").fetchone()["n"], 0
                )
                self.assertEqual(db.get_archive_overview()["people"], [])
                self.assertEqual(db.list_classes()[0]["member_ids"], [])

    def test_removed_tag_soft_negative_pushes_down_comparable_lookalikes(self) -> None:
        def unit(vals: dict[int, float]) -> tuple[float, ...]:
            arr = np.zeros((512,), dtype=np.float32)
            for k, v in vals.items():
                arr[k] = v
            arr /= float(np.linalg.norm(arr))
            return tuple(float(x) for x in arr.tolist())

        anchor = unit({0: 1.0})
        removed = unit({0: 1.0, 1: 1.0})  # sim 0.707 to the anchor
        # Equally close (0.60) to the child's anchor and to the removed face: below the hard
        # lookalike rule (needs neg > pos + 0.03) but caught by the soft negative tier.
        target = unit({0: 0.60, 1: 0.2485, 2: 0.7601})

        def run(with_rejection: bool) -> float:
            with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
                batch: list[Any] = []
                for i, vec in enumerate((anchor, removed, target)):
                    rel = f"Alex/Albums/2026/0{i + 1}/A{i}/00{i}.jpg"
                    batch.append(
                        (
                            PhotoProvenance(rel, "Alex", "2026", f"0{i + 1}", f"A{i}", f"00{i}.jpg"),
                            400,
                            300,
                            10 + i,
                            20 + i,
                            "scrfd",
                            [
                                DetectedFace(
                                    box=BoundingBox(top=10, right=110, bottom=110, left=10),
                                    embedding=vec,
                                    confidence=0.95,
                                )
                            ],
                        )
                    )
                saved = db.upsert_scanned_batch(batch)
                leo = db.ensure_person("Leo")
                db.tag_single_face(photo_id=saved[0][0], face_id=saved[0][1][0], person_id=leo)
                if with_rejection:
                    db.tag_single_face(photo_id=saved[1][0], face_id=saved[1][1][0], person_id=leo)
                    db.untag_face(photo_id=saved[1][0], face_id=saved[1][1][0])
                recompute_active_matching(db)
                row = db._conn.execute(
                    "SELECT similarity FROM assignments WHERE photo_id = ? AND person_id = ?",
                    (saved[2][0], leo),
                ).fetchone()
                return float(row["similarity"]) if row is not None else 0.0

        base = run(False)
        pushed = run(True)
        self.assertGreater(base, 0.55)
        self.assertLess(pushed, base - 0.04)

    def test_temporal_cohesion_boosts_faces_near_confirmed_days(self) -> None:
        def unit(vals: dict[int, float]) -> tuple[float, ...]:
            arr = np.zeros((512,), dtype=np.float32)
            for k, v in vals.items():
                arr[k] = v
            arr /= float(np.linalg.norm(arr))
            return tuple(float(x) for x in arr.tolist())

        # Leo's usual look (many confirmed January anchors) vs. a one-off "costume day" look on Mar 10:
        usual = [unit({0: 1.0, 100 + i: 0.15}) for i in range(10)]
        costume = unit({0: 0.45, 1: 0.89})
        lookalike = unit({0: 0.45, 1: 0.80, 2: 0.39})  # sim ~0.97 to the costume face
        specs: list[tuple[str, tuple[float, ...]]] = [
            *[(f"2026-01-{10 + i:02d}", v) for i, v in enumerate(usual)],
            ("2026-03-10", costume),  # confirmed
            ("2026-03-11", lookalike),  # next day -> temporal boost
            ("2026-06-20", lookalike),  # months later -> no temporal boost
        ]
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            batch: list[Any] = []
            for i, (ad, vec) in enumerate(specs):
                mo = ad[5:7]
                batch.append(
                    (
                        PhotoProvenance(
                            f"Alex/Albums/2026/{mo}/A{i}/{i:03d}.jpg", "Alex", "2026", mo, f"A{i}", f"{i:03d}.jpg",
                            album_date=ad,
                        ),
                        400,
                        300,
                        100 + i,
                        200 + i,
                        "scrfd",
                        [
                            DetectedFace(
                                box=BoundingBox(top=10, right=110, bottom=110, left=10),
                                embedding=vec,
                                confidence=0.95,
                            )
                        ],
                    )
                )
            saved = db.upsert_scanned_batch(batch)
            leo = db.ensure_person("Leo")
            for p_id, f_ids in saved[:11]:
                db.tag_single_face(photo_id=p_id, face_id=f_ids[0], person_id=leo)

            def sim_of(idx: int) -> float:
                row = db._conn.execute(
                    "SELECT similarity FROM assignments WHERE photo_id = ? AND person_id = ?",
                    (saved[idx][0], leo),
                ).fetchone()
                return float(row["similarity"]) if row is not None else 0.0

            recompute_active_matching(db, temporal_window_days=0)
            near_off, far_off = sim_of(11), sim_of(12)
            recompute_active_matching(db)
            near_on, far_on = sim_of(11), sim_of(12)
            self.assertAlmostEqual(far_on, far_off, places=3)
            self.assertGreater(near_on, near_off + 0.02)
            self.assertGreater(near_on, far_on + 0.02)

    def test_learned_label_vote_trusts_agreeing_matches_and_vetoes_background(self) -> None:
        def unit(vals: dict[int, float]) -> tuple[float, ...]:
            arr = np.zeros((512,), dtype=np.float32)
            for k, v in vals.items():
                arr[k] = v
            arr /= float(np.linalg.norm(arr))
            return tuple(float(x) for x in arr.tolist())

        leo_faces = [unit({0: 1.0, 100 + i: 0.12}) for i in range(20)]
        bg_faces = [unit({0: 0.8, 5: 0.6, 200 + i: 0.12}) for i in range(20)]
        open_leo = unit({0: 1.0, 300: 0.12})
        open_bg = unit({0: 0.8, 5: 0.6, 301: 0.12})
        vecs = [*leo_faces, *bg_faces, open_leo, open_bg]
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            batch: list[Any] = []
            for i, vec in enumerate(vecs):
                mo = f"{(i % 12) + 1:02d}"
                batch.append(
                    (
                        PhotoProvenance(
                            f"Alex/Albums/2026/{mo}/A{i}/{i:03d}.jpg", "Alex", "2026", mo, f"A{i}", f"{i:03d}.jpg"
                        ),
                        400,
                        300,
                        100 + i,
                        200 + i,
                        "scrfd",
                        [
                            DetectedFace(
                                box=BoundingBox(top=10, right=110, bottom=110, left=10),
                                embedding=vec,
                                confidence=0.95,
                            )
                        ],
                    )
                )
            saved = db.upsert_scanned_batch(batch)
            leo = db.ensure_person("Leo")
            bg = db.ensure_person("_Background", role="ignore")
            for p_id, f_ids in saved[:20]:
                db.tag_single_face(photo_id=p_id, face_id=f_ids[0], person_id=leo)
            for p_id, f_ids in saved[20:40]:
                db.tag_single_face(photo_id=p_id, face_id=f_ids[0], person_id=bg)

            def row_of(idx: int) -> Any:
                return db._conn.execute(
                    "SELECT status, source FROM assignments WHERE photo_id = ? AND person_id = ?",
                    (saved[idx][0], leo),
                ).fetchone()

            recompute_active_matching(db)
            on_leo = row_of(40)
            self.assertEqual((str(on_leo["status"]), str(on_leo["source"])), ("auto_high", "model_trusted"))
            self.assertIsNone(row_of(41))  # vetoed: nearest labelled faces are background
            leo_ov = next(p for p in db.list_people() if p["person_id"] == leo)
            self.assertEqual(leo_ov["review_ready_count"], 0)

            # Incremental neighbour cache: labelling one more face must not require a cold rebuild
            # and must still produce the same decisions.
            cache_before = db._knn_neighbor_cache
            db.tag_single_face(photo_id=saved[40][0], face_id=saved[40][1][0], person_id=leo)
            recompute_active_matching(db)
            self.assertIs(db._knn_neighbor_cache, cache_before)
            self.assertIsNone(row_of(41))

    def test_batch_proposal_and_matrix_load_do_not_deadlock(self) -> None:
        # Regression: get_batch_proposal(mode="impact") used to hold `_lock` while taking
        # `_faces_matrix_lock`, the opposite order of load_all_faces_matrix() (bg rematch worker).
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            batch: list[Any] = []
            for i in range(6):
                batch.append(
                    (
                        PhotoProvenance(f"Alex/Albums/2026/05/A/{i:03d}.jpg", "Alex", "2026", "05", "A", f"{i:03d}.jpg"),
                        800,
                        600,
                        1000 + i,
                        1 + i,
                        "scrfd",
                        [
                            DetectedFace(
                                box=BoundingBox(top=15, right=115, bottom=115, left=15),
                                embedding=_basis_vec(i % 2),
                                confidence=0.95,
                            )
                        ],
                    )
                )
            saved = db.upsert_scanned_batch(batch)
            pid = db.ensure_person("Leo")
            db.replace_model_assignments(
                [(ph, pid, fids[0], "candidate", 0.40, "model_match") for ph, fids in saved], set()
            )
            db.load_all_faces_matrix()
            orig_mask = db._query_suppressed_mask

            def slow_mask(face_ids: Any) -> Any:
                time.sleep(0.3)  # widen the window between the two lock acquisitions
                return orig_mask(face_ids)

            db._query_suppressed_mask = slow_mask  # type: ignore[method-assign]
            done = {"a": False, "b": False}

            def run_a() -> None:
                time.sleep(0.05)
                db.get_batch_proposal(pid, batch_size=4, mode="impact")
                done["a"] = True

            def run_b() -> None:
                db.load_all_faces_matrix()
                done["b"] = True

            threads = [threading.Thread(target=run_a, daemon=True), threading.Thread(target=run_b, daemon=True)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)
            self.assertEqual(done, {"a": True, "b": True})

    def test_people_classes_and_cluster_http_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup_dir = Path(tmp) / "bk"
            db_path = backup_dir / "Metadata" / "faces_index.sqlite"
            with FacesIndexDB(db_path) as db:
                saved = db.upsert_scanned_batch(
                    [
                        (
                            PhotoProvenance(
                                f"Alex/Albums/2026/0{i + 1}/A{i}/00{i}.jpg",
                                "Alex",
                                "2026",
                                f"0{i + 1}",
                                f"A{i}",
                                f"00{i}.jpg",
                                album_date=f"2026-0{i + 1}-10",
                            ),
                            400,
                            300,
                            100 + i,
                            200 + i,
                            "scrfd",
                            [
                                DetectedFace(
                                    box=BoundingBox(top=10, right=110, bottom=110, left=10),
                                    embedding=_basis_vec(0, jitter=0.05, jitter_idx=300 + i),
                                    confidence=0.95,
                                )
                            ],
                        )
                        for i in range(4)
                    ]
                )
                sofia = db.ensure_person("Sofia", role="family")
                db.tag_single_face(photo_id=saved[0][0], face_id=saved[0][1][0], person_id=sofia)
                recompute_active_matching(db)

                ctx = _AppContext(
                    backup_dir=backup_dir,
                    output_dir=backup_dir / "ByChild",
                    db=db,
                    csrf_token="tok-v5",
                    rematch_debounce_sec=0.05,
                )
                httpd = _FacesHTTPServer(("127.0.0.1", 0), ctx)
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{int(httpd.server_address[1])}"
                hdr = {"X-KP-Token": "tok-v5"}
                try:
                    r = requests.get(f"{base}/api/people/{sofia}/clusters?threshold=0.5&min_size=2", timeout=5)
                    r.raise_for_status()
                    clusters = r.json()["clusters"]
                    self.assertEqual(sum(c["size"] for c in clusters), 4)
                    faces = [{"photo_id": f["photo_id"], "face_id": f["face_id"]} for f in clusters[0]["faces"]]

                    r = requests.post(
                        f"{base}/api/people/{sofia}/cluster-action",
                        json={"action": "boost", "faces": faces},
                        headers=hdr,
                        timeout=5,
                    )
                    r.raise_for_status()
                    self.assertEqual(r.json()["applied"], len(faces))

                    r = requests.patch(
                        f"{base}/api/people/{sofia}",
                        json={"birth_date": "2026-02-01"},
                        headers=hdr,
                        timeout=5,
                    )
                    self.assertEqual(r.status_code, 200)
                    self.assertEqual(r.json()["birthday_removed"], 1)
                    bad = requests.patch(
                        f"{base}/api/people/{sofia}", json={"birth_date": "not-a-date"}, headers=hdr, timeout=5
                    )
                    self.assertEqual(bad.status_code, 400)

                    r = requests.post(
                        f"{base}/api/classes",
                        json={"name": "Daisies", "archive_child": "Alex", "start_date": "2025-09-01", "end_date": "2026-08-31"},
                        headers=hdr,
                        timeout=5,
                    )
                    r.raise_for_status()
                    cid = int(r.json()["class"]["class_id"])
                    r = requests.post(
                        f"{base}/api/classes/{cid}/members", json={"person_ids": [sofia]}, headers=hdr, timeout=5
                    )
                    r.raise_for_status()
                    self.assertEqual(r.json()["class"]["member_ids"], [sofia])
                    r = requests.patch(f"{base}/api/classes/{cid}", json={"name": "Daisies B"}, headers=hdr, timeout=5)
                    self.assertEqual(r.json()["class"]["name"], "Daisies B")
                    mx = requests.get(f"{base}/api/classes/matrix", timeout=5).json()
                    self.assertEqual(mx["classes"][0]["photo_count"], 4)

                    r = requests.delete(f"{base}/api/classes/{cid}", headers=hdr, timeout=5)
                    r.raise_for_status()
                    self.assertEqual(r.json()["overview"]["classes"], [])
                    self.assertEqual(requests.delete(f"{base}/api/people/{sofia}", timeout=5).status_code, 403)
                    r = requests.delete(f"{base}/api/people/{sofia}", headers=hdr, timeout=5)
                    r.raise_for_status()
                    self.assertEqual(r.json()["overview"]["people"], [])
                    self.assertEqual(requests.delete(f"{base}/api/people/{sofia}", headers=hdr, timeout=5).status_code, 404)
                finally:
                    httpd.shutdown()
                    httpd.server_close()


if __name__ == "__main__":
    unittest.main()
