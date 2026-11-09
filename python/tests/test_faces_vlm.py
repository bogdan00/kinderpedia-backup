"""Unit tests for the generic VLM provider interfaces, migration v8, and clothing continuity pipeline."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kinderpedia_backup.faces.db.schema import SCHEMA_VERSION
from kinderpedia_backup.faces.index_db import FacesIndexDB
from kinderpedia_backup.faces.models import BoundingBox, DetectedFace, PhotoProvenance
from kinderpedia_backup.faces.vlm.base import (
    ClothingMatchResult,
    ClothingSignature,
    VlmProvider,
)
from kinderpedia_backup.faces.vlm.loader import get_vlm_provider
from kinderpedia_backup.faces.vlm.null_provider import NullVlmProvider
from kinderpedia_backup.faces.vlm.pipeline import analyze_album_continuity


class MockVlmProvider:
    """Mock VLM provider for deterministic unit testing."""

    def __init__(
        self,
        *,
        clothing_sig: ClothingSignature | None = None,
        match_result: ClothingMatchResult | None = None,
    ) -> None:
        self._available = True
        self._clothing_sig = clothing_sig or ClothingSignature(
            clothing_summary="Yellow hoodie with blue jeans",
            top_color="yellow",
            bottom_color="blue",
            apparent_gender="girl",
        )
        self._match_result = match_result or ClothingMatchResult(
            is_match=True,
            confidence=0.92,
            apparent_gender="girl",
            gender_conflict=False,
            notes="Same yellow hoodie",
        )

    def is_available(self) -> bool:
        return self._available

    def extract_clothing(
        self,
        image_path: Path | str,
        face_box: tuple[int, int, int, int] | list[int],
        person_name: str,
        target_gender: str = "unknown",
    ) -> ClothingSignature | None:
        return self._clothing_sig

    def verify_clothing_match(
        self,
        image_path: Path | str,
        face_box: tuple[int, int, int, int] | list[int],
        target_person_name: str,
        target_gender: str,
        signature: ClothingSignature,
    ) -> ClothingMatchResult:
        return self._match_result

    def batch_verify_album_candidates(
        self,
        candidates: list[dict[str, Any]],
        target_person_name: str,
        target_gender: str,
        signature: ClothingSignature,
    ) -> list[dict[str, Any]]:
        results = []
        for c in candidates:
            results.append({
                "face_id": c.get("face_id"),
                "apparent_gender": self._match_result.apparent_gender,
                "gender_conflict": self._match_result.gender_conflict,
                "clothing_match": self._match_result.is_match,
                "confidence": self._match_result.confidence,
                "notes": self._match_result.notes,
            })
        return results


class VlmInterfacesTests(unittest.TestCase):
    def test_null_provider_behavior(self) -> None:
        null = NullVlmProvider()
        self.assertFalse(null.is_available())
        self.assertIsNone(null.extract_clothing(Path("/fake.jpg"), [0, 0, 10, 10], "Alice"))
        res = null.verify_clothing_match(
            Path("/fake.jpg"),
            [0, 0, 10, 10],
            "Alice",
            "girl",
            ClothingSignature("red shirt"),
        )
        self.assertFalse(res.is_match)
        self.assertEqual(res.confidence, 0.0)
        self.assertEqual(null.batch_verify_album_candidates([], "Alice", "girl", ClothingSignature("red shirt")), [])

    def test_loader_returns_provider(self) -> None:
        p = get_vlm_provider()
        self.assertTrue(isinstance(p, VlmProvider))

    def test_schema_version_is_eight(self) -> None:
        self.assertEqual(SCHEMA_VERSION, 8)


class VlmPipelineAndDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.db_path = self.tmp_path / "faces.sqlite"
        self.db = FacesIndexDB(self.db_path)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_gender_and_appearances_storage(self) -> None:
        # Migration v8 applied
        self.assertEqual(self.db.get_schema_version(), 8)

        # Check gender operations
        pid = self.db.ensure_person("ChildA", role="family")
        self.db.set_person_gender(pid, "girl")

        people = self.db.list_people()
        target = next(p for p in people if p["person_id"] == pid)
        self.assertEqual(target["gender"], "girl")

        # Record album appearance
        self.db.record_album_appearance(
            archive_child="ChildA",
            album_leaf="2026-11-01 Party",
            person_id=pid,
            sample_photo_id=1,
            sample_face_id=10,
            clothing_summary="Red dress with white polka dots",
            top_color="red",
            bottom_color="red",
            apparent_gender="girl",
            verified_count=2,
        )

        apps = self.db.get_album_appearances("ChildA", "2026-11-01 Party")
        self.assertEqual(len(apps), 1)
        self.assertEqual(apps[0]["clothing_summary"], "Red dress with white polka dots")
        self.assertEqual(apps[0]["person_name"], "ChildA")

    def test_pipeline_clothing_continuity_and_gender_guard(self) -> None:
        # Seed child and educator
        child_id = self.db.ensure_person("Maya", role="family")
        self.db.set_person_gender(child_id, "girl")

        # Create dummy photo files on disk
        photo_dir = self.tmp_path / "Maya" / "2026-11-01 Party"
        photo_dir.mkdir(parents=True, exist_ok=True)
        img1 = photo_dir / "p1.jpg"
        img2 = photo_dir / "p2.jpg"
        img1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        img2.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Insert photos and faces in DB
        prov1 = PhotoProvenance(
            rel_path="Maya/2026-11-01 Party/p1.jpg",
            archive_child="Maya",
            year="2026",
            month="11",
            album_leaf="2026-11-01 Party",
            filename="p1.jpg",
        )
        prov2 = PhotoProvenance(
            rel_path="Maya/2026-11-01 Party/p2.jpg",
            archive_child="Maya",
            year="2026",
            month="11",
            album_leaf="2026-11-01 Party",
            filename="p2.jpg",
        )
        f1_det = DetectedFace(box=BoundingBox(10, 50, 50, 10), embedding=(0.0,) * 512)
        f2_det = DetectedFace(box=BoundingBox(10, 50, 50, 10), embedding=(0.0,) * 512)

        batch = [
            (prov1, 200, 200, 104, 0, "test", [f1_det]),
            (prov2, 200, 200, 104, 0, "test", [f2_det]),
        ]
        ((ph1, (f1,)), (ph2, (f2,))) = self.db.upsert_scanned_batch(batch)

        # Tag f1 as confirmed for Maya
        self.db.tag_single_face(photo_id=ph1, face_id=f1, person_id=child_id, status="confirmed")

        # Assign f2 as candidate
        self.db.tag_single_face(photo_id=ph2, face_id=f2, person_id=child_id, status="candidate", source="model_match")

        # 1. Run pipeline with mock matching clothing -> upgrades f2 to confirmed (album_cohesion)
        mock_match = MockVlmProvider(
            match_result=ClothingMatchResult(
                is_match=True,
                confidence=0.95,
                apparent_gender="girl",
                gender_conflict=False,
                notes="Exact outfit match",
            )
        )
        res = analyze_album_continuity(
            self.db,
            mock_match,
            "Maya",
            "2026-11-01 Party",
            self.tmp_path,
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["verified_count"], 1)
        self.assertEqual(res["gender_rejected_count"], 0)

        # 2. Run pipeline with gender conflict -> rejects candidate
        mock_conflict = MockVlmProvider(
            match_result=ClothingMatchResult(
                is_match=False,
                confidence=0.1,
                apparent_gender="boy",
                gender_conflict=True,
                notes="Clearly depicts a boy",
            )
        )
        # Reset f2 to candidate
        self.db.tag_single_face(photo_id=ph2, face_id=f2, person_id=child_id, status="candidate", source="model_match")
        res2 = analyze_album_continuity(
            self.db,
            mock_conflict,
            "Maya",
            "2026-11-01 Party",
            self.tmp_path,
        )
        self.assertEqual(res2["gender_rejected_count"], 1)


class VlmServerEndpointsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.db = FacesIndexDB(self.tmp_path / "faces.sqlite")
        self.child_id = self.db.ensure_person("Emma", role="family")

        # Fake photo & face
        prov = PhotoProvenance(
            rel_path="Emma/2026-11-01/a.jpg",
            archive_child="Emma",
            year="2026",
            month="11",
            album_leaf="2026-11-01",
            filename="a.jpg",
        )
        face = DetectedFace(box=BoundingBox(10, 50, 50, 10), embedding=(0.0,) * 512)
        ((self.ph_id, (self.fid,)),) = self.db.upsert_scanned_batch([(prov, 200, 200, 100, 0, "test", [face])])

        # Create photo file on disk
        img = self.tmp_path / "Emma" / "2026-11-01" / "a.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        from kinderpedia_backup.faces.server import _AppContext, _FacesHTTPServer
        self.token = "tok-vlm-test"
        self.ctx = _AppContext(
            backup_dir=self.tmp_path,
            output_dir=self.tmp_path / "ByChild",
            db=self.db,
            csrf_token=self.token,
            rematch_debounce_sec=0.01,
            external_poll_sec=0.0,
        )
        # Inject mock VLM provider into ctx
        self.ctx.vlm = MockVlmProvider()

        self.httpd = _FacesHTTPServer(("127.0.0.1", 0), self.ctx)
        self.port = int(self.httpd.server_address[1])
        self.base = f"http://127.0.0.1:{self.port}"
        self.th = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.th.start()
        self.hdr = {"X-KP-Token": self.token}

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.ctx.close()
        self.db.close()
        self.tmp.cleanup()

    def test_vlm_status_endpoint(self) -> None:
        import requests
        r = requests.get(f"{self.base}/api/vlm/status", headers=self.hdr, timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["available"])
        self.assertEqual(data["provider"], "MockVlmProvider")

    def test_set_gender_endpoint(self) -> None:
        import requests
        r = requests.post(
            f"{self.base}/api/people/{self.child_id}/gender",
            json={"gender": "girl"},
            headers=self.hdr,
            timeout=5,
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["gender"], "girl")

    def test_inspect_face_endpoint(self) -> None:
        import requests
        r = requests.post(
            f"{self.base}/api/vlm/inspect-face",
            json={
                "photo_id": self.ph_id,
                "box": [10, 10, 50, 50],
                "person_name": "Emma",
                "gender": "girl",
            },
            headers=self.hdr,
            timeout=5,
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsNotNone(data["signature"])
        self.assertIn("Yellow hoodie", data["signature"]["clothing_summary"])

    def test_analyze_album_endpoint(self) -> None:
        import requests
        r = requests.post(
            f"{self.base}/api/vlm/analyze-album",
            json={"album_key": "Emma/2026-11-01"},
            headers=self.hdr,
            timeout=5,
        )
        self.assertEqual(r.status_code, 200)
        job = r.json()
        self.assertIn("job_id", job)
        self.assertEqual(job["kind"], "vlm_album")


if __name__ == "__main__":
    unittest.main()
