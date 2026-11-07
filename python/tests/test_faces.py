"""Unit tests for the facial-recognition and smart photo-organization pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinderpedia_backup.faces import (
    BoundingBox,
    DetectedFace,
    FaceOrganizationPipeline,
    cosine_similarity,
    discover_backup_photos,
)


class _StubRecognizer:
    """Deterministic fake recognizer keyed by filename substring for fast offline tests."""

    def __init__(self) -> None:
        self.calls: int = 0

    @property
    def model_id(self) -> str:
        return "stub-v1"

    def detect_and_embed(self, image_path: Path) -> list[DetectedFace]:
        self.calls += 1
        name = image_path.name.lower()
        box = BoundingBox(10, 110, 110, 10)
        faces: list[DetectedFace] = []
        if "alex" in name:
            faces.append(DetectedFace(box=box, embedding=(1.0, 0.0, 0.0)))
        if "maya" in name:
            faces.append(DetectedFace(box=box, embedding=(0.0, 1.0, 0.0)))
        if "other" in name:
            faces.append(DetectedFace(box=box, embedding=(0.0, 0.0, 1.0)))
        return faces


class FacesPipelineTests(unittest.TestCase):
    def test_cosine_similarity_math(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_end_to_end_pipeline_hardlinks_and_uses_incremental_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup_dir = root / "backup"
            album_dir = backup_dir / "Alex" / "Albums" / "2025" / "06" / "2025-06-01 Trip"
            album_dir.mkdir(parents=True)

            p_alex = album_dir / "001_alex_solo.jpg"
            p_both = album_dir / "002_alex_and_maya.jpg"
            p_other = album_dir / "003_other_kid.jpg"
            for p in (p_alex, p_both, p_other):
                p.write_bytes(b"fake-jpeg-bytes")
            (album_dir / "Classroom Song.mp4").write_bytes(b"co-located-mp4-ignored-by-faces")

            refs_dir = root / "refs"
            (refs_dir / "Alex").mkdir(parents=True)
            (refs_dir / "Maya").mkdir(parents=True)
            (refs_dir / "Alex" / "ref_alex.jpg").write_bytes(b"ref-a")
            (refs_dir / "Maya" / "ref_maya.jpg").write_bytes(b"ref-m")

            out_dir = backup_dir / "ByChild"
            cache_file = root / "state" / "face_scan_cache.json"

            rec = _StubRecognizer()
            pipeline = FaceOrganizationPipeline(
                rec,
                threshold=0.80,
                link_mode="hardlink",
                cache_file=cache_file,
            )

            stats1 = pipeline.run(backup_dir=backup_dir, refs_dir=refs_dir, output_dir=out_dir)
            self.assertEqual(stats1.reference_profiles, 2)
            self.assertEqual(stats1.photos_discovered, 3)
            self.assertEqual(stats1.photos_scanned, 3)
            self.assertEqual(stats1.photos_cached, 0)
            self.assertEqual(stats1.photos_matched, 2)
            # p_alex -> Alex (1 link), p_both -> Alex + Maya (2 links) => 3 total
            self.assertEqual(stats1.links_created, 3)

            alex_files = sorted(f.name for f in (out_dir / "Alex").iterdir())
            maya_files = sorted(f.name for f in (out_dir / "Maya").iterdir())
            self.assertEqual(len(alex_files), 2)
            self.assertEqual(len(maya_files), 1)
            # Verify hardlink shares the exact same inode as source photo
            self.assertEqual((out_dir / "Maya" / maya_files[0]).stat().st_ino, p_both.stat().st_ino)

            # Ensure ByChild output directory is never re-ingested as input album photos
            self.assertEqual(len(discover_backup_photos(backup_dir, exclude_dir=out_dir)), 3)

            # Second run should hit incremental cache for all 3 album photos (only 2 ref calls)
            calls_before = rec.calls
            stats2 = pipeline.run(backup_dir=backup_dir, refs_dir=refs_dir, output_dir=out_dir)
            self.assertEqual(stats2.photos_scanned, 0)
            self.assertEqual(stats2.photos_cached, 3)
            self.assertEqual(stats2.links_created, 0)
            self.assertEqual(stats2.links_skipped_existing, 3)
            self.assertEqual(rec.calls - calls_before, 2)


if __name__ == "__main__":
    unittest.main()
