"""Unit tests for Bookmarklet-compatible filename sanitization, `Metadata/` layout, and resume logic."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from kinderpedia_backup.models import ChildContext
from kinderpedia_backup.sync import (
    AdaptiveRateController,
    KinderpediaClient,
    StatusTracker,
    assign_album_folder_names,
    backup_child_context,
    candidate_image_urls,
    cleanup_stale_parts,
    parse_date_prefix,
    plan_video_targets,
    resolve_child_folder_names,
    sanitize_extension,
    sanitize_name,
    split_year_month,
)


class SyncHelpersTests(unittest.TestCase):
    def test_sanitize_name_matches_bookmarklet_conventions(self) -> None:
        self.assertEqual(sanitize_name('“Serbare”: Grupa <Mare>? 🌟...'), "Serbare_ Grupa _Mare_")
        self.assertEqual(sanitize_name("CON"), "_CON")
        self.assertEqual(sanitize_name(".."), "unnamed")
        self.assertEqual(sanitize_name("."), "unnamed")
        self.assertEqual(sanitize_name("../../../etc/passwd"), "_.._.._etc_passwd")
        self.assertEqual(sanitize_name(""), "unnamed")
        # Default max_len=60 matching bookmarklet sanitizeName(name, maxLen = 60)
        self.assertEqual(len(sanitize_name("a" * 90)), 60)

    def test_sanitize_extension_strips_slashes_and_dots(self) -> None:
        self.assertEqual(sanitize_extension(".JPG"), "jpg")
        self.assertEqual(sanitize_extension("../../sh"), "sh")
        self.assertEqual(sanitize_extension(""), "bin")

    def test_candidate_image_urls_preserves_priority_and_normalizes(self) -> None:
        img = {
            "download_url": "https://images.kinderpedia.co/files/photos/1.jpg",
            "fullsize": "https://images.kinderpedia.co/900x1200/files/photos/1.jpg",
            "thumb": "//images.kinderpedia.co/300x300/files/photos/1.jpg",
            "url": "/relative/path.jpg",
        }
        urls = candidate_image_urls(img)
        self.assertEqual(
            urls,
            [
                "https://images.kinderpedia.co/files/photos/1.jpg",
                "https://images.kinderpedia.co/900x1200/files/photos/1.jpg",
                "https://images.kinderpedia.co/relative/path.jpg".replace("images.", "apiv20."),
                "https://images.kinderpedia.co/300x300/files/photos/1.jpg",
            ],
        )

    def test_parse_date_prefix_formats(self) -> None:
        d1, ts1 = parse_date_prefix("2025-11-06 14:30:00")
        self.assertEqual(d1, "2025-11-06")
        self.assertIsNotNone(ts1)

        d2, _ = parse_date_prefix("2023-12-14T09:15:00Z")
        self.assertEqual(d2, "2023-12-14")

        d3, ts3 = parse_date_prefix(None)
        self.assertEqual(d3, "unknown_date")
        self.assertIsNone(ts3)
        self.assertEqual(split_year_month("2025-11-06"), ("2025", "11"))
        self.assertEqual(split_year_month("unknown_date"), ("Unknown", "00"))

    def test_resolve_child_folder_names_short_and_disambiguated(self) -> None:
        siblings = [
            ChildContext(child_id=1, kindergarten_id=10, name="Alex Smith", first_name="Alex"),
            ChildContext(child_id=2, kindergarten_id=10, name="Maya Smith", first_name="Maya"),
        ]
        self.assertEqual(
            resolve_child_folder_names(siblings),
            {(1, 10): "Alex", (2, 10): "Maya"},
        )

        multi_school = [
            ChildContext(child_id=1, kindergarten_id=10, name="Alex Smith", first_name="Alex", kindergarten_name="Nursery"),
            ChildContext(child_id=1, kindergarten_id=20, name="Alex Smith", first_name="Alex", kindergarten_name="Primary"),
        ]
        self.assertEqual(
            resolve_child_folder_names(multi_school),
            {(1, 10): "Alex (Nursery)", (1, 20): "Alex (Primary)"},
        )

    def test_assign_album_folder_names_year_month_and_same_day_collisions(self) -> None:
        # Input in DESC order (like Kinderpedia gallery pagination): chronological #1 is 09:00, #2 is 15:00
        albums = [
            {"id": 200, "name": "Joaca afara!", "date": "2025-06-12T15:00:00+0300"},
            {"id": 100, "name": "Joaca afara!", "date": "2025-06-12T09:00:00+0300"},
            {"id": 300, "name": "Atelier pictura", "date": "2025-06-12T11:00:00+0300"},
        ]
        assigned = assign_album_folder_names(albums)
        self.assertEqual(assigned[100], "2025/06/2025-06-12 Joaca afara!")
        self.assertEqual(assigned[200], "2025/06/2025-06-12 Joaca afara! (2)")
        self.assertEqual(assigned[300], "2025/06/2025-06-12 Atelier pictura")

    def test_plan_video_targets_matches_single_multi_and_standalone_days(self) -> None:
        albums = [
            {"id": 10, "name": "Serbare de vara", "date": "2025-06-12T10:00:00+0300", "owner": {"user_id": 5}},
            {"id": 20, "name": "Activitate matematica", "date": "2025-06-13T09:30:00+0300", "owner": {"user_id": 5}},
            {"id": 21, "name": "Pictură cu acuarele", "date": "2025-06-13T11:30:00+0300", "owner": {"user_id": 5}},
        ]
        folder_map = assign_album_folder_names(albums)
        videos = [
            # 1-album day + duplicate title on same day -> Song.mp4 & Song (2).mp4
            {"id": 902, "name": "Song", "date": "2025-06-12T10:20:00+0300"},
            {"id": 901, "name": "Song", "date": "2025-06-12T10:05:00+0300"},
            # 2-album day -> matches "Pictură cu acuarele" via diacritic-stripped stem ("pictura")
            {"id": 903, "name": "Pictura - prezentare", "date": "2025-06-13T11:35:00+0300", "owner": {"user_id": 5}},
            # 0-album day -> standalone dated folder under <Year>/<Month>/
            {"id": 904, "name": "Dans in curte", "date": "2025-07-04T10:00:00+0300"},
        ]
        targets = plan_video_targets(videos, albums, folder_map)
        self.assertEqual(targets[901], ("2025/06/2025-06-12 Serbare de vara", "Song.mp4"))
        self.assertEqual(targets[902], ("2025/06/2025-06-12 Serbare de vara", "Song (2).mp4"))
        self.assertEqual(targets[903], ("2025/06/2025-06-13 Pictură cu acuarele", "Pictura - prezentare.mp4"))
        self.assertEqual(targets[904], ("2025/07/2025-07-04 Dans in curte", "Dans in curte.mp4"))

    def test_cleanup_stale_parts_removes_only_part_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep = root / "photo.jpg"
            part1 = root / "photo.jpg.123.456.part"
            part2 = root / "video.123.456.part.mp4"
            venv_part = root / ".venv" / "ignored.part"
            venv_part.parent.mkdir()
            for p in (keep, part1, part2, venv_part):
                p.write_bytes(b"data")

            removed = cleanup_stale_parts(root)
            self.assertEqual(removed, 2)
            self.assertTrue(keep.exists())
            self.assertFalse(part1.exists())
            self.assertFalse(part2.exists())
            self.assertTrue(venv_part.exists())
            _ = os.listdir(root)

    def test_backup_child_context_matches_bookmarklet_metadata_format_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup_dir = root / "backup"
            state_dir = root / "state"

            client = MagicMock(spec=KinderpediaClient)
            client.rate_ctrl = AdaptiveRateController(state_dir / "rate.json")

            def fake_list_albums(
                kg_id: int, child_id: int, *, partial_file: Path | None = None, known: Any = None
            ) -> list[dict[str, Any]]:
                items = [{"id": 555, "name": "Summer Fest", "date": "2025-06-12", "count_images": 2}]
                if partial_file is not None:
                    partial_file.parent.mkdir(parents=True, exist_ok=True)
                    partial_file.write_text(json.dumps({"complete": True, "items": items}), encoding="utf-8")
                return items

            def fake_list_videos(
                kg_id: int, child_id: int, *, partial_file: Path | None = None, known_ids: Any = None
            ) -> list[dict[str, Any]]:
                return [{"id": 777, "name": "Song", "date": "2025-06-12", "url": "https://cdn.kinderpedia.co/v.mp4"}]

            def fake_list_docs(
                kg_id: int, child_id: int, *, partial_file: Path | None = None, known_ids: Any = None
            ) -> list[dict[str, Any]]:
                return [{"id": 888, "name": "Weekly Menu.pdf", "extension": ".pdf", "url": "https://cdn.kinderpedia.co/m.pdf", "size": 120}]

            client.list_albums.side_effect = fake_list_albums
            client.list_videos.side_effect = fake_list_videos
            client.list_documents.side_effect = fake_list_docs
            client.get_album_images.return_value = (
                {"id": 555},
                [
                    {"id": 101, "extension": "jpg", "download_url": "https://images.kinderpedia.co/101.jpg"},
                    {"id": 102, "extension": "jpg", "download_url": "https://images.kinderpedia.co/102.jpg"},
                ],
            )

            def fake_dl_fallbacks(urls: list[str], dest: Path, retries_per_url: int = 3) -> int:
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = b"jpeg-payload-bytes"
                dest.write_bytes(data)
                return len(data)

            def fake_dl_binary(
                url: str, dest: Path, extra_headers: dict[str, str] | None = None, retries: int = 6
            ) -> int:
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Videos require >= 10 KB for Bookmarklet skip parity
                payload = b"x" * 12_000 if dest.suffix == ".mp4" else b"d" * 200
                dest.write_bytes(payload)
                return len(payload)

            client.download_with_fallbacks.side_effect = fake_dl_fallbacks
            client.download_binary.side_effect = fake_dl_binary

            ctx = ChildContext(
                child_id=10,
                kindergarten_id=20,
                name="Alex Kid",
                first_name="Alex",
                group_name="Bears",
                kindergarten_name="Sunny School",
                avatar_url="https://images.kinderpedia.co/avatar.png",
            )
            tracker = StatusTracker(None, client.rate_ctrl, workers=2, video_workers=1)

            stats1 = backup_child_context(client, ctx, backup_dir, tracker, workers=2, video_workers=1)
            self.assertEqual(stats1.photos_downloaded, 2)
            self.assertEqual(stats1.videos_downloaded, 1)
            self.assertEqual(stats1.documents_downloaded, 1)

            # 1. Short child folder (`Alex`), no separate `Videos/`, and ZERO `.json` files inside media folders!
            child_media_dir = backup_dir / "Alex"
            self.assertTrue(child_media_dir.is_dir())
            self.assertFalse((child_media_dir / "Videos").exists())
            leaked_jsons = list(child_media_dir.rglob("*.json"))
            self.assertEqual(leaked_jsons, [])

            # 2. Verify `<Year>/<Month>/<YYYY>-<MM>-<DD> <Title>` hierarchy and co-located day video
            album_dir = child_media_dir / "Albums" / "2025" / "06" / "2025-06-12 Summer Fest"
            self.assertTrue((child_media_dir / "avatar.png").exists())
            self.assertTrue((album_dir / "001_101.jpg").exists())
            self.assertTrue((album_dir / "002_102.jpg").exists())
            self.assertTrue((album_dir / "Song.mp4").exists())
            self.assertTrue((child_media_dir / "Documents" / "Weekly Menu.pdf").exists())

            # 3. Verify all support files live under `<Backup>/Metadata/Alex/`
            meta_child = backup_dir / "Metadata" / "Alex"
            self.assertTrue((meta_child / "albums_index.json").exists())
            self.assertTrue((meta_child / "videos_index.json").exists())
            self.assertTrue((meta_child / "documents_index.json").exists())
            self.assertTrue((meta_child / "backup_progress.json").exists())
            album_meta_path = meta_child / "Albums" / "2025-06-12 Summer Fest.json"
            self.assertTrue(album_meta_path.exists())
            album_meta = json.loads(album_meta_path.read_text(encoding="utf-8"))
            self.assertTrue(album_meta["completed"])
            self.assertEqual(album_meta["photo_count"], 2)

            # 4. Second run (simulating Bookmarklet <-> Python cross-resume): one incremental listing
            #    check, but zero manifest fetches and zero downloads.
            client.list_albums.reset_mock()
            client.get_album_images.reset_mock()
            client.download_with_fallbacks.reset_mock()

            stats2 = backup_child_context(client, ctx, backup_dir, tracker, workers=2, video_workers=1)
            self.assertEqual(stats2.photos_downloaded, 0)
            self.assertEqual(stats2.photos_skipped_existing, 2)
            self.assertEqual(stats2.videos_skipped_existing, 1)
            self.assertEqual(stats2.documents_skipped_existing, 1)
            client.list_albums.assert_called_once()
            self.assertIn("known", client.list_albums.call_args.kwargs)
            client.get_album_images.assert_not_called()
            client.download_with_fallbacks.assert_not_called()

            # 5. Co-located `.mp4` does NOT fool numbered photo count when 1 photo is missing!
            (album_dir / "002_102.jpg").unlink()
            stats3 = backup_child_context(client, ctx, backup_dir, tracker, workers=2, video_workers=1)
            self.assertEqual(stats3.photos_downloaded, 1)
            self.assertEqual(stats3.photos_skipped_existing, 1)
            self.assertTrue((album_dir / "002_102.jpg").exists())


if __name__ == "__main__":
    unittest.main()
