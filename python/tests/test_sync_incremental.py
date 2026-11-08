"""Incremental sync (D1) and persisted folder identity (D2) against a mocked Kinderpedia API.

Uses a real `KinderpediaClient` whose `_http_json` is replaced by an in-memory fake gallery, so the
listing early-stop logic, manifest reuse and folder-map persistence are exercised end to end
without any network access.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

from kinderpedia_backup.models import ChildContext
from kinderpedia_backup.sync import (
    FOLDER_MAP_FILE,
    KinderpediaClient,
    StatusTracker,
    backup_child_context,
    load_folder_map,
    merge_incremental_listing,
    rebuild_folder_map,
)

PAGE = 30


def _album(aid: int, updated: str, *, count: int = 2, name: str | None = None, date: str | None = None) -> dict[str, Any]:
    return {
        "id": aid,
        "name": name or f"Album {aid}",
        "date": date or f"2025-0{1 + aid % 9}-{1 + aid % 27:02d}T10:00:00+0000",
        "updated_at": updated,
        "count_images": count,
    }


def _ts(n: int) -> str:
    """Monotonic fake `updated_at` values (larger n = more recent)."""
    return f"2025-{1 + n // 600:02d}-{1 + (n // 24) % 25:02d}T{n % 24:02d}:00:00+0000"


class FakeGallery:
    """Minimal stand-in for the apiv20 web-api endpoints used by the sync engine."""

    def __init__(self, albums: list[dict[str, Any]], videos: list[dict[str, Any]] | None = None) -> None:
        self.albums = albums
        self.videos = videos or []
        self.images: dict[int, list[dict[str, Any]]] = {}
        self.calls: list[tuple[str, dict[str, str]]] = []

    def images_for(self, album: dict[str, Any]) -> list[dict[str, Any]]:
        aid = int(album["id"])
        if aid in self.images:
            return self.images[aid]
        return [
            {"id": aid * 1000 + k, "extension": "jpg", "download_url": f"https://images.example.invalid/{aid}/{k}.jpg"}
            for k in range(int(album["count_images"]))
        ]

    def __call__(self, method: str, url: str, *, params: dict[str, str] | None = None, **_: Any) -> Any:
        params = dict(params or {})
        self.calls.append((url, params))
        if url.endswith("/web-api/data/gallery") and params.get("mode") in ("albums", "videos"):
            page = int(params["page"])
            if params["mode"] == "albums":
                ordered = sorted(self.albums, key=lambda a: str(a.get("updated_at") or ""), reverse=True)
                return {"response": {"albums": ordered[(page - 1) * PAGE : page * PAGE]}}
            ordered_v = sorted(self.videos, key=lambda v: str(v.get("date") or ""), reverse=True)
            return {"response": {"videos": ordered_v[(page - 1) * PAGE : page * PAGE]}}
        if "/web-api/data/images-plugin/" in url:
            aid = int(url.rsplit("/", 1)[1])
            album = next(a for a in self.albums if int(a["id"]) == aid)
            return {"response": {"images": self.images_for(album)}}
        if url.endswith("/web-api/data/documents"):
            return {"response": {"documents": []}}
        raise AssertionError(f"unexpected API call {url} {params}")

    def count(self, pred: Callable[[str, dict[str, str]], bool]) -> int:
        return sum(1 for u, p in self.calls if pred(u, p))

    def album_pages(self) -> int:
        return self.count(lambda u, p: u.endswith("/gallery") and p.get("mode") == "albums")

    def image_calls(self) -> list[int]:
        return [int(u.rsplit("/", 1)[1]) for u, _ in self.calls if "/images-plugin/" in u]


class _Harness:
    def __init__(self, tmp: Path, api: FakeGallery) -> None:
        self.backup_dir = tmp / "backup"
        self.client = KinderpediaClient(state_dir=tmp / "state", persist_session=False)
        self.client._http_json = api  # type: ignore[method-assign]
        self.downloads: list[Path] = []
        self.binary_downloads: list[Path] = []
        self.tracker = StatusTracker(None, self.client.rate_ctrl, 2, 1)
        self.ctx = ChildContext(child_id=10, kindergarten_id=20, name="Alex Kid", first_name="Alex")

        def fake_fallbacks(urls: list[str], dest: Path) -> int:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"jpeg:" + urls[0].encode())
            self.downloads.append(dest)
            return dest.stat().st_size

        def fake_binary(url: str, dest: Path, extra_headers: Any = None, retries: int = 6) -> int:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"v" * 20_000)
            self.binary_downloads.append(dest)
            return dest.stat().st_size

        self.client.download_with_fallbacks = fake_fallbacks  # type: ignore[method-assign]
        self.client.download_binary = fake_binary  # type: ignore[method-assign]

    def run(self, **kw: Any) -> Any:
        with patch("kinderpedia_backup.sync.engine.log"), patch("kinderpedia_backup.sync.client.log"):
            return backup_child_context(
                self.client, self.ctx, self.backup_dir, self.tracker, workers=2, video_workers=1, **kw
            )

    @property
    def meta_dir(self) -> Path:
        return self.backup_dir / "Metadata" / "Alex"

    @property
    def albums_dir(self) -> Path:
        return self.backup_dir / "Alex" / "Albums"


def _library(n: int = 65) -> list[dict[str, Any]]:
    return [_album(i, _ts(i)) for i in range(1, n + 1)]


class IncrementalSyncTests(unittest.TestCase):
    def test_first_run_lists_everything_then_no_changes_costs_one_listing_page(self) -> None:
        api = FakeGallery(_library())
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            st1 = h.run()
            self.assertEqual(api.album_pages(), 3)  # 30 + 30 + 5
            self.assertEqual(len(api.image_calls()), 65)
            self.assertEqual(st1.photos_downloaded, 130)

            api.calls.clear()
            h.downloads.clear()
            st2 = h.run()
            self.assertEqual(api.album_pages(), 1, "only the first listing page is requested")
            self.assertEqual(api.image_calls(), [], "zero album-image calls")
            self.assertEqual(h.downloads, [])
            self.assertEqual(st2.photos_skipped_existing, 130)
            self.assertEqual(st2.total_failures, 0)

    def test_one_new_album_on_first_page_triggers_exactly_one_manifest_fetch(self) -> None:
        api = FakeGallery(_library())
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            h.run()
            api.albums.append(_album(500, _ts(900), count=3, name="Brand new"))
            api.calls.clear()
            h.downloads.clear()

            st = h.run()
            self.assertEqual(api.album_pages(), 2, "page 1 has the new album; page 2 is fully known")
            self.assertEqual(api.image_calls(), [500])
            self.assertEqual(st.photos_downloaded, 3)
            index = json.loads((h.meta_dir / "albums_index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index), 66)
            self.assertEqual(index[0]["id"], 500)

    def test_updated_album_is_refetched_and_renumbered_photos_are_renamed_not_redownloaded(self) -> None:
        api = FakeGallery(_library())
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            h.run()
            target = next(a for a in api.albums if a["id"] == 7)
            old_images = api.images_for(target)
            # Teacher adds one photo at the front: count_images and updated_at change.
            target["count_images"] = 3
            target["updated_at"] = _ts(901)
            api.images[7] = [
                {"id": 7999, "extension": "jpg", "download_url": "https://images.example.invalid/7/new.jpg"},
                *old_images,
            ]
            api.calls.clear()
            h.downloads.clear()

            st = h.run()
            self.assertEqual(api.image_calls(), [7])
            self.assertEqual([d.name for d in h.downloads], ["001_7999.jpg"])
            folder = next(p for p in h.albums_dir.rglob("*Album 7") if p.is_dir())
            self.assertEqual(
                sorted(p.name for p in folder.iterdir()),
                ["001_7999.jpg", "002_7000.jpg", "003_7001.jpg"],
            )
            self.assertEqual(st.photos_downloaded, 1)
            meta = json.loads((h.meta_dir / "Albums" / f"{folder.name}.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["album_listing"]["count_images"], 3)
            self.assertTrue(meta["completed"])

    def test_count_only_change_on_first_page_is_detected(self) -> None:
        api = FakeGallery(_library())
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            h.run()
            newest = max(api.albums, key=lambda a: a["updated_at"])
            newest["count_images"] = 5
            api.calls.clear()
            h.run()
            self.assertEqual(api.image_calls(), [newest["id"]])

    def test_refresh_index_forces_full_listing_and_manifest_refetch(self) -> None:
        api = FakeGallery(_library())
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            h.run()
            api.calls.clear()
            h.downloads.clear()
            st = h.run(refresh_index=True)
            self.assertEqual(api.album_pages(), 3)
            self.assertEqual(len(api.image_calls()), 65)
            self.assertEqual(h.downloads, [], "photos already on disk are not downloaded again")
            self.assertEqual(st.photos_skipped_existing, 130)

    def test_empty_manifest_for_non_empty_album_is_not_marked_complete(self) -> None:
        api = FakeGallery(_library(3))
        api.images[2] = []
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            h.run()
            api.calls.clear()
            h.run()
            self.assertEqual(api.image_calls(), [2], "an empty manifest is refetched next run")

    def test_videos_listing_stops_at_first_known_page(self) -> None:
        videos = [
            {"id": 9000 + i, "name": f"Clip {i}", "date": f"2025-02-{1 + i % 27:02d}T{i % 24:02d}:00:00+0000",
             "url": f"https://cdn.kinderpedia.co/{i}.mp4"}
            for i in range(40)
        ]
        api = FakeGallery(_library(5), videos)
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            h.run()
            self.assertEqual(api.count(lambda u, p: p.get("mode") == "videos"), 2)
            api.calls.clear()
            h.binary_downloads.clear()
            st = h.run()
            self.assertEqual(api.count(lambda u, p: p.get("mode") == "videos"), 1)
            self.assertEqual(h.binary_downloads, [])
            self.assertEqual(st.videos_skipped_existing, 40)

    def test_merge_incremental_listing(self) -> None:
        merged = merge_incremental_listing(
            [{"id": 3, "v": "new"}, {"id": 1, "v": "fresh"}],
            [{"id": 1, "v": "old"}, {"id": 2, "v": "old"}, {"name": "no id"}],
        )
        self.assertEqual([(m["id"], m["v"]) for m in merged], [(3, "new"), (1, "fresh"), (2, "old")])


class FolderMapTests(unittest.TestCase):
    def _two_siblings(self) -> list[dict[str, Any]]:
        return [
            _album(1, _ts(1), name="Outdoor play", date="2025-06-12T09:00:00+0000"),
            _album(2, _ts(2), name="Outdoor play", date="2025-06-12T15:00:00+0000"),
        ]

    def test_folder_map_is_persisted_and_sibling_deletion_does_not_rename(self) -> None:
        api = FakeGallery(self._two_siblings())
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            h.run()
            fm = load_folder_map(h.meta_dir / FOLDER_MAP_FILE)
            assert fm is not None
            self.assertEqual(fm.albums[2], "2025/06/2025-06-12 Outdoor play (2)")

            # The teacher deletes the first sibling; a full refresh must not rename the survivor.
            api.albums = [a for a in api.albums if a["id"] != 1]
            h.downloads.clear()
            api.calls.clear()
            h.run(refresh_index=True)
            self.assertEqual(h.downloads, [])
            self.assertFalse((h.albums_dir / "2025" / "06" / "2025-06-12 Outdoor play" / "001_2000.jpg").exists())
            self.assertTrue((h.albums_dir / "2025" / "06" / "2025-06-12 Outdoor play (2)" / "001_2000.jpg").exists())

            # A new same-day sibling gets the next free name, never the deleted album's folder.
            api.albums.append(_album(3, _ts(3), name="Outdoor play", date="2025-06-12T08:00:00+0000"))
            h.run()
            fm2 = load_folder_map(h.meta_dir / FOLDER_MAP_FILE)
            assert fm2 is not None
            self.assertEqual(fm2.albums[1], "2025/06/2025-06-12 Outdoor play")
            self.assertEqual(fm2.albums[3], "2025/06/2025-06-12 Outdoor play (3)")

    def test_standalone_video_folder_stays_when_an_album_lands_on_that_day(self) -> None:
        video = {"id": 77, "name": "Song", "date": "2025-08-01T08:00:00+0000", "url": "https://cdn.kinderpedia.co/v.mp4"}
        api = FakeGallery([_album(1, _ts(1), date="2025-07-01T10:00:00+0000")], [video])
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            h.run()
            standalone = h.albums_dir / "2025" / "08" / "2025-08-01 Song" / "Song.mp4"
            self.assertTrue(standalone.exists())

            api.albums.append(_album(2, _ts(2), name="Garden", date="2025-08-01T09:00:00+0000"))
            h.binary_downloads.clear()
            h.run()
            self.assertEqual(h.binary_downloads, [], "the video is not re-downloaded into the new album")
            fm = load_folder_map(h.meta_dir / FOLDER_MAP_FILE)
            assert fm is not None
            self.assertEqual(fm.videos[77], ("2025/08/2025-08-01 Song", "Song.mp4"))

    def test_existing_archive_without_map_is_rebuilt_from_album_metadata(self) -> None:
        # An older run named album 2 "(2)" while a since-deleted sibling held the base name.
        api = FakeGallery([_album(2, _ts(2), name="Outdoor play", date="2025-06-12T15:00:00+0000")])
        with tempfile.TemporaryDirectory() as tmp:
            h = _Harness(Path(tmp), api)
            leaf = "2025-06-12 Outdoor play (2)"
            album_dir = h.albums_dir / "2025" / "06" / leaf
            album_dir.mkdir(parents=True)
            for k in range(2):
                (album_dir / f"{k + 1:03d}_{2000 + k}.jpg").write_bytes(b"x")
            (h.meta_dir / "Albums").mkdir(parents=True)
            (h.meta_dir / "Albums" / f"{leaf}.json").write_text(
                json.dumps({"album_listing": api.albums[0], "images": api.images_for(api.albums[0]), "completed": True}),
                encoding="utf-8",
            )
            (h.meta_dir / "albums_index.json").write_text(json.dumps(api.albums), encoding="utf-8")

            rebuilt = rebuild_folder_map(h.meta_dir / "Albums", h.albums_dir)
            self.assertEqual(rebuilt.albums, {2: f"2025/06/{leaf}"})

            st = h.run()
            self.assertEqual(api.image_calls(), [])
            self.assertEqual(h.downloads, [])
            self.assertEqual(st.photos_skipped_existing, 2)
            fm = load_folder_map(h.meta_dir / FOLDER_MAP_FILE)
            assert fm is not None
            self.assertEqual(fm.albums[2], f"2025/06/{leaf}")
            self.assertFalse((h.albums_dir / "2025" / "06" / "2025-06-12 Outdoor play").exists())

    def test_unsafe_folder_map_entries_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / FOLDER_MAP_FILE
            path.write_text(
                json.dumps({
                    "version": 1,
                    "albums": {"1": "../../etc", "2": "2025/06/ok", "3": "/abs", "x": "2025/01/a", "4": "C:/win"},
                    "videos": {"9": {"folder": "2025/06/ok", "file": "a/b.mp4"}, "10": {"folder": "2025/06/ok", "file": "S.mp4"}},
                }),
                encoding="utf-8",
            )
            fm = load_folder_map(path)
            assert fm is not None
            self.assertEqual(fm.albums, {2: "2025/06/ok"})
            self.assertEqual(fm.videos, {10: ("2025/06/ok", "S.mp4")})


if __name__ == "__main__":
    unittest.main()
