"""Resumable media synchronization with short names and videos co-located in day albums."""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import os
import re
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ..console import LiveProgress, log
from ..models import ChildBackupStats, ChildContext
from ..security import ensure_within_directory, redact_secrets, scrub_metadata, write_json_atomic
from .client import WEB_API_BASE, KinderpediaClient
from .naming import (
    assign_album_folder_names,
    parse_date_prefix,
    plan_video_targets,
    sanitize_extension,
    sanitize_name,
)
from .rate_limit import AdaptiveRateController

_ALLOWED_IMAGE_EXTS = frozenset({"jpg", "jpeg", "png", "webp", "heic", "gif", "mp4", "mov"})
_NUMBERED_PHOTO_RE = re.compile(r"^\d{3}_")


class _IndexFetcher(Protocol):
    def __call__(
        self,
        kindergarten_id: int,
        child_id: int,
        *,
        partial_file: Path | None = None,
    ) -> list[dict[str, Any]]: ...


def cleanup_stale_parts(root_dir: Path) -> int:
    """Delete incomplete .part / .part.mp4 files left by interrupted runs."""
    removed = 0
    if not root_dir.exists():
        return 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [
            d for d in dirnames if d not in (".venv", "venv", ".git", "__pycache__", "node_modules")
        ]
        for fn in filenames:
            if fn.endswith(".part") or fn.endswith(".part.mp4"):
                fp = Path(dirpath) / fn
                with contextlib.suppress(OSError):
                    fp.unlink()
                    removed += 1
    return removed


def set_file_mtime(path: Path, ts: float | None) -> None:
    if ts and ts > 0 and path.exists():
        with contextlib.suppress(OSError):
            os.utime(path, (ts, ts))


def candidate_image_urls(img: dict[str, Any]) -> list[str]:
    """Return ordered fallback URLs (handles pre-2024 S3 albums where download_url returns 404)."""
    urls: list[str] = []
    for key in ("download_url", "fullsize", "url_original", "url_large", "src", "url", "thumb"):
        val = img.get(key)
        if isinstance(val, str) and val.strip():
            u = val.strip()
            if u.startswith("//"):
                u = "https:" + u
            elif u.startswith("/"):
                u = WEB_API_BASE + u
            if u not in urls:
                urls.append(u)
    return urls


@dataclass
class _AlbumState:
    album_id: int
    album_name: str
    meta_file: Path
    meta_payload: dict[str, Any]
    total_photos: int
    remaining: int
    failed: int = 0


class StatusTracker:
    """Thread-safe status aggregator writing live telemetry JSON and updating TTY progress."""

    def __init__(
        self,
        path: Path | None,
        rate_ctrl: AdaptiveRateController,
        workers: int,
        video_workers: int,
        live_progress: LiveProgress | None = None,
    ) -> None:
        self.path = path
        self.rate_ctrl = rate_ctrl
        self.workers = workers
        self.video_workers = video_workers
        self.live_progress = live_progress
        self.started_at = datetime.now().isoformat()
        self.phase = "initializing"
        self.current_child = ""
        self.children_stats: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._last_write = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pid": os.getpid(),
                "started_at": self.started_at,
                "updated_at": datetime.now().isoformat(),
                "phase": self.phase,
                "current_child": self.current_child,
                "workers": {"meta": 1, "photos": self.workers, "videos": self.video_workers},
                "adaptive_rate": self.rate_ctrl.snapshot().to_dict(),
                "children": {k: dict(v) for k, v in self.children_stats.items()},
            }

    def update(
        self,
        *,
        phase: str | None = None,
        child: str | None = None,
        stats: ChildBackupStats | None = None,
        force: bool = False,
    ) -> None:
        should_write_disk = False
        with self._lock:
            if phase is not None:
                self.phase = phase
            if child is not None:
                self.current_child = child
            if stats is not None:
                self.children_stats[stats.context_key] = asdict(stats)
            now = time.monotonic()
            if force or (now - self._last_write) >= 2.5:
                self._last_write = now
                should_write_disk = True

        if self.live_progress is not None:
            self.live_progress.update(self, force=force)

        if should_write_disk and self.path is not None:
            with contextlib.suppress(Exception):
                write_json_atomic(self.path, self.snapshot())


def _count_numbered_photos(directory: Path) -> int:
    """Count numbered album photos (`001_<id>.jpg`), ignoring co-located `.mp4` videos."""
    if not directory.is_dir():
        return 0
    count = 0
    with contextlib.suppress(OSError):
        for entry in os.scandir(directory):
            if (
                entry.is_file()
                and _NUMBERED_PHOTO_RE.match(entry.name)
                and not entry.name.endswith(".json")
                and not entry.name.endswith(".part")
            ):
                count += 1
    return count


def backup_child_context(
    client: KinderpediaClient,
    ctx: ChildContext,
    backup_dir: Path,
    tracker: StatusTracker,
    *,
    child_folder_name: str | None = None,
    disambiguate_school: bool = False,
    workers: int = 6,
    video_workers: int = 2,
    discover_only: bool = False,
    refresh_index: bool = False,
    skip_photos: bool = False,
    skip_videos: bool = False,
    skip_documents: bool = False,
) -> ChildBackupStats:
    cid = ctx.child_id
    kid = ctx.kindergarten_id
    child_name = ctx.name
    group_name = ctx.group_name

    if child_folder_name is None:
        short = sanitize_name(ctx.short_name, 60)
        if disambiguate_school:
            school_tag = sanitize_name(ctx.kindergarten_name, 30) if ctx.kindergarten_name else f"KG_{kid}"
            child_folder_name = sanitize_name(f"{short} ({school_tag})", 60)
        else:
            child_folder_name = short

    if group_name:
        log(f"\n=== Child: {child_name} -> folder '{child_folder_name}' (ID={cid}, Kindergarten={kid}, Group={group_name}) ===")
    else:
        log(f"\n=== Child: {child_name} -> folder '{child_folder_name}' (ID={cid}, Kindergarten={kid}) ===")

    # Media directories: `Albums/` holds both photos and day-matched videos; `Documents/` holds docs
    child_dir = ensure_within_directory(backup_dir, backup_dir / child_folder_name)
    albums_dir = child_dir / "Albums"
    docs_dir = child_dir / "Documents"

    # Support metadata directories: `<Backup Root>/Metadata/<Child>/...`
    metadata_root = ensure_within_directory(backup_dir, backup_dir / "Metadata")
    child_meta_dir = ensure_within_directory(metadata_root, metadata_root / child_folder_name)
    meta_albums_dir = ensure_within_directory(child_meta_dir, child_meta_dir / "Albums")
    meta_videos_dir = ensure_within_directory(child_meta_dir, child_meta_dir / "Videos")

    if not discover_only:
        child_dir.mkdir(parents=True, exist_ok=True)
        child_meta_dir.mkdir(parents=True, exist_ok=True)

    tracker.update(phase="indexing", child=child_folder_name, force=True)

    def _load_or_fetch_index(fname: str, fetcher: _IndexFetcher) -> list[dict[str, Any]]:
        idx_file = ensure_within_directory(child_meta_dir, child_meta_dir / fname)
        stem = fname.removesuffix(".json")
        partial_file = ensure_within_directory(child_meta_dir, child_meta_dir / f"{stem}_partial.json")

        if idx_file.exists() and not refresh_index:
            with contextlib.suppress(Exception):
                cached_list = json.loads(idx_file.read_text(encoding="utf-8"))
                if isinstance(cached_list, list) and (len(cached_list) > 0 or fname == "documents_index.json"):
                    log(f"  [Resume] Loaded cached Metadata/{child_folder_name}/{fname} ({len(cached_list)} items)")
                    return cached_list

        use_partial = None if (discover_only or refresh_index) else partial_file
        items = fetcher(kid, cid, partial_file=use_partial)
        if not discover_only:
            write_json_atomic(idx_file, scrub_metadata(items))
        return items

    # Load album listings whenever photos OR videos are enabled (videos match against day albums!)
    need_album_index = (not skip_photos) or (not skip_videos)
    albums_for_matching = _load_or_fetch_index("albums_index.json", client.list_albums) if need_album_index else []
    albums = [] if skip_photos else albums_for_matching
    videos = [] if skip_videos else _load_or_fetch_index("videos_index.json", client.list_videos)
    docs = [] if skip_documents else _load_or_fetch_index("documents_index.json", client.list_documents)

    album_folder_by_id = assign_album_folder_names(albums_for_matching)
    video_targets = plan_video_targets(videos, albums_for_matching, album_folder_by_id)

    total_expected_photos = sum(int(a.get("count_images") or 0) for a in albums)
    log(
        f"Summary for {child_folder_name} (KG {kid}): "
        f"{len(albums)} photo albums (~{total_expected_photos} photos), "
        f"{len(videos)} videos (matched into Albums/), {len(docs)} documents."
    )

    stats = ChildBackupStats(
        context_key=child_folder_name,
        child_name=child_name,
        child_id=cid,
        kindergarten_id=kid,
        group_name=group_name,
        albums_count=len(albums),
        expected_photos_count=total_expected_photos,
        videos_count=len(videos),
        documents_count=len(docs),
    )
    tracker.update(stats=stats, force=True)

    if discover_only:
        return stats

    if ctx.avatar_url and not (child_dir / "avatar.png").exists():
        with contextlib.suppress(Exception):
            client.download_binary(ctx.avatar_url, ensure_within_directory(child_dir, child_dir / "avatar.png"))

    album_dirs_to_touch: dict[Path, float | None] = {}

    # 1. Serialized Adaptive Album Metadata + Gentle Parallel Photo Downloads
    if albums:
        albums_dir.mkdir(parents=True, exist_ok=True)
        meta_albums_dir.mkdir(parents=True, exist_ok=True)
        progress_file = ensure_within_directory(child_meta_dir, child_meta_dir / "backup_progress.json")

        tracker.update(phase="albums", child=child_folder_name, stats=stats, force=True)
        log(
            f"Starting adaptive backup of {len(albums)} albums for {child_folder_name} "
            f"(meta_workers=1, photo_workers={workers}, initial_pace={client.rate_ctrl.interval:.2f}s)..."
        )
        prog_lock = threading.Lock()
        in_flight_sem = threading.Semaphore(max(16, workers * 8))
        live_done = 0

        def _finalize_album(astate: _AlbumState) -> None:
            is_complete = astate.failed == 0 and astate.total_photos >= 0
            astate.meta_payload["completed"] = is_complete
            astate.meta_payload["completed_at"] = datetime.now().isoformat()
            astate.meta_payload["photo_count"] = astate.total_photos
            write_json_atomic(astate.meta_file, astate.meta_payload)
            write_json_atomic(
                progress_file,
                {
                    "last_completed_album_id": astate.album_id,
                    "last_completed_album_name": astate.album_name,
                    "total_photos_processed": stats.photos_downloaded + stats.photos_skipped_existing,
                    "updated_at": datetime.now().isoformat(),
                },
            )

        def _prepare_album(
            album: dict[str, Any],
        ) -> tuple[bool, bool, _AlbumState, list[tuple[list[str], Path, float | None]]]:
            aid = int(album["id"])
            aname = str(album.get("name") or f"album_{aid}")
            expected_cnt = int(album.get("count_images") or 0)
            adate_raw = str(album.get("date") or album.get("updated_at") or "")
            _, album_ts = parse_date_prefix(adate_raw)

            folder_rel = album_folder_by_id[aid]
            leaf_name = folder_rel.rsplit("/", 1)[-1]
            album_path = ensure_within_directory(albums_dir, albums_dir / folder_rel)
            album_path.mkdir(parents=True, exist_ok=True)
            meta_file = ensure_within_directory(meta_albums_dir, meta_albums_dir / f"{leaf_name}.json")
            album_dirs_to_touch[album_path] = album_ts

            cached_meta: dict[str, Any] | None = None
            images: list[dict[str, Any]] | None = None
            from_cache = False

            if meta_file.exists() and not refresh_index:
                with contextlib.suppress(Exception):
                    loaded = json.loads(meta_file.read_text(encoding="utf-8"))
                    if (
                        isinstance(loaded, dict)
                        and isinstance(loaded.get("images"), list)
                        and (len(loaded["images"]) >= expected_cnt or expected_cnt == 0)
                    ):
                        cached_meta = loaded
                        images = loaded["images"]
                        from_cache = True
                        if (
                            loaded.get("completed") is True
                            and len(images) > 0
                            and _count_numbered_photos(album_path) >= len(images)
                        ):
                            astate_fast = _AlbumState(
                                album_id=aid,
                                album_name=aname,
                                meta_file=meta_file,
                                meta_payload=loaded,
                                total_photos=len(images),
                                remaining=0,
                            )
                            return True, True, astate_fast, []

            if images is None:
                album_meta, fetched_images = client.get_album_images(kid, cid, aid, expected_count=expected_cnt)
                images = fetched_images
                meta_out: dict[str, Any] = scrub_metadata({
                    "album_listing": album,
                    "album_details": album_meta,
                    "images": images,
                    "completed": False,
                })
                write_json_atomic(meta_file, meta_out)
            else:
                meta_out = cached_meta if cached_meta is not None else {}

            tasks: list[tuple[list[str], Path, float | None]] = []
            for i_pos, img in enumerate(images, 1):
                urls = candidate_image_urls(img)
                if not urls:
                    continue
                img_id = sanitize_name(str(img.get("id") or i_pos), 32)
                raw_ext = str(img.get("extension") or "")
                if not raw_ext.strip():
                    parsed_path = urllib.parse.urlparse(urls[0]).path
                    raw_ext = Path(parsed_path).suffix
                ext = sanitize_extension(raw_ext, default="jpg")
                if ext not in _ALLOWED_IMAGE_EXTS:
                    ext = "jpg"
                _, img_ts = parse_date_prefix(img.get("dateadd"))
                fname = f"{i_pos:03d}_{img_id}.{ext}"
                dest = ensure_within_directory(album_path, album_path / fname)
                tasks.append((urls, dest, img_ts or album_ts))

            astate = _AlbumState(
                album_id=aid,
                album_name=aname,
                meta_file=meta_file,
                meta_payload=meta_out,
                total_photos=len(tasks),
                remaining=0,
            )
            return from_cache, False, astate, tasks

        def _dl_photo(item: tuple[list[str], Path, float | None]) -> int:
            urls, d, ts = item
            try:
                sz = client.download_with_fallbacks(urls, d)
                set_file_mtime(d, ts)
                return sz
            finally:
                in_flight_sem.release()

        photo_futs: dict[concurrent.futures.Future[int], tuple[_AlbumState, tuple[list[str], Path, float | None]]] = {}
        failed_photo_items: list[tuple[_AlbumState, tuple[list[str], Path, float | None]]] = []
        total_queued_photos = 0

        def _on_photo_done(fut: concurrent.futures.Future[int]) -> None:
            nonlocal live_done
            astate_to_finalize: _AlbumState | None = None
            with prog_lock:
                astate, _ = photo_futs[fut]
                live_done += 1
                cur = live_done
                exc = fut.exception()
                if exc is None:
                    sz = fut.result()
                    stats.photos_downloaded += 1
                    stats.photos_bytes += sz
                else:
                    astate.failed += 1
                astate.remaining -= 1
                if astate.remaining == 0 and astate.failed == 0:
                    astate_to_finalize = astate

                if cur % 500 == 0:
                    mb = stats.photos_bytes / (1024 * 1024)
                    snap = client.rate_ctrl.snapshot()
                    log(
                        f"  [Photos - {child_folder_name}] +{stats.photos_downloaded} new, "
                        f"{stats.photos_skipped_existing} resumed "
                        f"({mb:.1f} MiB | pace={snap.api_interval_s:.2f}s/req, 429s={snap.total_429s})"
                    )
            if astate_to_finalize is not None:
                with contextlib.suppress(Exception):
                    _finalize_album(astate_to_finalize)
            tracker.update(stats=stats)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as photo_pool:
            logged_resume_transition = False
            for idx, album in enumerate(albums, 1):
                if not logged_resume_transition and stats.albums_resumed_cached > 0:
                    aid_peek = int(album["id"])
                    leaf_peek = album_folder_by_id[aid_peek].rsplit("/", 1)[-1]
                    if not (meta_albums_dir / f"{leaf_peek}.json").exists() or refresh_index:
                        logged_resume_transition = True
                        log(
                            f"  [Resume - {child_folder_name}] Fast-forwarded past {stats.albums_resumed_cached} cached albums "
                            f"({stats.photos_skipped_existing} existing photos verified, {total_queued_photos} missing queued); "
                            f"resuming live API fetch at album {idx}/{len(albums)}."
                        )

                try:
                    from_cache, fast_skipped, astate, tasks = _prepare_album(album)
                except Exception as exc:
                    log(f"  [ERROR] Album {idx}/{len(albums)} (ID={album.get('id')}) metadata failed: {redact_secrets(str(exc))}")
                    continue

                with prog_lock:
                    stats.albums_indexed = idx
                    if from_cache:
                        stats.albums_resumed_cached += 1
                    if fast_skipped:
                        stats.photos_skipped_existing += astate.total_photos

                if fast_skipped:
                    tracker.update(stats=stats)
                    continue

                pending_for_album: list[tuple[list[str], Path, float | None]] = []
                for urls, d, ts in tasks:
                    if d.exists() and d.stat().st_size > 0:
                        for alias in urls:
                            client.register_cached_url(alias, d)
                        with prog_lock:
                            stats.photos_skipped_existing += 1
                    else:
                        pending_for_album.append((urls, d, ts))

                if not pending_for_album:
                    if astate.meta_payload.get("completed") is not True:
                        with contextlib.suppress(Exception):
                            _finalize_album(astate)
                else:
                    with prog_lock:
                        astate.remaining = len(pending_for_album)
                    for item_task in pending_for_album:
                        in_flight_sem.acquire()
                        pf = photo_pool.submit(_dl_photo, item_task)
                        with prog_lock:
                            photo_futs[pf] = (astate, item_task)
                        pf.add_done_callback(_on_photo_done)
                        total_queued_photos += 1

                tracker.update(stats=stats)
                if not from_cache and (idx % 20 == 0 or idx == len(albums)):
                    snap = client.rate_ctrl.snapshot()
                    log(
                        f"  [Albums - {child_folder_name}] {idx}/{len(albums)} indexed "
                        f"(resumed_albums={stats.albums_resumed_cached}, "
                        f"new_photos={stats.photos_downloaded}, existing_photos={stats.photos_skipped_existing} "
                        f"| pace={snap.api_interval_s:.2f}s/req, streak={snap.clean_streak}, 429s={snap.total_429s})"
                    )

            if not logged_resume_transition and stats.albums_resumed_cached > 0:
                log(
                    f"  [Resume - {child_folder_name}] All {stats.albums_resumed_cached} albums loaded from local Metadata cache "
                    f"({stats.photos_skipped_existing} existing photos verified)."
                )

            concurrent.futures.wait(photo_futs.keys())
            for pfut, (astate, item) in photo_futs.items():
                fut_exc = pfut.exception()
                if fut_exc is not None:
                    failed_photo_items.append((astate, item))
                    log(f"    [WARN] Photo retry queued for {item[1].name}: {redact_secrets(str(fut_exc))}")

            if failed_photo_items:
                log(f"  Retrying {len(failed_photo_items)} failed photo(s)...")
                recovered_albums: set[int] = set()
                retried_states: dict[int, _AlbumState] = {}
                for astate, (urls, d, ts) in failed_photo_items:
                    retried_states[astate.album_id] = astate
                    try:
                        sz = client.download_with_fallbacks(urls, d)
                        set_file_mtime(d, ts)
                        with prog_lock:
                            stats.photos_downloaded += 1
                            stats.photos_bytes += sz
                            astate.failed = max(0, astate.failed - 1)
                            if astate.failed == 0:
                                recovered_albums.add(astate.album_id)
                    except Exception as exc:
                        with prog_lock:
                            stats.photos_failed += 1
                        log(f"    [ERROR] Photo failed permanently {d.name}: {redact_secrets(str(exc))}")
                for rec_aid in recovered_albums:
                    with contextlib.suppress(Exception):
                        _finalize_album(retried_states[rec_aid])

        tracker.update(stats=stats, force=True)

    # 2. Gentle Parallel Video Downloads — placed directly into matched day `Albums/<folder>/`!
    if videos:
        tracker.update(phase="videos", child=child_folder_name, stats=stats, force=True)
        log(f"Starting download of {len(videos)} videos into day albums for {child_folder_name} (video_workers={video_workers})...")
        albums_dir.mkdir(parents=True, exist_ok=True)
        meta_videos_dir.mkdir(parents=True, exist_ok=True)
        dir_lock = threading.Lock()

        def _dl_one_video(idx_vid: tuple[int, dict[str, Any]]) -> tuple[str, int, str]:
            idx, vid_item = idx_vid
            vid = int(vid_item["id"])
            vtitle_raw = str(vid_item.get("name") or "video")
            _, v_ts = parse_date_prefix(str(vid_item.get("date") or vid_item.get("updated_at") or ""))

            target_folder, video_filename = video_targets[vid]
            target_album_dir = ensure_within_directory(albums_dir, albums_dir / target_folder)
            with dir_lock:
                target_album_dir.mkdir(parents=True, exist_ok=True)
                if target_album_dir not in album_dirs_to_touch:
                    album_dirs_to_touch[target_album_dir] = v_ts

            mp4_dest = ensure_within_directory(target_album_dir, target_album_dir / video_filename)
            json_dest = ensure_within_directory(meta_videos_dir, meta_videos_dir / f"{vid}.json")

            if mp4_dest.exists() and mp4_dest.stat().st_size >= 10_000:
                vu_existing = vid_item.get("url") or vid_item.get("share_url")
                if vu_existing:
                    client.register_cached_url(str(vu_existing), mp4_dest)
                return ("skipped", 0, vtitle_raw)

            video_url = str(vid_item.get("url") or vid_item.get("share_url") or "")
            if not video_url and json_dest.exists() and not refresh_index:
                with contextlib.suppress(Exception):
                    cand = json.loads(json_dest.read_text(encoding="utf-8"))
                    if isinstance(cand, dict):
                        video_url = str(cand.get("url") or cand.get("share_url") or "")

            if not video_url:
                vdet = client.get_video_details(kid, cid, vid)
                merged = {**vid_item, **vdet}
                write_json_atomic(json_dest, scrub_metadata(merged))
                video_url = str(merged.get("url") or merged.get("share_url") or "")

            if not video_url:
                raise RuntimeError("No url or share_url in video details")

            if "vimeo.com" in video_url:
                sz = client.download_vimeo_video(video_url, mp4_dest)
            elif video_url.lower().endswith((".mp4", ".mov", ".m4v")) or "kinderpedia.co" in video_url:
                sz = client.download_binary(video_url, mp4_dest)
            else:
                write_json_atomic(json_dest, scrub_metadata({**vid_item, "external_url": video_url}))
                log(f"    [WARN] External video link saved in Metadata/{child_folder_name}/Videos/{json_dest.name}")
                return ("external", 0, vtitle_raw)

            set_file_mtime(mp4_dest, v_ts)
            log(f"  [Video {idx}/{len(videos)}] Saved '{target_folder}/{video_filename}' ({sz / (1024 * 1024):.1f} MiB)")
            return ("downloaded", sz, vtitle_raw)

        failed_videos: list[tuple[int, dict[str, Any]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=video_workers) as vpool:
            vfuts = {vpool.submit(_dl_one_video, (i, v)): (i, v) for i, v in enumerate(videos, 1)}
            for vfut in concurrent.futures.as_completed(vfuts):
                iv = vfuts[vfut]
                try:
                    status, sz, _ = vfut.result()
                    if status == "downloaded":
                        stats.videos_downloaded += 1
                        stats.videos_bytes += sz
                    elif status == "skipped":
                        stats.videos_skipped_existing += 1
                    tracker.update(stats=stats)
                except FileNotFoundError as exc:
                    stats.videos_failed += 1
                    log(f"    [SKIP] Video '{iv[1].get('name')}' skipped ({redact_secrets(str(exc))})")
                except Exception as exc:
                    failed_videos.append(iv)
                    log(f"    [WARN] Video retry queued for '{iv[1].get('name')}' (ID={iv[1].get('id')}): {redact_secrets(str(exc))}")

        if failed_videos:
            log(f"  Retrying {len(failed_videos)} failed video(s)...")
            for iv in failed_videos:
                try:
                    status, sz, _ = _dl_one_video(iv)
                    if status == "downloaded":
                        stats.videos_downloaded += 1
                        stats.videos_bytes += sz
                except Exception as exc:
                    stats.videos_failed += 1
                    log(f"    [ERROR] Video failed permanently '{iv[1].get('name')}' (ID={iv[1].get('id')}): {redact_secrets(str(exc))}")
        tracker.update(stats=stats, force=True)

    # Restore album folder mtimes after both photos & co-located videos finish writing
    for apath, ats in album_dirs_to_touch.items():
        set_file_mtime(apath, ats)

    # 3. Download Documents / Child Portfolio files
    if docs:
        tracker.update(phase="documents", child=child_folder_name, stats=stats, force=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        used_doc_names: dict[str, str] = {}

        for idx, doc in enumerate(docs, 1):
            durl = str(doc.get("url") or doc.get("file_url") or doc.get("document_url") or doc.get("download_url") or "").strip()
            if not durl:
                continue
            if durl.startswith("/"):
                durl = WEB_API_BASE + durl

            did = sanitize_name(str(doc.get("id") or idx), 32)
            raw_name = str(doc.get("name") or doc.get("title") or f"doc_{did}")
            dname = sanitize_name(raw_name, 60)

            raw_ext = str(doc.get("extension") or Path(urllib.parse.urlparse(durl).path).suffix or "pdf")
            ext = sanitize_extension(raw_ext, default="pdf")
            suffix_ci = f".{ext.lower()}"
            if dname.lower().endswith(suffix_ci) and len(dname) > len(suffix_ci):
                dname = dname[: -len(suffix_ci)].rstrip(". ") or f"doc_{did}"

            fname = f"{dname}.{ext}"
            prev_did = used_doc_names.get(fname.lower())
            if prev_did is not None and prev_did != did:
                fname = f"{dname}_{did}.{ext}"
            used_doc_names[fname.lower()] = did

            raw_size = doc.get("size")
            expected_bytes = int(raw_size) if isinstance(raw_size, int) and raw_size > 0 else 100
            _, d_ts = parse_date_prefix(str(doc.get("date") or doc.get("created_date") or doc.get("dateadd") or ""))

            dest = ensure_within_directory(docs_dir, docs_dir / fname)
            if dest.exists() and dest.stat().st_size >= expected_bytes:
                client.register_cached_url(durl, dest)
                stats.documents_skipped_existing += 1
                continue
            try:
                client.download_binary(durl, dest)
                set_file_mtime(dest, d_ts)
                stats.documents_downloaded += 1
                tracker.update(stats=stats)
            except Exception as exc:
                log(f"    [WARN] Document '{raw_name}': {redact_secrets(str(exc))}")

    tracker.update(stats=stats, force=True)
    return stats
