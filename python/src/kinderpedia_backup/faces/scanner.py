"""Incremental multi-threaded classroom photo scanner, seed loader, and cluster builder."""

from __future__ import annotations

import concurrent.futures
import contextlib
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..console import LiveProgress, log
from ..security import ensure_within_directory
from .alignment import crop_face_portrait
from .db.scan_lock import ScanLock
from .index_db import FacesIndexDB
from .matcher import rebuild_yearly_clusters, recompute_active_matching
from .models import (
    BoundingBox,
    DetectedFace,
    PersonRole,
    PhotoProvenance,
    PhotoScanState,
    RelinkReport,
    ScanSummary,
)
from .provenance import ProvenanceResolver, discover_child_avatars
from .recognizer import FaceRecognizer
from .storage import discover_backup_photos, discover_reference_photos, file_fingerprint

_REF_YEAR_RE = re.compile(r"^(20\d{2})\b")
_PORTRAIT_EXPAND = 0.45  # must match `crop_face_portrait(expand_ratio=...)` default
_SWAPPING_ORIENTATIONS = frozenset({5, 6, 7, 8})


def face_thumb_path(thumbs_root: Path, face_id: int) -> Path:
    """Shard face thumbnails into 256 hex subdirectories (`00`..`ff`) for fast APFS/NTFS lookups."""
    bucket = f"{face_id & 0xFF:02x}"
    return thumbs_root / bucket / f"{face_id}.jpg"


def save_face_thumb_jpeg(thumbs_root: Path, face_id: int, img: Image.Image) -> Path:
    """Write the thumbnail atomically (temp file + `os.replace`) so readers never see a partial JPEG."""
    dst = face_thumb_path(thumbs_root, face_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{face_id}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        img.save(tmp, format="JPEG", quality=83, optimize=False)
        os.replace(tmp, dst)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return dst


def remove_face_thumbs(thumbs_root: Path, face_ids: list[int]) -> int:
    """Delete cached thumbnails (e.g. faces removed or moved by a rescan); returns files removed."""
    removed = 0
    for fid in face_ids:
        with contextlib.suppress(OSError):
            face_thumb_path(thumbs_root, int(fid)).unlink()
            removed += 1
    return removed


def _render_face_thumb(src: Path, box: BoundingBox, size: int) -> Image.Image:
    """Crop a portrait thumbnail decoding as few pixels as possible.

    JPEGs are decoded at a reduced DCT scale (`Image.draft`) when the face is large relative to the
    output, and only the portrait region is converted to RGB (crop-before-convert). The result is the
    same crop `crop_face_portrait` would produce on the full-resolution image.
    """
    with Image.open(src) as raw:
        orientation = 1
        with contextlib.suppress(Exception):
            orientation = int(raw.getexif().get(0x0112, 1) or 1)
        raw_w, raw_h = raw.size
        t_w, t_h = (raw_h, raw_w) if orientation in _SWAPPING_ORIENTATIONS else (raw_w, raw_h)
        side_full = max(float(box.width), float(box.height), 32.0) * (1.0 + 2.0 * _PORTRAIT_EXPAND)
        scale_req = float(size) / max(side_full, 1.0)
        if raw.format == "JPEG" and scale_req < 0.5:
            raw.draft("RGB", (max(1, math.ceil(raw_w * scale_req)), max(1, math.ceil(raw_h * scale_req))))
        img = ImageOps.exif_transpose(raw)
        sx = img.width / max(1, t_w)
        sy = img.height / max(1, t_h)
        s_box = BoundingBox(
            top=round(box.top * sy),
            right=round(box.right * sx),
            bottom=round(box.bottom * sy),
            left=round(box.left * sx),
        )
        cx = 0.5 * (s_box.left + s_box.right)
        cy = 0.5 * (s_box.top + s_box.bottom)
        half = 0.5 * max(float(s_box.width), float(s_box.height), 32.0 * min(sx, sy)) * (
            1.0 + 2.0 * _PORTRAIT_EXPAND
        )
        x1 = max(0, math.floor(cx - half) - 2)
        y1 = max(0, math.floor(cy - half) - 2)
        x2 = min(img.width, math.ceil(cx + half) + 2)
        y2 = min(img.height, math.ceil(cy + half) + 2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return crop_face_portrait(img.convert("RGB"), s_box, size=size)
        region = img.crop((x1, y1, x2, y2)).convert("RGB")
        local = BoundingBox(
            top=s_box.top - y1, right=s_box.right - x1, bottom=s_box.bottom - y1, left=s_box.left - x1
        )
        return crop_face_portrait(region, local, size=size)


def ensure_face_thumb_on_disk(
    *,
    backup_root: Path,
    thumbs_root: Path,
    face_id: int,
    rel_path: str,
    box: Any,
    size: int = 160,
) -> Path | None:
    """Return cached thumbnail path for `face_id`, lazily cropping from original photo on cache miss."""
    dst = face_thumb_path(thumbs_root, face_id)
    if dst.is_file():
        return dst
    src = ensure_within_directory(backup_root, backup_root / rel_path)
    if not src.is_file():
        return None
    bbox = (
        box
        if isinstance(box, BoundingBox)
        else BoundingBox(top=int(box.top), right=int(box.right), bottom=int(box.bottom), left=int(box.left))
    )
    with contextlib.suppress(OSError, ValueError):
        thumb = _render_face_thumb(src, bbox, size)
        return save_face_thumb_jpeg(thumbs_root, face_id, thumb)
    return None


def _inspect_single_photo(
    recognizer: FaceRecognizer,
    photo_path: Path,
    *,
    thumb_size: int = 160,
) -> tuple[Path, int, int, int, int, list[DetectedFace], list[Image.Image], str]:
    st = photo_path.stat()
    c_hash = file_fingerprint(photo_path, size=int(st.st_size))
    analyze_fn = getattr(recognizer, "analyze_photo", None)
    if callable(analyze_fn):
        w, h, faces, thumbs = analyze_fn(photo_path, thumb_size=thumb_size)
        return photo_path, int(w), int(h), int(st.st_size), int(st.st_mtime_ns), faces, thumbs, c_hash

    # Fallback for generic `FaceRecognizer` implementations (e.g. unit-test stubs):
    faces = recognizer.detect_and_embed(photo_path)
    w, h = 0, 0
    fb_thumbs: list[Image.Image] = []
    with contextlib.suppress(Exception), Image.open(photo_path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        w, h = rgb.size
        fb_thumbs = [crop_face_portrait(rgb, f.box, size=thumb_size) for f in faces]
    return photo_path, w, h, int(st.st_size), int(st.st_mtime_ns), faces, fb_thumbs, c_hash


def enroll_avatars_and_references(
    db: FacesIndexDB,
    recognizer: FaceRecognizer,
    *,
    backup_dir: Path,
    refs_dir: Path | None = None,
) -> int:
    """
    Auto-enroll `<Child>/avatar.png` as `family` anchors plus optional `References/<Person>/*.jpg`.
    Returns total number of seeded people.
    """
    seeded_pids: set[int] = set()
    avatars = discover_child_avatars(backup_dir)
    for child_name, avatar_path in avatars.items():
        pid = db.ensure_person(child_name, role="family", archive_hint=child_name)
        faces = recognizer.detect_and_embed(avatar_path)
        if faces:
            best = max(faces, key=lambda f: f.box.area * f.confidence)
            db.upsert_reference_seed(
                person_id=pid,
                source_key=f"avatar:{child_name}:{avatar_path.name}",
                embedding=best.embedding,
                anchor_year=None,
            )
            seeded_pids.add(pid)
        else:
            log(f"[Faces] Note: no clear face detected in {avatar_path.name} for {child_name}; use Web UI clusters to seed.")

    eff_refs = refs_dir if refs_dir is not None else (backup_dir / "References")
    ref_map = discover_reference_photos(eff_refs)
    for person_name, portrait_paths in ref_map.items():
        is_fam = person_name in avatars
        default_role: PersonRole = (
            "family" if is_fam else ("ignore" if person_name.startswith("_") else "classmate")
        )
        pid = db.ensure_person(
            person_name,
            role=default_role,
            archive_hint=person_name if is_fam else "",
        )
        for p in portrait_paths:
            faces = recognizer.detect_and_embed(p)
            if not faces:
                log(f"[Faces] Warning: 0 faces detected in reference portrait {p}; skipping.")
                continue
            if len(faces) >= 2:
                by_area = sorted(faces, key=lambda f: f.box.area, reverse=True)
                if by_area[0].box.area < 2.2 * max(1, by_area[1].box.area):
                    log(
                        f"[Faces] Warning: multiple similarly-sized faces in {p.name} for {person_name}; "
                        "crop to a solo portrait for cleanest accuracy."
                    )
            best = max(faces, key=lambda f: f.box.area * f.confidence)
            m_yr = _REF_YEAR_RE.match(p.stem)
            anchor_yr = int(m_yr.group(1)) if m_yr else None
            db.upsert_reference_seed(
                person_id=pid,
                source_key=f"ref:{person_name}:{p.name}",
                embedding=best.embedding,
                anchor_year=anchor_yr,
            )
            seeded_pids.add(pid)

    return len(seeded_pids)


def _hash_many(paths: list[tuple[Path, int]], workers: int) -> list[str | None]:
    """Fingerprint `(path, size)` pairs concurrently; `None` for unreadable files."""

    def _one(item: tuple[Path, int]) -> str | None:
        try:
            return file_fingerprint(item[0], size=item[1])
        except OSError:
            return None

    if not paths:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_one, paths))


def run_archive_face_scan(
    *,
    backup_dir: Path,
    recognizer: FaceRecognizer,
    refs_dir: Path | None = None,
    output_dir: Path | None = None,
    workers: int | None = None,
    auto_high_threshold: float = 0.45,
    candidate_threshold: float = 0.33,
    batch_commit_size: int = 80,
    progress: LiveProgress | None = None,
    force: bool = False,
    prune_missing: bool | None = None,
    lock_timeout_s: float = 0.0,
) -> ScanSummary:
    """
    Incrementally scan all classroom photos in `backup_dir`, populate `Metadata/faces_index.sqlite`
    with embeddings + Kinderpedia provenance/URLs, discover per-year face clusters, and run matching.

    Change detection: unchanged `(size, mtime, model_id)` -> skip; same size but new mtime -> verify the
    head/tail content fingerprint (only an actual content change rescans); `model_id` change (new
    detector / `--det-size` / embedder bump) or `force=True` -> label-preserving rescan (faces are
    re-linked to their old `face_id`s). Paths that vanished are re-keyed to renamed/moved files (same
    fingerprint, or same Kinderpedia image id + size); the rest are deleted unless that would remove
    an implausibly large share of the archive (`prune_missing=None` -> auto guard, `True` -> always,
    `False` -> never). Holds `Metadata/faces_scan.lock` for the whole run (`ScanLockBusyError`).
    """
    backup_root = backup_dir.resolve()
    excl_dir = (output_dir or (backup_root / "ByChild")).resolve()
    meta_dir = ensure_within_directory(backup_root, backup_root / "Metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)
    db_path = ensure_within_directory(meta_dir, meta_dir / "faces_index.sqlite")
    thumbs_root = ensure_within_directory(meta_dir, meta_dir / "face_thumbs")
    portable_json = ensure_within_directory(meta_dir, meta_dir / "categorization_index.json")

    with ScanLock(meta_dir, timeout_s=lock_timeout_s):
        return _run_locked_scan(
            backup_root=backup_root,
            excl_dir=excl_dir,
            db_path=db_path,
            thumbs_root=thumbs_root,
            portable_json=portable_json,
            recognizer=recognizer,
            refs_dir=refs_dir,
            workers=workers,
            auto_high_threshold=auto_high_threshold,
            candidate_threshold=candidate_threshold,
            batch_commit_size=batch_commit_size,
            progress=progress,
            force=force,
            prune_missing=prune_missing,
        )


def _run_locked_scan(
    *,
    backup_root: Path,
    excl_dir: Path,
    db_path: Path,
    thumbs_root: Path,
    portable_json: Path,
    recognizer: FaceRecognizer,
    refs_dir: Path | None,
    workers: int | None,
    auto_high_threshold: float,
    candidate_threshold: float,
    batch_commit_size: int,
    progress: LiveProgress | None,
    force: bool,
    prune_missing: bool | None,
) -> ScanSummary:
    photos = discover_backup_photos(backup_root, exclude_dir=excl_dir)
    summary = ScanSummary(photos_total=len(photos))
    resolver = ProvenanceResolver(backup_root)
    det_id = recognizer.model_id
    default_cap = 8 if getattr(recognizer, "engine_kind", "") == "onnxruntime" else 6
    eff_workers = max(1, workers if workers is not None else min(default_cap, os.cpu_count() or 4))
    io_workers = max(4, eff_workers)
    report = RelinkReport()

    with FacesIndexDB(db_path) as db:
        summary.people_seeded = enroll_avatars_and_references(
            db,
            recognizer,
            backup_dir=backup_root,
            refs_dir=refs_dir,
        )
        states = db.get_photo_scan_states()
        disk_rel: dict[str, Path] = {}
        for p in photos:
            disk_rel["/".join(p.resolve().relative_to(backup_root).parts)] = p
        missing: dict[str, PhotoScanState] = {r: s for r, s in states.items() if r not in disk_rel}

        to_scan: list[Path] = []
        verify: list[tuple[Path, int, PhotoScanState, int]] = []  # (path, size, prev, mtime_ns)
        backfill: list[tuple[Path, int, PhotoScanState]] = []
        new_files: list[tuple[Path, int, int]] = []  # (path, size, mtime_ns)
        model_changed = 0
        for rel_key, p in disk_rel.items():
            try:
                st = p.stat()
            except OSError:
                continue
            prev = states.get(rel_key)
            if prev is None:
                new_files.append((p, int(st.st_size), int(st.st_mtime_ns)))
            elif force or prev.detector_id != det_id or prev.file_size != st.st_size:
                model_changed += int(prev.detector_id != det_id)
                to_scan.append(p)
            elif prev.mtime_ns == st.st_mtime_ns:
                summary.photos_cached += 1
                if not prev.content_hash:
                    backfill.append((p, int(st.st_size), prev))
            else:
                verify.append((p, int(st.st_size), prev, int(st.st_mtime_ns)))

        if model_changed:
            log(
                f"[Faces] Model changed to {det_id} for {model_changed:,} photo(s): re-detecting with "
                "label-preserving face re-link (confirmed tags, anchors and pressure stay on their faces)."
            )

        # Same size, new mtime: compare fingerprints instead of re-running detection.
        fp_updates: list[tuple[int, str, int]] = []
        if verify:
            hashes = _hash_many([(p, sz) for p, sz, _, _ in verify], io_workers)
            for (p, _sz, prev, mtime_ns), h in zip(verify, hashes, strict=True):
                if h is not None and prev.content_hash and h == prev.content_hash:
                    fp_updates.append((prev.photo_id, h, mtime_ns))
                    summary.photos_verified += 1
                    summary.photos_cached += 1
                else:
                    to_scan.append(p)
        if backfill:
            log(f"[Faces] Fingerprinting {len(backfill):,} previously indexed photo(s) (one-time)...")
            hashes = _hash_many([(p, sz) for p, sz, _ in backfill], io_workers)
            fp_updates.extend(
                (prev.photo_id, h, prev.mtime_ns)
                for (_, _, prev), h in zip(backfill, hashes, strict=True)
                if h is not None
            )
        db.set_photo_fingerprints(fp_updates)

        # Renamed / moved photos: re-key vanished rows onto new paths, keeping photo_id + labels.
        if missing and new_files:
            by_hash: dict[str, str] = {}
            by_kp: dict[tuple[int, int], list[str]] = {}
            for rel, s in missing.items():
                if s.content_hash:
                    by_hash.setdefault(s.content_hash, rel)
                if s.kinderpedia_image_id is not None:
                    by_kp.setdefault((s.kinderpedia_image_id, s.file_size), []).append(rel)
            new_hashes = _hash_many([(p, sz) for p, sz, _ in new_files], io_workers)
            still_new: list[tuple[Path, int, int]] = []
            for (p, sz, mtime_ns), h in zip(new_files, new_hashes, strict=True):
                prov = resolver.resolve(p)
                old_rel = by_hash.get(h) if h else None
                if old_rel is None or old_rel not in missing:
                    old_rel = None
                    if prov.kinderpedia_image_id is not None:
                        cands = [
                            r for r in by_kp.get((prov.kinderpedia_image_id, sz), []) if r in missing
                        ]
                        same_name = [r for r in cands if missing[r].filename == prov.filename]
                        pick = same_name or cands
                        old_rel = pick[0] if pick else None
                if old_rel is None:
                    still_new.append((p, sz, mtime_ns))
                    continue
                prev = missing.pop(old_rel)
                db.rekey_photo(prev.photo_id, prov, mtime_ns=mtime_ns, content_hash=h or "")
                summary.photos_rekeyed += 1
                if force or prev.detector_id != det_id or prev.file_size != sz:
                    to_scan.append(p)
                else:
                    summary.photos_cached += 1
            new_files = still_new
            if summary.photos_rekeyed:
                log(f"[Faces] Re-keyed {summary.photos_rekeyed:,} renamed/moved photo(s) (labels kept).")
        to_scan.extend(p for p, _, _ in new_files)

        # Photos gone from disk: delete (cascade-safe) unless it looks like an unmounted/partial archive.
        if missing:
            n_missing = len(missing)
            guard_ok = bool(photos) and n_missing <= max(100, int(0.10 * len(states)))
            do_prune = prune_missing if prune_missing is not None else guard_ok
            if do_prune and photos:
                res = db.delete_photos([s.photo_id for s in missing.values()])
                summary.photos_removed = int(res["photos"])
                summary.labels_removed_with_photos = int(res["human_labels"])
                remove_face_thumbs(thumbs_root, list(res["face_ids"]))
                log(
                    f"[Faces] Removed {summary.photos_removed:,} photo(s) no longer on disk "
                    f"({res['faces']:,} faces, {summary.labels_removed_with_photos:,} human labels)."
                )
            else:
                summary.photos_missing_kept = n_missing
                log(
                    f"[Faces] Warning: {n_missing:,} indexed photo(s) are missing on disk; kept them "
                    "(looks like a partial/unmounted archive). Re-run with `--prune-missing` to delete."
                )

        log(
            f"[Faces] Archive: {len(photos):,} photos ({summary.photos_cached:,} cached, "
            f"{len(to_scan):,} to scan with {eff_workers} worker(s), model={det_id})"
        )

        if to_scan:
            t_start = time.monotonic()
            pending_rows: list[tuple[PhotoProvenance, int, int, int, int, str, list[DetectedFace]]] = []
            pending_thumbs: list[list[Image.Image]] = []
            pending_hashes: list[str] = []

            def _flush_pending() -> None:
                if not pending_rows:
                    return
                n_removed = len(report.removed_face_ids)
                n_moved = len(report.moved_face_ids)
                saved_ids = db.upsert_scanned_batch(
                    pending_rows, content_hashes=pending_hashes, report=report
                )
                remove_face_thumbs(thumbs_root, report.removed_face_ids[n_removed:])
                remove_face_thumbs(thumbs_root, report.moved_face_ids[n_moved:])
                for (_, face_ids), thumb_list in zip(saved_ids, pending_thumbs, strict=True):
                    for fid, t_img in zip(face_ids, thumb_list, strict=False):
                        with contextlib.suppress(OSError):
                            save_face_thumb_jpeg(thumbs_root, fid, t_img)
                pending_rows.clear()
                pending_thumbs.clear()
                pending_hashes.clear()

            with concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers) as pool:
                in_flight: set[concurrent.futures.Future[Any]] = set()
                max_queue = eff_workers * 4
                photo_iter = iter(to_scan)
                done_count = 0

                def _submit_next() -> bool:
                    try:
                        nxt = next(photo_iter)
                    except StopIteration:
                        return False
                    in_flight.add(pool.submit(_inspect_single_photo, recognizer, nxt))
                    return True

                for _ in range(min(max_queue, len(to_scan))):
                    _submit_next()

                while in_flight:
                    done_set, in_flight = concurrent.futures.wait(
                        in_flight,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in done_set:
                        _submit_next()
                        done_count += 1
                        try:
                            p_path, w, h, fsize, mtime_ns, faces, thumbs, c_hash = fut.result()
                            prov = resolver.resolve(p_path, width=w, height=h)
                            pending_rows.append((prov, w, h, fsize, mtime_ns, det_id, faces))
                            pending_thumbs.append(thumbs)
                            pending_hashes.append(c_hash)
                            summary.photos_scanned += 1
                        except Exception as exc:
                            summary.photos_failed += 1
                            log(f"[Faces] Warning: skipped unreadable image ({exc})")

                        if len(pending_rows) >= batch_commit_size:
                            _flush_pending()
                            if done_count % 5000 < batch_commit_size and done_count >= 5000:
                                recompute_active_matching(
                                    db,
                                    auto_high_threshold=auto_high_threshold,
                                    candidate_threshold=candidate_threshold,
                                )

                        if progress is not None and progress.enabled:
                            elapsed = max(0.001, time.monotonic() - t_start)
                            rate = done_count / elapsed
                            rem_s = int((len(to_scan) - done_count) / max(rate, 0.01))
                            progress.render_line(
                                f"  [Scan] {done_count:,}/{len(to_scan):,} "
                                f"({100.0 * done_count / len(to_scan):.1f}%) | "
                                f"{rate:.1f} img/s | ETA {rem_s // 60}m{rem_s % 60:02d}s"
                            )
                        elif done_count % 1000 == 0 or done_count == len(to_scan):
                            elapsed = max(0.001, time.monotonic() - t_start)
                            rate = done_count / elapsed
                            rem_s = int((len(to_scan) - done_count) / max(rate, 0.01))
                            log(
                                f"[Faces] Progress: {done_count:,}/{len(to_scan):,} "
                                f"({100.0 * done_count / len(to_scan):.1f}%) | "
                                f"{rate:.1f} img/s | ETA {rem_s // 60}m{rem_s % 60:02d}s"
                            )

                _flush_pending()
            if progress is not None:
                progress.close()

        summary.faces_relinked = report.faces_relinked
        summary.faces_removed = report.faces_removed
        summary.labels_orphaned = report.labels_orphaned
        if report.faces_relinked or report.faces_removed:
            log(
                f"[Faces] Re-link: {report.faces_relinked:,} face(s) kept their id, "
                f"{report.faces_new:,} new, {report.faces_removed:,} removed"
                + (
                    f" — {report.labels_orphaned:,} human label(s) lost their face box "
                    "(kept as photo-level tags; see meta.orphaned_face_labels_total)"
                    if report.labels_orphaned
                    else ""
                )
            )

        log("[Faces] Building per-(classroom, year) recurring child clusters...")
        summary.clusters_created = rebuild_yearly_clusters(db)
        log("[Faces] Running active-learning multi-anchor matcher (with sibling guard & hard negatives)...")
        summary.auto_high_matches, summary.candidate_matches = recompute_active_matching(
            db,
            auto_high_threshold=auto_high_threshold,
            candidate_threshold=candidate_threshold,
        )
        overview = db.get_archive_overview()
        summary.faces_total = int(overview["faces_total"])
        db.export_portable_index(portable_json)
        db.bump_data_version(scan_finished=True)

    return summary
