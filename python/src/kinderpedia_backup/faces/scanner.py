"""Incremental multi-threaded classroom photo scanner, seed loader, and cluster builder."""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..console import LiveProgress, log
from ..security import ensure_within_directory
from .alignment import crop_face_portrait
from .index_db import FacesIndexDB
from .matcher import rebuild_yearly_clusters, recompute_active_matching
from .models import DetectedFace, PersonRole, PhotoProvenance, ScanSummary
from .provenance import ProvenanceResolver, discover_child_avatars
from .recognizer import FaceRecognizer
from .storage import discover_backup_photos, discover_reference_photos

_REF_YEAR_RE = re.compile(r"^(20\d{2})\b")


def face_thumb_path(thumbs_root: Path, face_id: int) -> Path:
    """Shard face thumbnails into 256 hex subdirectories (`00`..`ff`) for fast APFS/NTFS lookups."""
    bucket = f"{face_id & 0xFF:02x}"
    return thumbs_root / bucket / f"{face_id}.jpg"


def save_face_thumb_jpeg(thumbs_root: Path, face_id: int, img: Image.Image) -> Path:
    dst = face_thumb_path(thumbs_root, face_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=83, optimize=False)
    return dst


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
    with contextlib.suppress(OSError, ValueError), Image.open(src) as raw:
        rgb = ImageOps.exif_transpose(raw).convert("RGB")
        thumb = crop_face_portrait(rgb, box, size=size)
        return save_face_thumb_jpeg(thumbs_root, face_id, thumb)
    return None


def _inspect_single_photo(
    recognizer: FaceRecognizer,
    photo_path: Path,
    *,
    thumb_size: int = 160,
) -> tuple[Path, int, int, int, int, list[DetectedFace], list[Image.Image]]:
    st = photo_path.stat()
    analyze_fn = getattr(recognizer, "analyze_photo", None)
    if callable(analyze_fn):
        w, h, faces, thumbs = analyze_fn(photo_path, thumb_size=thumb_size)
        return photo_path, int(w), int(h), int(st.st_size), int(st.st_mtime_ns), faces, thumbs

    # Fallback for generic `FaceRecognizer` implementations (e.g. unit-test stubs):
    faces = recognizer.detect_and_embed(photo_path)
    w, h = 0, 0
    fb_thumbs: list[Image.Image] = []
    with contextlib.suppress(Exception), Image.open(photo_path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        w, h = rgb.size
        fb_thumbs = [crop_face_portrait(rgb, f.box, size=thumb_size) for f in faces]
    return photo_path, w, h, int(st.st_size), int(st.st_mtime_ns), faces, fb_thumbs


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
) -> ScanSummary:
    """
    Incrementally scan all classroom photos in `backup_dir`, populate `Metadata/faces_index.sqlite`
    with embeddings + Kinderpedia provenance/URLs, discover per-year face clusters, and run matching.
    """
    backup_root = backup_dir.resolve()
    excl_dir = (output_dir or (backup_root / "ByChild")).resolve()
    meta_dir = ensure_within_directory(backup_root, backup_root / "Metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)
    db_path = ensure_within_directory(meta_dir, meta_dir / "faces_index.sqlite")
    thumbs_root = ensure_within_directory(meta_dir, meta_dir / "face_thumbs")
    portable_json = ensure_within_directory(meta_dir, meta_dir / "categorization_index.json")

    photos = discover_backup_photos(backup_root, exclude_dir=excl_dir)
    summary = ScanSummary(photos_total=len(photos))
    resolver = ProvenanceResolver(backup_root)
    det_id = recognizer.model_id
    default_cap = 8 if getattr(recognizer, "engine_kind", "") == "onnxruntime" else 6
    eff_workers = max(1, workers if workers is not None else min(default_cap, os.cpu_count() or 4))

    with FacesIndexDB(db_path) as db:
        summary.people_seeded = enroll_avatars_and_references(
            db,
            recognizer,
            backup_dir=backup_root,
            refs_dir=refs_dir,
        )
        cached_states = db.get_cached_photo_states()
        to_scan: list[Path] = []
        for p in photos:
            rel_key = "/".join(p.resolve().relative_to(backup_root).parts)
            try:
                st = p.stat()
            except OSError:
                continue
            prev = cached_states.get(rel_key)
            if prev is not None and prev[1] == st.st_size and prev[2] == st.st_mtime_ns and prev[3] == det_id:
                summary.photos_cached += 1
            else:
                to_scan.append(p)

        log(
            f"[Faces] Archive: {len(photos):,} photos ({summary.photos_cached:,} cached, "
            f"{len(to_scan):,} to scan with {eff_workers} worker(s), model={det_id})"
        )

        if to_scan:
            t_start = time.monotonic()
            pending_rows: list[tuple[PhotoProvenance, int, int, int, int, str, list[DetectedFace]]] = []
            pending_thumbs: list[list[Image.Image]] = []

            def _flush_pending() -> None:
                if not pending_rows:
                    return
                saved_ids = db.upsert_scanned_batch(pending_rows)
                for (_, face_ids), thumb_list in zip(saved_ids, pending_thumbs, strict=True):
                    for fid, t_img in zip(face_ids, thumb_list, strict=False):
                        with contextlib.suppress(OSError):
                            save_face_thumb_jpeg(thumbs_root, fid, t_img)
                pending_rows.clear()
                pending_thumbs.clear()

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
                            p_path, w, h, fsize, mtime_ns, faces, thumbs = fut.result()
                            prov = resolver.resolve(p_path, width=w, height=h)
                            pending_rows.append((prov, w, h, fsize, mtime_ns, det_id, faces))
                            pending_thumbs.append(thumbs)
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

    return summary
