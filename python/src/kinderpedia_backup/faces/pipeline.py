"""Incremental facial-recognition scan & per-child photo organization pipeline."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from ..security import ensure_within_directory, write_json_atomic
from ..sync.naming import sanitize_name
from .models import FaceMatch, LinkMode, OrganizeStats, ReferenceProfile
from .recognizer import FaceRecognizer, cosine_similarity
from .storage import discover_backup_photos, discover_reference_photos, link_or_copy_photo


class FaceOrganizationPipeline:
    """
    Scans backed-up classroom albums and organizes photos containing each enrolled child
    into `<output_dir>/<Child Name>/` using zero-extra-space hardlinks (or symlinks/copies).
    """

    def __init__(
        self,
        recognizer: FaceRecognizer,
        *,
        threshold: float = 0.45,
        link_mode: LinkMode = "hardlink",
        cache_file: Path | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Similarity threshold must be in (0.0, 1.0].")
        self.recognizer = recognizer
        self.threshold = threshold
        self.link_mode: LinkMode = link_mode
        self.cache_file = cache_file

    def enroll_references(self, refs_dir: Path) -> list[ReferenceProfile]:
        """Build reference embedding profiles from `<refs_dir>/<Child Name>/*` portraits."""
        ref_map = discover_reference_photos(refs_dir)
        profiles: list[ReferenceProfile] = []
        for person_name, paths in ref_map.items():
            prof = ReferenceProfile(name=person_name)
            for p in paths:
                faces = self.recognizer.detect_and_embed(p)
                if not faces:
                    continue
                # Pick highest-confidence / primary face in the reference portrait
                best_face = max(faces, key=lambda f: f.confidence)
                prof.embeddings.append(best_face.embedding)
                prof.source_photos.append(p.name)
            if prof.embeddings:
                profiles.append(prof)
        return profiles

    def match_photo(self, image_path: Path, profiles: list[ReferenceProfile]) -> list[FaceMatch]:
        """Detect faces in `image_path` and match them against enrolled `profiles`."""
        detected = self.recognizer.detect_and_embed(image_path)
        matches_by_person: dict[str, FaceMatch] = {}

        for face in detected:
            best_person: str | None = None
            best_score = -1.0
            for prof in profiles:
                for ref_emb in prof.embeddings:
                    score = cosine_similarity(face.embedding, ref_emb)
                    if score > best_score:
                        best_score = score
                        best_person = prof.name
            if best_person is not None and best_score >= self.threshold:
                prev = matches_by_person.get(best_person)
                if prev is None or best_score > prev.similarity:
                    matches_by_person[best_person] = FaceMatch(
                        person_name=best_person,
                        similarity=best_score,
                        box=face.box,
                    )

        return sorted(matches_by_person.values(), key=lambda m: m.similarity, reverse=True)

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self.cache_file is None or not self.cache_file.exists():
            return {}
        with contextlib.suppress(OSError, ValueError):
            raw = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if (
                isinstance(raw, dict)
                and raw.get("model_id") == self.recognizer.model_id
                and raw.get("threshold") == self.threshold
                and isinstance(raw.get("items"), dict)
            ):
                return dict(raw["items"])
        return {}

    def _save_cache(self, items: dict[str, dict[str, Any]]) -> None:
        if self.cache_file is None:
            return
        write_json_atomic(
            self.cache_file,
            {
                "model_id": self.recognizer.model_id,
                "threshold": self.threshold,
                "items": items,
            },
        )

    def run(self, *, backup_dir: Path, refs_dir: Path, output_dir: Path) -> OrganizeStats:
        """Scan `backup_dir` against `refs_dir` and populate `output_dir/<Person>/<Album>_<Photo>`."""
        backup_root = backup_dir.resolve()
        out_root = output_dir.resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        stats = OrganizeStats()
        profiles = self.enroll_references(refs_dir)
        stats.reference_profiles = len(profiles)
        if not profiles:
            return stats

        enrolled_names = {p.name for p in profiles}
        photos = discover_backup_photos(backup_root, exclude_dir=out_root)
        stats.photos_discovered = len(photos)

        cache = self._load_cache()
        updated_cache: dict[str, dict[str, Any]] = dict(cache)

        for photo in photos:
            rel_key = str(photo.resolve().relative_to(backup_root))
            st = photo.stat()
            cached = cache.get(rel_key)
            matches: list[FaceMatch]

            if (
                isinstance(cached, dict)
                and cached.get("size") == st.st_size
                and cached.get("mtime_ns") == st.st_mtime_ns
                and isinstance(cached.get("matches"), list)
            ):
                matches = [
                    m
                    for m in (FaceMatch.from_dict(x) for x in cached["matches"] if isinstance(x, dict))
                    if m.person_name in enrolled_names
                ]
                stats.photos_cached += 1
            else:
                matches = self.match_photo(photo, profiles)
                updated_cache[rel_key] = {
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "matches": [m.to_dict() for m in matches],
                }
                stats.photos_scanned += 1

            if not matches:
                continue

            stats.photos_matched += 1
            album_prefix = sanitize_name(photo.parent.name, max_len=60)
            target_filename = f"{album_prefix}_{photo.name}"

            for m in matches:
                person_dir = ensure_within_directory(out_root, out_root / sanitize_name(m.person_name, 60))
                dst = ensure_within_directory(person_dir, person_dir / target_filename)
                if link_or_copy_photo(photo, dst, root_dir=out_root, mode=self.link_mode):
                    stats.links_created += 1
                else:
                    stats.links_skipped_existing += 1

        self._save_cache(updated_cache)
        return stats
