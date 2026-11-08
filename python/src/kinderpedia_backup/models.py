"""Typed domain models for Kinderpedia backup."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ChildContext:
    """Represents one child enrolled in one kindergarten/school context."""

    child_id: int
    kindergarten_id: int
    name: str
    first_name: str = ""
    group_name: str = ""
    kindergarten_name: str = ""
    avatar_url: str = ""

    @property
    def short_name(self) -> str:
        raw = self.first_name.strip()
        if raw:
            return raw
        parts = self.name.strip().split()
        return parts[0] if parts else f"child_{self.child_id}"


@dataclass(frozen=True)
class RateSnapshot:
    """Point-in-time telemetry from AdaptiveRateController."""

    api_interval_s: float
    effective_rpm: float
    soft_floor_s: float
    clean_streak: int
    total_api_calls: int
    total_429s: int
    cooldown_remaining_s: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class ChildBackupStats:
    """Mutable per-child backup counters serialized to status & summary JSON."""

    context_key: str
    child_name: str
    child_id: int
    kindergarten_id: int
    group_name: str
    albums_count: int = 0
    albums_indexed: int = 0
    albums_resumed_cached: int = 0
    albums_failed: int = 0
    expected_photos_count: int = 0
    photos_downloaded: int = 0
    photos_skipped_existing: int = 0
    photos_failed: int = 0
    photos_bytes: int = 0
    videos_count: int = 0
    videos_downloaded: int = 0
    videos_skipped_existing: int = 0
    videos_failed: int = 0
    videos_bytes: int = 0
    documents_count: int = 0
    documents_downloaded: int = 0
    documents_skipped_existing: int = 0
    documents_failed: int = 0

    @property
    def total_failures(self) -> int:
        """Photos, videos, documents and album manifests that could not be backed up this run."""
        return self.photos_failed + self.videos_failed + self.documents_failed + self.albums_failed

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("context_key", None)
        return data
