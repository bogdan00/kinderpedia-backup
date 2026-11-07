"""Data models for facial recognition and per-child photo organization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

LinkMode = Literal["hardlink", "symlink", "copy"]


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coordinates of a detected face region."""

    top: int
    right: int
    bottom: int
    left: int

    def to_dict(self) -> dict[str, int]:
        return {"top": self.top, "right": self.right, "bottom": self.bottom, "left": self.left}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundingBox:
        return cls(
            top=int(data.get("top", 0)),
            right=int(data.get("right", 0)),
            bottom=int(data.get("bottom", 0)),
            left=int(data.get("left", 0)),
        )


@dataclass(frozen=True)
class DetectedFace:
    """A detected face and its L2-normalized or raw embedding vector."""

    box: BoundingBox
    embedding: tuple[float, ...]
    confidence: float = 1.0


@dataclass
class ReferenceProfile:
    """Enrolled reference embeddings for a single child/person."""

    name: str
    embeddings: list[tuple[float, ...]] = field(default_factory=list)
    source_photos: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FaceMatch:
    """A positive match associating a photo face with an enrolled person."""

    person_name: str
    similarity: float
    box: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_name": self.person_name,
            "similarity": round(self.similarity, 4),
            "box": self.box.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceMatch:
        raw_box = data.get("box")
        box = BoundingBox.from_dict(raw_box) if isinstance(raw_box, dict) else BoundingBox(0, 0, 0, 0)
        return cls(
            person_name=str(data["person_name"]),
            similarity=float(data.get("similarity", 0.0)),
            box=box,
        )


@dataclass
class OrganizeStats:
    """Telemetry summary for a facial-recognition scan & organize run."""

    reference_profiles: int = 0
    photos_discovered: int = 0
    photos_scanned: int = 0
    photos_cached: int = 0
    photos_matched: int = 0
    links_created: int = 0
    links_skipped_existing: int = 0
