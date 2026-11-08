"""Data models for facial recognition, provenance indexing, and active-learning categorization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

LinkMode = Literal["hardlink", "symlink", "copy"]
PersonRole = Literal["family", "classmate", "educator", "ignore"]
AssignmentStatus = Literal["confirmed", "auto_high", "candidate", "unsure", "rejected"]
AssignmentSource = Literal[
    "avatar_seed",
    "reference_file",
    "cluster_confirm",
    "batch_approve",
    "reassigned",
    "manual_tag",
    "album_cohesion",
    "model_trusted",
    "model_match",
]


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coordinates of a detected face region."""

    top: int
    right: int
    bottom: int
    left: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

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
    """A detected face, its bounding box, and its L2-normalized embedding vector."""

    box: BoundingBox
    embedding: tuple[float, ...]
    confidence: float = 1.0
    landmarks: tuple[tuple[float, float], ...] | None = None


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
    links_pruned: int = 0


@dataclass(frozen=True)
class PhotoProvenance:
    """Origin and Kinderpedia CDN metadata associated with an archived photo."""

    rel_path: str
    archive_child: str
    year: str
    month: str
    album_leaf: str
    filename: str
    kinderpedia_album_id: int | None = None
    album_title: str = ""
    album_date: str = ""
    classroom_groups: str = ""
    kinderpedia_image_id: int | None = None
    seq_in_album: int | None = None
    image_dateadd: str = ""
    best_live_url: str = ""
    download_url: str = ""
    fullsize_url: str = ""
    thumb_url: str = ""


@dataclass
class ScanSummary:
    """Summary metrics returned by `faces scan`."""

    photos_total: int = 0
    photos_scanned: int = 0
    photos_cached: int = 0
    photos_failed: int = 0
    faces_total: int = 0
    clusters_created: int = 0
    people_seeded: int = 0
    auto_high_matches: int = 0
    candidate_matches: int = 0
    photos_verified: int = 0
    photos_rekeyed: int = 0
    photos_removed: int = 0
    photos_missing_kept: int = 0
    faces_relinked: int = 0
    faces_removed: int = 0
    labels_orphaned: int = 0
    labels_removed_with_photos: int = 0


@dataclass(frozen=True)
class PhotoScanState:
    """Per-photo fingerprint stored in the index, used to decide skip / verify / rescan / re-key."""

    photo_id: int
    file_size: int
    mtime_ns: int
    detector_id: str
    content_hash: str
    kinderpedia_image_id: int | None
    filename: str
    width: int  # EXIF-normalized pixel size from the last detection pass (feeds `best_live_url`)
    height: int


@dataclass
class RelinkReport:
    """Accumulates what `upsert_scanned_batch` did to previously stored faces (label-preserving rescan)."""

    faces_relinked: int = 0
    faces_new: int = 0
    faces_removed: int = 0
    labels_orphaned: int = 0
    removed_face_ids: list[int] = field(default_factory=list)
    moved_face_ids: list[int] = field(default_factory=list)
