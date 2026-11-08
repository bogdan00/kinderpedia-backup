"""Facial recognition, provenance indexing, and active-learning categorization package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import (
    AssignmentSource,
    AssignmentStatus,
    BoundingBox,
    DetectedFace,
    FaceMatch,
    LinkMode,
    OrganizeStats,
    PersonRole,
    PhotoProvenance,
    ReferenceProfile,
    ScanSummary,
)
from .provenance import ProvenanceResolver, discover_child_avatars, resolve_best_live_url
from .recognizer import (
    FaceRecognizer,
    MissingFacesBackendError,
    cosine_similarity,
    load_default_recognizer,
)
from .storage import discover_backup_photos, discover_reference_photos, link_or_copy_photo

if TYPE_CHECKING:
    from .index_db import FacesIndexDB
    from .matcher import rebuild_yearly_clusters, recompute_active_matching
    from .scanner import run_archive_face_scan
    from .server import serve_faces_ui

__all__ = [
    "AssignmentSource",
    "AssignmentStatus",
    "BoundingBox",
    "DetectedFace",
    "FaceMatch",
    "FaceRecognizer",
    "FacesIndexDB",
    "LinkMode",
    "MissingFacesBackendError",
    "OrganizeStats",
    "PersonRole",
    "PhotoProvenance",
    "ProvenanceResolver",
    "ReferenceProfile",
    "ScanSummary",
    "cosine_similarity",
    "discover_backup_photos",
    "discover_child_avatars",
    "discover_reference_photos",
    "link_or_copy_photo",
    "load_default_recognizer",
    "rebuild_yearly_clusters",
    "recompute_active_matching",
    "resolve_best_live_url",
    "run_archive_face_scan",
    "serve_faces_ui",
]


def __getattr__(name: str) -> Any:
    """Lazy-load numpy/Pillow-backed modules so `kinderpedia-backup sync` requires only `requests`."""
    try:
        if name == "FacesIndexDB":
            from .index_db import FacesIndexDB

            return FacesIndexDB
        if name in ("rebuild_yearly_clusters", "recompute_active_matching"):
            from . import matcher

            return getattr(matcher, name)
        if name == "run_archive_face_scan":
            from .scanner import run_archive_face_scan

            return run_archive_face_scan
        if name == "serve_faces_ui":
            from .server import serve_faces_ui

            return serve_faces_ui
    except ImportError as exc:
        raise MissingFacesBackendError(
            f"Optional `faces` dependencies are not installed ({exc}). "
            'Install them (Python 3.10-3.14) via: pip install "./python[faces]"'
        ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
