"""Facial recognition and smart per-child photo organization package."""

from __future__ import annotations

from .models import BoundingBox, DetectedFace, FaceMatch, LinkMode, OrganizeStats, ReferenceProfile
from .pipeline import FaceOrganizationPipeline
from .recognizer import (
    FaceRecognizer,
    MissingFacesBackendError,
    cosine_similarity,
    load_default_recognizer,
)
from .storage import discover_backup_photos, discover_reference_photos, link_or_copy_photo

__all__ = [
    "BoundingBox",
    "DetectedFace",
    "FaceMatch",
    "FaceOrganizationPipeline",
    "FaceRecognizer",
    "LinkMode",
    "MissingFacesBackendError",
    "OrganizeStats",
    "ReferenceProfile",
    "cosine_similarity",
    "discover_backup_photos",
    "discover_reference_photos",
    "link_or_copy_photo",
    "load_default_recognizer",
]
