"""Generic Vision-Language Model (VLM) provider protocol and data structures.

This module defines clean, abstract interfaces for visual clothing continuity,
gender plausibility, and candidate verification. Concrete implementations may be
backed by local scripts, multimodal language models, or remote vision endpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class ClothingSignature:
    """Detected appearance and clothing profile of a person in a specific album/session."""

    clothing_summary: str
    top_color: str = ""
    bottom_color: str = ""
    distinctive_features: str = ""
    apparent_gender: str = "unknown"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClothingMatchResult:
    """Result of checking whether a candidate photo/face matches an expected appearance."""

    is_match: bool
    confidence: float
    apparent_gender: str = "unknown"
    gender_conflict: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FaceInspectionResult:
    """Direct visual inspection result of a single face crop in context."""

    apparent_gender: str = "unknown"
    clothing_summary: str = ""
    gender_conflict: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class VlmProvider(Protocol):
    """Abstract interface that all VLM backends implement."""

    def is_available(self) -> bool:
        """Return True if the underlying model/daemon is reachable and ready."""
        ...

    def extract_clothing(
        self,
        image_path: Path | str,
        face_box: tuple[int, int, int, int] | list[int],
        person_name: str,
        target_gender: str = "unknown",
    ) -> ClothingSignature | None:
        """Inspect a confirmed face and extract the child's clothing & appearance signature."""
        ...

    def verify_clothing_match(
        self,
        image_path: Path | str,
        face_box: tuple[int, int, int, int] | list[int],
        target_person_name: str,
        target_gender: str,
        signature: ClothingSignature,
    ) -> ClothingMatchResult:
        """Verify if a candidate face matches the expected clothing signature and gender."""
        ...

    def batch_verify_album_candidates(
        self,
        candidates: list[dict[str, Any]],
        target_person_name: str,
        target_gender: str,
        signature: ClothingSignature,
    ) -> list[dict[str, Any]]:
        """Verify multiple candidate photos in an album against the clothing signature."""
        ...
