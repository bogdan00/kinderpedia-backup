"""No-op VLM provider used when no vision backend is configured."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ClothingMatchResult, ClothingSignature


class NullVlmProvider:
    """Safe fallback provider that gracefully reports unavailable."""

    def is_available(self) -> bool:
        return False

    def extract_clothing(
        self,
        image_path: Path | str,
        face_box: tuple[int, int, int, int] | list[int],
        person_name: str,
        target_gender: str = "unknown",
    ) -> ClothingSignature | None:
        return None

    def verify_clothing_match(
        self,
        image_path: Path | str,
        face_box: tuple[int, int, int, int] | list[int],
        target_person_name: str,
        target_gender: str,
        signature: ClothingSignature,
    ) -> ClothingMatchResult:
        return ClothingMatchResult(
            is_match=False,
            confidence=0.0,
            apparent_gender="unknown",
            gender_conflict=False,
            notes="No VLM backend configured.",
        )

    def batch_verify_album_candidates(
        self,
        candidates: list[dict[str, Any]],
        target_person_name: str,
        target_gender: str,
        signature: ClothingSignature,
    ) -> list[dict[str, Any]]:
        return []
