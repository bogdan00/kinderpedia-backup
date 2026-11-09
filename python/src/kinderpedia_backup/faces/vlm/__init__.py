"""Generic Vision-Language Model (VLM) interfaces and pipeline."""

from .base import (
    ClothingMatchResult,
    ClothingSignature,
    FaceInspectionResult,
    VlmProvider,
)
from .loader import get_vlm_provider
from .null_provider import NullVlmProvider
from .pipeline import analyze_album_continuity

__all__ = [
    "ClothingMatchResult",
    "ClothingSignature",
    "FaceInspectionResult",
    "NullVlmProvider",
    "VlmProvider",
    "analyze_album_continuity",
    "get_vlm_provider",
]
