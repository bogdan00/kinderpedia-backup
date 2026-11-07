"""Pluggable recognizer interface and pure-Python similarity math."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import DetectedFace


class MissingFacesBackendError(RuntimeError):
    """Raised when facial-recognition ML extras are not yet installed or configured."""


@runtime_checkable
class FaceRecognizer(Protocol):
    """Interface implemented by any local face detection + embedding backend."""

    @property
    def model_id(self) -> str:
        """Stable identifier used to invalidate cached scan results on model upgrades."""

    def detect_and_embed(self, image_path: Path) -> list[DetectedFace]:
        """Return all faces and embedding vectors detected in `image_path`."""


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity in [-1.0, 1.0] between two equal-length vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        raise ValueError("Embedding vectors must be non-empty and have matching dimensions.")
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_default_recognizer(backend: str = "auto") -> FaceRecognizer:
    """
    Resolve an installed local face-recognition backend.

    Designed as the extension seam for upcoming InsightFace (ONNX) / dlib / Vision adapters
    so core `sync` users never have to install heavy ML wheels.
    """
    raise MissingFacesBackendError(
        f"Facial recognition backend ({backend!r}) is not yet installed/enabled. "
        "The `faces` pipeline architecture (reference enrollment, incremental cache, "
        "cosine matching, and zero-copy hardlink organization) is ready — plug a "
        "`FaceRecognizer` adapter into `kinderpedia_backup.faces.recognizer` and install "
        "optional ML extras via `pip install \"./python[faces]\"`."
    )
