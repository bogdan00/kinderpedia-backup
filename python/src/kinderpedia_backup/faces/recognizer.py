"""Pluggable recognizer interface and pure-Python similarity math."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import DetectedFace


class MissingFacesBackendError(RuntimeError):
    """Raised when facial-recognition ML extras are not installed or configured."""


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


def load_default_recognizer(*, det_size: int = 640) -> FaceRecognizer:
    """Load the local SCRFD-10GF + ArcFace R50 recognizer running on `onnxruntime`."""
    try:
        from .onnx_backend import ScrfdArcFaceRecognizer
    except ImportError as exc:
        raise MissingFacesBackendError(
            f"Facial recognition dependencies missing ({exc}). The faces engine (onnxruntime) "
            'needs Python 3.10-3.14; install it with: pip install "./python[faces]"'
        ) from exc
    return ScrfdArcFaceRecognizer(det_size=det_size)
