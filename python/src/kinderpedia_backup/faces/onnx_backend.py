"""Dual-runtime ONNX inference backend (`onnxruntime` or `opencv-python-headless` `cv2.dnn`)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from ..console import log
from .alignment import (
    ALIGNMENT_VERSION,
    SCRFD_STRIDE_HEADS,
    align_face_112,
    crop_face_portrait,
    decode_scrfd_outputs,
    letterbox_scrfd_input,
)
from .models import DetectedFace
from .recognizer import MissingFacesBackendError
from .weights import ensure_buffalo_l_weights

_SCRFD_OUT_NAMES: tuple[str, ...] = tuple(
    name for _, s_name, b_name, k_name in SCRFD_STRIDE_HEADS for name in (s_name, b_name, k_name)
)


class _EngineRunner:
    """Internal runtime adapter wrapping either `onnxruntime` or `cv2.dnn`."""

    def run_detector(self, blob: NDArray[np.float32]) -> dict[str, NDArray[Any]]:
        raise NotImplementedError

    def run_embedder(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError


class _OrtRunner(_EngineRunner):
    def __init__(self, det_path: Path, rec_path: Path, *, det_size: int = 640) -> None:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        det_opts = ort.SessionOptions()
        rec_opts = ort.SessionOptions()
        # Suppress verbose C++ graph partitioning warnings during CLI scans:
        det_opts.log_severity_level = 3
        rec_opts.log_severity_level = 3

        providers: list[Any]
        if "CoreMLExecutionProvider" in avail:
            # Pin SCRFD dynamic spatial dims ('?') and ArcFace batch dim ('None') so
            # CoreMLExecutionProvider constant-folds shape ops onto Apple Neural Engine / GPU:
            det_opts.add_free_dimension_override_by_name("?", det_size)
            rec_opts.add_free_dimension_override_by_name("None", 1)
            det_opts.intra_op_num_threads = 1
            rec_opts.intra_op_num_threads = 1

            cache_dir = det_path.parent / "coreml_cache" / f"s{det_size}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            coreml_opts = {
                "ModelFormat": "NeuralNetwork",
                "MLComputeUnits": "ALL",
                "ModelCacheDirectory": str(cache_dir),
            }
            providers = [
                ("CoreMLExecutionProvider", coreml_opts),
                "CPUExecutionProvider",
            ]
        else:
            det_opts.intra_op_num_threads = 2
            rec_opts.intra_op_num_threads = 2
            providers = [
                p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail
            ] or ["CPUExecutionProvider"]

        self._det = ort.InferenceSession(str(det_path), sess_options=det_opts, providers=providers)
        self._rec = ort.InferenceSession(str(rec_path), sess_options=rec_opts, providers=providers)
        self._det_in = self._det.get_inputs()[0].name
        self._det_outs = [o.name for o in self._det.get_outputs()]
        self._rec_in = self._rec.get_inputs()[0].name
        self._rec_out = self._rec.get_outputs()[0].name
        self.active_providers: tuple[str, ...] = tuple(self._det.get_providers())

    def run_detector(self, blob: NDArray[np.float32]) -> dict[str, NDArray[Any]]:
        raw = self._det.run(self._det_outs, {self._det_in: blob})
        return {name: np.asarray(val) for name, val in zip(self._det_outs, raw, strict=True)}

    def run_embedder(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        if batch.shape[0] <= 1:
            raw = self._rec.run([self._rec_out], {self._rec_in: batch})[0]
            return np.asarray(raw, dtype=np.float32)
        # CoreMLExecutionProvider compiles model with fixed batch=1:
        outs = [
            self._rec.run([self._rec_out], {self._rec_in: batch[i : i + 1]})[0]
            for i in range(batch.shape[0])
        ]
        return np.concatenate(outs, axis=0, dtype=np.float32)


class _OpenCvDnnRunner(_EngineRunner):
    def __init__(self, det_path: Path, rec_path: Path) -> None:
        import cv2

        self._det = cv2.dnn.readNetFromONNX(str(det_path))
        self._rec = cv2.dnn.readNetFromONNX(str(rec_path))
        self._det_outs = list(self._det.getUnconnectedOutLayersNames())
        self._rec_outs = list(self._rec.getUnconnectedOutLayersNames())

    def run_detector(self, blob: NDArray[np.float32]) -> dict[str, NDArray[Any]]:
        self._det.setInput(blob)
        outs = self._det.forward(self._det_outs)
        result = {name: np.asarray(arr) for name, arr in zip(self._det_outs, outs, strict=True)}
        if not all(name in result for name in _SCRFD_OUT_NAMES):
            # OpenCV builds that do not expose the ONNX tensor names: identify heads by shape,
            # never by position (graph order is type-major, not the stride-major name table).
            result.update(map_scrfd_outputs_by_shape(list(outs), det_size=int(blob.shape[-1])))
        return result

    def run_embedder(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        self._rec.setInput(batch)
        out = self._rec.forward(self._rec_outs)[0]
        return np.asarray(out, dtype=np.float32)


def map_scrfd_outputs_by_shape(outs: list[NDArray[Any]], *, det_size: int) -> dict[str, NDArray[Any]]:
    """
    Name anonymous SCRFD output tensors from their shapes: the last dimension identifies the head
    (1 = score, 4 = bbox distances, 10 = keypoints) and the anchor count `2 * (det_size / stride)^2`
    identifies the stride. Returns `{}` unless all nine heads are found exactly once.
    """
    head_pos_by_width = {1: 0, 4: 1, 10: 2}  # (score, bbox, kps) position in SCRFD_STRIDE_HEADS
    names_by_rows = {2 * (det_size // stride) ** 2: (s, b, k) for stride, s, b, k in SCRFD_STRIDE_HEADS}
    mapped: dict[str, NDArray[Any]] = {}
    for arr_raw in outs:
        arr = np.asarray(arr_raw)
        if arr.ndim == 0:
            return {}
        width = int(arr.shape[-1])
        rows = int(arr.size // max(width, 1))
        if width not in head_pos_by_width or rows not in names_by_rows:
            return {}
        key = names_by_rows[rows][head_pos_by_width[width]]
        if key in mapped:
            return {}
        mapped[key] = arr
    return mapped if len(mapped) == len(_SCRFD_OUT_NAMES) else {}


def _probe_engine_kind(backend: str) -> str:
    norm = backend.strip().lower()
    if norm in ("auto", "onnx", "onnxruntime"):
        try:
            import onnxruntime  # noqa: F401

            return "onnxruntime"
        except ImportError:
            if norm == "onnxruntime":
                raise MissingFacesBackendError(
                    "Requested `--backend onnxruntime`, but `onnxruntime` is not installed."
                ) from None
    if norm in ("auto", "onnx", "opencv", "cv2"):
        try:
            import cv2  # noqa: F401

            return "opencv"
        except ImportError:
            pass
    raise MissingFacesBackendError(
        f"No ONNX runtime available for backend={backend!r}. "
        'Install the faces extras via: pip install "./python[faces]"'
    )


def describe_engine(engine_kind: str) -> str:
    """Human-readable engine name + version for scan logs (providers are logged once a session exists)."""
    try:
        if engine_kind == "onnxruntime":
            import onnxruntime as ort

            avail = ", ".join(ort.get_available_providers())
            return f"onnxruntime {ort.__version__} (available providers: {avail})"
        import cv2

        return f"OpenCV DNN {cv2.__version__} (CPU)"
    except Exception:  # pragma: no cover - purely informational
        return engine_kind


class ScrfdArcFaceRecognizer:
    """
    Local face detector + 512-D ArcFace embedder powered by SCRFD-10GF (`det_10g.onnx`)
    and WebFace600K ResNet-50 (`w600k_r50.onnx`).

    Shares a single C++-thread-safe `onnxruntime` session pair (compiled once onto CoreML/CUDA/CPU)
    across worker threads, or uses thread-local `cv2.dnn` nets when falling back to OpenCV.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        det_size: int = 640,
        score_thresh: float = 0.55,
        nms_thresh: float = 0.40,
        min_face_px: int = 28,
        models_dir: Path | None = None,
    ) -> None:
        self.engine_kind = _probe_engine_kind(backend)
        self.det_size = det_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.min_face_px = min_face_px
        self.det_path, self.rec_path = ensure_buffalo_l_weights(models_dir)
        self._tls = threading.local()
        self._ort_lock = threading.Lock()
        self._shared_ort_runner: _OrtRunner | None = None
        log(f"[Faces] Inference engine: {describe_engine(self.engine_kind)} (model {self.model_id})")
        if self.engine_kind == "opencv" and backend.strip().lower() == "auto":
            log(
                "[Faces] onnxruntime is not installed; using the slower OpenCV DNN fallback. "
                'Install it for CoreML/CUDA acceleration: pip install "onnxruntime>=1.17"'
            )

    @property
    def model_id(self) -> str:
        # `_align{N}` tracks the crop geometry: embeddings from a different alignment are not comparable.
        return f"scrfd_10g_{self.det_size}_arcface_r50_v1_align{ALIGNMENT_VERSION}"

    def _runner(self) -> _EngineRunner:
        if self.engine_kind == "onnxruntime":
            if self._shared_ort_runner is None:
                with self._ort_lock:
                    if self._shared_ort_runner is None:
                        self._shared_ort_runner = _OrtRunner(
                            self.det_path,
                            self.rec_path,
                            det_size=self.det_size,
                        )
                        log(
                            "[Faces] onnxruntime execution providers: "
                            + ", ".join(self._shared_ort_runner.active_providers)
                        )
            return self._shared_ort_runner

        runner: _EngineRunner | None = getattr(self._tls, "runner", None)
        if runner is None:
            runner = _OpenCvDnnRunner(self.det_path, self.rec_path)
            self._tls.runner = runner
        return runner

    def analyze_photo(
        self,
        image_path: Path,
        *,
        thumb_size: int | None = None,
    ) -> tuple[int, int, list[DetectedFace], list[Image.Image]]:
        """
        Single-pass EXIF-normalized decode returning `(width, height, detected_faces, portrait_thumbs)`.
        """
        with Image.open(image_path) as raw_im:
            pil_rgb = ImageOps.exif_transpose(raw_im).convert("RGB")
        width, height = pil_rgb.size

        blob, det_scale = letterbox_scrfd_input(pil_rgb, det_size=self.det_size)
        runner = self._runner()
        det_outs = runner.run_detector(blob)
        raw_dets = decode_scrfd_outputs(
            det_outs,
            det_size=self.det_size,
            det_scale=det_scale,
            img_width=width,
            img_height=height,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            min_face_px=self.min_face_px,
        )
        if not raw_dets:
            return width, height, [], []

        aligned_tensors = [align_face_112(pil_rgb, d.landmarks) for d in raw_dets]
        batch = np.stack(aligned_tensors, axis=0)
        raw_embs = runner.run_embedder(batch)
        norms = np.maximum(np.linalg.norm(raw_embs, axis=1, keepdims=True), 1e-8)
        normed = raw_embs / norms

        faces: list[DetectedFace] = []
        thumbs: list[Image.Image] = []
        for det_item, emb_vec in zip(raw_dets, normed, strict=True):
            kps_tuple = tuple((float(pt[0]), float(pt[1])) for pt in det_item.landmarks)
            faces.append(
                DetectedFace(
                    box=det_item.box,
                    embedding=tuple(float(v) for v in emb_vec.tolist()),
                    confidence=det_item.score,
                    landmarks=kps_tuple,
                )
            )
            if thumb_size is not None:
                thumbs.append(crop_face_portrait(pil_rgb, det_item.box, size=thumb_size))
        return width, height, faces, thumbs

    def detect_and_embed(self, image_path: Path) -> list[DetectedFace]:
        """Satisfy `FaceRecognizer` protocol."""
        _, _, faces, _ = self.analyze_photo(image_path, thumb_size=None)
        return faces
