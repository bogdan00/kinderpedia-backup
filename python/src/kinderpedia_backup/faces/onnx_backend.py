"""ONNX Runtime inference engine: SCRFD-10GF detection + ArcFace R50 embedding (CoreML / CUDA / CPU)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from PIL import Image, ImageOps

from ..console import log
from .alignment import (
    ALIGNMENT_VERSION,
    align_face_112,
    crop_face_portrait,
    decode_scrfd_outputs,
    letterbox_scrfd_input,
)
from .models import DetectedFace
from .weights import ensure_buffalo_l_weights


class _OrtRunner:
    """One thread-safe detector + embedder `InferenceSession` pair shared by all scan worker threads."""

    def __init__(self, det_path: Path, rec_path: Path, *, det_size: int) -> None:
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
        # SCRFD heads are addressed by their ONNX output names (see `alignment.SCRFD_STRIDE_HEADS`).
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


class ScrfdArcFaceRecognizer:
    """
    Local face detector + 512-D ArcFace embedder powered by SCRFD-10GF (`det_10g.onnx`)
    and WebFace600K ResNet-50 (`w600k_r50.onnx`).

    One `onnxruntime` session pair (compiled once onto CoreML / CUDA / CPU) is shared by all scan
    worker threads; `InferenceSession.run` is thread-safe.
    """

    def __init__(
        self,
        *,
        det_size: int = 640,
        score_thresh: float = 0.55,
        nms_thresh: float = 0.40,
        min_face_px: int = 28,
        models_dir: Path | None = None,
    ) -> None:
        self.det_size = det_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.min_face_px = min_face_px
        self.det_path, self.rec_path = ensure_buffalo_l_weights(models_dir)
        self._runner = _OrtRunner(self.det_path, self.rec_path, det_size=det_size)
        log(
            f"[Faces] Inference engine: onnxruntime {ort.__version__} "
            f"(providers: {', '.join(self._runner.active_providers)}; model {self.model_id})"
        )

    @property
    def model_id(self) -> str:
        # `_align{N}` tracks the crop geometry: embeddings from a different alignment are not comparable.
        return f"scrfd_10g_{self.det_size}_arcface_r50_v1_align{ALIGNMENT_VERSION}"

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
        det_outs = self._runner.run_detector(blob)
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
        raw_embs = self._runner.run_embedder(batch)
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
