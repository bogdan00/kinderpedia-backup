"""Numerical / golden tests for the faces ML layer: alignment, SCRFD decoding, backends and matcher."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from kinderpedia_backup.faces.alignment import (
    ARCFACE_DST_112,
    SCRFD_STRIDE_HEADS,
    align_face_112,
    decode_scrfd_outputs,
    umeyama_similarity_2d,
)
from kinderpedia_backup.faces.onnx_backend import _OpenCvDnnRunner, map_scrfd_outputs_by_shape
from kinderpedia_backup.faces.weights import default_models_dir

HAS_CV2 = importlib.util.find_spec("cv2") is not None


def _smooth_test_image(width: int = 320, height: int = 240, seed: int = 3) -> NDArray[np.uint8]:
    """Deterministic, smoothly textured RGB image (bilinear sampling differences stay sub-level)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    img = np.zeros((height, width, 3), dtype=np.float64)
    for ch in range(3):
        acc = np.zeros_like(xx)
        for _ in range(6):
            fx, fy = rng.uniform(0.05, 0.25, size=2)
            ph = rng.uniform(0, 2 * np.pi)
            acc += np.sin(fx * xx + fy * yy + ph)
        img[..., ch] = 127.5 + 20.0 * acc
    return np.clip(img, 0, 255).astype(np.uint8)


def _landmarks(angle: float, scale: float, offset: tuple[float, float]) -> NDArray[np.float32]:
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = (ARCFACE_DST_112.astype(np.float64) - 56.0) @ rot.T * scale + np.asarray(offset)
    return np.asarray(pts, dtype=np.float32)


def _reference_warp(img: NDArray[np.uint8], fwd: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pure-numpy `cv2.warpAffine(img, fwd, (112, 112))` semantics: src = inv(M) @ dst, bilinear, 0 border."""
    inv = np.linalg.inv(np.vstack([fwd, [0.0, 0.0, 1.0]]))[:2]
    yy, xx = np.mgrid[0:112, 0:112].astype(np.float64)
    sx = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
    sy = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
    h, w = img.shape[:2]
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    ax = (sx - x0)[..., None]
    ay = (sy - y0)[..., None]
    padded = np.zeros((h + 2, w + 2, 3), dtype=np.float64)
    padded[1:-1, 1:-1] = img

    def px(yi: NDArray[np.int64], xi: NDArray[np.int64]) -> NDArray[np.float64]:
        return padded[np.clip(yi + 1, 0, h + 1), np.clip(xi + 1, 0, w + 1)]

    top = px(y0, x0) * (1 - ax) + px(y0, x0 + 1) * ax
    bot = px(y0 + 1, x0) * (1 - ax) + px(y0 + 1, x0 + 1) * ax
    return np.asarray(top * (1 - ay) + bot * ay)


def _to_pixels(tensor: NDArray[np.float32]) -> NDArray[np.float64]:
    return np.asarray(tensor.transpose(1, 2, 0).astype(np.float64) * 127.5 + 127.5)


class AlignmentGoldenTests(unittest.TestCase):
    CASES = (
        (0.0, 1.0, (160.0, 120.0)),  # identity scale
        (0.21, 0.55, (100.0, 90.0)),  # small face, upscaled ~1.8x (group-shot case)
        (-0.35, 1.5, (165.0, 125.0)),  # large rotated face, downscaled
    )

    def test_eye_landmark_lands_on_canonical_template(self) -> None:
        # A bright Gaussian dot centred exactly on the source left-eye landmark must land on the
        # canonical ArcFace left-eye coordinate (OpenCV pixel convention) within 0.1 px.
        for angle, scale, offset in self.CASES:
            lm = _landmarks(angle, scale, offset)
            yy, xx = np.mgrid[0:240, 0:320].astype(np.float64)
            sigma = 2.5 * max(scale, 1.0)
            dot = np.exp(-((xx - lm[0, 0]) ** 2 + (yy - lm[0, 1]) ** 2) / (2 * sigma**2))
            img = np.repeat((dot * 255.0)[..., None], 3, axis=2).astype(np.uint8)
            out = _to_pixels(align_face_112(Image.fromarray(img), lm))[..., 0]
            gy, gx = np.mgrid[0:112, 0:112].astype(np.float64)
            win = (np.abs(gx - ARCFACE_DST_112[0, 0]) < 12) & (np.abs(gy - ARCFACE_DST_112[0, 1]) < 12)
            wts = np.where(win, out, 0.0)
            cx = float((wts * gx).sum() / wts.sum())
            cy = float((wts * gy).sum() / wts.sum())
            self.assertAlmostEqual(cx, float(ARCFACE_DST_112[0, 0]), delta=0.1, msg=f"case {angle, scale}")
            self.assertAlmostEqual(cy, float(ARCFACE_DST_112[0, 1]), delta=0.1, msg=f"case {angle, scale}")

    def test_matches_numpy_reference_warp(self) -> None:
        img = _smooth_test_image()
        for angle, scale, offset in self.CASES:
            lm = _landmarks(angle, scale, offset)
            ours = _to_pixels(align_face_112(Image.fromarray(img), lm))
            ref = _reference_warp(img, umeyama_similarity_2d(lm))
            inner = (slice(4, -4), slice(4, -4))  # border handling is irrelevant for the face crop
            diff = np.abs(ours[inner] - ref[inner])
            self.assertLess(float(diff.mean()), 0.75, msg=f"case {angle, scale}")
            self.assertLessEqual(float(diff.max()), 1.5, msg=f"case {angle, scale}")

    @unittest.skipUnless(HAS_CV2, "opencv not installed")
    def test_matches_cv2_warpaffine_golden(self) -> None:
        import cv2

        img = _smooth_test_image()
        for angle, scale, offset in self.CASES:
            lm = _landmarks(angle, scale, offset)
            ours = _to_pixels(align_face_112(Image.fromarray(img), lm))
            ref = cv2.warpAffine(img, umeyama_similarity_2d(lm), (112, 112), borderValue=0.0).astype(np.float64)
            inner = (slice(4, -4), slice(4, -4))
            diff = np.abs(ours[inner] - ref[inner])
            self.assertLess(float(diff.mean()), 0.75, msg=f"case {angle, scale}")
            self.assertLessEqual(float(diff.max()), 1.5, msg=f"case {angle, scale}")

    def test_uncorrected_pillow_coefficients_would_fail_the_golden(self) -> None:
        # Guard that the reference is sensitive enough to catch the half-pixel regression.
        img = _smooth_test_image()
        lm = _landmarks(0.21, 0.55, (100.0, 90.0))
        fwd = umeyama_similarity_2d(lm)
        inv = np.linalg.inv(np.vstack([fwd, [0.0, 0.0, 1.0]]))
        naive = Image.fromarray(img).transform(
            (112, 112), Image.Transform.AFFINE, tuple(float(v) for v in inv[:2].ravel()), Image.Resampling.BILINEAR
        )
        ref = _reference_warp(img, fwd)
        inner = (slice(4, -4), slice(4, -4))
        self.assertGreater(float(np.abs(np.asarray(naive, dtype=np.float64)[inner] - ref[inner]).mean()), 2.0)


def _encode_scrfd(
    faces: list[tuple[int, int, tuple[float, float, float, float], NDArray[np.float64], float]],
    det_size: int,
) -> dict[str, NDArray[np.float32]]:
    """Inverse of SCRFD decoding: place (stride, anchor_idx, ltrb distances px, kps px (5,2), score)."""
    outs: dict[str, NDArray[np.float32]] = {}
    for stride, s_key, b_key, k_key in SCRFD_STRIDE_HEADS:
        n = 2 * (det_size // stride) ** 2
        outs[s_key] = np.zeros((n, 1), dtype=np.float32)
        outs[b_key] = np.zeros((n, 4), dtype=np.float32)
        outs[k_key] = np.zeros((n, 10), dtype=np.float32)
    keys = {stride: (s, b, k) for stride, s, b, k in SCRFD_STRIDE_HEADS}
    for stride, a_idx, ltrb, kps, score in faces:
        s_key, b_key, k_key = keys[stride]
        feat = det_size // stride
        loc = a_idx // 2
        cx, cy = (loc % feat) * stride, (loc // feat) * stride
        outs[s_key][a_idx, 0] = score
        outs[b_key][a_idx] = np.asarray(ltrb, dtype=np.float32) / stride
        outs[k_key][a_idx] = ((kps - np.array([cx, cy])) / stride).reshape(-1).astype(np.float32)
    return outs


class ScrfdDecodeGoldenTests(unittest.TestCase):
    def test_decode_recovers_boxes_and_landmarks_in_original_pixels(self) -> None:
        det_size, det_scale = 640, 0.5  # e.g. a 1280x960 photo letterboxed to 640x480
        kps_a = np.array([[90.0, 95.0], [110.0, 95.0], [100.0, 105.0], [92.0, 115.0], [108.0, 115.0]])
        kps_b = np.array([[410.0, 300.0], [450.0, 300.0], [430.0, 320.0], [415.0, 340.0], [445.0, 340.0]])
        # Face A: stride 8, location (12, 12) -> centre (96, 96), second anchor of the pair.
        # Face B: stride 32, location (13, 9) -> centre (416, 288).
        faces = [
            (8, 2 * (12 * 80 + 12) + 1, (16.0, 14.0, 18.0, 30.0), kps_a, 0.91),
            (32, 2 * (9 * 20 + 13), (30.0, 40.0, 60.0, 80.0), kps_b, 0.77),
            (16, 2 * (5 * 40 + 5), (10.0, 10.0, 10.0, 10.0), kps_a, 0.30),  # below score threshold
        ]
        dets = decode_scrfd_outputs(
            _encode_scrfd(faces, det_size), det_size=det_size, det_scale=det_scale, img_width=1280, img_height=960
        )
        self.assertEqual(len(dets), 2)
        self.assertAlmostEqual(dets[0].score, 0.91, places=5)
        box_a = dets[0].box
        self.assertEqual((box_a.left, box_a.top, box_a.right, box_a.bottom), (160, 164, 228, 252))
        np.testing.assert_allclose(dets[0].landmarks, kps_a / det_scale, atol=1e-3)
        box_b = dets[1].box
        self.assertEqual((box_b.left, box_b.top, box_b.right, box_b.bottom), (772, 496, 952, 736))
        np.testing.assert_allclose(dets[1].landmarks, kps_b / det_scale, atol=1e-3)

    def test_decode_nms_and_min_face_gate(self) -> None:
        kps = np.tile(np.array([[100.0, 100.0]]), (5, 1))
        faces = [
            (8, 2 * (12 * 80 + 12), (20.0, 20.0, 20.0, 20.0), kps, 0.90),
            (8, 2 * (12 * 80 + 12) + 1, (21.0, 19.0, 20.0, 21.0), kps, 0.80),  # duplicate -> suppressed
            (8, 2 * (40 * 80 + 40), (5.0, 5.0, 5.0, 5.0), kps, 0.95),  # 10 px face -> below min_face_px
        ]
        dets = decode_scrfd_outputs(_encode_scrfd(faces, 640), det_size=640, det_scale=1.0, img_width=640, img_height=640)
        self.assertEqual([round(d.score, 2) for d in dets], [0.90])


class BackendOutputMappingTests(unittest.TestCase):
    def test_shape_mapping_is_independent_of_graph_order(self) -> None:
        det_size = 640
        named: dict[str, NDArray[np.float32]] = {}
        rng = np.random.default_rng(0)
        for stride, s_key, b_key, k_key in SCRFD_STRIDE_HEADS:
            n = 2 * (det_size // stride) ** 2
            named[s_key] = rng.random((n, 1)).astype(np.float32)
            named[b_key] = rng.random((n, 4)).astype(np.float32)
            named[k_key] = rng.random((n, 10)).astype(np.float32)
        # OpenCV / ONNX graph order is type-major (all scores, all boxes, all kps):
        graph_order = [str(h[pos]) for pos in (1, 2, 3) for h in SCRFD_STRIDE_HEADS]
        mapped = map_scrfd_outputs_by_shape([named[k] for k in graph_order], det_size=det_size)
        self.assertEqual(set(mapped), set(named))
        for key, arr in named.items():
            self.assertIs(mapped[key], arr)
        # Shuffled order and an extra batch axis still map correctly:
        shuffled = list(reversed(graph_order))
        mapped2 = map_scrfd_outputs_by_shape([named[k][None] for k in shuffled], det_size=det_size)
        for key, arr in named.items():
            np.testing.assert_array_equal(mapped2[key][0], arr)
        # Unknown shapes -> refuse to guess:
        self.assertEqual(map_scrfd_outputs_by_shape([named[k] for k in graph_order[:-1]], det_size=det_size), {})
        self.assertEqual(map_scrfd_outputs_by_shape([np.zeros((7, 3))], det_size=det_size), {})


def _cached_weights() -> tuple[Path, Path] | None:
    md = default_models_dir()
    det, rec = md / "det_10g.onnx", md / "w600k_r50.onnx"
    return (det, rec) if det.is_file() and rec.is_file() else None


@unittest.skipUnless(HAS_CV2 and _cached_weights() is not None, "needs opencv + cached ONNX weights")
class BackendParityTests(unittest.TestCase):
    """Runs only on machines with the model cache (never downloads)."""

    def test_opencv_runner_exposes_all_heads_and_matches_onnxruntime(self) -> None:
        weights = _cached_weights()
        assert weights is not None
        det_path, rec_path = weights
        blob = np.random.default_rng(1).uniform(-1.0, 1.0, size=(1, 3, 640, 640)).astype(np.float32)
        crops = np.random.default_rng(2).uniform(-1.0, 1.0, size=(2, 3, 112, 112)).astype(np.float32)
        cv_runner = _OpenCvDnnRunner(det_path, rec_path)
        cv_out = cv_runner.run_detector(blob)
        for stride, s_key, b_key, k_key in SCRFD_STRIDE_HEADS:
            n = 2 * (640 // stride) ** 2
            self.assertEqual(int(cv_out[s_key].size), n)
            self.assertEqual(int(cv_out[b_key].size), n * 4)
            self.assertEqual(int(cv_out[k_key].size), n * 10)
        cv_emb = cv_runner.run_embedder(crops[:1])
        if importlib.util.find_spec("onnxruntime") is None:
            return
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        det = ort.InferenceSession(str(det_path), so, providers=["CPUExecutionProvider"])
        rec = ort.InferenceSession(str(rec_path), so, providers=["CPUExecutionProvider"])
        names = [o.name for o in det.get_outputs()]
        ort_out = dict(zip(names, det.run(names, {det.get_inputs()[0].name: blob}), strict=True))
        for key in ort_out:
            np.testing.assert_allclose(np.asarray(cv_out[key]).reshape(-1), ort_out[key].reshape(-1), atol=1e-3)
        ort_emb = rec.run(None, {rec.get_inputs()[0].name: crops[:1]})[0]
        a = cv_emb[0] / np.linalg.norm(cv_emb[0])
        b = ort_emb[0] / np.linalg.norm(ort_emb[0])
        self.assertGreater(float(a @ b), 0.999)


if __name__ == "__main__":
    unittest.main()
