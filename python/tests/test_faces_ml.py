"""Numerical / golden tests for the faces ML layer: alignment, SCRFD decoding, ONNX Runtime engine and matcher."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from PIL import Image

from kinderpedia_backup.faces import BoundingBox, DetectedFace, FacesIndexDB, recompute_active_matching
from kinderpedia_backup.faces.alignment import (
    ARCFACE_DST_112,
    SCRFD_STRIDE_HEADS,
    align_face_112,
    decode_scrfd_outputs,
    letterbox_scrfd_input,
    umeyama_similarity_2d,
)
from kinderpedia_backup.faces.models import PhotoProvenance
from kinderpedia_backup.faces.onnx_backend import ScrfdArcFaceRecognizer, _OrtRunner
from kinderpedia_backup.faces.weights import (
    DET_MODEL_FILENAME,
    REC_MODEL_FILENAME,
    default_models_dir,
    ensure_buffalo_l_weights,
)


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

    def test_matches_cv2_warpaffine_golden(self) -> None:
        # opencv-python-headless (dev extra) is only the golden reference; production warps with Pillow.
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


REQUIRE_ORT_PARITY = os.environ.get("KP_REQUIRE_ORT_PARITY") == "1"


def _parity_weights() -> tuple[Path, Path] | None:
    """Pinned weights: downloaded + SHA-verified when parity is required (CI), else only if cached."""
    if REQUIRE_ORT_PARITY:
        return ensure_buffalo_l_weights()
    md = default_models_dir()
    det, rec = md / DET_MODEL_FILENAME, md / REC_MODEL_FILENAME
    return (det, rec) if det.is_file() and rec.is_file() else None


class OrtEngineParityTests(unittest.TestCase):
    """
    The production `_OrtRunner` (CoreML on macOS, CUDA if available, else CPU) against a plain
    CPU `InferenceSession` on the pinned SCRFD / ArcFace weights. Skips without cached weights unless
    `KP_REQUIRE_ORT_PARITY=1`, which downloads them and turns any problem into a failure.
    """

    weights: tuple[Path, Path]

    @classmethod
    def setUpClass(cls) -> None:
        weights = _parity_weights()
        if weights is None:
            raise unittest.SkipTest("ONNX weights not cached (set KP_REQUIRE_ORT_PARITY=1 to download)")
        cls.weights = weights

    def _cpu_sessions(self) -> tuple[ort.InferenceSession, ort.InferenceSession]:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        det_path, rec_path = self.weights
        return (
            ort.InferenceSession(str(det_path), so, providers=["CPUExecutionProvider"]),
            ort.InferenceSession(str(rec_path), so, providers=["CPUExecutionProvider"]),
        )

    def test_runner_maps_scrfd_heads_by_name_and_matches_cpu_reference(self) -> None:
        det_path, rec_path = self.weights
        runner = _OrtRunner(det_path, rec_path, det_size=640)
        self.assertIn("CPUExecutionProvider", runner.active_providers)
        blob, _ = letterbox_scrfd_input(Image.fromarray(_smooth_test_image(960, 720)), det_size=640)
        out = runner.run_detector(blob)
        for stride, s_key, b_key, k_key in SCRFD_STRIDE_HEADS:
            n = 2 * (640 // stride) ** 2
            self.assertEqual(int(out[s_key].size), n)
            self.assertEqual(int(out[b_key].size), n * 4)
            self.assertEqual(int(out[k_key].size), n * 10)
        det_cpu, rec_cpu = self._cpu_sessions()
        names = [o.name for o in det_cpu.get_outputs()]
        ref = dict(zip(names, det_cpu.run(names, {det_cpu.get_inputs()[0].name: blob}), strict=True))
        self.assertEqual(set(ref), set(out))
        # CoreML runs in reduced precision: scores must agree tightly; box / keypoint distances (in
        # stride units, |v| up to ~3.5) on this synthetic input drift by <= ~0.25 at worst.
        score_keys = {s for _, s, _, _ in SCRFD_STRIDE_HEADS}
        for key, ref_arr in ref.items():
            diff = np.abs(out[key].reshape(-1) - ref_arr.reshape(-1))
            if key in score_keys:
                self.assertLess(float(diff.max()), 0.01, msg=key)
            else:
                self.assertLess(float(diff.mean()), 0.03, msg=key)
                self.assertLess(float(diff.max()), 0.35, msg=key)

        crops = np.random.default_rng(2).uniform(-1.0, 1.0, size=(3, 3, 112, 112)).astype(np.float32)
        emb = runner.run_embedder(crops)
        self.assertEqual(emb.shape, (3, 512))
        ref_emb = np.concatenate(
            [rec_cpu.run(None, {rec_cpu.get_inputs()[0].name: crops[i : i + 1]})[0] for i in range(3)]
        )
        for i in range(3):
            a = emb[i] / np.linalg.norm(emb[i])
            b = ref_emb[i] / np.linalg.norm(ref_emb[i])
            self.assertGreater(float(a @ b), 0.995)

    def test_recognizer_end_to_end_on_a_faceless_image(self) -> None:
        rec = ScrfdArcFaceRecognizer(models_dir=self.weights[0].parent)
        self.assertTrue(rec.model_id.startswith("scrfd_10g_640_arcface_r50_v1_align"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.png"
            Image.fromarray(_smooth_test_image()).save(path)
            width, height, faces, thumbs = rec.analyze_photo(path, thumb_size=64)
        self.assertEqual((width, height, faces, thumbs), (320, 240, [], []))


class _SyntheticClassroom:
    """
    ArcFace-like synthetic embeddings: every identity is a random unit centre in 512-D and a face is
    `alpha * centre + sqrt(1 - alpha^2) * noise`, so same-person cosine ~ alpha^2 (0.5-0.8), strangers
    ~0 +- 0.05, and a look-alike (centre cosine `c`) lands at ~c * alpha^2 -- the ambiguous band the
    matcher's thresholds are tuned for. No real data involved.
    """

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.centres: dict[str, NDArray[np.float64]] = {}
        self.items: list[tuple[str, NDArray[np.float32]]] = []  # (identity, embedding) per photo

    def _unit(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(v / np.linalg.norm(v))

    def identity(self, name: str, *, like: str | None = None, centre_sim: float = 0.0) -> None:
        raw = self._unit(self.rng.standard_normal(512))
        if like is not None:
            base = self.centres[like]
            raw = self._unit(raw - (raw @ base) * base)
            raw = centre_sim * base + np.sqrt(1.0 - centre_sim**2) * raw
        self.centres[name] = self._unit(raw)

    def faces(self, name: str, count: int, *, alpha: tuple[float, float] = (0.74, 0.88)) -> None:
        c = self.centres[name]
        for _ in range(count):
            a = float(self.rng.uniform(*alpha))
            noise = self._unit(self.rng.standard_normal(512))
            noise = self._unit(noise - (noise @ c) * c)
            self.items.append((name, (a * c + np.sqrt(1.0 - a * a) * noise).astype(np.float32)))

    def save(self, db: FacesIndexDB) -> list[tuple[str, int, int]]:
        order = self.rng.permutation(len(self.items))
        batch: list[tuple[PhotoProvenance, int, int, int, int, str, list[DetectedFace]]] = []
        for slot, i in enumerate(order.tolist()):
            month, day = 1 + (slot * 5 // 28) % 12, 1 + (slot * 5) % 28
            leaf = f"2026-{month:02d}-{day:02d} Day {slot}"
            batch.append(
                (
                    PhotoProvenance(
                        f"Alex/Albums/2026/{month:02d}/{leaf}/{slot:03d}.jpg",
                        "Alex",
                        "2026",
                        f"{month:02d}",
                        leaf,
                        f"{slot:03d}.jpg",
                        album_date=f"2026-{month:02d}-{day:02d}",
                    ),
                    400,
                    300,
                    100 + slot,
                    1,
                    "synthetic",
                    [
                        DetectedFace(
                            box=BoundingBox(top=10, right=110, bottom=110, left=10),
                            embedding=tuple(float(x) for x in self.items[i][1].tolist()),
                            confidence=0.9,
                        )
                    ],
                )
            )
        saved = db.upsert_scanned_batch(batch)
        return [(self.items[i][0], p_id, f_ids[0]) for i, (p_id, f_ids) in zip(order.tolist(), saved, strict=True)]


def _model_rows(db: FacesIndexDB) -> dict[int, tuple[int, str, str]]:
    return {
        int(r[0]): (int(r[1]), str(r[2]), str(r[3]))
        for r in db._conn.execute(
            "SELECT face_id, person_id, status, source FROM assignments "
            "WHERE status IN ('auto_high', 'candidate') AND face_id IS NOT NULL"
        )
    }


class MatcherRealisticEmbeddingTests(unittest.TestCase):
    KIDS = ("Maya", "Leo", "Sofia", "Nina")

    def _classroom(self, seed: int) -> _SyntheticClassroom:
        cls = _SyntheticClassroom(seed)
        for kid in self.KIDS:
            cls.identity(kid)
            cls.faces(kid, 36)
        cls.identity("Theo", like="Leo", centre_sim=0.62)  # un-enrolled look-alike of Leo
        cls.faces("Theo", 14)
        for i in range(30):
            cls.identity(f"stranger{i}")
            cls.faces(f"stranger{i}", 1)
        return cls

    def test_auto_matches_are_precise_on_non_orthogonal_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            photos = self._classroom(seed=21).save(db)
            pid_of = {kid: db.ensure_person(kid) for kid in self.KIDS}
            labelled: set[int] = set()
            per_kid: dict[str, int] = dict.fromkeys(self.KIDS, 0)
            for who, p_id, f_id in photos:
                if who in per_kid and per_kid[who] < 12:
                    db.tag_single_face(photo_id=p_id, face_id=f_id, person_id=pid_of[who])
                    per_kid[who] += 1
                    labelled.add(f_id)
            recompute_active_matching(db)
            rows = _model_rows(db)
            auto_ok = auto_bad = kid_open = 0
            for who, _p_id, f_id in photos:
                if f_id in labelled:
                    continue
                got = rows.get(f_id)
                truth = pid_of.get(who)
                kid_open += truth is not None
                if got is not None and got[1] == "auto_high":
                    if got[0] == truth:
                        auto_ok += 1
                    else:
                        auto_bad += 1
                        self.assertNotEqual(who[:8], "stranger", "a stranger was auto-attributed")
            self.assertEqual(auto_bad, 0)
            self.assertGreaterEqual(auto_ok, int(0.8 * kid_open))
            # Look-alikes (cosine ~0.4-0.5 to Leo's faces) may be queued for review but never auto-accepted:
            theo = [f for who, _, f in photos if who == "Theo"]
            self.assertTrue(all(rows.get(f, (0, "none", ""))[1] != "auto_high" for f in theo))

    def test_calibration_band_requires_labelled_neighbour_agreement(self) -> None:
        cls = _SyntheticClassroom(seed=5)
        for kid in ("Maya", "Leo"):
            cls.identity(kid)
            cls.faces(kid, 30)
        # One ambiguous face, roughly equally close to both kids (scores ~0.45-0.6 for each):
        mix = cls._unit(0.62 * cls.centres["Maya"] + 0.58 * cls.centres["Leo"] + 0.53 * cls._unit(cls.rng.standard_normal(512)))
        cls.items.append(("ambiguous", mix.astype(np.float32)))
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            photos = cls.save(db)
            pid_of = {kid: db.ensure_person(kid) for kid in ("Maya", "Leo")}
            for who, p_id, f_id in photos:
                if who in pid_of:
                    db.tag_single_face(photo_id=p_id, face_id=f_id, person_id=pid_of[who])
            amb = next(f for who, _, f in photos if who == "ambiguous")
            recompute_active_matching(db)
            got = _model_rows(db).get(amb)
            assert got is not None
            self.assertEqual(got[1], "candidate")  # the split neighbourhood sends it to review


class OfflineDecisionReplayTests(unittest.TestCase):
    """
    Replays synthetic "historic" batch decisions (approvals of the child, rejections of look-alikes)
    with the decisions hidden from the matcher, and asserts the precision of what the matcher would
    have auto-accepted at `auto_high_threshold`.
    """

    def test_precision_at_auto_high_threshold_on_hidden_decisions(self) -> None:
        cls = _SyntheticClassroom(seed=8)
        for kid in ("Maya", "Leo", "Sofia"):
            cls.identity(kid)
            cls.faces(kid, 40, alpha=(0.70, 0.88))
        for i, (kid, sim) in enumerate((("Maya", 0.66), ("Leo", 0.60), ("Sofia", 0.70), ("Maya", 0.55))):
            cls.identity(f"lookalike{i}", like=kid, centre_sim=sim)
            cls.faces(f"lookalike{i}", 12)
        lookalike_of = {"lookalike0": "Maya", "lookalike1": "Leo", "lookalike2": "Sofia", "lookalike3": "Maya"}
        with tempfile.TemporaryDirectory() as tmp, FacesIndexDB(Path(tmp) / "faces_index.sqlite") as db:
            photos = cls.save(db)
            pid_of = {kid: db.ensure_person(kid) for kid in ("Maya", "Leo", "Sofia")}
            decisions: list[tuple[int, int, int, bool]] = []  # (photo, face, person, approved)
            for who, p_id, f_id in photos:
                if who in pid_of:
                    decisions.append((p_id, f_id, pid_of[who], True))
                else:
                    decisions.append((p_id, f_id, pid_of[lookalike_of[who]], False))
            for kid, pid in pid_of.items():
                del kid
                db.apply_batch_decisions(
                    person_id=pid,
                    approved=[(p, f) for p, f, q, ok in decisions if q == pid and ok],
                    disproved=[(p, f) for p, f, q, ok in decisions if q == pid and not ok],
                )
            hidden = decisions[::4]  # replay a quarter of the history
            with db._conn:
                db._conn.executemany(
                    "DELETE FROM assignments WHERE photo_id = ? AND person_id = ?", [(p, q) for p, _f, q, _ok in hidden]
                )
            diag: dict[str, Any] = {}
            recompute_active_matching(db, diagnostics=diag)
            final = {(a[2], a[1]): a[3] for a in diag["assignments"]}
            auto = [(ok) for _p, f, q, ok in hidden if final.get((f, q)) == "auto_high"]
            self.assertGreaterEqual(len(auto), 10)
            precision = sum(auto) / len(auto)
            self.assertGreaterEqual(precision, 0.95, f"precision at auto_high_threshold = {precision:.2f}")
            rejected_auto = sum(1 for ok in auto if not ok)
            self.assertLessEqual(rejected_auto, 0)


if __name__ == "__main__":
    unittest.main()
