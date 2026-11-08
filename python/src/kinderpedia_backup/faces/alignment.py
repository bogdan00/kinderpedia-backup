"""Pure numpy + Pillow SCRFD anchor decoding, NMS, 5-point Umeyama alignment, and UI crops."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .models import BoundingBox

# Standard canonical 5-keypoint coordinates for 112x112 ArcFace models:
# (left_eye, right_eye, nose_tip, left_mouth, right_mouth)
ARCFACE_DST_112: NDArray[np.float32] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

SCRFD_STRIDE_HEADS: tuple[tuple[int, str, str, str], ...] = (
    (8, "448", "451", "454"),
    (16, "471", "474", "477"),
    (32, "494", "497", "500"),
)


@dataclass(frozen=True)
class RawScrfdDetection:
    """Decoded face box, confidence score, and 5 facial keypoints in original image space."""

    box: BoundingBox
    score: float
    landmarks: NDArray[np.float32]  # shape (5, 2)


def letterbox_scrfd_input(
    rgb: Image.Image | NDArray[np.uint8],
    det_size: int = 640,
) -> tuple[NDArray[np.float32], float]:
    """
    Resize `rgb` (PIL `Image` or `H x W x 3` uint8 array) into a fixed `(1, 3, det_size, det_size)`
    tensor normalized as `(x - 127.5) / 128.0`. Returns `(input_blob, scale_factor)`.
    """
    if isinstance(rgb, Image.Image):
        w, h = rgb.size
        pil_src = rgb
    else:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        pil_src = None
    if h <= 0 or w <= 0:
        raise ValueError("Input image has invalid zero dimensions.")

    im_ratio = h / w
    if im_ratio > 1.0:
        new_h = det_size
        new_w = max(1, int(new_h / im_ratio))
    else:
        new_w = det_size
        new_h = max(1, int(new_w * im_ratio))

    det_scale = float(new_h) / float(h)
    pil_img = pil_src if pil_src is not None else Image.fromarray(np.asarray(rgb))
    pil_resized = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    resized_f32 = (np.asarray(pil_resized, dtype=np.float32) - np.float32(127.5)) / np.float32(128.0)
    blob = np.full((1, 3, det_size, det_size), np.float32(-127.5 / 128.0), dtype=np.float32)
    blob[0, :, :new_h, :new_w] = resized_f32.transpose(2, 0, 1)
    return blob, det_scale


@lru_cache(maxsize=16)
def _scrfd_anchor_centers(det_size: int, stride: int) -> NDArray[np.float32]:
    feat_dim = det_size // stride
    cy, cx = np.mgrid[:feat_dim, :feat_dim]
    centers = np.stack([cx, cy], axis=-1).reshape(-1, 2)
    centers_f32 = (np.repeat(centers, 2, axis=0).astype(np.float32) * float(stride))
    centers_f32.flags.writeable = False
    return centers_f32


def greedy_nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], iou_thresh: float = 0.40) -> list[int]:
    """Standard greedy Non-Maximum Suppression over `[x1, y1, x2, y2]` boxes."""
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[idx, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[idx, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[idx, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[idx, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = max(0.0, float((boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])))
        area_r = np.maximum(0.0, (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]))
        iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def decode_scrfd_outputs(
    outputs: dict[str, NDArray[Any]],
    *,
    det_size: int,
    det_scale: float,
    img_width: int,
    img_height: int,
    score_thresh: float = 0.55,
    nms_thresh: float = 0.40,
    min_face_px: int = 28,
) -> list[RawScrfdDetection]:
    """Decode SCRFD-10GF multi-stride outputs, run NMS, and enforce classroom quality gate."""
    boxes_parts: list[NDArray[np.float32]] = []
    scores_parts: list[NDArray[np.float32]] = []
    kps_parts: list[NDArray[np.float32]] = []

    for stride, score_key, bbox_key, kps_key in SCRFD_STRIDE_HEADS:
        if score_key not in outputs or bbox_key not in outputs or kps_key not in outputs:
            continue
        raw_scores = np.asarray(outputs[score_key], dtype=np.float32).reshape(-1)
        mask = raw_scores >= score_thresh
        if not np.any(mask):
            continue

        centers = _scrfd_anchor_centers(det_size, stride)

        c_sel = centers[mask]
        b_sel = np.asarray(outputs[bbox_key], dtype=np.float32).reshape(-1, 4)[mask] * float(stride)
        k_sel = np.asarray(outputs[kps_key], dtype=np.float32).reshape(-1, 10)[mask] * float(stride)

        x1 = (c_sel[:, 0] - b_sel[:, 0]) / det_scale
        y1 = (c_sel[:, 1] - b_sel[:, 1]) / det_scale
        x2 = (c_sel[:, 0] + b_sel[:, 2]) / det_scale
        y2 = (c_sel[:, 1] + b_sel[:, 3]) / det_scale

        boxes_parts.append(np.stack([x1, y1, x2, y2], axis=-1))
        scores_parts.append(raw_scores[mask])
        kps = (k_sel.reshape(-1, 5, 2) + c_sel[:, np.newaxis, :]) / det_scale
        kps_parts.append(kps.astype(np.float32))

    if not scores_parts:
        return []

    boxes = np.concatenate(boxes_parts, axis=0)
    scores = np.concatenate(scores_parts, axis=0)
    kps_all = np.concatenate(kps_parts, axis=0)

    keep_indices = greedy_nms(boxes, scores, iou_thresh=nms_thresh)
    detections: list[RawScrfdDetection] = []

    for idx in keep_indices:
        left = max(0, min(img_width, round(float(boxes[idx, 0]))))
        top = max(0, min(img_height, round(float(boxes[idx, 1]))))
        right = max(0, min(img_width, round(float(boxes[idx, 2]))))
        bottom = max(0, min(img_height, round(float(boxes[idx, 3]))))
        if min(right - left, bottom - top) < min_face_px:
            continue
        detections.append(
            RawScrfdDetection(
                box=BoundingBox(top=top, right=right, bottom=bottom, left=left),
                score=float(scores[idx]),
                landmarks=kps_all[idx],
            )
        )
    return detections


def umeyama_similarity_2d(src: NDArray[np.float32], dst: NDArray[np.float32] = ARCFACE_DST_112) -> NDArray[np.float64]:
    """
    Closed-form 2D similarity transform (Umeyama SVD) mapping `src` (5, 2) -> `dst` (5, 2).
    Returns the forward `2x3` affine matrix `M` such that `dst_pt = M[:, :2] @ src_pt + M[:, 2]`.
    """
    src_f = np.asarray(src, dtype=np.float64)
    dst_f = np.asarray(dst, dtype=np.float64)
    num = src_f.shape[0]
    src_mean = src_f.mean(axis=0)
    dst_mean = dst_f.mean(axis=0)
    src_demean = src_f - src_mean
    dst_demean = dst_f - dst_mean

    cov = (dst_demean.T @ src_demean) / float(num)
    d = np.ones((2,), dtype=np.float64)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0

    u_mat, s_vec, vt_mat = np.linalg.svd(cov)
    rank = int(np.linalg.matrix_rank(cov))
    if rank == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    if rank == 1:
        if np.linalg.det(u_mat) * np.linalg.det(vt_mat) > 0:
            rot = u_mat @ vt_mat
        else:
            saved = d[1]
            d[1] = -1.0
            rot = u_mat @ np.diag(d) @ vt_mat
            d[1] = saved
    else:
        rot = u_mat @ np.diag(d) @ vt_mat

    var_src = float(src_demean.var(axis=0).sum())
    scale = 1.0 if var_src <= 1e-10 else float(s_vec @ d) / var_src
    mat = np.zeros((2, 3), dtype=np.float64)
    mat[:, :2] = scale * rot
    mat[:, 2] = dst_mean - mat[:, :2] @ src_mean
    return mat


#: Version tag of the face-alignment warp, folded into the recognizer `model_id` so that any change
#: to the crop geometry invalidates cached embeddings. v2 = OpenCV pixel-index convention (see below).
ALIGNMENT_VERSION = 2


def pil_affine_coeffs_from_cv2(inv_2x3: NDArray[np.float64]) -> tuple[float, float, float, float, float, float]:
    """
    Convert an inverse (destination -> source) affine matrix written in OpenCV's pixel-index
    convention into Pillow `Image.transform(AFFINE)` coefficients.

    `cv2.warpAffine(..., WARP_INVERSE_MAP)` samples the source at `A @ dst + t`, where integer
    coordinates are pixel centres. Pillow treats pixel centres as `i + 0.5` and samples the source at
    `A @ (dst + 0.5) + t' - 0.5`. Equating both gives `t' = t + 0.5 - 0.5 * A @ (1, 1)`.
    Without this correction the crop is shifted by up to ~1 px (more on small, upscaled faces), which
    perturbs ArcFace embeddings versus the canonical `insightface.norm_crop` alignment.
    """
    a, b, c = float(inv_2x3[0, 0]), float(inv_2x3[0, 1]), float(inv_2x3[0, 2])
    d, e, f = float(inv_2x3[1, 0]), float(inv_2x3[1, 1]), float(inv_2x3[1, 2])
    return (a, b, c + 0.5 - 0.5 * (a + b), d, e, f + 0.5 - 0.5 * (d + e))


def align_face_112(rgb_img: Image.Image, landmarks: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Warp a detected face via 5-point Umeyama alignment onto a canonical `112x112` RGB crop
    and return a `(3, 112, 112)` float32 tensor normalized as `(pixel - 127.5) / 127.5`.

    Geometrically equivalent to `cv2.warpAffine(img, M, (112, 112))` with `M` the forward
    Umeyama matrix (the ArcFace training alignment), implemented with Pillow.
    """
    fwd_2x3 = umeyama_similarity_2d(landmarks, ARCFACE_DST_112)
    hom = np.vstack([fwd_2x3, np.array([0.0, 0.0, 1.0], dtype=np.float64)])
    try:
        inv = np.linalg.inv(hom)
    except np.linalg.LinAlgError:
        inv = np.eye(3, dtype=np.float64)

    coeffs = pil_affine_coeffs_from_cv2(inv[:2])
    aligned = rgb_img.transform(
        (112, 112),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BILINEAR,
    )
    arr = np.asarray(aligned, dtype=np.float32)
    return np.ascontiguousarray(((arr - 127.5) / 127.5).transpose(2, 0, 1), dtype=np.float32)


def crop_face_portrait(
    rgb_img: Image.Image,
    box: BoundingBox,
    *,
    size: int = 160,
    expand_ratio: float = 0.45,
) -> Image.Image:
    """
    Extract a generous square portrait crop around `box` (including hair/ears/headwear context)
    for fast human recognition in the interactive Web UI.
    """
    img_w, img_h = rgb_img.size
    cx = 0.5 * (box.left + box.right)
    cy = 0.5 * (box.top + box.bottom)
    side = max(float(box.width), float(box.height), 32.0) * (1.0 + 2.0 * expand_ratio)
    half = 0.5 * side

    x1 = max(0, round(cx - half))
    y1 = max(0, round(cy - half))
    x2 = min(img_w, round(cx + half))
    y2 = min(img_h, round(cy + half))
    if x2 <= x1 or y2 <= y1:
        return rgb_img.resize((size, size), resample=Image.Resampling.BILINEAR)

    cropped = rgb_img.crop((x1, y1, x2, y2))
    return cropped.resize((size, size), resample=Image.Resampling.LANCZOS)
