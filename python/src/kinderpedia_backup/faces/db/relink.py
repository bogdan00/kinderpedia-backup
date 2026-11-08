"""Pure helpers that re-link previously stored faces to fresh detections of the same photo.

Used by `FacesIndexDB.upsert_scanned_batch` so a rescan (content change, `--det-size`, `--force`, or a
`model_id` bump) keeps every `face_id` — and with it assignments, anchors, `suppress_auto`, pressure,
cluster membership and cover/medoid references — attached to the same physical face.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IOU_MIN = 0.5
COSINE_MIN = 0.9


@dataclass(frozen=True)
class StoredFace:
    face_id: int
    top: int
    right: int
    bottom: int
    left: int
    embedding: NDArray[np.float32] | None  # L2-normalized, or None when unavailable


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two `(top, right, bottom, left)` boxes."""
    a_t, a_r, a_b, a_l = a
    b_t, b_r, b_b, b_l = b
    iw = min(a_r, b_r) - max(a_l, b_l)
    ih = min(a_b, b_b) - max(a_t, b_t)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw) * float(ih)
    area_a = float(max(0, a_r - a_l)) * float(max(0, a_b - a_t))
    area_b = float(max(0, b_r - b_l)) * float(max(0, b_b - b_t))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_faces(
    old: list[StoredFace],
    new_boxes: list[tuple[int, int, int, int]],
    new_embs: NDArray[np.float32] | None,
    *,
    iou_min: float = IOU_MIN,
    cosine_min: float = COSINE_MIN,
) -> dict[int, int]:
    """Greedy one-to-one matching `{new_index: old_face_id}`.

    Pass 1 pairs by bbox IoU (`>= iou_min`, best first). Pass 2 pairs the leftovers by embedding cosine
    (`>= cosine_min`) when both sides have comparable (same-dimension) embeddings — this catches
    re-encoded / resized photos whose pixel coordinates moved.
    """
    out: dict[int, int] = {}
    if not old or not new_boxes:
        return out
    pairs: list[tuple[float, int, int]] = []
    for ni, nb in enumerate(new_boxes):
        for oi, of in enumerate(old):
            iou = bbox_iou(nb, (of.top, of.right, of.bottom, of.left))
            if iou >= iou_min:
                pairs.append((iou, ni, oi))
    used_new: set[int] = set()
    used_old: set[int] = set()
    for _, ni, oi in sorted(pairs, key=lambda t: (-t[0], t[1], t[2])):
        if ni in used_new or oi in used_old:
            continue
        used_new.add(ni)
        used_old.add(oi)
        out[ni] = old[oi].face_id

    if new_embs is None or len(used_new) == len(new_boxes) or len(used_old) == len(old):
        return out
    rest_new = [i for i in range(len(new_boxes)) if i not in used_new]
    rest_old: list[int] = []
    o_vecs: list[NDArray[np.float32]] = []
    for i, of in enumerate(old):
        emb = of.embedding
        if i in used_old or emb is None or emb.shape[0] != new_embs.shape[1]:
            continue
        rest_old.append(i)
        o_vecs.append(emb)
    if not rest_new or not rest_old:
        return out
    o_mat = np.stack(o_vecs).astype(np.float32)
    n_mat = new_embs[rest_new]
    sims = n_mat @ o_mat.T
    cos_pairs = [
        (float(sims[a, b]), rest_new[a], rest_old[b])
        for a in range(len(rest_new))
        for b in range(len(rest_old))
        if float(sims[a, b]) >= cosine_min
    ]
    for _, ni, oi in sorted(cos_pairs, key=lambda t: (-t[0], t[1], t[2])):
        if ni in used_new or oi in used_old:
            continue
        used_new.add(ni)
        used_old.add(oi)
        out[ni] = old[oi].face_id
    return out
