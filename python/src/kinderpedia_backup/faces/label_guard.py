"""Label-consistency guard: spot confirmed face labels that look like none of the person's other
confirmed faces of the same year (likely mislabels, e.g. a bulk cluster confirm that swept in a
classmate).

A confirmed label of person P in year Y is *suspicious* when
- P has at least `LABEL_OUTLIER_MIN_FACES` confirmed faces (with embeddings) in Y, and
- its highest cosine similarity to any OTHER confirmed face of P in Y is below `LABEL_OUTLIER_MIN_SIM`,
- and the user has not marked it reviewed-OK ("keep").

The matcher never uses a suspicious label as positive evidence (anchor, album/temporal anchor or kNN
voter); the face itself stays tagged and locked. The values are calibrated with
`python -m kinderpedia_backup.faces.evaluate` (see docs/faces-handover.md).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

LABEL_OUTLIER_MIN_SIM = 0.40
LABEL_OUTLIER_MIN_FACES = 8


def max_sim_to_others(
    embs: NDArray[np.float32], chunk_size: int = 2048
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """For each row, the highest cosine similarity to any OTHER row and that row's index
    (chunked; `(-1, -1)` for a single row)."""
    n = int(embs.shape[0])
    best = np.full((n,), -1.0, dtype=np.float32)
    arg = np.full((n,), -1, dtype=np.int64)
    if n < 2:
        return best, arg
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = embs[start:end] @ embs.T
        sims[np.arange(end - start), np.arange(start, end)] = -1.0
        arg[start:end] = np.argmax(sims, axis=1)
        best[start:end] = sims[np.arange(end - start), arg[start:end]]
    return best, arg


def label_outlier_mask(
    embs: NDArray[np.float32],
    *,
    min_sim: float = LABEL_OUTLIER_MIN_SIM,
    min_faces: int = LABEL_OUTLIER_MIN_FACES,
) -> tuple[NDArray[np.bool_], NDArray[np.float32], NDArray[np.int64]]:
    """(suspicious mask, max same-person similarity, index of that nearest same-person face) for one
    person-year's confirmed face embeddings (row-aligned, L2-normalized)."""
    best, arg = max_sim_to_others(embs)
    if int(embs.shape[0]) < max(2, min_faces):
        return np.zeros((int(embs.shape[0]),), dtype=bool), best, arg
    return best < np.float32(min_sim), best, arg
