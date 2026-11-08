"""Persistent SQLite WAL categorization & provenance index (`Metadata/faces_index.sqlite`).

`FacesIndexDB` is a thin facade over cohesive mixin modules in `faces/db/` (each building on the
previous one): `core` (connection, locks, meta, change signal) -> `people` (people, classes,
eligibility) -> `matrix_cache` (RAM embedding matrix) -> `scan_repo` (scan persistence, re-link,
reconciliation) -> `assignments` (state machine on `_claim_face_locked`) -> `clusters` -> `queries_ui`
-> `exports`; DDL and migrations live in `db/schema.py`. Import from this module only.
"""

from __future__ import annotations

from .db.codec import (
    UNKNOWN_DATE_ORD,
    _now_iso,
    date_ord_to_day_num,
    date_to_ord,
    normalize_iso_date,
    pack_embedding_f16,
    photo_date_ord,
    unpack_embeddings_f32,
)
from .db.exports import ExportsMixin, _prune_orphan_person_folders
from .db.queries_ui import _pack_person_cluster


class FacesIndexDB(ExportsMixin):
    """Thread-safe SQLite WAL database backing `faces scan`, `faces ui`, and URL/folder exports.

    See `faces.db.core.DBCore` for the concurrency / lock-order rules.
    """


__all__ = [
    "UNKNOWN_DATE_ORD",
    "FacesIndexDB",
    "_now_iso",
    "_pack_person_cluster",
    "_prune_orphan_person_folders",
    "date_ord_to_day_num",
    "date_to_ord",
    "normalize_iso_date",
    "pack_embedding_f16",
    "photo_date_ord",
    "unpack_embeddings_f32",
]
