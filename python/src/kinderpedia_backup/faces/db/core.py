"""Connection, locks, caches, meta / change-signal helpers and the shared cover-refresh primitive."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, TypeVar

from .codec import (
    _now_iso,
)
from .scan_lock import read_scan_lock
from .schema import apply_migrations, read_schema_version

_SelfDB = TypeVar("_SelfDB", bound="DBCore")


class DBCore:
    """Thread-safe SQLite WAL database backing `faces scan`, `faces ui`, and URL/folder exports.

    Concurrency rules:
    - One shared connection; every cursor touch happens under `_lock` (an `RLock`), and rows are
      materialized before it is released.
    - Lock order: `_faces_matrix_lock` is the OUTER lock and `_lock` the INNER one. Code holding
      `_faces_matrix_lock` may take `_lock` (e.g. the cold matrix load); code holding `_lock` must
      NEVER take `_faces_matrix_lock` — so never call `load_all_faces_matrix()` (or anything that
      reaches it, e.g. `_select_impactful_rows`) while holding `_lock`; release it first.
    - Long CPU/filesystem work (SGEMMs, `materialize_flat_folders` I/O) runs without `_lock`.
    - Cross-process: `faces scan` holds `Metadata/faces_scan.lock` and bumps `get_data_version()`
      when it finishes; a long-lived reader (the Web UI) polls that value to drop its caches.
    """

    _lock: threading.RLock
    _conn: sqlite3.Connection

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._faces_matrix_lock = threading.Lock()
        self._faces_matrix_cache: dict[str, Any] | None = None
        self._photo_summary_cache: dict[str, Any] | None = None
        self._knn_neighbor_cache: dict[str, Any] | None = None
        self._label_outliers: set[tuple[int, int]] = set()  # (face_id, person_id) from the last rematch
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.commit()
            apply_migrations(self._conn)

    def close(self) -> None:
        with self._faces_matrix_lock:
            self._faces_matrix_cache = None
        with self._lock:
            self._photo_summary_cache = None
            self._conn.close()

    def invalidate_caches(self) -> None:
        """Clear in-memory face matrix and static photo overview caches (e.g. after a new scan)."""
        with self._faces_matrix_lock:
            self._faces_matrix_cache = None
        with self._lock:
            self._photo_summary_cache = None

    def __enter__(self: _SelfDB) -> _SelfDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_data_version(self) -> str:
        """Opaque change token for cross-process cache invalidation (compare for equality only).

        Combines SQLite's `PRAGMA data_version` — which changes only when ANOTHER connection or
        process commits (never on this connection's own writes) — with `get_scan_generation()`,
        bumped when a `faces scan` ends. A long-running process (the Web UI) polls this cheaply and
        drops its in-memory caches (`invalidate_caches()`) when the value changes; its own label
        writes never change it.
        """
        with self._lock:
            pragma = self._conn.execute("PRAGMA data_version").fetchone()
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'data_version'").fetchone()
        return f"{row['value'] if row is not None else 0}:{int(pragma[0])}"

    def get_scan_generation(self) -> int:
        """Monotonic counter (`meta.data_version`) incremented at the end of every `faces scan`."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'data_version'").fetchone()
        try:
            return int(row["value"]) if row is not None else 0
        except (TypeError, ValueError):
            return 0

    def bump_data_version(self, *, scan_finished: bool = False) -> int:
        """Atomically increment `meta.data_version` (and stamp `meta.last_scan_at` for scans)."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO meta (key, value) VALUES ('data_version', '1')
                ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                """
            )
            if scan_finished:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('last_scan_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (_now_iso(),),
                )
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'data_version'").fetchone()
        return int(row["value"])

    def get_scan_status(self) -> dict[str, Any] | None:
        """Live `faces scan` lock info (`pid`, `host`, `started_at`, `heartbeat_at`) or None."""
        return read_scan_lock(self.db_path.parent)

    def get_schema_version(self) -> int:
        with self._lock:
            return read_schema_version(self._conn)

    def _refresh_cover_face_if_needed(self, person_id: int, lost_face_id: int) -> None:
        self._conn.execute(
            """
            UPDATE people
            SET cover_face_id = (
                SELECT a.face_id
                FROM assignments a
                LEFT JOIN faces f ON f.face_id = a.face_id
                WHERE a.person_id = ?
                  AND a.status = 'confirmed'
                  AND a.face_id IS NOT NULL
                  AND a.face_id != ?
                ORDER BY COALESCE(f.is_manual, 1) ASC, a.is_anchor DESC, a.updated_at DESC
                LIMIT 1
            )
            WHERE person_id = ? AND cover_face_id = ?
            """,
            (person_id, lost_face_id, person_id, lost_face_id),
        )
