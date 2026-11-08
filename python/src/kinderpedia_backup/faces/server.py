"""Local HTTP backend (`127.0.0.1`) serving the interactive active-learning face categorization Web UI."""

from __future__ import annotations

import contextlib
import hashlib
import json
import mimetypes
import queue
import secrets
import sqlite3
import threading
import time
import urllib.parse
import webbrowser
from collections import OrderedDict
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from ..console import log
from ..security import ensure_within_directory
from .index_db import FacesIndexDB
from .matcher import rebuild_yearly_clusters, recompute_active_matching
from .models import AssignmentStatus, BoundingBox, LinkMode, PersonRole
from .scanner import ensure_face_thumb_on_disk, face_thumb_path

_WEB_DIR = Path(__file__).resolve().parent / "web"

#: Faces shipped per child-cluster card in `GET /api/people/<pid>/clusters`; the rest is fetched on demand.
CHILD_CLUSTER_FACES_PREVIEW = 12
#: Distinct (person, threshold, min_size, data version) child-cluster results kept in memory.
_CHILD_CLUSTER_CACHE_MAX = 6
#: Finished background jobs remembered for `GET /api/jobs/<id>`.
_FINISHED_JOBS_KEEP = 20

_JOB_LABELS = {
    "recluster": "Rebuilding clusters & re-matching…",
    "materialize": "Syncing ByChild/ folders…",
}


class JobConflictError(RuntimeError):
    """A background job of the same kind is already running (mapped to HTTP 409)."""

    def __init__(self, job: dict[str, Any]) -> None:
        super().__init__(f"{job.get('label') or job.get('kind')} is already running.")
        self.job = job


def _is_db_locked_error(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


class _AppContext:
    def __init__(
        self,
        *,
        backup_dir: Path,
        output_dir: Path,
        db: FacesIndexDB,
        csrf_token: str,
        auto_high_threshold: float = 0.45,
        candidate_threshold: float = 0.33,
        link_mode: LinkMode = "hardlink",
        rematch_debounce_sec: float = 1.6,
        snapshot_debounce_sec: float = 20.0,
        warm_cache_on_start: bool = True,
        warm_rematch_on_start: bool = False,
        external_poll_sec: float = 3.0,
    ) -> None:
        self.backup_root = backup_dir.resolve()
        self.output_root = output_dir.resolve()
        self.meta_dir = ensure_within_directory(self.backup_root, self.backup_root / "Metadata")
        self.thumbs_root = ensure_within_directory(self.meta_dir, self.meta_dir / "face_thumbs")
        self.portable_json = ensure_within_directory(self.meta_dir, self.meta_dir / "categorization_index.json")
        self.db = db
        self.csrf_token = csrf_token
        self.auto_high_threshold = auto_high_threshold
        self.candidate_threshold = candidate_threshold
        self.link_mode: LinkMode = link_mode
        self.rematch_debounce_sec = max(0.0, float(rematch_debounce_sec))
        self.snapshot_debounce_sec = max(0.0, float(snapshot_debounce_sec))
        self._rematch_lock = threading.Lock()

        self._sse_lock = threading.Lock()
        self._sse_clients: set[queue.Queue[str | None]] = set()

        self._bg_cond = threading.Condition()
        self._closed = False
        self._warm_requested = bool(warm_cache_on_start)
        self._warm_rematch = bool(warm_rematch_on_start)
        self._rematch_requested_seq = 0
        self._rematch_completed_seq = 0
        self._rematch_due_at = 0.0
        self._rematch_running = False
        self._snapshot_dirty = False
        self._snapshot_due_at = 0.0
        self._snapshot_running = False
        self._rematch_started_at = 0.0
        self._last_rematch_ms: float | None = None
        self._last_rematch_wall: float | None = None
        self._last_bg_error = ""
        self._warming = bool(warm_cache_on_start)
        self._sync_op = ""

        # Long-running jobs (`/api/recluster`, `/api/materialize`) run on their own threads:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_jobs: dict[str, str] = {}
        self._job_threads: dict[str, threading.Thread] = {}
        self._job_seq = 0

        # Bumped whenever data changes outside the rematch sequence (HTTP mutations, recluster,
        # external `faces scan` writes) so derived caches keyed on it are never served stale.
        self._data_epoch = 0
        self._external_changes = 0
        self._pc_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._pc_lock = threading.Lock()
        self._pc_compute_lock = threading.Lock()

        self._closed_event = threading.Event()
        self.external_poll_sec = max(0.0, float(external_poll_sec))
        self._ext_signal: tuple[Any, ...] | None = None
        self._ext_pending = False
        with contextlib.suppress(Exception):
            self._ext_signal = self._read_external_signal()

        self._worker_thread = threading.Thread(
            target=self._bg_worker_loop,
            name="kp-faces-rematch-worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._watch_thread: threading.Thread | None = None
        if self.external_poll_sec > 0:
            self._watch_thread = threading.Thread(
                target=self._external_watch_loop,
                name="kp-faces-external-watch",
                daemon=True,
            )
            self._watch_thread.start()

    @property
    def is_closed(self) -> bool:
        with self._bg_cond:
            return self._closed

    @property
    def completed_rematch_seq(self) -> int:
        with self._bg_cond:
            return self._rematch_completed_seq

    @property
    def data_epoch(self) -> int:
        with self._bg_cond:
            return self._data_epoch

    def note_data_changed(self) -> None:
        """Invalidate request-level derived caches after any write (cheap; called per mutation)."""
        with self._bg_cond:
            self._data_epoch += 1

    @property
    def has_pending_rematch(self) -> bool:
        with self._bg_cond:
            return self._rematch_running or (self._rematch_requested_seq > self._rematch_completed_seq)

    def _job_public_locked(self, info: dict[str, Any], *, with_result: bool = True) -> dict[str, Any]:
        pub = {k: v for k, v in info.items() if not k.startswith("_")}
        if info.get("state") == "running":
            pub["running_ms"] = round((time.monotonic() - float(info["_t0"])) * 1000.0)
        if not with_result:
            pub.pop("result", None)
        return pub

    def bg_status(self) -> dict[str, Any]:
        """Snapshot of background work for the UI status chip (`GET /api/status` + SSE `bg_status`)."""
        now = time.monotonic()
        with self._bg_cond:
            pending = self._rematch_requested_seq > self._rematch_completed_seq
            if self._rematch_running:
                rematch_state = "running"
            elif pending:
                rematch_state = "queued"
            else:
                rematch_state = "idle"
            if self._snapshot_running:
                snap_state = "saving"
            elif self._snapshot_dirty:
                snap_state = "pending"
            else:
                snap_state = "idle"
            jobs = [
                self._job_public_locked(self._jobs[jid], with_result=False)
                for jid in self._active_jobs.values()
                if jid in self._jobs
            ]
            sync_op = self._sync_op or " · ".join(str(j["label"]) for j in jobs)
            return {
                "rematch": rematch_state,
                "rematch_due_in_ms": round(max(0.0, self._rematch_due_at - now) * 1000.0)
                if rematch_state == "queued"
                else 0,
                "rematch_running_ms": round((now - self._rematch_started_at) * 1000.0)
                if rematch_state == "running"
                else 0,
                "last_rematch_ms": self._last_rematch_ms,
                "last_rematch_at": self._last_rematch_wall,
                "snapshot": snap_state,
                "snapshot_due_in_ms": round(max(0.0, self._snapshot_due_at - now) * 1000.0)
                if snap_state == "pending"
                else 0,
                "warming": self._warming,
                "sync_op": sync_op,
                "jobs": jobs,
                "external_changes": self._external_changes,
                "last_error": self._last_bg_error,
                "busy": rematch_state != "idle"
                or snap_state == "saving"
                or self._warming
                or bool(sync_op),
            }

    def _emit_status(self) -> None:
        self.broadcast_sse("bg_status", self.bg_status())

    @contextlib.contextmanager
    def sync_operation(self, label: str) -> Iterator[None]:
        """Mark a long synchronous request (recluster / materialize) as visible background work."""
        with self._bg_cond:
            self._sync_op = label
        self._emit_status()
        try:
            yield
        finally:
            with self._bg_cond:
                self._sync_op = ""
            self._emit_status()

    def subscribe_sse(self) -> queue.Queue[str | None]:
        q: queue.Queue[str | None] = queue.Queue(maxsize=64)
        with self._sse_lock:
            if self._closed:
                q.put_nowait(None)
            else:
                self._sse_clients.add(q)
        return q

    def unsubscribe_sse(self, q: queue.Queue[str | None]) -> None:
        with self._sse_lock:
            self._sse_clients.discard(q)

    def broadcast_sse(self, event: str, data: dict[str, Any]) -> None:
        frame = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        with self._sse_lock:
            clients = list(self._sse_clients)
        for q in clients:
            try:
                q.put_nowait(frame)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    q.get_nowait()
                with contextlib.suppress(queue.Full):
                    q.put_nowait(frame)

    def close_sse_streams(self) -> None:
        with self._sse_lock:
            clients = list(self._sse_clients)
        for q in clients:
            try:
                q.put_nowait(None)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    q.get_nowait()
                with contextlib.suppress(queue.Full):
                    q.put_nowait(None)

    def schedule_rematch(self, *, delay: float | None = None) -> int:
        """Schedule a debounced background global rematch and throttled portable JSON snapshot."""
        now = time.monotonic()
        eff_delay = self.rematch_debounce_sec if delay is None else max(0.0, float(delay))
        with self._bg_cond:
            if self._closed:
                return self._rematch_completed_seq
            had_unstarted = self._rematch_requested_seq > self._rematch_completed_seq
            self._rematch_requested_seq += 1
            seq = self._rematch_requested_seq
            target_due = now + eff_delay
            if delay is not None and had_unstarted and eff_delay < self.rematch_debounce_sec:
                self._rematch_due_at = min(self._rematch_due_at, target_due)
            else:
                self._rematch_due_at = target_due
            self._snapshot_dirty = True
            self._snapshot_due_at = now + self.snapshot_debounce_sec
            running = self._rematch_running
            self._bg_cond.notify_all()
        self.broadcast_sse("rematch_scheduled", {"seq": seq, "running": running})
        self._emit_status()
        return seq

    def schedule_snapshot(self, *, delay: float | None = None) -> None:
        """Schedule a throttled background `categorization_index.json` export without rematching."""
        now = time.monotonic()
        eff_delay = self.snapshot_debounce_sec if delay is None else max(0.0, float(delay))
        with self._bg_cond:
            if self._closed:
                return
            self._snapshot_dirty = True
            self._snapshot_due_at = now + eff_delay
            self._bg_cond.notify_all()
        self._emit_status()

    def wait_for_rematch(self, timeout: float = 15.0) -> bool:
        """Expedite any pending debounced rematch and wait until it finishes."""
        deadline = time.monotonic() + max(0.05, timeout)
        with self._bg_cond:
            target_seq = self._rematch_requested_seq
            if target_seq <= self._rematch_completed_seq and not self._rematch_running:
                return True
            self._rematch_due_at = 0.0
            self._bg_cond.notify_all()
            while not self._closed and (
                self._rematch_completed_seq < target_seq or self._rematch_running
            ):
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return False
                self._bg_cond.wait(timeout=rem)
            return self._rematch_completed_seq >= target_seq

    def rematch_and_snapshot(self) -> tuple[int, int]:
        """Run an immediate synchronous rematch + portable JSON snapshot (e.g. for `/api/recluster`)."""
        with self._rematch_lock:
            ah, cand = recompute_active_matching(
                self.db,
                auto_high_threshold=self.auto_high_threshold,
                candidate_threshold=self.candidate_threshold,
            )
            self.db.export_portable_index(self.portable_json)
        with self._bg_cond:
            self._snapshot_dirty = False
        return ah, cand

    # ------------------------------------------------------------------
    # Background jobs (recluster / materialize): 202 + job id, SSE `job_status` / `job_complete`
    # ------------------------------------------------------------------

    def start_job(self, kind: str, fn: Callable[[], dict[str, Any]], *, label: str | None = None) -> dict[str, Any]:
        """Run `fn` on a dedicated thread; raise `JobConflictError` if a `kind` job is already running."""
        eff_label = label or _JOB_LABELS.get(kind, kind)
        with self._bg_cond:
            if self._closed:
                raise RuntimeError("Server is shutting down.")
            running_id = self._active_jobs.get(kind)
            if running_id is not None and running_id in self._jobs:
                raise JobConflictError(self._job_public_locked(self._jobs[running_id], with_result=False))
            self._job_seq += 1
            job_id = f"{kind}-{self._job_seq}-{secrets.token_hex(3)}"
            info: dict[str, Any] = {
                "job_id": job_id,
                "kind": kind,
                "label": eff_label,
                "state": "running",
                "started_at": time.time(),
                "finished_at": None,
                "elapsed_ms": None,
                "result": None,
                "error": "",
                "_t0": time.monotonic(),
            }
            self._jobs[job_id] = info
            self._active_jobs[kind] = job_id
            finished = [jid for jid, j in self._jobs.items() if j["state"] != "running"]
            for jid in finished[: max(0, len(finished) - _FINISHED_JOBS_KEEP)]:
                del self._jobs[jid]
            public = self._job_public_locked(info, with_result=False)
            thread = threading.Thread(
                target=self._run_job, args=(job_id, fn), name=f"kp-faces-job-{kind}", daemon=True
            )
            self._job_threads[job_id] = thread
        thread.start()
        self.broadcast_sse("job_status", public)
        self._emit_status()
        return public

    def _run_job(self, job_id: str, fn: Callable[[], dict[str, Any]]) -> None:
        result: dict[str, Any] | None = None
        error = ""
        try:
            result = fn()
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            if _is_db_locked_error(exc):
                error = "The face index is busy (another process may be writing). Please retry in a moment."
            log(f"[Faces UI] Background job {job_id} failed: {exc}")
        with self._bg_cond:
            info = self._jobs.get(job_id)
            if info is None:
                return
            info["state"] = "error" if error else "done"
            info["error"] = error
            info["result"] = result
            info["finished_at"] = time.time()
            info["elapsed_ms"] = round((time.monotonic() - float(info["_t0"])) * 1000.0, 1)
            if self._active_jobs.get(str(info["kind"])) == job_id:
                del self._active_jobs[str(info["kind"])]
            self._job_threads.pop(job_id, None)
            if error:
                self._last_bg_error = f"{info['label']} failed: {error}"
            public = self._job_public_locked(info)
            self._bg_cond.notify_all()
        self.broadcast_sse("job_complete", {**public, "ok": not error})
        self._emit_status()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._bg_cond:
            info = self._jobs.get(job_id)
            return self._job_public_locked(info) if info is not None else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._bg_cond:
            return [self._job_public_locked(j, with_result=False) for j in self._jobs.values()]

    def wait_for_job(self, job_id: str, timeout: float = 30.0) -> dict[str, Any] | None:
        """Block until `job_id` finishes (tests / CLI helpers); returns the final public job record."""
        deadline = time.monotonic() + max(0.05, timeout)
        with self._bg_cond:
            while True:
                info = self._jobs.get(job_id)
                if info is None or info["state"] != "running":
                    return self._job_public_locked(info) if info is not None else None
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return self._job_public_locked(info)
                self._bg_cond.wait(timeout=rem)

    def run_recluster_job(self) -> dict[str, Any]:
        """Job body for `/api/recluster`: rebuild yearly clusters + full rematch under `_rematch_lock`."""
        with self._rematch_lock:
            if self.is_closed:
                raise RuntimeError("Server is shutting down.")
            self.db.invalidate_caches()
            n_clusters = rebuild_yearly_clusters(self.db)
            ah, cand = recompute_active_matching(
                self.db,
                auto_high_threshold=self.auto_high_threshold,
                candidate_threshold=self.candidate_threshold,
            )
            self.db.export_portable_index(self.portable_json)
        with self._bg_cond:
            self._snapshot_dirty = False
            self._data_epoch += 1
        ov = self.db.get_archive_overview()
        self.broadcast_sse(
            "rematch_complete",
            {
                "seq": self.completed_rematch_seq,
                "pending": self.has_pending_rematch,
                "auto_high": ah,
                "candidates": cand,
                "overview": ov,
            },
        )
        return {"clusters_created": n_clusters, "auto_high": ah, "candidates": cand, "overview": ov}

    def run_materialize_job(self, person_filter: str | None, include_candidates: bool) -> dict[str, Any]:
        """Job body for `/api/materialize`: flush pending rematch, sync ByChild/, refresh the snapshot."""
        self.wait_for_rematch(timeout=15.0)
        stats = self.db.materialize_flat_folders(
            backup_dir=self.backup_root,
            output_dir=self.output_root,
            link_mode=self.link_mode,
            person_filter=person_filter,
            include_candidates=include_candidates,
            prune=True,
        )
        with self._rematch_lock:
            self.db.export_portable_index(self.portable_json)
        with self._bg_cond:
            self._snapshot_dirty = False
        return {
            "output_dir": str(self.output_root),
            "person": person_filter or "",
            "profiles": stats.reference_profiles,
            "photos_matched": stats.photos_matched,
            "links_created": stats.links_created,
            "links_existing": stats.links_skipped_existing,
            "links_pruned": stats.links_pruned,
        }

    # ------------------------------------------------------------------
    # Per-child look-alike clusters: cached per (person, params, rematch seq, data epoch)
    # ------------------------------------------------------------------

    def person_face_clusters(self, person_id: int, *, threshold: float, min_size: int) -> dict[str, Any]:
        """Return the cached `{full, by_key, signature}` entry for one child's look-alike clusters."""
        thr = round(float(min(0.95, max(0.20, float(threshold)))), 3)
        min_sz = max(2, int(min_size))
        key = (int(person_id), thr, min_sz, self.completed_rematch_seq, self.data_epoch)
        with self._pc_lock:
            hit = self._pc_cache.get(key)
            if hit is not None:
                self._pc_cache.move_to_end(key)
                return hit
        with self._pc_compute_lock:
            with self._pc_lock:
                hit = self._pc_cache.get(key)
                if hit is not None:
                    return hit
            full = self.db.get_person_face_clusters(int(person_id), threshold=thr, min_size=min_sz)
            sig = hashlib.sha1(
                json.dumps(full, sort_keys=True, ensure_ascii=False).encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:16]
            entry = {
                "full": full,
                "by_key": {str(c["cluster_key"]): c for c in full.get("clusters", [])},
                "signature": sig,
            }
            with self._pc_lock:
                self._pc_cache[key] = entry
                while len(self._pc_cache) > _CHILD_CLUSTER_CACHE_MAX:
                    self._pc_cache.popitem(last=False)
            return entry

    @staticmethod
    def truncated_person_clusters(entry: dict[str, Any], faces_limit: int) -> dict[str, Any]:
        full = entry["full"]
        out = {k: v for k, v in full.items() if k != "clusters"}
        clusters = []
        for c in full.get("clusters", []):
            faces = c.get("faces", [])
            if faces_limit > 0 and len(faces) > faces_limit:
                clusters.append({**c, "faces": faces[:faces_limit], "faces_truncated": True})
            else:
                clusters.append({**c, "faces_truncated": False})
        out["clusters"] = clusters
        out["signature"] = entry["signature"]
        out["faces_preview"] = faces_limit
        return out

    # ------------------------------------------------------------------
    # External writers (a concurrent `faces scan` process) -> invalidate caches + rematch
    # ------------------------------------------------------------------

    def _read_external_signal(self) -> tuple[Any, ...]:
        """Cheap DB signal that changes when *another* process commits (own writes do not count)."""
        db = self.db
        getter = getattr(db, "get_data_version", None)
        if callable(getter):
            return ("data_version", getter(), db.get_meta("last_scan_at"))
        with db._lock:
            row = db._conn.execute("PRAGMA data_version").fetchone()
        return ("pragma", int(row[0]) if row is not None else 0, db.get_meta("last_scan_at"))

    def check_external_changes(self) -> bool:
        """Poll once; on change invalidate caches, reset the kNN cache and schedule a rematch."""
        if self.is_closed:
            return False
        try:
            sig = self._read_external_signal()
        except sqlite3.Error:
            return False  # e.g. locked mid-scan: try again on the next tick
        if self._ext_signal is None:
            self._ext_signal = sig
            return False
        changed = sig != self._ext_signal
        self._ext_signal = sig
        if changed:
            self._ext_pending = True
        if not self._ext_pending:
            return False
        # While a `faces scan` holds its lock the token changes on every batch commit: defer the
        # (expensive) cache drop + rematch until the scan has finished.
        status_getter = getattr(self.db, "get_scan_status", None)
        if callable(status_getter):
            try:
                if status_getter() is not None:
                    return False
            except Exception:  # noqa: S110 - unreadable lock file: fall through and refresh now
                pass
        self._ext_pending = False
        log("[Faces UI] Face index changed on disk (e.g. `faces scan`): refreshing caches and re-matching.")
        with self._rematch_lock:
            if self.is_closed:
                return False
            self.db.invalidate_caches()
            if hasattr(self.db, "_knn_neighbor_cache"):
                self.db._knn_neighbor_cache = None
        with self._bg_cond:
            self._data_epoch += 1
            self._external_changes += 1
        self.broadcast_sse("data_changed", {"source": "external", "count": self._external_changes})
        self.schedule_rematch(delay=0.5)
        return True

    def _external_watch_loop(self) -> None:
        while not self._closed_event.wait(self.external_poll_sec):
            try:
                self.check_external_changes()
            except Exception as exc:
                log(f"[Faces UI] External change check failed: {exc}")

    def _bg_worker_loop(self) -> None:
        while True:
            do_warm = False
            run_seq: int | None = None
            do_snapshot = False

            with self._bg_cond:
                while not self._closed:
                    now = time.monotonic()
                    if self._warm_requested:
                        self._warm_requested = False
                        do_warm = True
                        break
                    if self._rematch_requested_seq > self._rematch_completed_seq:
                        wait_rem = self._rematch_due_at - now
                        if wait_rem <= 0:
                            run_seq = self._rematch_requested_seq
                            self._rematch_running = True
                            self._rematch_started_at = time.monotonic()
                            break
                        self._bg_cond.wait(timeout=wait_rem)
                        continue
                    if self._snapshot_dirty:
                        wait_snap = self._snapshot_due_at - now
                        if wait_snap <= 0:
                            self._snapshot_dirty = False
                            self._snapshot_running = True
                            do_snapshot = True
                            break
                        self._bg_cond.wait(timeout=wait_snap)
                        continue
                    self._bg_cond.wait()

                if self._closed:
                    return

            if do_warm:
                self._emit_status()
                with contextlib.suppress(Exception):
                    self.db.load_all_faces_matrix()
                if not self._warm_rematch:
                    with self._bg_cond:
                        self._warming = False
                    self._emit_status()
                if self._warm_rematch:
                    # Build the learned-label neighbour cache (and apply the latest matcher) right away
                    # so the first real click only pays the cheap incremental update.
                    self.schedule_rematch(delay=0.0)
                continue

            if run_seq is not None:
                self.broadcast_sse("rematch_status", {"running": True, "seq": run_seq})
                self._emit_status()
                t0 = time.perf_counter()
                try:
                    with self._rematch_lock:
                        if self.is_closed:
                            return
                        ah, cand = recompute_active_matching(
                            self.db,
                            auto_high_threshold=self.auto_high_threshold,
                            candidate_threshold=self.candidate_threshold,
                        )
                        overview = self.db.get_archive_overview()
                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    with self._bg_cond:
                        self._rematch_completed_seq = max(self._rematch_completed_seq, run_seq)
                        still_pending = self._rematch_requested_seq > self._rematch_completed_seq
                        self._rematch_running = False
                        self._last_rematch_ms = elapsed_ms
                        self._last_rematch_wall = time.time()
                        self._last_bg_error = ""
                        self._warming = False
                        self._bg_cond.notify_all()
                    self.broadcast_sse(
                        "rematch_complete",
                        {
                            "seq": run_seq,
                            "pending": still_pending,
                            "auto_high": ah,
                            "candidates": cand,
                            "elapsed_ms": elapsed_ms,
                            "overview": overview,
                        },
                    )
                    self._emit_status()
                except Exception as exc:
                    log(f"[Faces UI] Background rematch error: {exc}")
                    with self._bg_cond:
                        self._rematch_completed_seq = max(self._rematch_completed_seq, run_seq)
                        self._rematch_running = False
                        self._last_bg_error = f"Re-match failed: {exc}"
                        self._warming = False
                        self._bg_cond.notify_all()
                    self.broadcast_sse("rematch_status", {"running": False, "seq": run_seq})
                    self._emit_status()
                continue

            if do_snapshot:
                self._emit_status()
                try:
                    with self._rematch_lock:
                        if not self.is_closed:
                            self.db.export_portable_index(self.portable_json)
                except Exception as exc:
                    log(f"[Faces UI] Background snapshot warning: {exc}")
                    with self._bg_cond:
                        self._last_bg_error = f"Saving categorization_index.json failed: {exc}"
                finally:
                    with self._bg_cond:
                        self._snapshot_running = False
                self._emit_status()

    def close(self, *, flush_snapshot: bool = False) -> None:
        with self._bg_cond:
            already_closed = self._closed
            was_snapshot_dirty = self._snapshot_dirty
            self._closed = True
            self._snapshot_dirty = False
            self._bg_cond.notify_all()
            job_threads = list(self._job_threads.values())
        self._closed_event.set()
        self.close_sse_streams()
        if not already_closed and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        if not already_closed and self._watch_thread is not None and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=5.0)
        for t in job_threads:
            if t.is_alive():
                t.join(timeout=30.0)
        if flush_snapshot and was_snapshot_dirty:
            try:
                with self._rematch_lock:
                    self.db.export_portable_index(self.portable_json)
            except Exception as exc:
                log(f"[Faces UI] Final snapshot warning: {exc}")


class _FacesHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], ctx: _AppContext) -> None:
        super().__init__(server_address, _FacesRequestHandler)
        self.ctx = ctx

    def shutdown(self) -> None:
        self.ctx.close_sse_streams()
        super().shutdown()

    def server_close(self) -> None:
        self.ctx.close()
        super().server_close()


class _FacesRequestHandler(BaseHTTPRequestHandler):
    server_version = "KinderpediaFacesLocal/1.0"
    # Keep-alive: the grid/strip/roster fetch dozens of thumbnails per view. Every response carries a
    # Content-Length (or closes the connection, e.g. SSE), so HTTP/1.1 persistent connections are safe.
    protocol_version = "HTTP/1.1"
    # Idle keep-alive sockets are closed after this many seconds (frees the handler thread).
    timeout = 120

    _body_consumed = False
    _mutating = False
    _response_started = False

    @property
    def ctx(self) -> _AppContext:
        return cast(_FacesHTTPServer, self.server).ctx

    def log_message(self, format: str, *args: Any) -> None:
        # Keep terminal quiet during rapid thumbnail/API fetches.
        return

    def send_response(self, code: int, message: str | None = None) -> None:
        self._response_started = True
        super().send_response(code, message)

    def _declared_length(self) -> int:
        try:
            return max(0, int(self.headers.get("Content-Length") or "0"))
        except (TypeError, ValueError):
            return -1

    def _dispatch(self, handler: Callable[[], None], *, mutating: bool, csrf: bool) -> None:
        self._body_consumed = False
        self._mutating = mutating
        self._response_started = False
        try:
            if csrf and not self._verify_csrf():
                return
            handler()
        except Exception as exc:
            self._send_exception(exc)
        finally:
            pending = self._declared_length()
            if pending != 0 and not self._body_consumed:
                # Unread request body would corrupt the next request on this keep-alive connection:
                # drain small bodies, hang up on anything else.
                if 0 < pending <= 1024 * 1024:
                    try:
                        self.rfile.read(pending)
                    except OSError:
                        self.close_connection = True
                else:
                    self.close_connection = True

    def do_GET(self) -> None:
        self._dispatch(self._handle_get, mutating=False, csrf=False)

    def do_POST(self) -> None:
        self._dispatch(self._handle_post, mutating=True, csrf=True)

    def do_PATCH(self) -> None:
        self._dispatch(self._handle_patch, mutating=True, csrf=True)

    def do_DELETE(self) -> None:
        self._dispatch(self._handle_delete, mutating=True, csrf=True)

    def _send_exception(self, exc: BaseException) -> None:
        """Map handler exceptions to HTTP: 404 KeyError, 400 validation, 409 job, 503 DB busy, 500 bugs."""
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            self.close_connection = True
            return
        if self._response_started:
            # Headers (maybe a partial body) already went out; the only safe option is to hang up.
            self.close_connection = True
            return
        if isinstance(exc, JobConflictError):
            self._send_json({"error": str(exc), "job": exc.job}, status=HTTPStatus.CONFLICT)
        elif _is_db_locked_error(exc):
            self._send_json(
                {
                    "error": "The face index is busy (another process such as `faces scan` is writing). "
                    "Please retry in a moment.",
                    "retryable": True,
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                extra_headers={"Retry-After": "2"},
            )
        elif isinstance(exc, KeyError):
            msg = str(exc.args[0]) if exc.args else "Not found."
            self._send_json({"error": msg}, status=HTTPStatus.NOT_FOUND)
        elif isinstance(exc, (ValueError, TypeError)):
            self._send_json({"error": str(exc) or "Invalid request."}, status=HTTPStatus.BAD_REQUEST)
        else:
            log(f"[Faces UI] Internal error on {self.command} {self.path}: {exc!r}")
            self._send_json(
                {"error": f"Internal server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _send_json(
        self,
        payload: Any,
        status: int = HTTPStatus.OK,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self._mutating:
            # Bump before the client can observe the response, so follow-up GETs never hit stale caches.
            self.ctx.note_data_changed()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for hk, hv in (extra_headers or {}).items():
            self.send_header(hk, hv)
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        status: int = HTTPStatus.OK,
        cache_control: str = "no-cache",
        disposition: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        self.wfile.write(body)

    def _verify_csrf(self) -> bool:
        tok = self.headers.get("X-KP-Token", "")
        if not secrets.compare_digest(tok.encode("utf-8"), self.ctx.csrf_token.encode("utf-8")):
            self._send_json({"error": "Invalid or missing X-KP-Token header."}, status=HTTPStatus.FORBIDDEN)
            return False
        origin = self.headers.get("Origin")
        if origin:
            parsed = urllib.parse.urlparse(origin)
            if parsed.hostname not in ("127.0.0.1", "localhost"):
                self._send_json({"error": "Cross-origin request rejected."}, status=HTTPStatus.FORBIDDEN)
                return False
        return True

    def _read_json_body(self) -> dict[str, Any]:
        length = self._declared_length()
        if length == 0:
            self._body_consumed = True
            return {}
        if length < 0 or length > 8 * 1024 * 1024:
            raise ValueError("Request body missing a valid Content-Length or larger than 8 MB.")
        raw = self.rfile.read(length)
        self._body_consumed = True
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def _handle_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            html = (_WEB_DIR / "index.html").read_text(encoding="utf-8")
            injected = html.replace("__KP_CSRF_TOKEN__", self.ctx.csrf_token)
            self._send_bytes(injected.encode("utf-8"), content_type="text/html; charset=utf-8")
            return

        if path == "/assets/styles.css":
            self._send_bytes(
                (_WEB_DIR / "styles.css").read_bytes(),
                content_type="text/css; charset=utf-8",
            )
            return

        if path == "/assets/app.js":
            self._send_bytes(
                (_WEB_DIR / "app.js").read_bytes(),
                content_type="application/javascript; charset=utf-8",
            )
            return

        if path.startswith("/media/face/") and path.endswith(".jpg"):
            fid_str = path[len("/media/face/") : -len(".jpg")]
            if not fid_str.isdigit():
                self._send_json({"error": "Invalid face id."}, status=HTTPStatus.BAD_REQUEST)
                return
            fid = int(fid_str)
            spec = self.ctx.db.get_face_crop_spec(fid)
            if spec is None:
                self._send_json({"error": "Face not found."}, status=HTTPStatus.NOT_FOUND)
                return
            rel_path, box = spec
            thumb_file = ensure_face_thumb_on_disk(
                backup_root=self.ctx.backup_root,
                thumbs_root=self.ctx.thumbs_root,
                face_id=fid,
                rel_path=rel_path,
                box=box,
            )
            if thumb_file is None or not thumb_file.is_file():
                self._send_json({"error": "Source image unavailable."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(
                thumb_file.read_bytes(),
                content_type="image/jpeg",
                cache_control="public, max-age=86400",
            )
            return

        if path.startswith("/media/photo/"):
            pid_str = path[len("/media/photo/") :]
            if not pid_str.isdigit():
                self._send_json({"error": "Invalid photo id."}, status=HTTPStatus.BAD_REQUEST)
                return
            rel_p = self.ctx.db.get_photo_rel_path(int(pid_str))
            if rel_p is None:
                self._send_json({"error": "Photo not found."}, status=HTTPStatus.NOT_FOUND)
                return
            src = ensure_within_directory(self.ctx.backup_root, self.ctx.backup_root / rel_p)
            if not src.is_file():
                self._send_json({"error": "Photo file missing on disk."}, status=HTTPStatus.NOT_FOUND)
                return
            mime = mimetypes.guess_type(src.name)[0] or "image/jpeg"
            self._send_bytes(
                src.read_bytes(),
                content_type=mime,
                cache_control="public, max-age=86400",
            )
            return

        if path == "/api/events":
            # Streamed until either side hangs up: no Content-Length, so this response is
            # close-delimited and the keep-alive connection must not be reused afterwards.
            self.close_connection = True
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = self.ctx.subscribe_sse()
            try:
                init_info = {
                    "seq": self.ctx.completed_rematch_seq,
                    "pending": self.ctx.has_pending_rematch,
                    "status": self.ctx.bg_status(),
                }
                self.wfile.write(
                    f": connected\nevent: connected\ndata: {json.dumps(init_info)}\n\n".encode()
                )
                self.wfile.flush()
                while not self.ctx.is_closed:
                    try:
                        frame = q.get(timeout=15.0)
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        continue
                    if frame is None:
                        break
                    self.wfile.write(frame.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                self.ctx.unsubscribe_sse(q)
            return

        if path == "/api/overview":
            self._send_json(self.ctx.db.get_archive_overview())
            return

        if path == "/api/status":
            self._send_json(self.ctx.bg_status())
            return

        if path == "/api/clusters":
            year = (qs.get("year") or [""])[0].strip() or None
            arch = (qs.get("archive_child") or [""])[0].strip() or None
            inc_assigned = (qs.get("include_assigned") or ["0"])[0] in ("1", "true")
            limit = min(500, max(1, int((qs.get("limit") or ["180"])[0])))
            self._send_json(
                {
                    "clusters": self.ctx.db.list_clusters(
                        year=year,
                        archive_child=arch,
                        include_assigned=inc_assigned,
                        limit=limit,
                    )
                }
            )
            return

        if path == "/api/batch":
            pid = int((qs.get("person_id") or ["0"])[0])
            size = min(96, max(4, int((qs.get("size") or ["24"])[0])))
            b_mode = (qs.get("mode") or ["impact"])[0]
            b_mode = b_mode if b_mode in ("impact", "confident") else "impact"
            self._send_json(self.ctx.db.get_batch_proposal(pid, batch_size=size, mode=b_mode))
            return

        if path == "/api/gallery":
            pid = int((qs.get("person_id") or ["0"])[0])
            st_filter = (qs.get("status") or ["included"])[0]
            year = (qs.get("year") or [""])[0].strip() or None
            offset = max(0, int((qs.get("offset") or ["0"])[0]))
            limit = min(200, max(1, int((qs.get("limit") or ["96"])[0])))
            self._send_json(
                self.ctx.db.list_person_photos(
                    pid,
                    status_filter=st_filter,
                    year=year,
                    offset=offset,
                    limit=limit,
                )
            )
            return

        if path.startswith("/api/photos/"):
            pid_str = path[len("/api/photos/") :]
            if not pid_str.isdigit():
                self._send_json({"error": "Invalid photo id."}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.ctx.db.get_photo_detail(int(pid_str)))
            return

        if path.startswith("/api/faces/") and path.endswith("/photo"):
            fid_str = path[len("/api/faces/") : -len("/photo")]
            if not fid_str.isdigit():
                self._send_json({"error": "Invalid face id."}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.ctx.db.get_photo_detail_by_face(int(fid_str)))
            return

        if path.startswith("/api/clusters/") and path.endswith("/faces"):
            cid_str = path[len("/api/clusters/") : -len("/faces")]
            if not cid_str.isdigit():
                self._send_json({"error": "Invalid cluster id."}, status=HTTPStatus.BAD_REQUEST)
                return
            limit = min(200, max(1, int((qs.get("limit") or ["120"])[0])))
            self._send_json(self.ctx.db.get_cluster_faces(int(cid_str), limit=limit))
            return

        if path.startswith("/api/people/") and path.endswith("/clusters/faces"):
            pid_str = path[len("/api/people/") : -len("/clusters/faces")]
            if not pid_str.isdigit():
                self._send_json({"error": "Invalid person id."}, status=HTTPStatus.BAD_REQUEST)
                return
            ckey = (qs.get("key") or [""])[0]
            thr = float((qs.get("threshold") or ["0.50"])[0])
            min_sz = int((qs.get("min_size") or ["3"])[0])
            offset = max(0, int((qs.get("offset") or ["0"])[0]))
            entry = self.ctx.person_face_clusters(int(pid_str), threshold=thr, min_size=min_sz)
            cluster = entry["by_key"].get(ckey)
            if cluster is None:
                raise KeyError(f"Cluster {ckey!r} no longer exists — reload the clusters view.")
            faces = cluster.get("faces", [])
            self._send_json(
                {
                    "cluster_key": ckey,
                    "size": len(faces),
                    "offset": offset,
                    "faces": faces[offset:],
                    "signature": entry["signature"],
                }
            )
            return

        if path.startswith("/api/people/") and path.endswith("/clusters"):
            pid_str = path[len("/api/people/") : -len("/clusters")]
            if not pid_str.isdigit():
                self._send_json({"error": "Invalid person id."}, status=HTTPStatus.BAD_REQUEST)
                return
            thr = float((qs.get("threshold") or ["0.50"])[0])
            min_sz = int((qs.get("min_size") or ["3"])[0])
            faces_limit = int((qs.get("faces_limit") or [str(CHILD_CLUSTER_FACES_PREVIEW)])[0])
            entry = self.ctx.person_face_clusters(int(pid_str), threshold=thr, min_size=min_sz)
            self._send_json(self.ctx.truncated_person_clusters(entry, max(0, faces_limit)))
            return

        if path == "/api/jobs":
            self._send_json({"jobs": self.ctx.list_jobs()})
            return

        if path.startswith("/api/jobs/"):
            job = self.ctx.get_job(path[len("/api/jobs/") :])
            if job is None:
                raise KeyError("Unknown job id.")
            self._send_json(job)
            return

        if path == "/api/classes/matrix":
            self._send_json(self.ctx.db.get_class_matrix())
            return

        if path == "/api/classes":
            self._send_json({"classes": self.ctx.db.list_classes()})
            return

        if path == "/api/export-urls":
            self.ctx.wait_for_rematch(timeout=10.0)
            person_q = (qs.get("person") or [""])[0].strip()
            fmt = (qs.get("format") or ["txt"])[0].strip().lower()
            inc_cand = (qs.get("include_candidates") or ["0"])[0] in ("1", "true")
            disp_name, content, _ = self.ctx.db.format_person_urls(
                person_q,
                fmt=fmt,
                include_candidates=inc_cand,
            )
            safe_slug = disp_name.replace(" ", "_")
            ext = fmt if fmt in ("txt", "csv", "json") else "txt"
            ctype = (
                "application/json; charset=utf-8"
                if ext == "json"
                else "text/csv; charset=utf-8"
                if ext == "csv"
                else "text/plain; charset=utf-8"
            )
            self._send_bytes(
                content.encode("utf-8"),
                content_type=ctype,
                disposition=f'attachment; filename="{safe_slug}_urls.{ext}"',
            )
            return

        self._send_json({"error": f"Unknown route: {path}"}, status=HTTPStatus.NOT_FOUND)

    def _handle_post(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        body = self._read_json_body()

        if path == "/api/people":
            name = str(body.get("name") or "").strip()
            role_raw = str(body.get("role") or "classmate").strip()
            role: PersonRole = (
                cast(PersonRole, role_raw)
                if role_raw in ("family", "classmate", "educator", "ignore")
                else "classmate"
            )
            pid = self.ctx.db.ensure_person(name, role=role)
            self.ctx.schedule_snapshot()
            ov = self.ctx.db.get_archive_overview()
            self.ctx.broadcast_sse("overview_updated", {"overview": ov})
            self._send_json(
                {
                    "person_id": pid,
                    "overview": ov,
                }
            )
            return

        if path.startswith("/api/clusters/") and path.endswith("/assign"):
            cid_str = path[len("/api/clusters/") : -len("/assign")]
            cid = int(cid_str)
            new_name = str(body.get("new_name") or "").strip()
            role_raw = str(body.get("role") or "classmate").strip()
            role_val: PersonRole = (
                cast(PersonRole, role_raw)
                if role_raw in ("family", "classmate", "educator", "ignore")
                else "classmate"
            )
            pid = (
                self.ctx.db.ensure_person(new_name, role=role_val)
                if new_name
                else int(body["person_id"])
            )
            confirmed_n = self.ctx.db.assign_cluster_to_person(cid, pid)
            batch_size = min(96, max(4, int(body.get("batch_size") or 24)))
            ov = self.ctx.db.get_archive_overview()
            next_batch = self.ctx.db.get_batch_proposal(
                pid,
                batch_size=batch_size,
                mode="confident" if str(body.get("mode") or "") == "confident" else "impact",
            )
            self.ctx.schedule_rematch(delay=0.1 if role_val != "ignore" else None)
            self._send_json(
                {
                    "person_id": pid,
                    "confirmed_from_cluster": confirmed_n,
                    "overview": ov,
                    "next_batch": next_batch,
                    "rematch_scheduled": True,
                }
            )
            return

        if path.startswith("/api/clusters/") and path.endswith("/remove-face"):
            cid_str = path[len("/api/clusters/") : -len("/remove-face")]
            cid = int(cid_str)
            rem_face_id = int(body["face_id"])
            updated_cluster = self.ctx.db.remove_face_from_cluster(
                cluster_id=cid,
                face_id=rem_face_id,
            )
            self._send_json(
                {
                    "cluster": updated_cluster,
                    "overview": self.ctx.db.get_archive_overview(),
                }
            )
            return

        if path == "/api/batch/submit":
            pid = int(body["person_id"])
            batch_size = min(96, max(4, int(body.get("batch_size") or 24)))
            approved_in = body.get("approved") or []
            disproved_in = body.get("disproved") or []
            unsure_in = body.get("unsure") or []
            reassigned_in = body.get("reassigned") or []

            approved = [
                (
                    int(it["photo_id"]),
                    int(it["face_id"]) if it.get("face_id") is not None else None,
                )
                for it in approved_in
                if isinstance(it, dict)
            ]
            disproved = [
                (
                    int(it["photo_id"]),
                    int(it["face_id"]) if it.get("face_id") is not None else None,
                )
                for it in disproved_in
                if isinstance(it, dict)
            ]
            unsure = [
                (
                    int(it["photo_id"]),
                    int(it["face_id"]) if it.get("face_id") is not None else None,
                )
                for it in unsure_in
                if isinstance(it, dict)
            ]
            reassigned: list[tuple[int, int, int]] = []
            for it in reassigned_in:
                if not isinstance(it, dict) or it.get("face_id") is None:
                    continue
                target_name = str(it.get("new_target_name") or "").strip()
                t_pid = (
                    self.ctx.db.ensure_person(target_name, role="classmate")
                    if target_name
                    else int(it["target_person_id"])
                )
                reassigned.append((int(it["photo_id"]), int(it["face_id"]), t_pid))

            applied = self.ctx.db.apply_batch_decisions(
                person_id=pid,
                approved=approved,
                disproved=disproved,
                unsure=unsure,
                reassigned=reassigned,
            )
            ov = self.ctx.db.get_archive_overview()
            next_batch = self.ctx.db.get_batch_proposal(
                pid,
                batch_size=batch_size,
                mode="confident" if str(body.get("mode") or "") == "confident" else "impact",
            )
            self.ctx.schedule_rematch(delay=0.6)
            self._send_json(
                {
                    "applied": applied,
                    "overview": ov,
                    "next_batch": next_batch,
                    "rematch_scheduled": True,
                }
            )
            return

        if path == "/api/faces/manual":
            photo_id = int(body["photo_id"])
            raw_box = body.get("box")
            if not isinstance(raw_box, dict):
                raise ValueError("Missing bounding box (`box`) for manual face tag.")
            bbox = BoundingBox(
                top=int(raw_box["top"]),
                right=int(raw_box["right"]),
                bottom=int(raw_box["bottom"]),
                left=int(raw_box["left"]),
            )
            new_name = str(body.get("new_name") or "").strip()
            role_raw = str(body.get("role") or "classmate").strip()
            role_m: PersonRole = (
                cast(PersonRole, role_raw)
                if role_raw in ("family", "classmate", "educator", "ignore")
                else "classmate"
            )
            pid = (
                self.ctx.db.ensure_person(new_name, role=role_m)
                if new_name
                else int(body["person_id"])
            )
            new_fid = self.ctx.db.create_manual_face_and_tag(
                photo_id=photo_id,
                box=bbox,
                person_id=pid,
            )
            photo_detail = self.ctx.db.get_photo_detail(photo_id)
            ov = self.ctx.db.get_archive_overview()
            self.ctx.schedule_rematch()
            self._send_json(
                {
                    "person_id": pid,
                    "face_id": new_fid,
                    "photo": photo_detail,
                    "overview": ov,
                    "rematch_scheduled": True,
                }
            )
            return

        if path == "/api/faces/untag":
            photo_id = int(body["photo_id"])
            face_id_req = int(body["face_id"])
            removed_pids = self.ctx.db.untag_face(
                photo_id=photo_id,
                face_id=face_id_req,
            )
            photo_detail = self.ctx.db.get_photo_detail(photo_id)
            if not any(int(f["face_id"]) == face_id_req for f in photo_detail["faces"]):
                face_thumb_path(self.ctx.thumbs_root, face_id_req).unlink(missing_ok=True)
            ov = self.ctx.db.get_archive_overview()
            self.ctx.schedule_rematch()
            self._send_json(
                {
                    "removed_person_ids": removed_pids,
                    "photo": photo_detail,
                    "overview": ov,
                    "rematch_scheduled": True,
                }
            )
            return

        if path == "/api/faces/tag":
            photo_id = int(body["photo_id"])
            face_id = int(body["face_id"]) if body.get("face_id") is not None else None
            st_raw = str(body.get("status") or "confirmed")
            if st_raw == "untagged" and face_id is not None:
                removed_pids = self.ctx.db.untag_face(
                    photo_id=photo_id,
                    face_id=face_id,
                )
                photo_detail = self.ctx.db.get_photo_detail(photo_id)
                if not any(int(f["face_id"]) == face_id for f in photo_detail["faces"]):
                    face_thumb_path(self.ctx.thumbs_root, face_id).unlink(missing_ok=True)
                ov = self.ctx.db.get_archive_overview()
                self.ctx.schedule_rematch()
                self._send_json(
                    {
                        "removed_person_ids": removed_pids,
                        "photo": photo_detail,
                        "overview": ov,
                        "rematch_scheduled": True,
                    }
                )
                return

            new_name = str(body.get("new_name") or "").strip()
            role_raw = str(body.get("role") or "classmate").strip()
            role_t: PersonRole = (
                cast(PersonRole, role_raw)
                if role_raw in ("family", "classmate", "educator", "ignore")
                else "classmate"
            )
            pid = (
                self.ctx.db.ensure_person(new_name, role=role_t)
                if new_name
                else int(body["person_id"])
            )
            if face_id is None and isinstance(body.get("box"), dict) and st_raw == "confirmed":
                raw_b = body["box"]
                manual_fid = self.ctx.db.create_manual_face_and_tag(
                    photo_id=photo_id,
                    box=BoundingBox(
                        top=int(raw_b["top"]),
                        right=int(raw_b["right"]),
                        bottom=int(raw_b["bottom"]),
                        left=int(raw_b["left"]),
                    ),
                    person_id=pid,
                )
                photo_detail = self.ctx.db.get_photo_detail(photo_id)
                ov = self.ctx.db.get_archive_overview()
                self.ctx.schedule_rematch()
                self._send_json(
                    {
                        "person_id": pid,
                        "face_id": manual_fid,
                        "photo": photo_detail,
                        "overview": ov,
                        "rematch_scheduled": True,
                    }
                )
                return

            st_val: AssignmentStatus = (
                cast(AssignmentStatus, st_raw)
                if st_raw in ("confirmed", "rejected", "unsure")
                else "confirmed"
            )
            replace_pid = (
                int(body["replace_person_id"])
                if body.get("replace_person_id") is not None
                else None
            )
            resolved_fid = self.ctx.db.tag_single_face(
                photo_id=photo_id,
                face_id=face_id,
                person_id=pid,
                status=st_val,
                replace_person_id=replace_pid,
            )
            photo_detail = self.ctx.db.get_photo_detail(photo_id)
            if (
                face_id is not None
                and resolved_fid is None
                and not any(int(f["face_id"]) == face_id for f in photo_detail["faces"])
            ):
                face_thumb_path(self.ctx.thumbs_root, face_id).unlink(missing_ok=True)
            ov = self.ctx.db.get_archive_overview()
            self.ctx.schedule_rematch()
            self._send_json(
                {
                    "person_id": pid,
                    "face_id": resolved_fid,
                    "photo": photo_detail,
                    "overview": ov,
                    "rematch_scheduled": True,
                }
            )
            return

        if path.startswith("/api/people/") and path.endswith("/cluster-action"):
            pid = int(path[len("/api/people/") : -len("/cluster-action")])
            action = str(body.get("action") or "").strip()
            faces_in = body.get("faces") or []
            face_pairs = [
                (int(it["photo_id"]), int(it["face_id"]))
                for it in faces_in
                if isinstance(it, dict) and it.get("face_id") is not None
            ]
            applied_n = self.ctx.db.apply_person_cluster_action(
                person_id=pid, action=action, faces=face_pairs
            )
            ov = self.ctx.db.get_archive_overview()
            self.ctx.schedule_rematch(delay=0.3)
            self._send_json({"applied": applied_n, "overview": ov, "rematch_scheduled": True})
            return

        if path == "/api/classes/auto":
            created = self.ctx.db.auto_create_classes()
            ov = self.ctx.db.get_archive_overview()
            self.ctx.broadcast_sse("overview_updated", {"overview": ov})
            self._send_json({"created": created, "overview": ov})
            return

        if path.startswith("/api/classes/") and path.endswith("/members"):
            cid = int(path[len("/api/classes/") : -len("/members")])
            raw_ids = body.get("person_ids") or []
            cls = self.ctx.db.set_class_members(cid, [int(x) for x in raw_ids])
            ov = self.ctx.db.get_archive_overview()
            self.ctx.schedule_rematch(delay=0.8)
            self._send_json({"class": cls, "overview": ov, "rematch_scheduled": True})
            return

        if path == "/api/classes":
            cls = self.ctx.db.create_class(
                name=str(body.get("name") or ""),
                archive_child=str(body.get("archive_child") or ""),
                start_date=str(body.get("start_date") or ""),
                end_date=str(body.get("end_date") or ""),
            )
            ov = self.ctx.db.get_archive_overview()
            self.ctx.broadcast_sse("overview_updated", {"overview": ov})
            self._send_json({"class": cls, "overview": ov})
            return

        if path == "/api/recluster":
            job = self.ctx.start_job("recluster", self.ctx.run_recluster_job)
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return

        if path == "/api/materialize":
            person_f = str(body.get("person") or "").strip() or None
            inc_cand = bool(body.get("include_candidates", False))
            label = f"Syncing ByChild/{person_f}/ folder…" if person_f else None
            job = self.ctx.start_job(
                "materialize",
                lambda: self.ctx.run_materialize_job(person_f, inc_cand),
                label=label,
            )
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return

        self._send_json({"error": f"Unknown POST route: {path}"}, status=HTTPStatus.NOT_FOUND)

    def _handle_patch(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        body = self._read_json_body()
        if path.startswith("/api/people/"):
            pid = int(path[len("/api/people/") :])
            if body.get("merge_into_person_id") is not None:
                target_pid = int(body["merge_into_person_id"])
                self.ctx.db.merge_people(pid, target_pid)
                ov = self.ctx.db.get_archive_overview()
                self.ctx.schedule_rematch()
                self._send_json(
                    {
                        "merged_into": target_pid,
                        "overview": ov,
                        "rematch_scheduled": True,
                    }
                )
                return

            disp = str(body["display_name"]).strip() if "display_name" in body else None
            role_raw = str(body["role"]).strip() if "role" in body else None
            role_opt: PersonRole | None = (
                cast(PersonRole, role_raw)
                if role_raw in ("family", "classmate", "educator", "ignore")
                else None
            )
            cover_fid = int(body["cover_face_id"]) if body.get("cover_face_id") is not None else None
            birth_in = str(body["birth_date"] or "") if "birth_date" in body else None
            updated = self.ctx.db.update_person(
                pid,
                display_name=disp,
                role=role_opt,
                cover_face_id=cover_fid,
                birth_date=birth_in,
            )
            ov = self.ctx.db.get_archive_overview()
            needs_rematch = role_opt is not None or birth_in is not None
            if needs_rematch:
                self.ctx.schedule_rematch()
            else:
                self.ctx.schedule_snapshot()
                self.ctx.broadcast_sse("overview_updated", {"overview": ov})
            self._send_json(
                {
                    "person": updated,
                    "overview": ov,
                    "rematch_scheduled": needs_rematch,
                    "birthday_removed": int(updated.get("birthday_removed", 0)),
                }
            )
            return

        if path.startswith("/api/classes/"):
            cid = int(path[len("/api/classes/") :])
            cls = self.ctx.db.update_class(
                cid,
                name=str(body["name"]) if "name" in body else None,
                archive_child=str(body["archive_child"] or "") if "archive_child" in body else None,
                start_date=str(body["start_date"] or "") if "start_date" in body else None,
                end_date=str(body["end_date"] or "") if "end_date" in body else None,
            )
            ov = self.ctx.db.get_archive_overview()
            self.ctx.schedule_rematch(delay=0.8)
            self._send_json({"class": cls, "overview": ov, "rematch_scheduled": True})
            return

        self._send_json({"error": f"Unknown PATCH route: {path}"}, status=HTTPStatus.NOT_FOUND)

    def _handle_delete(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/people/"):
            pid = int(path[len("/api/people/") :])
            res = self.ctx.db.delete_person(pid)
            for mfid in res.get("deleted_manual_face_ids", []):
                face_thumb_path(self.ctx.thumbs_root, int(mfid)).unlink(missing_ok=True)
            ov = self.ctx.db.get_archive_overview()
            self.ctx.schedule_rematch()
            self._send_json(
                {
                    "deleted_person_id": pid,
                    "removed_assignments": res["removed_assignments"],
                    "overview": ov,
                    "rematch_scheduled": True,
                }
            )
            return

        if path.startswith("/api/classes/"):
            cid = int(path[len("/api/classes/") :])
            self.ctx.db.delete_class(cid)
            ov = self.ctx.db.get_archive_overview()
            self.ctx.schedule_rematch(delay=0.8)
            self._send_json({"overview": ov, "rematch_scheduled": True})
            return

        self._send_json({"error": f"Unknown DELETE route: {path}"}, status=HTTPStatus.NOT_FOUND)


def serve_faces_ui(
    *,
    backup_dir: Path,
    output_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    auto_high_threshold: float = 0.45,
    candidate_threshold: float = 0.33,
    link_mode: LinkMode = "hardlink",
) -> None:
    """Run the local interactive face categorization Web UI until Ctrl+C."""
    backup_root = backup_dir.resolve()
    eff_out = (output_dir or (backup_root / "ByChild")).resolve()
    db_path = ensure_within_directory(backup_root, backup_root / "Metadata" / "faces_index.sqlite")
    if not db_path.is_file():
        raise FileNotFoundError(
            f"No face index found at {db_path}. Run `kinderpedia-backup faces scan` first."
        )

    with FacesIndexDB(db_path) as db:
        csrf_token = db.get_meta("ui_csrf_token")
        if not csrf_token:
            csrf_token = secrets.token_urlsafe(32)
            db.set_meta("ui_csrf_token", csrf_token)
        ctx = _AppContext(
            backup_dir=backup_root,
            output_dir=eff_out,
            db=db,
            csrf_token=csrf_token,
            auto_high_threshold=auto_high_threshold,
            candidate_threshold=candidate_threshold,
            link_mode=link_mode,
            warm_rematch_on_start=True,
        )
        httpd = _FacesHTTPServer((host, port), ctx)
        actual_port = int(httpd.server_address[1])
        url = f"http://{host}:{actual_port}/"
        log(f"[Faces UI] Serving active-learning classroom categorization studio at {url}")
        log("[Faces UI] Press Ctrl+C to stop.")
        if open_browser:
            threading.Timer(0.35, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log("\n[Faces UI] Shutting down cleanly.")
        finally:
            ctx.close(flush_snapshot=True)
            httpd.server_close()
