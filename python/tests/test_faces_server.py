"""HTTP-level tests for the faces UI backend: jobs, child-cluster cache, keep-alive/SSE, error mapping."""

from __future__ import annotations

import http.client
import json
import queue
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import requests

from kinderpedia_backup.faces import BoundingBox, DetectedFace, FacesIndexDB, recompute_active_matching
from kinderpedia_backup.faces.models import PhotoProvenance
from kinderpedia_backup.faces.server import CHILD_CLUSTER_FACES_PREVIEW, _AppContext, _FacesHTTPServer

TOKEN = "tok-server-tests"
HDR = {"X-KP-Token": TOKEN}


def _unit(vals: dict[int, float]) -> tuple[float, ...]:
    arr = np.zeros((512,), dtype=np.float32)
    for k, v in vals.items():
        arr[k] = v
    arr /= float(np.linalg.norm(arr))
    return tuple(float(x) for x in arr.tolist())


def _seed(db: FacesIndexDB, *, n_tight: int = 20, n_cand: int = 6) -> dict[str, Any]:
    """Leo: one confirmed anchor + `n_tight` near-identical faces (one big look-alike cluster) and
    `n_cand` weaker look-alikes that land in the review queue."""
    vecs: list[tuple[float, ...]] = [_unit({0: 1.0})]
    vecs += [_unit({0: 1.0, 100 + i: 0.06}) for i in range(n_tight)]
    vecs += [_unit({0: 0.40, 200 + i: 0.9165}) for i in range(n_cand)]
    batch: list[Any] = []
    for i, vec in enumerate(vecs):
        mo = f"{(i % 9) + 1:02d}"
        batch.append(
            (
                PhotoProvenance(
                    f"Alex/Albums/2026/{mo}/A{i}/{i:03d}.jpg",
                    "Alex",
                    "2026",
                    mo,
                    f"A{i}",
                    f"{i:03d}.jpg",
                    album_date=f"2026-{mo}-10",
                ),
                400,
                300,
                100 + i,
                200 + i,
                "scrfd",
                [DetectedFace(box=BoundingBox(top=10, right=110, bottom=110, left=10), embedding=vec, confidence=0.95)],
            )
        )
    saved = db.upsert_scanned_batch(batch)
    leo = db.ensure_person("Leo", role="classmate")
    db.tag_single_face(photo_id=saved[0][0], face_id=saved[0][1][0], person_id=leo)
    recompute_active_matching(db)
    return {"saved": saved, "leo": leo}


class _SSEReader:
    """Background reader for `GET /api/events` that parses frames into a queue of (event, data)."""

    def __init__(self, port: int) -> None:
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        self.conn.request("GET", "/api/events")
        self.resp = self.conn.getresponse()
        self.events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        ev_name = "message"
        data_lines: list[str] = []
        try:
            while True:
                raw = self.resp.fp.readline()
                if not raw:
                    return
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data_lines:
                        try:
                            payload = json.loads("\n".join(data_lines))
                        except ValueError:
                            payload = {}
                        self.events.put((ev_name, payload))
                    ev_name, data_lines = "message", []
                elif line.startswith("event:"):
                    ev_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())
        except (OSError, ValueError):
            return

    def wait_for(
        self, name: str, pred: Callable[[dict[str, Any]], bool] | None = None, timeout: float = 20.0
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise AssertionError(f"SSE event {name!r} not received")
            try:
                ev, data = self.events.get(timeout=rem)
            except queue.Empty as exc:
                raise AssertionError(f"SSE event {name!r} not received") from exc
            if ev == name and (pred is None or pred(data)):
                return data

    def close(self) -> None:
        self.conn.close()


class FacesServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.backup_dir = Path(self._tmp.name) / "bk"
        self.db = FacesIndexDB(self.backup_dir / "Metadata" / "faces_index.sqlite")
        self.seed = _seed(self.db)
        self.ctx = _AppContext(
            backup_dir=self.backup_dir,
            output_dir=self.backup_dir / "ByChild",
            db=self.db,
            csrf_token=TOKEN,
            rematch_debounce_sec=0.05,
            snapshot_debounce_sec=0.2,
            external_poll_sec=0.0,
        )
        self.httpd = _FacesHTTPServer(("127.0.0.1", 0), self.ctx)
        self.port = int(self.httpd.server_address[1])
        self.base = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        self._sse: list[_SSEReader] = []

    def tearDown(self) -> None:
        for r in self._sse:
            r.close()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.db.close()
        self._tmp.cleanup()

    def sse(self) -> _SSEReader:
        r = _SSEReader(self.port)
        self._sse.append(r)
        r.wait_for("connected")
        return r

    # ------------------------------------------------------------------ C5
    def test_keep_alive_and_sse_still_streams(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", "/api/status")
            r1 = conn.getresponse()
            self.assertEqual(r1.version, 11)
            self.assertIsNotNone(r1.getheader("Content-Length"))
            r1.read()
            self.assertFalse(r1.will_close)
            sock1 = conn.sock
            self.assertIsNotNone(sock1)
            conn.request("GET", "/api/overview")
            r2 = conn.getresponse()
            self.assertEqual(r2.status, 200)
            json.loads(r2.read())
            self.assertIs(conn.sock, sock1)  # same TCP connection reused
            # A POST with a rejected CSRF token must not leave its body in the pipe for the next request:
            conn.request("POST", "/api/people", body=json.dumps({"name": "Nina"}), headers={"X-KP-Token": "bad"})
            r3 = conn.getresponse()
            r3.read()
            self.assertEqual(r3.status, 403)
            conn.request("POST", "/api/people", body=json.dumps({"name": "Nina"}), headers=HDR)
            r4 = conn.getresponse()
            self.assertEqual(r4.status, 200, r4.read())
        finally:
            conn.close()

        reader = self.sse()
        self.assertTrue(reader.resp.getheader("Connection", "").lower() == "close")
        self.ctx.schedule_rematch(delay=0.0)
        reader.wait_for("rematch_scheduled")
        reader.wait_for("rematch_complete")
        # Normal requests keep working while an SSE stream is open:
        self.assertEqual(requests.get(f"{self.base}/api/status", timeout=5).status_code, 200)

    # ------------------------------------------------------------------ C1
    def test_recluster_and_materialize_are_background_jobs(self) -> None:
        reader = self.sse()
        release = threading.Event()

        def slow_job() -> dict[str, Any]:
            release.wait(10)
            return {"ok": 1}

        blocker = self.ctx.start_job("recluster", slow_job)
        st = requests.get(f"{self.base}/api/status", timeout=5).json()
        self.assertTrue(st["busy"])
        self.assertEqual([j["job_id"] for j in st["jobs"]], [blocker["job_id"]])
        self.assertIn("Rebuilding clusters", st["sync_op"])

        r = requests.post(f"{self.base}/api/recluster", json={}, headers=HDR, timeout=5)
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["job"]["job_id"], blocker["job_id"])
        release.set()
        reader.wait_for("job_complete", lambda d: d.get("job_id") == blocker["job_id"])

        t0 = time.monotonic()
        r = requests.post(f"{self.base}/api/recluster", json={}, headers=HDR, timeout=5)
        self.assertEqual(r.status_code, 202, r.text)
        self.assertLess(time.monotonic() - t0, 2.0)
        job_id = r.json()["job_id"]
        reader.wait_for("job_status", lambda d: d.get("job_id") == job_id)
        done = reader.wait_for("job_complete", lambda d: d.get("job_id") == job_id)
        self.assertTrue(done["ok"], done)
        self.assertIn("clusters_created", done["result"])
        self.assertEqual(requests.get(f"{self.base}/api/jobs/{job_id}", timeout=5).json()["state"], "done")
        self.assertEqual(requests.get(f"{self.base}/api/jobs/nope", timeout=5).status_code, 404)

        r = requests.post(f"{self.base}/api/materialize", json={"person": "Leo"}, headers=HDR, timeout=5)
        self.assertEqual(r.status_code, 202, r.text)
        m_id = r.json()["job_id"]
        done = reader.wait_for("job_complete", lambda d: d.get("job_id") == m_id)
        self.assertTrue(done["ok"], done)
        self.assertEqual(done["result"]["person"], "Leo")
        self.assertFalse(requests.get(f"{self.base}/api/status", timeout=5).json()["jobs"])

    def test_recluster_waits_for_running_rematch(self) -> None:
        # Recluster must hold `_rematch_lock`: while it is taken the job cannot progress.
        self.ctx._rematch_lock.acquire()
        try:
            r = requests.post(f"{self.base}/api/recluster", json={}, headers=HDR, timeout=5)
            self.assertEqual(r.status_code, 202)
            job_id = r.json()["job_id"]
            time.sleep(0.3)
            self.assertEqual(self.ctx.get_job(job_id)["state"], "running")  # type: ignore[index]
        finally:
            self.ctx._rematch_lock.release()
        job = self.ctx.wait_for_job(job_id, timeout=20)
        self.assertIsNotNone(job)
        self.assertEqual(job["state"], "done")  # type: ignore[index]

    # ------------------------------------------------------------------ C2
    def test_child_clusters_truncated_cached_and_expandable(self) -> None:
        leo = self.seed["leo"]
        url = f"{self.base}/api/people/{leo}/clusters?threshold=0.5&min_size=3"
        with mock.patch.object(self.db, "get_person_face_clusters", wraps=self.db.get_person_face_clusters) as spy:
            data = requests.get(url, timeout=5).json()
            big = max(data["clusters"], key=lambda c: c["size"])
            self.assertGreater(big["size"], CHILD_CLUSTER_FACES_PREVIEW)
            self.assertEqual(len(big["faces"]), CHILD_CLUSTER_FACES_PREVIEW)
            self.assertTrue(big["faces_truncated"])
            self.assertTrue(data["signature"])

            again = requests.get(url, timeout=5).json()
            self.assertEqual(again["signature"], data["signature"])
            self.assertEqual(spy.call_count, 1)  # served from the cache

            rest = requests.get(
                f"{self.base}/api/people/{leo}/clusters/faces",
                params={"key": big["cluster_key"], "threshold": "0.5", "min_size": "3", "offset": "12"},
                timeout=5,
            ).json()
            self.assertEqual(rest["size"], big["size"])
            self.assertEqual(len(rest["faces"]), big["size"] - CHILD_CLUSTER_FACES_PREVIEW)
            shown = {f["face_id"] for f in big["faces"]}
            self.assertFalse(shown & {f["face_id"] for f in rest["faces"]})
            self.assertEqual(spy.call_count, 1)

            missing = requests.get(
                f"{self.base}/api/people/{leo}/clusters/faces", params={"key": "1999-nope"}, timeout=5
            )
            self.assertEqual(missing.status_code, 404)

            # Any write invalidates the cache, so the follow-up view reflects it immediately:
            victim = big["faces"][-1]
            r = requests.post(
                f"{self.base}/api/people/{leo}/cluster-action",
                json={"action": "exclude", "faces": [{"photo_id": victim["photo_id"], "face_id": victim["face_id"]}]},
                headers=HDR,
                timeout=5,
            )
            r.raise_for_status()
            after = requests.get(url, timeout=5).json()
            self.assertEqual(spy.call_count, 2)
            self.assertNotEqual(after["signature"], data["signature"])
            self.assertEqual(sum(c["size"] for c in after["clusters"]), data["total_faces"] - 1)

    # ------------------------------------------------------------------ batch loop
    def test_batch_submit_next_batch_loop(self) -> None:
        leo = self.seed["leo"]
        b1 = requests.get(f"{self.base}/api/batch", params={"person_id": leo, "size": 4}, timeout=5).json()
        self.assertTrue(b1["cards"])
        first, rest = b1["cards"][0], b1["cards"][1:]
        r = requests.post(
            f"{self.base}/api/batch/submit",
            json={
                "person_id": leo,
                "batch_size": 4,
                "approved": [{"photo_id": first["photo_id"], "face_id": first["face_id"]}],
                "disproved": [{"photo_id": c["photo_id"], "face_id": c["face_id"]} for c in rest],
            },
            headers=HDR,
            timeout=10,
        )
        r.raise_for_status()
        res = r.json()
        self.assertEqual(res["applied"]["approved"], 1)
        self.assertEqual(res["applied"]["disproved"], len(rest))
        self.assertIn("cards", res["next_batch"])
        reviewed = {c["face_id"] for c in b1["cards"]}
        self.assertFalse(reviewed & {c["face_id"] for c in res["next_batch"]["cards"]})
        self.assertTrue(self.ctx.wait_for_rematch(timeout=15))
        b2 = requests.get(f"{self.base}/api/batch", params={"person_id": leo, "size": 4}, timeout=5)
        self.assertEqual(b2.status_code, 200)
        self.assertFalse(reviewed & {c["face_id"] for c in b2.json()["cards"]})

    # ------------------------------------------------------------------ C6
    def test_error_mapping(self) -> None:
        with mock.patch.object(
            self.db, "get_archive_overview", side_effect=sqlite3.OperationalError("database is locked")
        ):
            r = requests.get(f"{self.base}/api/overview", timeout=5)
            self.assertEqual(r.status_code, 503)
            self.assertTrue(r.json()["retryable"])
            self.assertEqual(r.headers.get("Retry-After"), "2")
        with mock.patch.object(self.db, "get_archive_overview", side_effect=RuntimeError("boom")):
            r = requests.get(f"{self.base}/api/overview", timeout=5)
            self.assertEqual(r.status_code, 500)
        self.assertEqual(requests.get(f"{self.base}/api/batch?person_id=abc", timeout=5).status_code, 400)
        r = requests.patch(f"{self.base}/api/people/999999", json={"display_name": "Theo"}, headers=HDR, timeout=5)
        self.assertEqual(r.status_code, 404)
        r = requests.post(
            f"{self.base}/api/people",
            data=b"{not json",
            headers={**HDR, "Content-Type": "application/json"},
            timeout=5,
        )
        self.assertEqual(r.status_code, 400)

    def test_external_writer_invalidates_caches_and_schedules_rematch(self) -> None:
        self.assertTrue(self.ctx.wait_for_rematch(timeout=15))
        self.db.load_all_faces_matrix()
        self.assertFalse(self.ctx.check_external_changes())
        self.db._knn_neighbor_cache = {"stale": True}
        epoch0 = self.ctx.data_epoch
        reader = self.sse()

        other = sqlite3.connect(str(self.db.db_path))
        try:
            other.execute(
                "INSERT INTO meta (key, value) VALUES ('last_scan_at', '2026-11-08T10:00:00') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
            other.commit()
        finally:
            other.close()

        self.assertTrue(self.ctx.check_external_changes())
        self.assertIsNone(self.db._faces_matrix_cache)
        self.assertNotEqual(self.db._knn_neighbor_cache, {"stale": True})
        self.assertGreater(self.ctx.data_epoch, epoch0)
        reader.wait_for("data_changed")
        reader.wait_for("rematch_complete")
        self.assertFalse(self.ctx.check_external_changes())
        # The UI's own writes never look "external":
        self.db.set_meta("ui_probe", "1")
        self.assertFalse(self.ctx.check_external_changes())

    def test_external_watch_thread_polls(self) -> None:
        ctx2 = _AppContext(
            backup_dir=self.backup_dir,
            output_dir=self.backup_dir / "ByChild",
            db=self.db,
            csrf_token=TOKEN,
            warm_cache_on_start=False,
            external_poll_sec=0.05,
        )
        try:
            other = sqlite3.connect(str(self.db.db_path))
            try:
                other.execute("INSERT INTO meta (key, value) VALUES ('ext_probe', '1')")
                other.commit()
            finally:
                other.close()
            deadline = time.monotonic() + 5
            while ctx2.bg_status()["external_changes"] == 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(ctx2.bg_status()["external_changes"], 1)
        finally:
            ctx2.close()


def _find_chrome() -> str | None:
    for cand in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
        shutil.which("chromium-browser") or "",
    ):
        if cand and Path(cand).exists():
            return cand
    return None


_JS_SMOKE = Path(__file__).resolve().parent / "js" / "faces_ui_smoke.mjs"


@unittest.skipUnless(_find_chrome() and shutil.which("node") and _JS_SMOKE.is_file(), "needs Chrome + node")
class FacesUiBrowserSmokeTests(unittest.TestCase):
    """Drive the real `app.js` in headless Chrome over CDP: batch -> submit -> next batch."""

    def test_batch_flow_in_headless_chrome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup_dir = Path(tmp) / "bk"
            db = FacesIndexDB(backup_dir / "Metadata" / "faces_index.sqlite")
            try:
                _seed(db)
                ctx = _AppContext(
                    backup_dir=backup_dir,
                    output_dir=backup_dir / "ByChild",
                    db=db,
                    csrf_token=TOKEN,
                    rematch_debounce_sec=0.05,
                    external_poll_sec=0.0,
                )
                httpd = _FacesHTTPServer(("127.0.0.1", 0), ctx)
                th = threading.Thread(target=httpd.serve_forever, daemon=True)
                th.start()
                try:
                    proc = subprocess.run(
                        [str(shutil.which("node")), str(_JS_SMOKE), f"http://127.0.0.1:{int(httpd.server_address[1])}/", str(_find_chrome())],
                        capture_output=True,
                        text=True,
                        timeout=90,
                    )
                    self.assertEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
                    self.assertIn("SMOKE OK", proc.stdout)
                finally:
                    httpd.shutdown()
                    httpd.server_close()
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
