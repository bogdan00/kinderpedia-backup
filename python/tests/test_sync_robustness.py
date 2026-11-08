"""Exit codes / auth-expiry circuit breaker (D4) and resumable downloads / ffmpeg watchdog (D5)."""

from __future__ import annotations

import http.server
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

from kinderpedia_backup.cli import main
from kinderpedia_backup.models import ChildBackupStats, ChildContext
from kinderpedia_backup.sync import (
    AdaptiveRateController,
    AuthExpiredError,
    KinderpediaClient,
    StatusTracker,
    backup_child_context,
    cleanup_stale_parts,
    hls_playlist_duration,
)
from kinderpedia_backup.sync.client import _resumable_part_path

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


# ----------------------------------------------------------------------------- D4: client & engine


def _resp(status: int, text: str = "") -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.headers = {}
    r.json.return_value = {}
    return r


class AuthExpiryTests(unittest.TestCase):
    def _client(self, tmp: str) -> KinderpediaClient:
        c = KinderpediaClient(state_dir=Path(tmp), persist_session=False)
        c.rate_ctrl.acquire = lambda *a, **k: None  # type: ignore[method-assign]
        c.token = "expired-token"
        return c

    def test_401_without_refresh_raises_auth_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            with (
                patch.object(c.http, "request", return_value=_resp(401, "expired")),
                self.assertRaises(AuthExpiredError),
            ):
                c.list_albums(1, 2)

    def test_401_after_failed_refresh_raises_auth_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            c.refresh_token = "refresh-me"
            with (
                patch.object(c.http, "request", return_value=_resp(401, "expired")),
                patch.object(c, "refresh_jwt", return_value=False) as refresh,
            ):
                with self.assertRaises(AuthExpiredError):
                    c.get_video_details(1, 2, 3)
                refresh.assert_called_once()

    def test_401_on_unauthenticated_login_call_is_not_auth_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            with patch.object(c.http, "request", return_value=_resp(401, "bad credentials")):
                with self.assertRaises(RuntimeError) as cm:
                    c.login("parent@example.invalid", "pw")
                self.assertNotIsInstance(cm.exception, AuthExpiredError)

    def test_album_images_reraise_auth_expiry_and_fail_loudly_on_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            with (
                patch.object(c, "_http_json", side_effect=AuthExpiredError("401")),
                self.assertRaises(AuthExpiredError),
            ):
                c.get_album_images(1, 2, 3, expected_count=4)
            with patch.object(c, "_http_json", side_effect=RuntimeError("HTTP 500")), patch(
                "kinderpedia_backup.sync.client.log"
            ):
                with self.assertRaises(RuntimeError):
                    c.get_album_images(1, 2, 3, expected_count=4)
                self.assertEqual(c.get_album_images(1, 2, 3, expected_count=0), ({}, []))

    def _engine(self, tmp: Path, images_side_effect: Any) -> tuple[MagicMock, Any]:
        client = MagicMock(spec=KinderpediaClient)
        client.rate_ctrl = AdaptiveRateController(None)
        albums = [
            {"id": i, "name": f"Album {i}", "date": f"2025-06-{i:02d}", "updated_at": f"2025-06-{i:02d}", "count_images": 1}
            for i in range(1, 6)
        ]
        client.list_albums.return_value = albums
        client.list_videos.return_value = []
        client.list_documents.return_value = []
        client.get_album_images.side_effect = images_side_effect

        def fake_dl(urls: list[str], dest: Path) -> int:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return 1

        client.download_with_fallbacks.side_effect = fake_dl
        tracker = StatusTracker(None, client.rate_ctrl, 2, 1)
        ctx = ChildContext(child_id=1, kindergarten_id=2, name="Maya Kid", first_name="Maya")

        def run() -> ChildBackupStats:
            with patch("kinderpedia_backup.sync.engine.log"):
                return backup_child_context(client, ctx, tmp / "bk", tracker, workers=2, video_workers=1)

        return client, run

    def test_album_metadata_failures_are_counted(self) -> None:
        def images(kid: int, cid: int, aid: int, expected_count: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            if aid == 3:
                raise RuntimeError("HTTP 500")
            return {}, [{"id": aid * 10, "extension": "jpg", "download_url": f"https://images.example.invalid/{aid}.jpg"}]

        with tempfile.TemporaryDirectory() as tmp:
            _client, run = self._engine(Path(tmp), images)
            st = run()
            self.assertEqual(st.albums_failed, 1)
            self.assertEqual(st.photos_downloaded, 4)
            self.assertEqual(st.total_failures, 1)

    def test_auth_expiry_aborts_instead_of_failing_every_album(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client, run = self._engine(Path(tmp), AuthExpiredError("HTTP 401"))
            with self.assertRaises(AuthExpiredError):
                run()
            self.assertEqual(client.get_album_images.call_count, 1)


# ----------------------------------------------------------------------------- D4: run_sync exit codes


class _FakeClient:
    instances: ClassVar[list[_FakeClient]] = []

    def __init__(self, **_: Any) -> None:
        self.session_file = Path("session.json")
        self.rate_ctrl = AdaptiveRateController(None)
        self.logins: list[str] = []
        _FakeClient.instances.append(self)

    def load_session(self, legacy_session_file: Path | None = None) -> bool:
        return True

    def get_me_and_children(self) -> tuple[dict[str, Any], list[ChildContext]]:
        return {}, [
            ChildContext(child_id=1, kindergarten_id=9, name="Alex Kid", first_name="Alex"),
            ChildContext(child_id=2, kindergarten_id=9, name="Maya Kid", first_name="Maya"),
        ]

    def refresh_jwt(self, failed_token: str | None = None) -> bool:
        return False

    def login(self, email: str, password: str, confirmation_code: str | None = None) -> None:
        self.logins.append(email)


def _stats(ctx: ChildContext, **kw: int) -> ChildBackupStats:
    return ChildBackupStats(
        context_key=ctx.short_name, child_name=ctx.name, child_id=ctx.child_id,
        kindergarten_id=ctx.kindergarten_id, group_name="", **kw,
    )


class RunSyncExitCodeTests(unittest.TestCase):
    @contextmanager
    def _env(self, backup_side_effect: Any, *, tty: bool = False) -> Iterator[tuple[list[str], MagicMock, Path]]:
        logs: list[str] = []
        _FakeClient.instances.clear()
        with tempfile.TemporaryDirectory() as tmp:
            env = {"KINDERPEDIA_STATE_DIR": str(Path(tmp) / "state"), "KINDERPEDIA_EMAIL": "parent@example.invalid"}
            with (
                patch.dict(os.environ, env),
                patch("kinderpedia_backup.cli.load_dotenv", return_value=({}, [])),
                patch("kinderpedia_backup.cli.KinderpediaClient", _FakeClient),
                patch("kinderpedia_backup.cli.backup_child_context", side_effect=backup_side_effect) as bcc,
                patch("kinderpedia_backup.cli.log", side_effect=lambda m: logs.append(str(m))),
                patch("kinderpedia_backup.cli.sys.stdin.isatty", return_value=tty),
                patch("kinderpedia_backup.cli.getpass.getpass", return_value="pw"),
            ):
                yield logs, bcc, Path(tmp)

    def _main(self, tmp: Path) -> int:
        return main(["sync", "--output", str(tmp / "bk"), "--no-progress", "--no-keep-awake", "--no-save-session"])

    def test_clean_run_exits_zero_and_reports_complete(self) -> None:
        with self._env(lambda client, ctx, **kw: _stats(ctx)) as (logs, _, tmp):
            self.assertEqual(self._main(tmp), 0)
            self.assertTrue(any("BACKUP COMPLETE" in m for m in logs))

    def test_any_failure_exits_non_zero_and_reports_partial(self) -> None:
        for field in ("photos_failed", "videos_failed", "documents_failed", "albums_failed"):
            with self.subTest(field=field):
                def fx(client: Any, ctx: ChildContext, _f: str = field, **kw: Any) -> ChildBackupStats:
                    return _stats(ctx, **{_f: 1})

                with self._env(fx) as (logs, _, tmp):
                    self.assertEqual(self._main(tmp), 1)
                    self.assertTrue(any("BACKUP PARTIAL" in m for m in logs))
                    self.assertFalse(any("BACKUP COMPLETE" in m for m in logs))
                    summary = (tmp / "bk" / "Metadata" / "backup_summary.json").read_text(encoding="utf-8")
                    self.assertIn('"status": "partial"', summary)

    def test_auth_expiry_headless_aborts_with_exit_3(self) -> None:
        with self._env(AuthExpiredError("HTTP 401 on https://apiv20.kinderpedia.co/x")) as (logs, bcc, tmp):
            self.assertEqual(self._main(tmp), 3)
            self.assertEqual(bcc.call_count, 1, "the second child is not attempted")
            self.assertTrue(any("BACKUP FAILED" in m and "session expired" in m for m in logs))
            self.assertEqual(_FakeClient.instances[0].logins, [])

    def test_auth_expiry_on_tty_reprompts_once_and_resumes(self) -> None:
        calls: list[int] = []

        def fx(client: Any, ctx: ChildContext, **kw: Any) -> ChildBackupStats:
            calls.append(ctx.child_id)
            if len(calls) == 1:
                raise AuthExpiredError("HTTP 401")
            return _stats(ctx)

        with self._env(fx, tty=True) as (_logs, _, tmp):
            self.assertEqual(self._main(tmp), 0)
            self.assertEqual(calls, [1, 1, 2])
            self.assertEqual(_FakeClient.instances[0].logins, ["parent@example.invalid"])

    def test_auth_expiry_on_tty_gives_up_after_one_reprompt(self) -> None:
        with self._env(AuthExpiredError("HTTP 401"), tty=True) as (_, bcc, tmp):
            self.assertEqual(self._main(tmp), 3)
            self.assertEqual(bcc.call_count, 2)


# ----------------------------------------------------------------------------- D5: Range resume


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    mode = "range"  # "range" | "norange" | "drop_once"
    seen_ranges: ClassVar[list[str | None]] = []
    dropped = False

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        rng = self.headers.get("Range")
        type(self).seen_ranges.append(rng)
        cls = type(self)
        if cls.mode == "drop_once" and not cls.dropped:
            cls.dropped = True
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(PAYLOAD[: len(PAYLOAD) // 3])
            self.wfile.flush()
            self.close_connection = True
            return
        if rng and cls.mode != "norange":
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(PAYLOAD)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = PAYLOAD[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)


class RangeResumeTests(unittest.TestCase):
    @contextmanager
    def _server(self, mode: str) -> Iterator[str]:
        handler: type[_RangeHandler] = type(
            "H", (_RangeHandler,), {"mode": mode, "seen_ranges": [], "dropped": False}
        )
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self.handler = handler
        try:
            yield f"http://127.0.0.1:{srv.server_address[1]}/media/video.mp4?token=rotating"
        finally:
            srv.shutdown()
            srv.server_close()

    def _client(self, tmp: str) -> KinderpediaClient:
        return KinderpediaClient(state_dir=Path(tmp) / "state", persist_session=False)

    def test_existing_part_is_resumed_with_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self._server("range") as url:
            dest = Path(tmp) / "video.mp4"
            part = _resumable_part_path(url, dest)
            part.write_bytes(PAYLOAD[:300_000])
            size = self._client(tmp).download_binary(url, dest)
            self.assertEqual(size, len(PAYLOAD))
            self.assertEqual(dest.read_bytes(), PAYLOAD)
            self.assertEqual(self.handler.seen_ranges, ["bytes=300000-"])
            self.assertFalse(part.exists())

    def test_dropped_connection_resumes_on_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self._server("drop_once") as url, patch("time.sleep"):
            dest = Path(tmp) / "video.mp4"
            size = self._client(tmp).download_binary(url, dest)
            self.assertEqual(size, len(PAYLOAD))
            self.assertEqual(dest.read_bytes(), PAYLOAD)
            self.assertEqual(self.handler.seen_ranges[0], None)
            self.assertEqual(len(self.handler.seen_ranges), 2)
            resumed_from = int(str(self.handler.seen_ranges[1]).split("=")[1].rstrip("-"))
            self.assertGreater(resumed_from, 0)

    def test_server_without_range_support_restarts_from_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self._server("norange") as url:
            dest = Path(tmp) / "video.mp4"
            part = _resumable_part_path(url, dest)
            part.write_bytes(b"garbage-from-elsewhere")
            self.assertEqual(self._client(tmp).download_binary(url, dest), len(PAYLOAD))
            self.assertEqual(dest.read_bytes(), PAYLOAD)

    def test_complete_part_is_finalized_on_416(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self._server("range") as url:
            dest = Path(tmp) / "video.mp4"
            _resumable_part_path(url, dest).write_bytes(PAYLOAD)
            self.assertEqual(self._client(tmp).download_binary(url, dest), len(PAYLOAD))
            self.assertEqual(dest.read_bytes(), PAYLOAD)

    def test_truncated_body_is_never_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"Content-Length": "100"}
            resp.iter_content.return_value = [b"x" * 40]
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            dest = Path(tmp) / "a.jpg"
            with (
                patch.object(c.http, "get", return_value=resp),
                patch("time.sleep"),
                self.assertRaises(RuntimeError),
            ):
                c.download_binary("https://images.example.invalid/a.jpg", dest, retries=2)
            self.assertFalse(dest.exists())

    def test_cleanup_keeps_recent_resumable_parts_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fresh = root / "v.mp4.0123456789.part"
            stale = root / "w.mp4.abcdef0123.part"
            legacy = root / "x.jpg.123.456.part"
            ff = root / "y.123.456.part.mp4"
            for p in (fresh, stale, legacy, ff):
                p.write_bytes(b"d")
            old = time.time() - 30 * 24 * 3600
            os.utime(stale, (old, old))
            self.assertEqual(cleanup_stale_parts(root), 3)
            self.assertTrue(fresh.exists())


# ----------------------------------------------------------------------------- D5: ffmpeg watchdog


class FfmpegWatchdogTests(unittest.TestCase):
    def _script(self, tmp: str, body: str) -> list[str]:
        path = Path(tmp) / "fake_ffmpeg.py"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return [sys.executable, str(path)]

    def test_stalled_ffmpeg_is_killed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = self._script(tmp, """
                import sys, time
                print("total_size=10\\nprogress=continue", flush=True)
                time.sleep(60)
            """)
            started = time.monotonic()
            with self.assertRaises(RuntimeError) as cm:
                KinderpediaClient._run_ffmpeg_watchdog(cmd, stall_timeout=1.0, poll_s=0.05)
            self.assertIn("stalled", str(cm.exception))
            self.assertLess(time.monotonic() - started, 20)

    def test_slow_but_progressing_ffmpeg_is_not_killed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = self._script(tmp, """
                import sys, time
                for i in range(8):
                    print(f"total_size={i * 1000}\\nout_time_us={i * 500000}\\nprogress=continue", flush=True)
                    time.sleep(0.15)
                print("progress=end", flush=True)
                sys.stderr.write("done\\n")
            """)
            rc, err = KinderpediaClient._run_ffmpeg_watchdog(cmd, stall_timeout=2.0, poll_s=0.05)
            self.assertEqual(rc, 0)
            self.assertIn("done", err)

    def test_runtime_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = self._script(tmp, """
                import time
                i = 0
                while True:
                    i += 1
                    print(f"total_size={i}", flush=True)
                    time.sleep(0.1)
            """)
            with self.assertRaises(RuntimeError) as cm:
                KinderpediaClient._run_ffmpeg_watchdog(cmd, stall_timeout=5.0, max_runtime=1.0, poll_s=0.05)
            self.assertIn("ceiling", str(cm.exception))

    def test_hls_playlist_duration(self) -> None:
        m3u8 = "#EXTM3U\n#EXTINF:6.006,\nseg1.ts\n#EXTINF:4.5,\nseg2.ts\n#EXT-X-ENDLIST\n"
        self.assertAlmostEqual(hls_playlist_duration(m3u8), 10.506)
        self.assertEqual(hls_playlist_duration("#EXTM3U\n"), 0.0)


if __name__ == "__main__":
    unittest.main()
