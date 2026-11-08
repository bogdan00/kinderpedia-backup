"""Advisory cross-process lock so only one `faces scan` writes the index at a time.

The lock is a small JSON file (`Metadata/faces_scan.lock`) holding `pid`, `host`, a random `token`,
`started_at` and a `heartbeat_at` refreshed by a daemon thread. A lock whose holder is gone (dead pid on
this host, or no heartbeat for `stale_after_s`) is taken over. A running `faces ui` never takes this
lock: it keeps working against the WAL database and can show `read_scan_lock()` as "scan running".
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCAN_LOCK_NAME = "faces_scan.lock"


class ScanLockBusyError(RuntimeError):
    """Another live `faces scan` holds the lock."""

    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info
        super().__init__(
            f"Another `faces scan` is already running (pid {info.get('pid')} on {info.get('host')}, "
            f"started {info.get('started_at')}). Wait for it to finish or pass a lock timeout."
        )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":  # os.kill(pid, 0) would terminate the process on Windows
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_stale(info: dict[str, Any] | None, stale_after_s: float) -> bool:
    if info is None:
        return True
    try:
        beat = float(info.get("heartbeat_ts", 0.0))
        pid = int(info.get("pid", 0))
    except (TypeError, ValueError):
        return True
    if time.time() - beat > stale_after_s:
        return True
    return info.get("host") == socket.gethostname() and not _pid_alive(pid)


def read_scan_lock(meta_dir: Path, *, stale_after_s: float = 120.0) -> dict[str, Any] | None:
    """Return the live scan-lock info (`pid`, `host`, `started_at`, `heartbeat_at`) or None."""
    info = _read(meta_dir / SCAN_LOCK_NAME)
    if info is None or _is_stale(info, stale_after_s):
        return None
    info.pop("token", None)
    return info


class ScanLock:
    """Context manager: `with ScanLock(meta_dir, timeout_s=0): ...` (raises `ScanLockBusyError`)."""

    def __init__(
        self,
        meta_dir: Path,
        *,
        timeout_s: float = 0.0,
        heartbeat_s: float = 15.0,
        stale_after_s: float = 120.0,
    ) -> None:
        self.path = meta_dir / SCAN_LOCK_NAME
        self.timeout_s = max(0.0, float(timeout_s))
        self.heartbeat_s = max(0.05, float(heartbeat_s))
        self.stale_after_s = float(stale_after_s)
        self._token = secrets.token_hex(8)
        self._started = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _payload(self) -> dict[str, Any]:
        now = time.time()
        return {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "token": self._token,
            "started_at": self._started,
            "heartbeat_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
            "heartbeat_ts": now,
        }

    def _write_heartbeat(self) -> None:
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        with contextlib.suppress(OSError):
            tmp.write_text(json.dumps(self._payload()), encoding="utf-8")
            os.replace(tmp, self.path)

    def _beat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            info = _read(self.path)
            if info is None or info.get("token") != self._token:
                return  # lock was taken over (we were considered stale); stop touching it
            self._write_heartbeat()

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            self._started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = _read(self.path)
                if _is_stale(info, self.stale_after_s):
                    again = _read(self.path)
                    if (again or {}).get("token") != (info or {}).get("token"):
                        continue  # someone else just replaced it; re-evaluate
                    grave = self.path.with_name(f"{self.path.name}.stale.{os.getpid()}")
                    with contextlib.suppress(OSError):
                        os.replace(self.path, grave)
                        grave.unlink()
                    continue
                if time.monotonic() >= deadline:
                    raise ScanLockBusyError(dict(info or {})) from None
                time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._payload(), fh)
            break
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat_loop, name="faces-scan-lock", daemon=True)
        self._thread.start()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        info = _read(self.path)
        if info is not None and info.get("token") == self._token:
            with contextlib.suppress(OSError):
                self.path.unlink()

    def __enter__(self) -> ScanLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
