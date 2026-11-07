"""Console output, live TTY progress bar, OS sleep inhibitor, and ffmpeg lookup."""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Protocol


class SnapshotProvider(Protocol):
    def snapshot(self) -> dict[str, Any]: ...


CONSOLE_LOCK = threading.RLock()
_ACTIVE_PROGRESS: LiveProgress | None = None


def set_active_progress(progress: LiveProgress | None) -> None:
    global _ACTIVE_PROGRESS
    with CONSOLE_LOCK:
        _ACTIVE_PROGRESS = progress


def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    with contextlib.suppress(Exception):
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)


class LiveProgress:
    """Thread-safe single-line TTY status bar coordinated with log()."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stdout.isatty()
        if self.enabled:
            _enable_windows_ansi()
        self._last_line = ""
        self._last_render = 0.0
        self._hist: deque[tuple[float, int]] = deque(maxlen=30)

    def clear_unlocked(self) -> None:
        if self.enabled and self._last_line:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def redraw_unlocked(self) -> None:
        if self.enabled and self._last_line:
            cols = max(40, shutil.get_terminal_size((100, 20)).columns - 1)
            sys.stdout.write(f"\r\033[K{self._last_line[:cols]}")
            sys.stdout.flush()

    def close(self) -> None:
        with CONSOLE_LOCK:
            self.clear_unlocked()
            self._last_line = ""

    def update(self, tracker: SnapshotProvider, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and (now - self._last_render) < 0.4:
            return
        self._last_render = now

        snap = tracker.snapshot()
        children = snap.get("children", {})
        if not children:
            return

        exp_photos = sum(c.get("expected_photos_count", 0) for c in children.values())
        done_photos = sum(
            c.get("photos_downloaded", 0) + c.get("photos_skipped_existing", 0)
            for c in children.values()
        )
        tot_videos = sum(c.get("videos_count", 0) for c in children.values())
        done_videos = sum(
            c.get("videos_downloaded", 0) + c.get("videos_skipped_existing", 0)
            for c in children.values()
        )
        dl_bytes = sum(
            c.get("photos_bytes", 0) + c.get("videos_bytes", 0)
            for c in children.values()
        )

        tot_units = max(1, exp_photos + tot_videos * 25)
        done_units = min(tot_units, done_photos + done_videos * 25)
        pct = (done_units / tot_units) * 100.0

        self._hist.append((now, dl_bytes))
        mbps = 0.0
        if len(self._hist) >= 2 and (now - self._hist[0][0]) >= 1.0:
            dt = now - self._hist[0][0]
            db = dl_bytes - self._hist[0][1]
            mbps = max(0.0, (db / (1024 * 1024)) / dt)

        width = 18
        filled = round((pct / 100.0) * width)
        bar = "█" * filled + "░" * (width - filled)
        rate = snap.get("adaptive_rate", {})
        pace = rate.get("api_interval_s", 0.0)
        r429 = rate.get("total_429s", 0)

        line = (
            f"[{bar}] {pct:5.1f}% | "
            f"Photos {done_photos:,}/{exp_photos:,} | "
            f"Videos {done_videos}/{tot_videos} | "
            f"{mbps:4.1f} MiB/s | "
            f"API {pace:.2f}s/req (429s={r429})"
        )
        with CONSOLE_LOCK:
            self._last_line = line
            self.redraw_unlocked()


def log(msg: str) -> None:
    with CONSOLE_LOCK:
        if _ACTIVE_PROGRESS is not None:
            _ACTIVE_PROGRESS.clear_unlocked()
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
        if _ACTIVE_PROGRESS is not None:
            _ACTIVE_PROGRESS.redraw_unlocked()


@contextlib.contextmanager
def keep_system_awake(enabled: bool = True) -> Iterator[None]:
    """Prevent OS idle sleep during long multi-gigabyte backups."""
    if not enabled:
        yield
        return

    sys_name = platform.system()
    proc: subprocess.Popen[bytes] | None = None
    win_restored = False

    try:
        if sys_name == "Darwin":
            caff = shutil.which("caffeinate") or "/usr/bin/caffeinate"
            if os.path.exists(caff):
                proc = subprocess.Popen(
                    [caff, "-i", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        elif sys_name == "Windows":
            with contextlib.suppress(Exception):
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # type: ignore[attr-defined]
                win_restored = True
        yield
    finally:
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.terminate()
        if win_restored:
            with contextlib.suppress(Exception):
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # type: ignore[attr-defined]


def find_ffmpeg() -> str | None:
    """Locate ffmpeg across PATH and standard macOS / Windows / Linux install paths."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/snap/bin/ffmpeg",
    ]
    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        prog_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        user_prof = os.environ.get("USERPROFILE", "")
        candidates.extend([
            os.path.join(local_app, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
            os.path.join(prog_data, "chocolatey", "bin", "ffmpeg.exe"),
            os.path.join(user_prof, "scoop", "shims", "ffmpeg.exe"),
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ])

    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def ffmpeg_install_hint() -> str:
    sys_name = platform.system()
    if sys_name == "Darwin":
        return "brew install ffmpeg"
    if sys_name == "Windows":
        return "winget install Gyan.FFmpeg"
    return "sudo apt install ffmpeg   # (or dnf/pacman equivalent)"
