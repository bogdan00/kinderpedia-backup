"""Adaptive AIMD + soft-floor rate limiter for Kinderpedia's web API."""

from __future__ import annotations

import contextlib
import json
import random
import threading
import time
from datetime import datetime
from pathlib import Path

from ..models import RateSnapshot
from ..security import write_json_atomic


class AdaptiveRateController:
    """
    AIMD + CUBIC-inspired soft-floor rate controller for apiv20.kinderpedia.co.

    - Gently decreases inter-request interval during clean streaks to find optimal pace.
    - Remembers a soft floor slightly above any observed HTTP 429 threshold so steady
      state stays below server rate-limit walls.
    - Applies multiplicative slowdown + jittered exponential backoff on HTTP 429.
    """

    def __init__(
        self,
        state_file: Path | None = None,
        *,
        initial_interval: float = 3.15,
        min_interval: float = 1.20,
        max_interval: float = 15.0,
    ) -> None:
        self.state_file = state_file
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.interval = min(max_interval, max(min_interval, initial_interval))
        self.soft_floor = 2.85
        self.clean_streak = 0
        self.total_calls = 0
        self.total_429s = 0
        self.cooldown_until_epoch = 0.0
        self._next_slot = 0.0
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.exists():
            return
        with contextlib.suppress(Exception):
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            saved_int = float(data.get("interval", self.interval))
            saved_floor = float(data.get("soft_floor", self.soft_floor))
            self.interval = min(self.max_interval, max(self.min_interval, saved_int))
            self.soft_floor = min(self.max_interval, max(self.min_interval, saved_floor))

    def _save_unlocked(self) -> None:
        if self.state_file is None:
            return
        with contextlib.suppress(Exception):
            write_json_atomic(
                self.state_file,
                {
                    "interval": round(self.interval, 4),
                    "soft_floor": round(self.soft_floor, 4),
                    "clean_streak": self.clean_streak,
                    "total_calls": self.total_calls,
                    "total_429s": self.total_429s,
                    "updated_at": datetime.now().isoformat(),
                },
                private=True,
            )

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.interval
            wait_s = slot - now
        if wait_s > 0:
            time.sleep(wait_s)

    def on_success(self) -> None:
        with self._lock:
            self.total_calls += 1
            self.clean_streak += 1
            if self.clean_streak % 25 == 0 and self.soft_floor > self.min_interval:
                self.soft_floor = max(self.min_interval, self.soft_floor - 0.02)

            if self.clean_streak >= 4:
                if self.interval > self.soft_floor + 0.20:
                    step = 0.04
                elif self.interval > self.soft_floor:
                    step = 0.01
                else:
                    step = 0.005 if (self.clean_streak >= 35 and self.clean_streak % 5 == 0) else 0.0
                self.interval = max(self.min_interval, self.interval - step)

            if self.total_calls % 15 == 0:
                self._save_unlocked()

    def on_rate_limit(self, attempt: int, retry_after_header: str = "") -> float:
        with self._lock:
            self.total_calls += 1
            self.total_429s += 1
            prev_interval = self.interval
            if self.total_calls >= 5:
                self.soft_floor = min(self.max_interval, max(self.soft_floor, prev_interval * 1.06, 3.05))
            self.interval = min(self.max_interval, max(prev_interval * 1.25, self.soft_floor + 0.12))
            self.clean_streak = 0

            if retry_after_header.strip().isdigit():
                base_wait = float(retry_after_header.strip())
            else:
                base_wait = min(180.0, 20.0 * (2**attempt))
            pause_s = base_wait + random.uniform(1.0, 5.0)

            target = time.monotonic() + pause_s
            if target > self._next_slot:
                self._next_slot = target
            self.cooldown_until_epoch = time.time() + pause_s
            self._save_unlocked()
            return pause_s

    def snapshot(self) -> RateSnapshot:
        with self._lock:
            rpm = 60.0 / self.interval if self.interval > 0 else 0.0
            rem_cd = max(0.0, self.cooldown_until_epoch - time.time())
            return RateSnapshot(
                api_interval_s=round(self.interval, 3),
                effective_rpm=round(rpm, 1),
                soft_floor_s=round(self.soft_floor, 3),
                clean_streak=self.clean_streak,
                total_api_calls=self.total_calls,
                total_429s=self.total_429s,
                cooldown_remaining_s=round(rem_cd, 1),
            )
