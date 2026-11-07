"""Unit tests for AdaptiveRateController."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kinderpedia_backup.sync import AdaptiveRateController


class RateLimitTests(unittest.TestCase):
    def test_clean_streak_decreases_interval_toward_floor(self) -> None:
        ctrl = AdaptiveRateController(None, initial_interval=3.20, min_interval=1.20)
        start = ctrl.interval
        for _ in range(20):
            ctrl.on_success()
        self.assertLess(ctrl.interval, start)
        self.assertEqual(ctrl.total_429s, 0)
        self.assertEqual(ctrl.clean_streak, 20)

    def test_rate_limit_increases_interval_and_persists_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "adaptive_rate.json"
            ctrl = AdaptiveRateController(state_path, initial_interval=2.00)
            for _ in range(6):
                ctrl.on_success()
            before = ctrl.interval
            pause_s = ctrl.on_rate_limit(attempt=0, retry_after_header="5")
            self.assertGreaterEqual(pause_s, 6.0)
            self.assertLessEqual(pause_s, 10.5)
            self.assertGreater(ctrl.interval, before)
            self.assertEqual(ctrl.clean_streak, 0)
            self.assertEqual(ctrl.total_429s, 1)

            reloaded = AdaptiveRateController(state_path, initial_interval=3.15)
            self.assertAlmostEqual(reloaded.interval, ctrl.interval, places=3)
            self.assertAlmostEqual(reloaded.soft_floor, ctrl.soft_floor, places=3)


if __name__ == "__main__":
    unittest.main()
