"""Unit tests for CLI subcommand routing (`sync`, `faces`, and implicit default `sync`)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kinderpedia_backup.cli import main


class CliRoutingTests(unittest.TestCase):
    def test_faces_dry_run_succeeds_on_empty_backup_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"XDG_CONFIG_HOME": tmp, "LOCALAPPDATA": tmp}
            with patch.dict(os.environ, env):
                rc = main(["faces", "--input", tmp, "--dry-run"])
            self.assertEqual(rc, 0)

    def test_implicit_sync_flag_routes_to_sync_logout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "cfg"
            out_dir = Path(tmp) / "bk"
            env = {"XDG_CONFIG_HOME": str(state_dir), "LOCALAPPDATA": str(state_dir)}
            with patch.dict(os.environ, env):
                rc = main(["--output", str(out_dir), "--logout"])
            self.assertEqual(rc, 0)

    def test_explicit_sync_subcommand_logout_removes_private_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "cfg"
            pkg_state = state_dir / "kinderpedia-backup"
            pkg_state.mkdir(parents=True)
            sess_file = pkg_state / "session.json"
            sess_file.write_text('{"token": "dummy"}', encoding="utf-8")

            out_dir = Path(tmp) / "bk"
            env = {"XDG_CONFIG_HOME": str(state_dir), "LOCALAPPDATA": str(state_dir)}
            with patch.dict(os.environ, env):
                rc = main(["sync", "--output", str(out_dir), "--logout"])
            self.assertEqual(rc, 0)
            self.assertFalse(sess_file.exists())


if __name__ == "__main__":
    unittest.main()
