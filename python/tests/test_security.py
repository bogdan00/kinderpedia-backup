"""Unit tests for secret redaction, private 0600/0700 storage, and path safety."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from kinderpedia_backup.security import (
    check_dotenv_permissions,
    ensure_within_directory,
    load_dotenv,
    redact_secrets,
    scrub_metadata,
    write_json_atomic,
)


class SecurityTests(unittest.TestCase):
    def test_redact_secrets_scrubs_query_params_headers_and_json(self) -> None:
        url = (
            "https://usergateway-services.kinderpedia.co/api/token/refresh"
            "?refreshToken=super_secret_rt_123&foo=bar"
        )
        redacted_url = redact_secrets(url)
        self.assertNotIn("super_secret_rt_123", redacted_url)
        self.assertIn("refreshToken=[REDACTED]", redacted_url)
        self.assertIn("foo=bar", redacted_url)

        presigned = "https://s3.amazonaws.com/bucket/photo.jpg?X-Amz-Signature=deadbeef999&X-Amz-Credential=AKIA123"
        redacted_s3 = redact_secrets(presigned)
        self.assertNotIn("deadbeef999", redacted_s3)
        self.assertNotIn("AKIA123", redacted_s3)

        bearer_msg = "Auth header: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig failed"
        self.assertEqual(redact_secrets(bearer_msg), "Auth header: Bearer [REDACTED] failed")

        json_err = '{"token": "jwt_abc", "refreshToken": "rt_xyz", "password": "pw!"}'
        redacted_json = redact_secrets(json_err)
        self.assertNotIn("jwt_abc", redacted_json)
        self.assertNotIn("rt_xyz", redacted_json)
        self.assertNotIn("pw!", redacted_json)

    def test_scrub_metadata_strips_credential_keys_recursively(self) -> None:
        raw = {
            "id": 42,
            "name": "Summer Album",
            "apiKey": "secret-key",
            "nested": [{"token": "secret-jwt", "url": "https://images.kinderpedia.co/1.jpg"}],
        }
        cleaned = scrub_metadata(raw)
        self.assertEqual(cleaned["id"], 42)
        self.assertEqual(cleaned["apiKey"], "[REDACTED]")
        self.assertEqual(cleaned["nested"][0]["token"], "[REDACTED]")
        self.assertEqual(cleaned["nested"][0]["url"], "https://images.kinderpedia.co/1.jpg")

    @unittest.skipIf(os.name == "nt", "POSIX file mode checks only")
    def test_private_write_sets_0700_dir_and_0600_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "private_state"
            sess_file = state_dir / "session.json"
            write_json_atomic(sess_file, {"token": "abc"}, private=True)

            dir_mode = state_dir.stat().st_mode & 0o777
            file_mode = sess_file.stat().st_mode & 0o777
            self.assertEqual(dir_mode, 0o700)
            self.assertEqual(file_mode, 0o600)

    @unittest.skipIf(os.name == "nt", "POSIX file mode checks only")
    def test_dotenv_permission_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("KINDERPEDIA_EMAIL=parent@example.com\nexport KINDERPEDIA_PASSWORD='s3cret'\n")
            os.chmod(env_file, 0o644)
            self.assertIsNotNone(check_dotenv_permissions(env_file))

            os.chmod(env_file, 0o600)
            self.assertIsNone(check_dotenv_permissions(env_file))

            parsed, warnings = load_dotenv(env_file)
            self.assertEqual(warnings, [])
            self.assertEqual(parsed["KINDERPEDIA_EMAIL"], "parent@example.com")
            self.assertEqual(parsed["KINDERPEDIA_PASSWORD"], "s3cret")

    def test_ensure_within_directory_blocks_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "backup"
            base.mkdir()
            safe = ensure_within_directory(base, base / "Child" / "Albums")
            self.assertTrue(str(safe).startswith(str(base)))

            with self.assertRaises(ValueError):
                ensure_within_directory(base, base / ".." / "escaped.txt")


if __name__ == "__main__":
    unittest.main()
