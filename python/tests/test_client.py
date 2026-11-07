"""Unit tests for KinderpediaClient: S3 404 fallback, 2FA detection, and error redaction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import requests

from kinderpedia_backup.sync import KinderpediaClient, TwoFactorRequiredError


def _mock_stream_response(status_code: int, body: bytes = b"", text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {}
    resp.iter_content.return_value = [body] if body else []
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class ClientTests(unittest.TestCase):
    def test_download_with_fallbacks_skips_404_immediately_without_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = KinderpediaClient(state_dir=Path(tmp), persist_session=False)
            r_404 = _mock_stream_response(404)
            r_200 = _mock_stream_response(200, body=b"jpeg-bytes-here")

            with (
                patch.object(client.http, "get", side_effect=[r_404, r_200]) as mock_get,
                patch("time.sleep") as mock_sleep,
            ):
                dest = Path(tmp) / "001_img.jpg"
                size = client.download_with_fallbacks(
                    [
                        "https://images.kinderpedia.co/files/photos/old.jpg",
                        "https://images.kinderpedia.co/900x1200/files/photos/old.jpg",
                    ],
                    dest,
                )
                self.assertEqual(size, len(b"jpeg-bytes-here"))
                self.assertEqual(dest.read_bytes(), b"jpeg-bytes-here")
                self.assertEqual(mock_get.call_count, 2)
                mock_sleep.assert_not_called()

    def test_login_detects_2fa_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = KinderpediaClient(state_dir=Path(tmp), persist_session=False)
            fake_resp = MagicMock()
            fake_resp.status_code = 200
            fake_resp.text = '{"requiresConfirmationCode": true, "message": "2FA code sent"}'
            fake_resp.json.return_value = {"requiresConfirmationCode": True, "message": "2FA code sent"}

            with patch.object(client.http, "request", return_value=fake_resp), self.assertRaises(TwoFactorRequiredError):
                client.login("parent@example.com", "pw123")

    def test_refresh_token_never_leaks_in_exception_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = KinderpediaClient(state_dir=Path(tmp), persist_session=False)
            client.refresh_token = "ultra_secret_refresh_token_999"

            def boom(*args: Any, **kwargs: Any) -> Any:
                raise requests.ConnectionError(
                    "HTTPSConnectionPool: Max retries exceeded with url: "
                    "/api/token/refresh?refreshToken=ultra_secret_refresh_token_999"
                )

            with (
                patch.object(client.http, "request", side_effect=boom),
                patch("time.sleep"),
                patch("kinderpedia_backup.sync.client.log") as mock_log,
            ):
                ok = client.refresh_jwt()
                self.assertFalse(ok)
                logged = " ".join(str(call.args[0]) for call in mock_log.call_args_list)
                self.assertNotIn("ultra_secret_refresh_token_999", logged)
                self.assertIn("[REDACTED]", logged)


if __name__ == "__main__":
    unittest.main()
