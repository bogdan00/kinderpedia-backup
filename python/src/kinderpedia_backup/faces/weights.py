"""SHA-256-verified model weight management for offline SCRFD + ArcFace ONNX inference."""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import zipfile
from pathlib import Path

import requests

from ..console import log
from ..security import replace_atomic

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

DET_MODEL_FILENAME = "det_10g.onnx"
DET_MODEL_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"

REC_MODEL_FILENAME = "w600k_r50.onnx"
REC_MODEL_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"

_WEIGHTS_LOCK = threading.Lock()


def default_models_dir() -> Path:
    """Return user cache directory storing verified ONNX face recognition models."""
    override = os.environ.get("KINDERPEDIA_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return (base / "kinderpedia-backup" / "models").resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else (Path.home() / ".cache")
    return (base / "kinderpedia-backup" / "models").resolve()


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, expected_sha256: str) -> bool:
    return path.is_file() and sha256_file(path).lower() == expected_sha256.lower()


def ensure_buffalo_l_weights(models_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Ensure `det_10g.onnx` (SCRFD-10GF) and `w600k_r50.onnx` (ArcFace R50) exist locally
    and match their pinned SHA-256 digests. Downloads once on cache miss.
    """
    target_dir = (models_dir or default_models_dir()).resolve()
    det_path = target_dir / DET_MODEL_FILENAME
    rec_path = target_dir / REC_MODEL_FILENAME

    with _WEIGHTS_LOCK:
        if _verified(det_path, DET_MODEL_SHA256) and _verified(rec_path, REC_MODEL_SHA256):
            return det_path, rec_path

        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_zip = target_dir / f"buffalo_l.{os.getpid()}.part.zip"
        log(f"[Faces] Downloading SCRFD-10GF + ArcFace R50 ONNX models to {target_dir} (~275 MB, one-time)...")

        try:
            with requests.get(BUFFALO_L_URL, stream=True, timeout=(15, 180)) as resp:
                resp.raise_for_status()
                with tmp_zip.open("wb") as out:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)

            with zipfile.ZipFile(tmp_zip, "r") as zf:
                for fname, expected_hash in (
                    (DET_MODEL_FILENAME, DET_MODEL_SHA256),
                    (REC_MODEL_FILENAME, REC_MODEL_SHA256),
                ):
                    tmp_onnx = target_dir / f"{fname}.{os.getpid()}.part"
                    with zf.open(fname) as src_stream, tmp_onnx.open("wb") as dst_stream:
                        while buf := src_stream.read(1024 * 1024):
                            dst_stream.write(buf)
                    actual_hash = sha256_file(tmp_onnx)
                    if actual_hash.lower() != expected_hash.lower():
                        tmp_onnx.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"SHA-256 mismatch for {fname}: expected {expected_hash}, got {actual_hash}"
                        )
                    replace_atomic(tmp_onnx, target_dir / fname)
        finally:
            with contextlib.suppress(OSError):
                tmp_zip.unlink(missing_ok=True)

        log("[Faces] Model weights verified (SHA-256 OK).")
        return det_path, rec_path
