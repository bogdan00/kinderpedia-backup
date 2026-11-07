"""Security utilities: secret redaction, private state storage, and path safety."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

_QUERY_SECRET_RE = re.compile(
    r"((?:refresh_?token|access_?token|token|api_?key|password|confirmation_?code|"
    r"signature|sig|x-amz-signature|x-amz-credential|x-amz-security-token)=)([^&\s\"']+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_JSON_SECRET_RE = re.compile(
    r'("(?:token|jwToken|refresh_?token|refreshToken|refreshJwToken|apiKey|api_key|'
    r'monolithSession|monolith_session|password|confirmationCode)"\s*:\s*)"([^"]*)"',
    re.IGNORECASE,
)

_SENSITIVE_DICT_KEYS = frozenset({
    "token",
    "jwtoken",
    "refreshtoken",
    "refresh_token",
    "refreshjwtoken",
    "apikey",
    "api_key",
    "monolithsession",
    "monolith_session",
    "password",
    "confirmationcode",
    "vimeo_jwt",
    "vimeojwt",
})


def redact_secrets(text: str) -> str:
    """Scrub tokens, passwords, Bearer headers, and presigned signatures from logs/errors."""
    if not text:
        return text
    out = _QUERY_SECRET_RE.sub(r"\1[REDACTED]", text)
    out = _BEARER_RE.sub(r"\1[REDACTED]", out)
    out = _JSON_SECRET_RE.sub(r'\1"[REDACTED]"', out)
    return out


def scrub_metadata(obj: Any) -> Any:
    """Recursively strip any credential-like keys before writing API metadata to disk."""
    if isinstance(obj, dict):
        return {
            k: ("[REDACTED]" if str(k).lower() in _SENSITIVE_DICT_KEYS else scrub_metadata(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [scrub_metadata(item) for item in obj]
    return obj


def default_state_dir() -> Path:
    """
    Return an OS-private directory for session tokens & adaptive rate state.
    Keeping session tokens out of --output prevents accidental upload when
    backing up directly into a cloud-synced folder (Google Drive, iCloud, Dropbox).
    """
    override = os.environ.get("KINDERPEDIA_STATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return (base / "kinderpedia-backup").resolve()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else (Path.home() / ".config")
    return (base / "kinderpedia-backup").resolve()


def ensure_private_dir(path: Path) -> Path:
    """Create directory with 0700 permissions on POSIX."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        with contextlib.suppress(OSError):
            os.chmod(path, 0o700)
    return path


def replace_atomic(src: Path, dst: Path) -> None:
    """Atomically replace dst with src, retrying brief Windows AV file locks."""
    for attempt in range(6):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if os.name == "nt" and attempt < 5:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise


def write_json_atomic(path: Path, data: Any, *, private: bool = False) -> None:
    """
    Write JSON atomically via a PID/thread-scoped .part file.
    When private=True, creates the file descriptor with 0600 from inception
    so secrets are never world-readable even transiently before rename.
    """
    if private:
        ensure_private_dir(path.parent)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.part")
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

    mode = 0o600 if private else 0o644
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(tmp), flags, mode)
    try:
        if private and os.name != "nt":
            with contextlib.suppress(OSError):
                os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise

    replace_atomic(tmp, path)


def check_dotenv_permissions(dotenv_path: Path) -> str | None:
    """Return a warning message if a POSIX .env file is group- or world-accessible."""
    if os.name == "nt" or not dotenv_path.is_file():
        return None
    with contextlib.suppress(OSError):
        st_mode = dotenv_path.stat().st_mode & 0o777
        if st_mode & 0o077:
            return (
                f"Permissions on {dotenv_path} are {oct(st_mode)} (accessible to other local users). "
                f"Run: chmod 600 '{dotenv_path}'"
            )
    return None


def load_dotenv(*paths: Path) -> tuple[dict[str, str], list[str]]:
    """Load key=value pairs from .env files (earlier paths take precedence) and check permissions."""
    env: dict[str, str] = {}
    warnings: list[str] = []
    seen_resolved: set[Path] = set()

    ordered_unique: list[Path] = []
    for p in paths:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen_resolved:
                seen_resolved.add(rp)
                ordered_unique.append(rp)

    for rp in ordered_unique:
        warn = check_dotenv_permissions(rp)
        if warn:
            warnings.append(warn)

    for rp in reversed(ordered_unique):
        for raw_line in rp.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'").strip('"')
    return env, warnings


def ensure_within_directory(base_dir: Path, target_path: Path) -> Path:
    """Guard against path traversal when joining server-derived names."""
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()
    if not resolved_target.is_relative_to(resolved_base):
        raise ValueError(f"Refusing to write outside backup directory: {target_path}")
    return target_path
