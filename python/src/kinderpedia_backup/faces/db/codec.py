"""Calendar / photo-date helpers and the float16 embedding codec shared by the index modules."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

UNKNOWN_DATE_ORD = 99991231


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_iso_date(value: str | None) -> str:
    """Validate a `YYYY-MM-DD` date string (or empty) and return it normalized; raises `ValueError`."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return date.fromisoformat(raw[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid date {raw!r}; expected YYYY-MM-DD.") from exc


def date_to_ord(iso: str) -> int:
    """`"2025-03-07"` -> `20250307` (0 when empty/invalid)."""
    s = iso[:10]
    if len(s) == 10 and s[4] == "-" and s[7] == "-" and (s[:4] + s[5:7] + s[8:10]).isdigit():
        return int(s[:4] + s[5:7] + s[8:10])
    return 0


def date_ord_to_day_num(o: int) -> int:
    """`20250307` -> proleptic Gregorian day number (`-1` when unknown); clamps day-of-month."""
    if o <= 0 or o >= UNKNOWN_DATE_ORD:
        return -1
    y, m, d = o // 10000, (o // 100) % 100, o % 100
    if not (1 <= m <= 12 and 1 <= y <= 9998):
        return -1
    first = date(y, m, 1)
    next_first = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    last_day = (next_first - first).days
    return first.toordinal() + min(max(d, 1), last_day) - 1


def photo_date_ord(album_date: str, image_dateadd: str, year: str, month: str) -> int:
    """
    Latest possible capture date of a photo as `YYYYMMDD`: the earliest of the album / upload
    timestamps (a photo is always taken on or before it is posted), falling back to the last day of
    the `<Year>/<Month>` folder. Unknown dates return `UNKNOWN_DATE_ORD` (never excluded).
    """
    ords = [o for o in (date_to_ord(album_date or ""), date_to_ord(image_dateadd or "")) if o > 0]
    if ords:
        return min(ords)
    y = str(year or "")
    m = str(month or "")
    if len(y) == 4 and y.isdigit():
        mm = int(m) if m.isdigit() and 1 <= int(m) <= 12 else 12
        return int(y) * 10000 + mm * 100 + 31
    return UNKNOWN_DATE_ORD


def pack_embedding_f16(vec: tuple[float, ...] | NDArray[Any]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr.astype(np.float16).tobytes()


def unpack_embeddings_f32(blobs: list[bytes], dim: int = 512) -> NDArray[np.float32]:
    if not blobs:
        return np.zeros((0, dim), dtype=np.float32)
    joined = b"".join(blobs)
    f16 = np.frombuffer(joined, dtype=np.float16).reshape(len(blobs), dim)
    f32 = f16.astype(np.float32)
    norms = np.maximum(np.linalg.norm(f32, axis=1, keepdims=True), np.float32(1e-8))
    return np.asarray(f32 / norms, dtype=np.float32)
