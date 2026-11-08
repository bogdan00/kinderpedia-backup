"""Short-name disambiguation and video-to-day-album matching shared with the Bookmarklet."""

from __future__ import annotations

import collections
import contextlib
import functools
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ..models import ChildContext

_STOPWORDS = frozenset({
    "in", "de", "la", "cu", "si", "pe", "din", "un", "una", "ale", "cel", "cea",
    "mai", "pentru", "dupa", "prin", "sub", "sau", "iar", "dar", "ca", "ce", "se",
    "au", "sa", "ne", "am", "ai", "va", "le", "ii", "the", "and", "for", "with", "from",
})


def sanitize_name(name: str, max_len: int = 60) -> str:
    """Produce a cross-platform filename matching `bookmarklet/kinderpedia-backup.js` `sanitizeName`."""
    if not name:
        return "unnamed"
    clean = name
    clean = re.sub(r"[\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]", " ", clean)
    clean = re.sub(r"[\u201c\u201d\u201e\u00ab\u00bb\u201f]", "", clean)
    clean = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", clean)
    clean = re.sub(r"\u2026", "", clean)
    clean = re.sub(r"[\u2012\u2013\u2014\u2015]", "-", clean)
    clean = re.sub(r"[\U00010000-\U0010ffff]", "", clean)
    clean = re.sub(r"[\u2300-\u27bf\u2b50\ufe00-\ufe0f\u200b-\u200d\ufeff]", "", clean)
    clean = re.sub(r'[/\\?%*:|"<>]', "_", clean)
    clean = re.sub(r"[\x00-\x1f\x7f]", "", clean)
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"_+", "_", clean).strip()
    clean = clean.lstrip(". ")
    clean = re.sub(r"[. ]+$", "", clean)

    if len(clean) > max_len:
        clean = re.sub(r"[. ]+$", "", clean[:max_len].strip())

    if re.match(r"^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])$", clean, re.I):
        clean = f"_{clean}"

    return clean or "unnamed"


def sanitize_extension(ext: str | None, default: str = "bin") -> str:
    """Allow strictly alphanumeric file extensions (preventing path traversal via extension)."""
    if not ext:
        return default
    cleaned = re.sub(r"[^a-z0-9]", "", ext.strip().lstrip(".").lower())
    return cleaned[:12] if cleaned else default


_ISO_DATE_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?\s*(Z|[+-]\d{2}:?\d{2})?)?"
)
_TEXT_DATE_RE = re.compile(
    r"^(?:[A-Za-z]+\.?,\s*)?(\d{1,2})\s+([A-Za-z]+|\d{1,2})\.?\s+(\d{4})(?:\s*,?\s*(\d{1,2}):(\d{2}))?$"
)
_ANY_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_MONTHS = {
    name: idx
    for idx, names in enumerate(
        (
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}


def _valid_ymd(y: int, mo: int, d: int) -> bool:
    if not (1 <= mo <= 12 and d >= 1):
        return False
    days = (31, 29 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return d <= days[mo - 1]


def _local_ts(y: int, mo: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> float | None:
    try:
        return datetime(y, mo, d, hh, mm, ss).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def parse_date_prefix(date_str: str | None) -> tuple[str, float | None]:
    """
    Extract a `YYYY-MM-DD` folder date prefix and a POSIX timestamp (or `unknown_date`).

    Canonical, locale-independent contract shared byte-for-byte with the Bookmarklet's
    `parseDatePrefix` (see `python/tests/fixtures/layout_contract.json`):
    1. ISO/SQL `YYYY-MM-DD[( |T)HH:MM[:SS[.fff]][Z|+HH[:]MM]]` anchored at the start. The prefix is
       always the literal digits (never shifted by time zones). Timestamps ignore fractional seconds;
       offset-less times are interpreted in local time. Invalid calendar dates/times keep the prefix
       but yield `None`.
    2. `[Weekday,] DD <Month|MM> YYYY[ [,] HH:MM]` with English month names/abbreviations.
    3. Any embedded `YYYY-MM-DD` (prefix only, no timestamp).
    """
    if not date_str:
        return "unknown_date", None
    s = str(date_str).strip()
    if not s:
        return "unknown_date", None

    m = _ISO_DATE_RE.match(s)
    if m:
        prefix = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not _valid_ymd(y, mo, d):
            return prefix, None
        if m.group(4) is None:
            return prefix, _local_ts(y, mo, d)
        hh, mi, ss = int(m.group(4)), int(m.group(5)), int(m.group(6) or 0)
        if hh > 23 or mi > 59 or ss > 59:
            return prefix, None
        tz = m.group(7)
        if not tz:
            return prefix, _local_ts(y, mo, d, hh, mi, ss)
        offset_min = 0
        if tz != "Z":
            digits = tz[1:].replace(":", "")
            oh, om = int(digits[:2]), int(digits[2:])
            if oh > 23 or om > 59:
                return prefix, None
            offset_min = (oh * 60 + om) * (-1 if tz[0] == "-" else 1)
        try:
            epoch = datetime(y, mo, d, hh, mi, ss, tzinfo=timezone.utc).timestamp()
        except (ValueError, OverflowError, OSError):
            return prefix, None
        return prefix, epoch - offset_min * 60

    m = _TEXT_DATE_RE.match(s)
    if m:
        d, y = int(m.group(1)), int(m.group(3))
        mon_raw = m.group(2).lower()
        mo = int(mon_raw) if mon_raw.isdigit() else _MONTHS.get(mon_raw, 0)
        if _valid_ymd(y, mo, d):
            hh = int(m.group(4)) if m.group(4) is not None else 0
            mi = int(m.group(5)) if m.group(5) is not None else 0
            if hh <= 23 and mi <= 59:
                return f"{y:04d}-{mo:02d}-{d:02d}", _local_ts(y, mo, d, hh, mi)

    m = _ANY_DATE_RE.search(s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", None
    return "unknown_date", None


def split_year_month(ymd: str) -> tuple[str, str]:
    """Return `(<Year>, <Month>)` folder segments for a `YYYY-MM-DD` date prefix."""
    if len(ymd) == 10 and ymd[4] == "-" and ymd[7] == "-" and ymd[:4].isdigit() and ymd[5:7].isdigit():
        return ymd[:4], ymd[5:7]
    return "Unknown", "00"


def resolve_child_folder_names(contexts: Sequence[ChildContext]) -> dict[tuple[int, int], str]:
    """
    Assign short first-name folder names (e.g. `Alex`, `Maya`), disambiguating only when
    multiple contexts share the same short name (e.g. same child across multiple schools).
    """
    short_counts: collections.Counter[str] = collections.Counter(
        sanitize_name(c.short_name, 60).lower() for c in contexts
    )
    full_counts: collections.Counter[str] = collections.Counter(
        sanitize_name(c.name, 60).lower() for c in contexts
    )

    out: dict[tuple[int, int], str] = {}
    used_ci: set[str] = set()
    for c in contexts:
        short = sanitize_name(c.short_name, 60)
        full = sanitize_name(c.name, 60)
        if short_counts[short.lower()] == 1:
            cand = short
        elif full_counts[full.lower()] == 1:
            cand = full
        else:
            school_tag = sanitize_name(c.kindergarten_name, 30) if c.kindergarten_name else f"KG_{c.kindergarten_id}"
            cand = sanitize_name(f"{short} ({school_tag})", 60)

        base = cand
        k = 2
        while cand.lower() in used_ci:
            cand = sanitize_name(f"{base} ({k})", 60)
            k += 1
        used_ci.add(cand.lower())
        out[(c.child_id, c.kindergarten_id)] = cand
    return out


def _leaf_of(folder_rel: str) -> str:
    return folder_rel.rsplit("/", 1)[-1]


def assign_album_folder_names(
    albums: Sequence[dict[str, Any]],
    existing: Mapping[int, str] | None = None,
) -> dict[int, str]:
    """
    Assign deterministic album relative paths (`<Year>/<Month>/<YYYY>-<MM>-<DD> <Sanitised Title>`),
    appending ` (2)` in chronological `(timestamp_asc, id_asc)` order on same-day duplicate titles.

    `existing` is the persisted `album_id -> folder` map (`Metadata/<Child>/folder_map.json`): albums
    already present there keep their folder verbatim (even if the teacher renamed/re-dated them), and
    every persisted leaf (including those of deleted albums) is reserved, so only unknown IDs receive
    new names and siblings are never renumbered. With no `existing` map the output is identical to
    the historical position-based assignment.
    """
    existing = existing or {}
    used_ci: set[str] = {_leaf_of(folder).lower() for folder in existing.values()}
    folder_by_id: dict[int, str] = {}
    seen_ids: set[int] = set()

    by_base: dict[str, list[tuple[float, int, str, str, str]]] = collections.defaultdict(list)
    for a in albums:
        if "id" not in a:
            continue
        aid = int(a["id"])
        if aid in existing:
            folder_by_id[aid] = existing[aid]
            continue
        if aid in seen_ids:
            continue  # duplicate listing entry: first occurrence wins
        seen_ids.add(aid)
        ymd, ts = parse_date_prefix(str(a.get("date") or a.get("updated_at") or ""))
        year_s, month_s = split_year_month(ymd)
        title = sanitize_name(str(a.get("name") or f"album_{aid}"), 60)
        leaf_base = f"{ymd} {title}"
        by_base[leaf_base.lower()].append((ts if ts is not None else float("inf"), aid, year_s, month_s, leaf_base))

    for group in by_base.values():
        group.sort(key=lambda t: (t[0], t[1]))
        for idx, (_, aid, year_s, month_s, leaf_base) in enumerate(group):
            cand_leaf = leaf_base if idx == 0 else f"{leaf_base} ({idx + 1})"
            k = idx + 2
            while cand_leaf.lower() in used_ci:
                cand_leaf = f"{leaf_base} ({k})"
                k += 1
            used_ci.add(cand_leaf.lower())
            folder_by_id[aid] = f"{year_s}/{month_s}/{cand_leaf}"
    return folder_by_id


def _normalize_match_text(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    no_acc = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", no_acc).strip()


def stem_similarity(video_text: str, album_text: str) -> int:
    """Compute diacritic-insensitive keyword/stem overlap score between a video and an album."""
    vn = _normalize_match_text(video_text)
    an = _normalize_match_text(album_text)
    if not vn or not an:
        return 0
    score = 0
    if len(vn) >= 4 and len(an) >= 4 and (vn in an or an in vn):
        score += 2
    v_tokens = [w for w in vn.split() if len(w) >= 3 and w not in _STOPWORDS]
    a_tokens = [w for w in an.split() if len(w) >= 3 and w not in _STOPWORDS]
    for w1 in v_tokens:
        for w2 in a_tokens:
            if w1 == w2 or (len(w1) >= 4 and len(w2) >= 4 and w1[:5] == w2[:5]):
                score += 1
                break
    return score


def _extract_owner_id(item: dict[str, Any]) -> int | None:
    owner = item.get("owner")
    if isinstance(owner, dict) and owner.get("user_id") is not None:
        with contextlib.suppress(ValueError, TypeError):
            return int(owner["user_id"])
    return None


def _score_album_candidate(
    cand: dict[str, Any],
    *,
    v_text: str,
    v_owner: int | None,
    v_ts: float | None,
) -> tuple[int, int, float, int]:
    _, a_ts = parse_date_prefix(str(cand.get("date") or cand.get("updated_at") or ""))
    a_owner = _extract_owner_id(cand)
    a_text = f"{cand.get('name') or ''} {cand.get('description') or ''}"
    st = stem_similarity(v_text, a_text)
    same_o = 1 if (v_owner is not None and a_owner is not None and v_owner == a_owner) else 0
    dt_diff = abs((v_ts or 0.0) - (a_ts or 0.0)) if (v_ts is not None and a_ts is not None) else 1e12
    return (st, same_o, -dt_diff, -int(cand["id"]))


def plan_video_targets(
    videos: Sequence[dict[str, Any]],
    albums: Sequence[dict[str, Any]],
    album_folder_by_id: Mapping[int, str],
    existing: Mapping[int, tuple[str, str]] | None = None,
) -> dict[int, tuple[str, str]]:
    """
    Map each video ID to `(target_album_rel_path, short_video_mp4_filename)` inside `Albums/`:
    - 1 album on same `YYYY-MM-DD`: placed into that album folder (`<Year>/<Month>/<YYYY>-<MM>-<DD> <Title>`).
    - 2+ albums on same `YYYY-MM-DD`: matched via `(stem_similarity, same_educator, -time_diff, -album_id)`.
    - 0 albums on same `YYYY-MM-DD`: placed into `Albums/<Year>/<Month>/<YYYY>-<MM>-<DD> <Video Title>/`.
    - Duplicate video titles inside the same target folder receive ` (2)`, ` (3)` chronologically.

    `existing` (persisted `video_id -> (folder, filename)`) pins already-placed videos; their file
    names are reserved in their folders so new videos never displace them.
    """
    existing = existing or {}
    albums_by_day: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for a in albums:
        if "id" not in a or int(a["id"]) not in album_folder_by_id:
            continue
        ymd, _ = parse_date_prefix(str(a.get("date") or a.get("updated_at") or ""))
        albums_by_day[ymd].append(a)

    out: dict[int, tuple[str, str]] = {}
    used_by_folder: dict[str, set[str]] = collections.defaultdict(set)
    for folder, fname in existing.values():
        used_by_folder[folder].add(fname.lower())

    videos_by_folder: dict[str, list[tuple[float, int, str]]] = collections.defaultdict(list)
    seen_vids: set[int] = set()
    for v in videos:
        if "id" not in v:
            continue
        vid = int(v["id"])
        if vid in existing:
            out[vid] = existing[vid]
            continue
        if vid in seen_vids:
            continue  # duplicate listing entry: first occurrence wins
        seen_vids.add(vid)
        v_ymd, v_ts = parse_date_prefix(str(v.get("date") or v.get("updated_at") or ""))
        v_clean_title = sanitize_name(str(v.get("name") or "video"), 60)
        cands = albums_by_day.get(v_ymd, [])

        if len(cands) == 1:
            target_folder = album_folder_by_id[int(cands[0]["id"])]
        elif len(cands) >= 2:
            v_owner = _extract_owner_id(v)
            v_text = f"{v.get('name') or ''} {v.get('description') or ''}"
            scorer = functools.partial(_score_album_candidate, v_text=v_text, v_owner=v_owner, v_ts=v_ts)
            best = max(cands, key=scorer)
            target_folder = album_folder_by_id[int(best["id"])]
        else:
            year_s, month_s = split_year_month(v_ymd)
            target_folder = f"{year_s}/{month_s}/{v_ymd} {v_clean_title}"

        videos_by_folder[target_folder].append(
            (v_ts if v_ts is not None else float("inf"), vid, v_clean_title)
        )

    for folder, group in videos_by_folder.items():
        group.sort(key=lambda t: (t[0], t[1]))
        used_names_ci = used_by_folder[folder]
        for _, vid, v_clean_title in group:
            cand_name = f"{v_clean_title}.mp4"
            k = 2
            while cand_name.lower() in used_names_ci:
                cand_name = f"{v_clean_title} ({k}).mp4"
                k += 1
            used_names_ci.add(cand_name.lower())
            out[vid] = (folder, cand_name)
    return out
