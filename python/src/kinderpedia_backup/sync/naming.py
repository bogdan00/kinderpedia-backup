"""Short-name disambiguation and video-to-day-album matching shared with the Bookmarklet."""

from __future__ import annotations

import collections
import contextlib
import functools
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime
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
    clean = re.sub(r"[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]", " ", clean)
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


def parse_date_prefix(date_str: str | None) -> tuple[str, float | None]:
    """Extract `YYYY-MM-DD` date prefix (or `unknown_date` matching the Bookmarklet)."""
    if not date_str:
        return "unknown_date", None
    s = str(date_str).strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %m %Y",
        "%A, %d %b %Y",
        "%A, %d %B %Y %H:%M",
        "%A, %d %B %Y, %H:%M",
        "%d %B %Y , %H:%M",
    ]
    for fmt in fmts:
        with contextlib.suppress(ValueError):
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d"), dt.timestamp()
    with contextlib.suppress(ValueError):
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d"), dt.timestamp()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
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


def assign_album_folder_names(albums: Sequence[dict[str, Any]]) -> dict[int, str]:
    """
    Assign deterministic album relative paths (`<Year>/<Month>/<YYYY>-<MM>-<DD> <Sanitised Title>`),
    appending ` (2)` in chronological `(timestamp_asc, id_asc)` order on same-day duplicate titles.
    """
    by_base: dict[str, list[tuple[float, int, str, str, str]]] = collections.defaultdict(list)
    for a in albums:
        if "id" not in a:
            continue
        aid = int(a["id"])
        ymd, ts = parse_date_prefix(str(a.get("date") or a.get("updated_at") or ""))
        year_s, month_s = split_year_month(ymd)
        title = sanitize_name(str(a.get("name") or f"album_{aid}"), 60)
        leaf_base = f"{ymd} {title}"
        by_base[leaf_base.lower()].append((ts if ts is not None else float("inf"), aid, year_s, month_s, leaf_base))

    folder_by_id: dict[int, str] = {}
    used_ci: set[str] = set()
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
    album_folder_by_id: dict[int, str],
) -> dict[int, tuple[str, str]]:
    """
    Map each video ID to `(target_album_rel_path, short_video_mp4_filename)` inside `Albums/`:
    - 1 album on same `YYYY-MM-DD`: placed into that album folder (`<Year>/<Month>/<YYYY>-<MM>-<DD> <Title>`).
    - 2+ albums on same `YYYY-MM-DD`: matched via `(stem_similarity, same_educator, -time_diff, -album_id)`.
    - 0 albums on same `YYYY-MM-DD`: placed into `Albums/<Year>/<Month>/<YYYY>-<MM>-<DD> <Video Title>/`.
    - Duplicate video titles inside the same target folder receive ` (2)`, ` (3)` chronologically.
    """
    albums_by_day: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for a in albums:
        if "id" not in a or int(a["id"]) not in album_folder_by_id:
            continue
        ymd, _ = parse_date_prefix(str(a.get("date") or a.get("updated_at") or ""))
        albums_by_day[ymd].append(a)

    videos_by_folder: dict[str, list[tuple[float, int, str]]] = collections.defaultdict(list)
    for v in videos:
        if "id" not in v:
            continue
        vid = int(v["id"])
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

    out: dict[int, tuple[str, str]] = {}
    for folder, group in videos_by_folder.items():
        group.sort(key=lambda t: (t[0], t[1]))
        used_names_ci: set[str] = set()
        for _, vid, v_clean_title in group:
            cand_name = f"{v_clean_title}.mp4"
            k = 2
            while cand_name.lower() in used_names_ci:
                cand_name = f"{v_clean_title} ({k}).mp4"
                k += 1
            used_names_ci.add(cand_name.lower())
            out[vid] = (folder, cand_name)
    return out
