// Pure formatting helpers (no DOM).

export function compareByDisplayName(a, b) {
  return String(a.display_name || "").localeCompare(String(b.display_name || ""), undefined, {
    sensitivity: "base",
    numeric: true,
  });
}

export function setsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const v of a) if (!b.has(v)) return false;
  return true;
}

export function fmtSecs(ms) {
  const s = Math.max(0, ms) / 1000;
  return s < 10 ? s.toFixed(1) : String(Math.round(s));
}

/** "1 face" / "3 faces" (`many` defaults to word + "s"). */
export function plural(n, word, many = `${word}s`) {
  return `${n} ${n === 1 ? word : many}`;
}

/** Face crop when the item has a face, else the full photo. */
export function thumbSrc(item) {
  return item.face_id ? `/media/face/${Number(item.face_id)}.jpg` : `/media/photo/${Number(item.photo_id)}`;
}
