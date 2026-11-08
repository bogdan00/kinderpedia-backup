/**
 * Kinderpedia Cloud Backup — Browser Engine & Bookmarklet
 * 
 * Features:
 * - Direct-to-disk streaming via File System Access API (showDirectoryPicker)
 * - Zero RAM bloat (<30MB) across 30+ GiB and 50k+ files
 * - Zero-Prompt Session Auto-Detection: Recognizes logged-in parent & kids instantly
 * - Highly Communicative Real-Time UI: Live page-by-page cataloging, ticker & progress
 * - Adaptive AIMD + Soft-Floor rate controller for Kinderpedia APIs
 * - Instant Resume: Skips already-downloaded photos and cached manifests in <1s
 * - Pure-JS Vimeo HLS to MP4 remuxer via mux.js (no native ffmpeg required)
 * - Isolated Shadow DOM glassmorphic UI overlay with live telemetry
 */

(() => {
  // ---------------------------------------------------------------------------
  // Layout contract (pure functions, no DOM): shared byte-for-byte with the Python CLI
  // (python/src/kinderpedia_backup/sync/naming.py + folder_map.py) and pinned by the golden fixture
  // python/tests/fixtures/layout_contract.json (run: node --test bookmarklet/tests/).
  // Defined before any browser checks so Node can load this file and test them headlessly.
  // ---------------------------------------------------------------------------
  const STOPWORDS = new Set([
    "in", "de", "la", "cu", "si", "pe", "din", "un", "una", "ale", "cel", "cea",
    "mai", "pentru", "dupa", "prin", "sub", "sau", "iar", "dar", "ca", "ce", "se",
    "au", "sa", "ne", "am", "ai", "va", "le", "ii", "the", "and", "for", "with", "from",
  ]);
  const NUMBERED_PHOTO_RE = /^\d{3}_/;
  const FOLDER_MAP_FILE = "folder_map.json";

  function sanitizeName(name, maxLen = 60) {
    if (!name) return "unnamed";
    let clean = String(name)
      // Normalize NEL, non-breaking spaces & wide spaces to standard space
      .replace(/[\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/g, " ")
      // Normalize / remove smart curly quotes, guillemets, and ellipsis
      .replace(/[\u201c\u201d\u201e\u00ab\u00bb\u201f]/g, "")
      .replace(/[\u2018\u2019\u201a\u201b]/g, "'")
      .replace(/\u2026/g, "")
      // Normalize en-dash, em-dash, horizontal bar to standard ASCII hyphen
      .replace(/[\u2012\u2013\u2014\u2015]/g, "-")
      // Strip supplementary-plane emojis/flags & BMP dingbats/variation selectors (matches Python naming.py)
      .replace(/[\u{10000}-\u{10FFFF}]/gu, "")
      .replace(/[\u2300-\u27BF\u2B50\uFE00-\uFE0F\u200B-\u200D\uFEFF]/g, "")
      // Replace Windows forbidden characters: < > : " / \ | ? * and ASCII control characters
      .replace(/[/\\?%*:|"<>]/g, "_")
      .replace(/[\x00-\x1f\x7f]/g, "")
      .replace(/\s+/g, " ")
      .replace(/_+/g, "_")
      .trim();

    // Strip leading/trailing dots and spaces (Windows & hidden-file safe)
    clean = clean.replace(/^[. ]+/, "").replace(/[. ]+$/, "");

    if (clean.length > maxLen) {
      clean = clean.substring(0, maxLen).trim().replace(/[. ]+$/, "");
    }

    // Protect against Windows reserved DOS device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    if (/^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])$/i.test(clean)) {
      clean = `_${clean}`;
    }

    return clean || "unnamed";
  }

  const ISO_DATE_RE = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?\s*(Z|[+-]\d{2}:?\d{2})?)?/;
  const TEXT_DATE_RE = /^(?:[A-Za-z]+\.?,\s*)?(\d{1,2})\s+([A-Za-z]+|\d{1,2})\.?\s+(\d{4})(?:\s*,?\s*(\d{1,2}):(\d{2}))?$/;
  const ANY_DATE_RE = /(\d{4})-(\d{2})-(\d{2})/;
  const MONTHS = new Map();
  [
    ["january", "jan"], ["february", "feb"], ["march", "mar"], ["april", "apr"], ["may"], ["june", "jun"],
    ["july", "jul"], ["august", "aug"], ["september", "sep", "sept"], ["october", "oct"], ["november", "nov"],
    ["december", "dec"],
  ].forEach((names, idx) => names.forEach((n) => MONTHS.set(n, idx + 1)));

  function validYmd(y, mo, d) {
    if (!(mo >= 1 && mo <= 12 && d >= 1)) return false;
    const leap = (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
    const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    return d <= days[mo - 1];
  }

  function localTs(y, mo, d, hh = 0, mm = 0, ss = 0) {
    const dt = new Date(y, mo - 1, d, hh, mm, ss);
    dt.setFullYear(y); // years 0-99 must not map to 19xx
    const ms = dt.getTime();
    return Number.isNaN(ms) ? null : ms / 1000;
  }

  /**
   * Canonical, locale-independent date parsing (mirrors Python `parse_date_prefix`):
   * 1. ISO/SQL `YYYY-MM-DD[( |T)HH:MM[:SS[.fff]][Z|+HH[:]MM]]` anchored at the start; the prefix is the
   *    literal digits; fractional seconds ignored; offset-less times are local; invalid dates -> null ts.
   * 2. `[Weekday,] DD <English month|MM> YYYY[ [,] HH:MM]` (local time).
   * 3. Any embedded `YYYY-MM-DD` (prefix only). Never falls back to engine-specific `Date.parse`.
   */
  function parseDatePrefix(dateStr) {
    if (dateStr === null || dateStr === undefined || dateStr === "") return { prefix: "unknown_date", timestamp: null };
    const s = String(dateStr).trim();
    if (!s) return { prefix: "unknown_date", timestamp: null };

    const iso = s.match(ISO_DATE_RE);
    if (iso) {
      const prefix = `${iso[1]}-${iso[2]}-${iso[3]}`;
      const y = Number(iso[1]);
      const mo = Number(iso[2]);
      const d = Number(iso[3]);
      if (!validYmd(y, mo, d)) return { prefix, timestamp: null };
      if (iso[4] === undefined) return { prefix, timestamp: localTs(y, mo, d) };
      const hh = Number(iso[4]);
      const mi = Number(iso[5]);
      const ss = iso[6] !== undefined ? Number(iso[6]) : 0;
      if (hh > 23 || mi > 59 || ss > 59) return { prefix, timestamp: null };
      const tz = iso[7];
      if (!tz) return { prefix, timestamp: localTs(y, mo, d, hh, mi, ss) };
      let offsetMin = 0;
      if (tz !== "Z") {
        const digits = tz.slice(1).replace(":", "");
        const oh = Number(digits.slice(0, 2));
        const om = Number(digits.slice(2));
        if (oh > 23 || om > 59) return { prefix, timestamp: null };
        offsetMin = (oh * 60 + om) * (tz[0] === "-" ? -1 : 1);
      }
      const utcDate = new Date(Date.UTC(y, mo - 1, d, hh, mi, ss));
      utcDate.setUTCFullYear(y);
      return { prefix, timestamp: utcDate.getTime() / 1000 - offsetMin * 60 };
    }

    const txt = s.match(TEXT_DATE_RE);
    if (txt) {
      const d = Number(txt[1]);
      const y = Number(txt[3]);
      const monRaw = txt[2].toLowerCase();
      const mo = /^\d+$/.test(monRaw) ? Number(monRaw) : MONTHS.get(monRaw) || 0;
      if (validYmd(y, mo, d)) {
        const hh = txt[4] !== undefined ? Number(txt[4]) : 0;
        const mi = txt[5] !== undefined ? Number(txt[5]) : 0;
        if (hh <= 23 && mi <= 59) {
          const prefix = `${String(y).padStart(4, "0")}-${String(mo).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
          return { prefix, timestamp: localTs(y, mo, d, hh, mi) };
        }
      }
    }

    const anyDate = s.match(ANY_DATE_RE);
    if (anyDate) {
      return { prefix: `${anyDate[1]}-${anyDate[2]}-${anyDate[3]}`, timestamp: null };
    }
    return { prefix: "unknown_date", timestamp: null };
  }

  function splitYearMonth(ymd) {
    if (typeof ymd === "string" && /^\d{4}-\d{2}-\d{2}$/.test(ymd)) {
      return [ymd.slice(0, 4), ymd.slice(5, 7)];
    }
    return ["Unknown", "00"];
  }

  function getChildShortName(ctx) {
    const fn = String(ctx.first_name || "").trim();
    if (fn) return fn;
    const parts = String(ctx.child_name || "").trim().split(/\s+/).filter(Boolean);
    return parts.length > 0 ? parts[0] : `child_${ctx.child_id}`;
  }

  function resolveChildFolderNames(contexts) {
    const shortCounts = new Map();
    const fullCounts = new Map();
    for (const c of contexts) {
      const sKey = sanitizeName(getChildShortName(c), 60).toLowerCase();
      const fKey = sanitizeName(c.child_name, 60).toLowerCase();
      shortCounts.set(sKey, (shortCounts.get(sKey) || 0) + 1);
      fullCounts.set(fKey, (fullCounts.get(fKey) || 0) + 1);
    }

    const out = new Map();
    const usedCi = new Set();
    for (const c of contexts) {
      const short = sanitizeName(getChildShortName(c), 60);
      const full = sanitizeName(c.child_name, 60);
      let cand;
      if (shortCounts.get(short.toLowerCase()) === 1) {
        cand = short;
      } else if (fullCounts.get(full.toLowerCase()) === 1) {
        cand = full;
      } else {
        const schoolTag = c.school_name ? sanitizeName(c.school_name, 30) : `KG_${c.kindergarten_id}`;
        cand = sanitizeName(`${short} (${schoolTag})`, 60);
      }

      const base = cand;
      let k = 2;
      while (usedCi.has(cand.toLowerCase())) {
        cand = sanitizeName(`${base} (${k})`, 60);
        k++;
      }
      usedCi.add(cand.toLowerCase());
      out.set(`${c.child_id}_${c.kindergarten_id}`, cand);
    }
    return out;
  }

  const leafOf = (folderRel) => String(folderRel).split("/").pop();

  /**
   * `existing`: persisted Map<albumId, folderRel> from Metadata/<Child>/folder_map.json. Known albums keep
   * their folder verbatim and every persisted leaf is reserved, so only unknown IDs get new names.
   */
  function assignAlbumFolderNames(albums, existing = new Map()) {
    const usedCi = new Set();
    for (const folder of existing.values()) usedCi.add(leafOf(folder).toLowerCase());
    const folderById = new Map();
    const seenIds = new Set();
    const byBase = new Map();
    for (const a of albums) {
      if (a == null || a.id == null) continue;
      const aid = Number(a.id);
      if (existing.has(aid)) {
        folderById.set(aid, existing.get(aid));
        continue;
      }
      if (seenIds.has(aid)) continue; // duplicate listing entry: first occurrence wins
      seenIds.add(aid);
      const { prefix: ymd, timestamp: ts } = parseDatePrefix(a.date || a.updated_at || "");
      const [yearS, monthS] = splitYearMonth(ymd);
      const title = sanitizeName(String(a.name || `album_${aid}`), 60);
      const leafBase = `${ymd} ${title}`;
      const key = leafBase.toLowerCase();
      if (!byBase.has(key)) byBase.set(key, []);
      byBase.get(key).push({
        ts: ts !== null ? ts : Number.POSITIVE_INFINITY,
        aid,
        yearS,
        monthS,
        leafBase,
      });
    }

    for (const group of byBase.values()) {
      group.sort((x, y) => (x.ts !== y.ts ? (x.ts < y.ts ? -1 : 1) : x.aid - y.aid));
      for (let idx = 0; idx < group.length; idx++) {
        const { aid, yearS, monthS, leafBase } = group[idx];
        let candLeaf = idx === 0 ? leafBase : `${leafBase} (${idx + 1})`;
        let k = idx + 2;
        while (usedCi.has(candLeaf.toLowerCase())) {
          candLeaf = `${leafBase} (${k})`;
          k++;
        }
        usedCi.add(candLeaf.toLowerCase());
        folderById.set(aid, `${yearS}/${monthS}/${candLeaf}`);
      }
    }
    return folderById;
  }

  function normalizeMatchText(text) {
    return String(text || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/\p{Mn}/gu, "")
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  }

  function stemSimilarity(videoText, albumText) {
    const vn = normalizeMatchText(videoText);
    const an = normalizeMatchText(albumText);
    if (!vn || !an) return 0;
    let score = 0;
    if (vn.length >= 4 && an.length >= 4 && (vn.includes(an) || an.includes(vn))) {
      score += 2;
    }
    const vTokens = vn.split(" ").filter((w) => w.length >= 3 && !STOPWORDS.has(w));
    const aTokens = an.split(" ").filter((w) => w.length >= 3 && !STOPWORDS.has(w));
    for (const w1 of vTokens) {
      for (const w2 of aTokens) {
        if (w1 === w2 || (w1.length >= 4 && w2.length >= 4 && w1.slice(0, 5) === w2.slice(0, 5))) {
          score += 1;
          break;
        }
      }
    }
    return score;
  }

  function extractOwnerId(item) {
    if (item && item.owner && typeof item.owner === "object" && item.owner.user_id != null) {
      const parsed = parseInt(item.owner.user_id, 10);
      return Number.isNaN(parsed) ? null : parsed;
    }
    return null;
  }

  /**
   * `existing`: persisted Map<videoId, {folderRel, fileName}>; pinned videos keep their target and their
   * file names stay reserved in their folders.
   */
  function planVideoTargets(videos, albums, albumFolderById, existing = new Map()) {
    const albumsByDay = new Map();
    for (const a of albums) {
      if (a == null || a.id == null) continue;
      const aid = Number(a.id);
      if (!albumFolderById.has(aid)) continue;
      const { prefix: ymd } = parseDatePrefix(a.date || a.updated_at || "");
      if (!albumsByDay.has(ymd)) albumsByDay.set(ymd, []);
      albumsByDay.get(ymd).push(a);
    }

    const out = new Map();
    const usedByFolder = new Map();
    for (const { folderRel, fileName } of existing.values()) {
      if (!usedByFolder.has(folderRel)) usedByFolder.set(folderRel, new Set());
      usedByFolder.get(folderRel).add(fileName.toLowerCase());
    }

    const videosByFolder = new Map();
    const seenVids = new Set();
    for (const v of videos) {
      if (v == null || v.id == null) continue;
      const vid = Number(v.id);
      if (existing.has(vid)) {
        out.set(vid, existing.get(vid));
        continue;
      }
      if (seenVids.has(vid)) continue; // duplicate listing entry: first occurrence wins
      seenVids.add(vid);
      const { prefix: vYmd, timestamp: vTs } = parseDatePrefix(v.date || v.updated_at || "");
      const vCleanTitle = sanitizeName(String(v.name || "video"), 60);
      const cands = albumsByDay.get(vYmd) || [];

      let targetFolder;
      if (cands.length === 1) {
        targetFolder = albumFolderById.get(Number(cands[0].id));
      } else if (cands.length >= 2) {
        const vOwner = extractOwnerId(v);
        const vText = `${v.name || ""} ${v.description || ""}`;
        let best = cands[0];
        let bestTuple = null;
        for (const cand of cands) {
          const { timestamp: aTs } = parseDatePrefix(cand.date || cand.updated_at || "");
          const aOwner = extractOwnerId(cand);
          const aText = `${cand.name || ""} ${cand.description || ""}`;
          const st = stemSimilarity(vText, aText);
          const sameO = vOwner !== null && aOwner !== null && vOwner === aOwner ? 1 : 0;
          const dtDiff = vTs !== null && aTs !== null ? Math.abs(vTs - aTs) : 1e12;
          const tuple = [st, sameO, -dtDiff, -Number(cand.id)];
          if (
            bestTuple === null ||
            tuple[0] > bestTuple[0] ||
            (tuple[0] === bestTuple[0] && tuple[1] > bestTuple[1]) ||
            (tuple[0] === bestTuple[0] && tuple[1] === bestTuple[1] && tuple[2] > bestTuple[2]) ||
            (tuple[0] === bestTuple[0] && tuple[1] === bestTuple[1] && tuple[2] === bestTuple[2] && tuple[3] > bestTuple[3])
          ) {
            bestTuple = tuple;
            best = cand;
          }
        }
        targetFolder = albumFolderById.get(Number(best.id));
      } else {
        const [yearS, monthS] = splitYearMonth(vYmd);
        targetFolder = `${yearS}/${monthS}/${vYmd} ${vCleanTitle}`;
      }

      if (!videosByFolder.has(targetFolder)) videosByFolder.set(targetFolder, []);
      videosByFolder.get(targetFolder).push({
        ts: vTs !== null ? vTs : Number.POSITIVE_INFINITY,
        vid,
        vCleanTitle,
      });
    }

    for (const [folder, group] of videosByFolder.entries()) {
      group.sort((x, y) => (x.ts !== y.ts ? (x.ts < y.ts ? -1 : 1) : x.vid - y.vid));
      if (!usedByFolder.has(folder)) usedByFolder.set(folder, new Set());
      const usedNamesCi = usedByFolder.get(folder);
      for (const { vid, vCleanTitle } of group) {
        let candName = `${vCleanTitle}.mp4`;
        let k = 2;
        while (usedNamesCi.has(candName.toLowerCase())) {
          candName = `${vCleanTitle} (${k}).mp4`;
          k++;
        }
        usedNamesCi.add(candName.toLowerCase());
        out.set(vid, { folderRel: folder, fileName: candName });
      }
    }
    return out;
  }

  /** `(updated_at, count_images)` change-detection key of a gallery album listing entry (Python parity). */
  function albumListingSignature(album) {
    if (!album || typeof album !== "object") return "\u0000-1";
    const count = parseInt(album.count_images, 10) || 0;
    return `${album.updated_at || ""}\u0000${count}`;
  }

  function mergeIncrementalListing(fresh, cached) {
    const seen = new Set();
    const out = [];
    for (const item of [...fresh, ...cached]) {
      if (!item || item.id == null) continue;
      const k = String(item.id);
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(item);
    }
    return out;
  }

  function albumFolderFromLeaf(leaf) {
    const m = String(leaf).match(/^(\d{4}-\d{2}-\d{2}) /);
    const [yearS, monthS] = splitYearMonth(m ? m[1] : "");
    return `${yearS}/${monthS}/${leaf}`;
  }

  function isSafeRelFolder(rel) {
    if (typeof rel !== "string" || !rel || rel.includes("\\") || rel.includes("\u0000")) return false;
    if (rel.startsWith("/") || /^[A-Za-z]:/.test(rel)) return false;
    return rel.split("/").every((part) => part !== "" && part !== "." && part !== "..");
  }

  /** Validate a decoded folder_map.json -> { albums: Map<id, folderRel>, videos: Map<id, {folderRel, fileName}> }. */
  function parseFolderMap(data) {
    if (!data || typeof data !== "object" || !data.albums || typeof data.albums !== "object") return null;
    const albums = new Map();
    for (const [k, v] of Object.entries(data.albums)) {
      const id = Number(k);
      if (Number.isInteger(id) && isSafeRelFolder(v)) albums.set(id, v);
    }
    const videos = new Map();
    if (data.videos && typeof data.videos === "object") {
      for (const [k, v] of Object.entries(data.videos)) {
        const id = Number(k);
        if (!Number.isInteger(id) || !v || typeof v !== "object") continue;
        if (isSafeRelFolder(v.folder) && isSafeRelFolder(v.file) && !v.file.includes("/")) {
          videos.set(id, { folderRel: v.folder, fileName: v.file });
        }
      }
    }
    return { albums, videos };
  }

  function serializeFolderMap(albums, videos) {
    const byId = (a, b) => a[0] - b[0];
    const out = { version: 1, albums: {}, videos: {} };
    for (const [id, folder] of [...albums.entries()].sort(byId)) out.albums[String(id)] = folder;
    for (const [id, t] of [...videos.entries()].sort(byId)) out.videos[String(id)] = { folder: t.folderRel, file: t.fileName };
    return out;
  }

  const layoutApi = {
    sanitizeName,
    parseDatePrefix,
    splitYearMonth,
    getChildShortName,
    resolveChildFolderNames,
    assignAlbumFolderNames,
    normalizeMatchText,
    stemSimilarity,
    planVideoTargets,
    albumListingSignature,
    mergeIncrementalListing,
    albumFolderFromLeaf,
    parseFolderMap,
    serializeFolderMap,
  };
  // Headless (Node `node --test`) consumers get the pure layout API; the browser UI below never runs there.
  if (typeof module === "object" && module && module.exports && typeof window === "undefined") {
    module.exports = layoutApi;
    return;
  }

  if (window.__kinderpedia_backup_instance) {
    if (typeof window.__kinderpedia_backup_instance.toggleUI === "function") {
      window.__kinderpedia_backup_instance.toggleUI();
    } else if (window.__kinderpedia_backup_instance.ui && typeof window.__kinderpedia_backup_instance.ui.toggleUI === "function") {
      window.__kinderpedia_backup_instance.ui.toggleUI();
    }
    return;
  }

  // --- Graceful Environment & Compatibility Dialog ---
  function showNoticeModal({ icon, title, messageHtml, primaryBtn, secondaryBtn }) {
    if (typeof document === "undefined") return;
    const existing = document.getElementById("kp-notice-overlay");
    if (existing) existing.remove();

    const host = document.createElement("div");
    host.id = "kp-notice-overlay";
    document.body.appendChild(host);
    const shadow = host.attachShadow({ mode: "open" });

    shadow.innerHTML = `
      <style>
        * { box-sizing: border-box; }
        .backdrop {
          position: fixed;
          inset: 0;
          z-index: 2147483647;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(10, 15, 29, 0.78);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          padding: 16px;
          animation: kpFadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes kpFadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .modal {
          background: #0f172a;
          border: 1px solid rgba(255, 255, 255, 0.14);
          border-radius: 16px;
          max-width: 470px;
          width: 100%;
          padding: 26px;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          color: #f1f5f9;
          animation: kpScaleIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes kpScaleIn {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
        .icon-badge {
          width: 52px;
          height: 52px;
          border-radius: 14px;
          background: rgba(56, 189, 248, 0.12);
          border: 1px solid rgba(56, 189, 248, 0.25);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 26px;
          margin-bottom: 16px;
        }
        .title {
          font-size: 18px;
          font-weight: 700;
          color: #ffffff;
          margin: 0 0 10px 0;
        }
        .body {
          font-size: 14px;
          line-height: 1.55;
          color: #94a3b8;
          margin: 0 0 22px 0;
        }
        .body b { color: #f1f5f9; }
        .body code {
          background: rgba(255, 255, 255, 0.1);
          padding: 2px 6px;
          border-radius: 4px;
          color: #38bdf8;
          font-size: 13px;
        }
        .body a { color: #38bdf8; text-decoration: underline; }
        .body ul, .body ol {
          margin: 10px 0 0 18px;
          padding: 0;
        }
        .body li {
          margin-bottom: 6px;
        }
        .actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
        }
        .btn {
          border: none;
          padding: 10px 18px;
          border-radius: 8px;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          font-family: inherit;
          transition: all 0.15s ease;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          text-decoration: none;
        }
        .btn-primary {
          background: #0284c7;
          color: #ffffff;
        }
        .btn-primary:hover {
          background: #0369a1;
        }
        .btn-secondary {
          background: rgba(255, 255, 255, 0.08);
          color: #cbd5e1;
          border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .btn-secondary:hover {
          background: rgba(255, 255, 255, 0.14);
          color: #ffffff;
        }
      </style>
      <div class="backdrop" id="backdrop">
        <div class="modal">
          <div class="icon-badge">${icon}</div>
          <div class="title">${title}</div>
          <div class="body">${messageHtml}</div>
          <div class="actions">
            ${secondaryBtn ? `<button class="btn btn-secondary" id="sec-btn">${secondaryBtn.text}</button>` : ""}
            <button class="btn btn-primary" id="pri-btn">${primaryBtn.text}</button>
          </div>
        </div>
      </div>
    `;

    const close = () => {
      host.remove();
      window.removeEventListener("keydown", onKeyDown);
    };

    const onKeyDown = (e) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKeyDown);

    shadow.getElementById("backdrop").addEventListener("click", (e) => {
      if (e.target.id === "backdrop") close();
    });

    const priEl = shadow.getElementById("pri-btn");
    priEl.addEventListener("click", () => {
      if (primaryBtn.action) {
        try { primaryBtn.action(); } catch (_) {}
      }
      close();
    });

    if (secondaryBtn) {
      const secEl = shadow.getElementById("sec-btn");
      secEl.addEventListener("click", () => {
        if (secondaryBtn.action) {
          try { secondaryBtn.action(); } catch (_) {}
        }
        close();
      });
    }
  }

  // --- 1. Validate Target Website / Domain ---
  const hostName = (typeof window !== "undefined" && window.location && window.location.hostname) || "";
  const safeHost = String(hostName).replace(/[^\w.-]/g, "") || "an external website";
  const isAllowedHost =
    /(^|\.)kinderpedia\.(co|ro|com)$/i.test(hostName) ||
    hostName === "localhost" ||
    hostName === "127.0.0.1" ||
    (typeof document !== "undefined" && !!document.getElementById("launch-test-btn")) ||
    (typeof window !== "undefined" && window.location && window.location.search && window.location.search.includes("kp_test=1"));

  if (!isAllowedHost) {
    showNoticeModal({
      icon: "🌐",
      title: "Open Kinderpedia to Use Backup",
      messageHtml: `This bookmarklet backs up media directly from your logged-in Kinderpedia account.<br><br>You are currently on <code>${safeHost}</code>. Please navigate to your Kinderpedia portal, log in, and click this bookmarklet again.`,
      primaryBtn: {
        text: "Go to app.kinderpedia.co ↗",
        action: () => {
          window.open("https://app.kinderpedia.co", "_blank");
        },
      },
      secondaryBtn: {
        text: "Close",
      },
    });
    return;
  }

  // --- 2. Validate Device & File System Capabilities ---
  const isMobileDevice = (() => {
    if (typeof navigator === "undefined") return false;
    const ua = navigator.userAgent || "";
    const isMobileUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|mobile/i.test(ua);
    const isIPadOS = navigator.maxTouchPoints > 1 && /Macintosh/.test(ua);
    return isMobileUA || isIPadOS;
  })();

  const isFileSystemSupported = typeof window !== "undefined" && typeof window.showDirectoryPicker === "function";

  if (isMobileDevice) {
    showNoticeModal({
      icon: "📱",
      title: "Desktop Computer Required",
      messageHtml: `Mobile browsers (iPhone, iPad, and Android) do not support the <b>File System Access API</b>, which is required to stream and organize gigabytes of high-resolution photos and videos directly into folders.<br><br><b>To download your family archive:</b><ol><li>Open a computer (Mac, Windows, Linux, or Chromebook).</li><li>Use <b>Google Chrome</b>, <b>Microsoft Edge</b>, or <b>Brave</b>.</li><li>Log into <b><a href="https://app.kinderpedia.co" target="_blank">app.kinderpedia.co</a></b> and click the bookmarklet.</li></ol>`,
      primaryBtn: {
        text: "Got It",
      },
    });
    return;
  }

  if (!isFileSystemSupported) {
    showNoticeModal({
      icon: "💻",
      title: "Chromium Browser Required",
      messageHtml: `Your current browser (such as Safari or Firefox) does not support direct folder saving (File System Access API).<br><br>Please open your Kinderpedia account in a Chromium-based browser to back up your photos, videos, and documents:<ul><li><b>Google Chrome</b> (Recommended)</li><li><b>Microsoft Edge</b></li><li><b>Brave</b> or <b>Opera</b></li></ul>`,
      primaryBtn: {
        text: "Got It",
      },
    });
    return;
  }

  const API_KEY = "Web01Pari3l4em|v1.02";
  const BASE_API = "https://app.kinderpedia.co/web-api";
  const USERGATEWAY_API = "https://usergateway-services.kinderpedia.co";
  const MUX_JS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/mux.js/6.3.0/mux.min.js";
  const MUX_JS_FALLBACK_CDN = "https://cdn.jsdelivr.net/npm/mux.js@6.3.0/dist/mux.min.js";
  const DEFAULT_WORKER_URL = "https://kinderpedia-backup.bogdan00.workers.dev";
  const DEFAULT_WORKER_SECRET = "kp_sec_98721";

  // --- Utility Functions ---
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function getNestedDirectoryHandle(baseDirHandle, relPath, dirCache) {
    const parts = String(relPath).split("/").filter(Boolean);
    let curPromise = Promise.resolve(baseDirHandle);
    let prefix = "";
    for (const part of parts) {
      prefix = prefix ? `${prefix}/${part}` : part;
      if (dirCache && dirCache.has(prefix)) {
        curPromise = dirCache.get(prefix);
      } else {
        const prevPromise = curPromise;
        curPromise = prevPromise.then((parent) => parent.getDirectoryHandle(part, { create: true }));
        if (dirCache) dirCache.set(prefix, curPromise);
      }
    }
    return curPromise;
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KiB", "MiB", "GiB", "TiB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function formatDuration(sec) {
    sec = Math.max(0, Math.floor(sec));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  // --- Adaptive Rate Controller (AIMD + Soft Floor) ---
  class AdaptiveRateController {
    constructor() {
      this.interval = 2.65;
      this.softFloor = 2.60;
      this.minInterval = 1.20;
      this.maxInterval = 15.0;
      this.cleanStreak = 0;
      this.totalCalls = 0;
      this.total429s = 0;
      this.nextSlot = performance.now();
      this.queue = Promise.resolve();
    }

    async acquire(onWait) {
      let resolveWait;
      const waitPromise = new Promise((r) => (resolveWait = r));

      this.queue = this.queue.then(async () => {
        const now = performance.now();
        const wait = Math.max(0, this.nextSlot - now);
        this.nextSlot = Math.max(now, this.nextSlot) + this.interval * 1000;
        if (wait > 0) {
          if (onWait && wait > 1000) onWait(wait / 1000);
          await sleep(wait);
        }
        resolveWait();
      });

      return waitPromise;
    }

    onSuccess() {
      this.totalCalls++;
      this.cleanStreak++;
      if (this.cleanStreak >= 4) {
        const step = this.interval > this.softFloor + 0.2 ? 0.04 : 0.01;
        this.interval = Math.max(this.minInterval, this.interval - step);
      }
      if (this.cleanStreak % 25 === 0 && this.softFloor > this.minInterval) {
        this.softFloor = Math.max(this.minInterval, this.softFloor - 0.02);
      }
    }

    onRateLimit(attempt) {
      this.totalCalls++;
      this.total429s++;
      this.cleanStreak = 0;
      this.softFloor = Math.min(this.maxInterval, Math.max(this.softFloor, this.interval * 1.08));
      this.interval = Math.min(this.maxInterval, this.interval * 1.35);

      const jitter = Math.random() * 3.0;
      const pauseS = Math.min(60.0, 18.0 + 8.0 * attempt + jitter);
      this.nextSlot = performance.now() + pauseS * 1000;
      return pauseS;
    }
  }

  // --- Kinderpedia Web API Client ---
  class KinderpediaClient {
    constructor(rateController, logger) {
      this.rate = rateController;
      this.log = logger;
      this.token = null;
      this.refreshToken = null;
      this.userData = null;
      this.childrenContexts = [];
    }

    /**
     * Broad search for JWT tokens in localStorage, sessionStorage, cookies, Vuex store, window globals
     */
    initAuthFromBrowser() {
      const jwtRegex = /eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/;

      // 1. Scan localStorage
      try {
        for (let i = 0; i < localStorage.length; i++) {
          const k = localStorage.key(i);
          const v = localStorage.getItem(k);
          if (!v) continue;
          const m = v.match(jwtRegex);
          if (m) {
            this.token = m[0];
            break;
          }
        }
      } catch (_) {}

      // 2. Scan sessionStorage
      if (!this.token) {
        try {
          for (let i = 0; i < sessionStorage.length; i++) {
            const k = sessionStorage.key(i);
            const v = sessionStorage.getItem(k);
            if (!v) continue;
            const m = v.match(jwtRegex);
            if (m) {
              this.token = m[0];
              break;
            }
          }
        } catch (_) {}
      }

      // 3. Scan document.cookie
      if (!this.token) {
        try {
          if (document.cookie) {
            const m = document.cookie.match(jwtRegex);
            if (m) this.token = m[0];
          }
        } catch (_) {}
      }

      // 4. Scan Vue root instance state
      if (!this.token) {
        try {
          const appEl = document.getElementById("app") || document.querySelector("[data-v-app]");
          const vue = appEl?.__vue__ || appEl?.__vue_app__;
          const store = vue?.$store || vue?.config?.globalProperties?.$store;
          if (store && store.state) {
            const stateStr = JSON.stringify(store.state);
            const m = stateStr.match(jwtRegex);
            if (m) this.token = m[0];
          }
        } catch (_) {}
      }

      // 5. Scan window global variables
      if (!this.token) {
        try {
          for (const k of ["token", "jwt", "access_token", "authToken", "auth_token"]) {
            if (typeof window[k] === "string" && window[k].match(jwtRegex)) {
              this.token = window[k];
              break;
            }
          }
        } catch (_) {}
      }

      return !!this.token;
    }

    /**
     * Parse /web-api/data/parent-app-core response to extract user and children profiles
     */
    parseAppCore(res) {
      const user = res.user_obj || res.user || {};
      this.userData = {
        id: user.user_id || user.id,
        name: [user.first_name, user.last_name].filter(Boolean).join(" ") || user.name || "Parent",
        email: user.email || "",
      };

      const accounts = res.available_accounts || [];
      const children = res.children || [];
      const contexts = [];
      const seen = new Set();

      // Pair children with kindergarten accounts
      if (Array.isArray(accounts) && accounts.length > 0) {
        for (const acc of accounts) {
          const cid = acc.child_id;
          const kid = acc.kindergarten_id;
          if (!cid || !kid) continue;
          const key = `${cid}_${kid}`;
          if (seen.has(key)) continue;
          seen.add(key);

          const ch = children.find((c) => (c.id || c.child_id) === cid) || {};
          const name = [ch.first_name, ch.last_name].filter(Boolean).join(" ") || acc.name || `Child_${cid}`;
          contexts.push({
            child_id: cid,
            child_name: name,
            first_name: String(ch.first_name || "").trim(),
            kindergarten_id: kid,
            school_name: acc.kindergarten_name || acc.name || "Kindergarten",
            avatar_url: ch.avatar || acc.avatar || null,
          });
        }
      }

      // Fallback if available_accounts is empty
      if (!contexts.length && Array.isArray(children) && children.length > 0) {
        const defaultKid = res.current_kindergarten?.id || 0;
        const defaultSchool = res.current_kindergarten?.name || "Kindergarten";
        for (const ch of children) {
          const cid = ch.id || ch.child_id;
          if (!cid) continue;
          contexts.push({
            child_id: cid,
            child_name: [ch.first_name, ch.last_name].filter(Boolean).join(" ") || `Child_${cid}`,
            first_name: String(ch.first_name || "").trim(),
            kindergarten_id: defaultKid,
            school_name: defaultSchool,
            avatar_url: ch.avatar || null,
          });
        }
      }

      this.childrenContexts = contexts;
      return contexts;
    }

    /**
     * Check if the user is already authenticated in the browser session
     */
    async checkActiveSession() {
      // 1. Look for existing JWT token
      this.initAuthFromBrowser();

      // 2. Probe /web-api/data/parent-app-core (uses browser cookies + optional token)
      try {
        const core = await this.httpJson("/data/parent-app-core", {}, 2);
        const res = core?.result || core?.response || core;
        if (res && (res.user_obj || res.children || res.available_accounts)) {
          this.parseAppCore(res);
          return true;
        }
      } catch (err) {
        // If parent-app-core failed, try usergateway /api/users/me if token exists
        if (this.token) {
          try {
            const meRes = await fetch(`${USERGATEWAY_API}/api/users/me`, {
              headers: {
                Authorization: `Bearer ${this.token}`,
                "user-type": "parent",
                Accept: "application/ld+json",
              },
              credentials: "include",
            });
            if (meRes.ok) {
              const me = await meRes.json();
              await this.parseGatewayMe(me);
              return true;
            }
          } catch (_) {}
        }
      }
      return false;
    }

    async parseGatewayMe(me) {
      const user = me.user || me;
      this.userData = {
        id: user.id,
        name: [user.firstName, user.lastName, user.first_name, user.last_name].filter(Boolean).join(" ") || "Parent",
        email: user.email || "",
      };

      const contexts = [];
      const seen = new Set();

      // Lookup family members from gateway
      try {
        const famRes = await fetch(`${USERGATEWAY_API}/api/families/my-members`, {
          headers: {
            Authorization: `Bearer ${this.token}`,
            "user-type": "parent",
            Accept: "application/ld+json",
          },
          credentials: "include",
        });
        if (famRes.ok) {
          const fam = await famRes.json();
          const members = fam["hydra:member"] || fam.members || fam.children || [];
          for (const m of members) {
            const prof = m.profile || m;
            const cid = prof.id || m.entityId || m.userId;
            if (!cid) continue;
            const mg = prof.mainGroup || {};
            const kg = mg.kindergarten || {};
            const kid = mg.kindergartenId || kg.id || (me.kindergartenIds && me.kindergartenIds[0]) || 0;
            const key = `${cid}_${kid}`;
            if (seen.has(key)) continue;
            seen.add(key);

            contexts.push({
              child_id: cid,
              child_name: [prof.firstName, prof.lastName].filter(Boolean).join(" ") || `Child_${cid}`,
              first_name: String(prof.firstName || prof.first_name || "").trim(),
              kindergarten_id: kid,
              school_name: kg.name || "Kindergarten",
              avatar_url: prof.photo || null,
            });
          }
        }
      } catch (_) {}

      this.childrenContexts = contexts;
      return contexts;
    }

    /**
     * Fallback login through usergateway-services (parents)
     */
    async login(email, password) {
      this.log("Connecting to Kinderpedia...");
      const res = await fetch(`${USERGATEWAY_API}/api/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "user-type": "parent",
          Accept: "application/ld+json",
        },
        body: JSON.stringify({
          email,
          password,
          uuid: (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : "web-" + Date.now(),
        }),
      });
      if (!res.ok) {
        throw new Error(`Login failed with HTTP ${res.status}: ${await res.text()}`);
      }
      const data = await res.json();
      this.token = data.token || data.jwToken;
      this.refreshToken = data.refresh_token || data.refreshToken || null;
      return await this.checkActiveSession();
    }

    async httpJson(endpoint, options = {}, retries = 6, onThrottleWait = null) {
      const url = endpoint.startsWith("http") ? endpoint : `${BASE_API}${endpoint}`;
      for (let attempt = 0; attempt < retries; attempt++) {
        await this.rate.acquire((waitS) => {
          if (onThrottleWait) onThrottleWait(waitS);
        });
        try {
          const headers = {
            Accept: "application/json, application/ld+json",
            "X-Requested-With": "XMLHttpRequest",
            "x-api-key": API_KEY,
            ...(options.headers || {}),
          };
          if (this.token && !headers["Authorization"]) {
            headers["Authorization"] = `Bearer ${this.token}`;
          }

          const res = await fetch(url, {
            ...options,
            headers,
            credentials: "include",
          });

          if (res.status === 429 && attempt + 1 < retries) {
            const pause = this.rate.onRateLimit(attempt);
            this.log(`⏳ Pausing briefly (${pause.toFixed(0)}s) to stay within server limits...`);
            await sleep(pause * 1000);
            continue;
          }

          if (res.status >= 500 && attempt + 1 < retries) {
            await sleep(2000 * Math.pow(1.5, attempt));
            continue;
          }

          if (!res.ok) throw new Error(`HTTP ${res.status} on ${url}: ${await res.text()}`);

          this.rate.onSuccess();
          return await res.json();
        } catch (err) {
          if (attempt + 1 >= retries) throw err;
          await sleep(1500 * (attempt + 1));
        }
      }
    }

    async listAlbums(kgId, childId, onPage) {
      const albums = [];
      const seen = new Set();
      let page = 1;
      let runningPhotos = 0;
      while (true) {
        const query = new URLSearchParams({
          mode: "albums",
          page: String(page),
          items_per_page: "30",
          order_by: "updated_at",
          order_type: "DESC",
        });
        const resp = await this.httpJson(`/data/gallery?${query}`, {
          headers: {
            "X-KINDERGARTEN-ID": String(kgId),
            "X-CHILD-ID": String(childId),
          },
        });
        const payload = resp.response || resp.result || resp;
        let items = [];
        if (payload && Array.isArray(payload.albums)) {
          items = payload.albums;
        } else if (payload && Array.isArray(payload.items)) {
          items = payload.items;
        } else if (Array.isArray(payload)) {
          items = payload;
        }

        if (!items.length) break;

        let added = 0;
        for (const item of items) {
          const aid = item.id;
          if (aid && !seen.has(aid)) {
            seen.add(aid);
            albums.push(item);
            runningPhotos += (parseInt(item.count_images, 10) || 0);
            added++;
          }
        }

        if (onPage) {
          onPage(page, albums.length, runningPhotos, items[0]?.name || "");
        }
        this.log(`📚 Page ${page}: Discovered ${albums.length} albums so far (~${runningPhotos.toLocaleString()} photos)...`);

        if (added === 0 || items.length < 30) break;
        page++;
      }
      return albums;
    }

    async getAlbumImages(kgId, childId, albumId, expectedCount = 0) {
      let albumDetails = null;
      let images = [];

      // 1. Primary: images-plugin/{albumId}?entity_type=media (fetches ALL images in 1 fast call!)
      try {
        const raw = await this.httpJson(`/data/images-plugin/${albumId}?entity_type=media`, {
          headers: {
            "X-KINDERGARTEN-ID": String(kgId),
            "X-CHILD-ID": String(childId),
          },
        });
        const payload = raw.response || raw.result || raw;
        if (payload && Array.isArray(payload.images)) {
          images = payload.images;
        }
        if (payload && payload.album) {
          albumDetails = payload.album;
        }
      } catch (err) {
        this.log(`⚠️ images-plugin/${albumId} note: ${err.message}`);
      }

      // 2. Fallback: gallery/{albumId} if images-plugin returned empty and images expected
      if (!images.length && expectedCount > 0) {
        try {
          const raw = await this.httpJson(`/data/gallery/${albumId}`, {
            headers: {
              "X-KINDERGARTEN-ID": String(kgId),
              "X-CHILD-ID": String(childId),
            },
          });
          const payload = raw.response || raw.result || raw;
          if (payload && Array.isArray(payload.images)) {
            images = payload.images;
          }
          if (payload && payload.album) {
            albumDetails = payload.album;
          }
        } catch (_) {}
      }

      return { albumDetails, images };
    }

    async listVideos(kgId, childId, onPage) {
      const videos = [];
      const seen = new Set();
      let page = 1;
      while (true) {
        const query = new URLSearchParams({
          mode: "videos",
          page: String(page),
          items_per_page: "30",
          order_by: "dateadd",
          order_type: "DESC",
        });
        try {
          const resp = await this.httpJson(`/data/gallery?${query}`, {
            headers: {
              "X-KINDERGARTEN-ID": String(kgId),
              "X-CHILD-ID": String(childId),
            },
          });
          const payload = resp.response || resp.result || resp;
          let items = [];
          if (payload && Array.isArray(payload.videos)) {
            items = payload.videos;
          } else if (payload && Array.isArray(payload.items)) {
            items = payload.items;
          } else if (Array.isArray(payload)) {
            items = payload;
          }
          if (!items.length) break;

          let added = 0;
          for (const item of items) {
            const vid = item.id;
            if (vid && !seen.has(vid)) {
              seen.add(vid);
              videos.push(item);
              added++;
            }
          }

          if (onPage) onPage(page, videos.length);
          this.log(`🎬 Page ${page}: Discovered ${videos.length} videos...`);

          if (added === 0 || items.length < 30) break;
          page++;
        } catch (_) {
          break;
        }
      }
      return videos;
    }

    async getVideoDetails(kgId, childId, videoId) {
      const resp = await this.httpJson(`/data/gallery/${videoId}?mode=video`, {
        headers: {
          "X-KINDERGARTEN-ID": String(kgId),
          "X-CHILD-ID": String(childId),
        },
      });
      const payload = resp.response || resp.result || resp;
      if (payload && typeof payload === "object") {
        if (payload.video && typeof payload.video === "object") return payload.video;
        return payload;
      }
      return {};
    }

    async listDocuments(kgId, childId, onProgress) {
      const docs = [];
      const seen = new Set();
      const entities = [
        ["documents", 0],
        ["child-portfolio-folders", childId],
        ["children_public", childId],
      ];

      for (const [entityType, entityId] of entities) {
        let page = 1;
        while (true) {
          const query = new URLSearchParams({
            entity_type: entityType,
            entity_id: String(entityId),
            items_per_page: "30",
            page: String(page),
          });
          try {
            const resp = await this.httpJson(`/data/documents?${query}`, {
              headers: {
                "X-KINDERGARTEN-ID": String(kgId),
                "X-CHILD-ID": String(childId),
              },
            });
            const payload = resp.response || resp.result || resp;
            const items = payload.items || (Array.isArray(payload) ? payload : []);
            if (!items.length) break;
            let added = 0;
            for (const item of items) {
              const did = item.id;
              if (did && !seen.has(did)) {
                seen.add(did);
                docs.push(item);
                added++;
              }
            }

            if (onProgress) onProgress(docs.length, entityType);

            if (added === 0 || items.length < 30) break;
            page++;
          } catch (_) {
            break;
          }
        }
      }
      return docs;
    }
  }

  // --- Vimeo Video Downloader & Remuxer ---
  class VimeoDownloader {
    constructor(logger, serverBase = "http://localhost:8080") {
      this.log = logger;
      this.serverBase = serverBase;
      this.muxLoaded = false;
    }

    async ensureMuxLoaded() {
      const getMux = () => (typeof window !== "undefined" ? (window.muxjs || window.mux) : null);
      if (getMux()) {
        if (typeof window !== "undefined") {
          window.mux = getMux();
          window.muxjs = getMux();
        }
        this.muxLoaded = true;
        return;
      }

      // Candidate script URLs: local vendored copy first, then CDNs
      const candidates = [];
      try {
        if (typeof document !== "undefined") {
          const scripts = Array.from(document.querySelectorAll("script"));
          for (const s of scripts) {
            if (s.src && s.src.includes("kinderpedia-backup")) {
              candidates.push(new URL("mux.min.js", s.src).href);
              break;
            }
          }
        }
      } catch (_) {}
      candidates.push(MUX_JS_CDN);
      candidates.push(MUX_JS_FALLBACK_CDN);

      let lastError = null;
      for (const url of candidates) {
        try {
          await new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.src = url;
            s.crossOrigin = "anonymous";
            s.onload = () => {
              const m = window.muxjs || window.mux;
              if (m) {
                window.mux = m;
                window.muxjs = m;
                this.muxLoaded = true;
                resolve();
              } else {
                reject(new Error("mux.js loaded but neither window.muxjs nor window.mux is defined"));
              }
            };
            s.onerror = () => reject(new Error(`Failed to load ${url}`));
            document.head.appendChild(s);
          });
          if (getMux()) break;
        } catch (err) {
          lastError = err;
        }
      }

      if (!getMux()) {
        throw new Error(`Could not load video transmuxer engine: ${lastError?.message || "All sources failed"}`);
      }
    }

    async downloadVideo(vimeoUrl, destFileHandle, workerConfig = null, onProgress = null, useLocalHelper = false) {
      const mId = vimeoUrl.match(/video\/(\d+)/) || vimeoUrl.match(/vimeo\.com\/(\d+)/);
      if (!mId) throw new Error(`Cannot parse Vimeo ID from ${vimeoUrl}`);
      const vid = mId[1];
      const mHash = vimeoUrl.match(/[?&]h=([a-zA-Z0-9]+)/) || vimeoUrl.match(/vimeo\.com\/\d+\/([a-zA-Z0-9]+)/);
      const hash = mHash ? mHash[1] : "";

      // 1. Primary path: Stream via local helper server (ONLY if helper was detected and available)
      if (useLocalHelper && this.serverBase) {
        try {
          const streamUrl = `${this.serverBase}/vimeo-stream?id=${vid}&h=${hash}`;
          const res = await fetch(streamUrl);
          if (res.ok) {
            let writable = null;
            try {
              writable = await destFileHandle.createWritable({ keepExistingData: false });
              await res.body.pipeTo(writable);
              writable = null;
            } catch (pErr) {
              if (writable) {
                try { await writable.abort(); } catch (_) {}
              }
              throw pErr;
            }
            const file = await destFileHandle.getFile();
            if (!file || file.size < 10000) {
              throw new Error(`Video stream incomplete (${file?.size || 0} bytes)`);
            }
            return file.size;
          }
        } catch (err) {
          // Local helper server unreachable; fall through to Cloudflare Worker or direct
        }
      }

      // 2. Cloudflare Worker path: Fetch config via worker
      let cfg = null;
      if (workerConfig && workerConfig.url) {
        try {
          const wUrl = `${workerConfig.url}?id=${vid}&h=${hash}&secret=${encodeURIComponent(workerConfig.secret || "kp_sec_98721")}`;
          const wRes = await fetch(wUrl);
          if (wRes.ok) {
            const wData = await wRes.json();
            cfg = wData.raw_files ? { request: { files: wData.raw_files } } : wData;
          }
        } catch (_) {}
      }

      // 3. Fallback path: In-browser direct fetching (if direct access is available)
      if (!cfg) {
        const cfgUrl = `https://player.vimeo.com/video/${vid}/config${hash ? `?h=${hash}` : ""}`;
        try {
          const cfgRes = await fetch(cfgUrl);
          if (cfgRes.ok) cfg = await cfgRes.json();
        } catch (_) {}
      }

      if (!cfg) throw new Error("Could not retrieve Vimeo video configuration");

      // Check for progressive MP4 files first (cleanest, fastest)
      const progressive = cfg?.request?.files?.progressive;
      if (Array.isArray(progressive) && progressive.length > 0) {
        progressive.sort((a, b) => (b.width || 0) - (a.width || 0));
        const bestMp4 = progressive[0].url;
        const res = await fetch(bestMp4);
        if (!res.ok) throw new Error(`Failed to fetch progressive MP4: HTTP ${res.status}`);
        let writable = null;
        try {
          writable = await destFileHandle.createWritable({ keepExistingData: false });
          await res.body.pipeTo(writable);
          writable = null;
        } catch (pipeErr) {
          if (writable) {
            try { await writable.abort(); } catch (_) {}
          }
          throw pipeErr;
        }
        return;
      }

      // Fallback: HLS master playlist remuxing via mux.js
      await this.ensureMuxLoaded();

      const hls = cfg?.request?.files?.hls;
      const defaultCdn = hls?.default_cdn;
      const masterUrl = hls?.cdns?.[defaultCdn]?.url || Object.values(hls?.cdns || {})[0]?.url;
      if (!masterUrl) throw new Error("No HLS or progressive stream found in Vimeo embed config");

      // Request standard MPEG-TS segments by ensuring sf=ts
      let tsMasterUrl = masterUrl;
      try {
        const u = new URL(masterUrl);
        u.searchParams.set("sf", "ts");
        tsMasterUrl = u.href;
      } catch (_) {}

      const masterRes = await fetch(tsMasterUrl);
      if (!masterRes.ok) throw new Error(`Failed to fetch HLS master playlist: HTTP ${masterRes.status}`);
      const masterText = await masterRes.text();

      let streamUrl = tsMasterUrl;
      const lines = masterText.split("\n");
      let bestBandwidth = -1;
      let variantUri = null;

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith("#EXT-X-STREAM-INF:")) {
          const bwMatch = line.match(/BANDWIDTH=(\d+)/);
          const bw = bwMatch ? parseInt(bwMatch[1], 10) : 0;
          const nextLine = lines[i + 1]?.trim();
          if (bw > bestBandwidth && nextLine && !nextLine.startsWith("#")) {
            bestBandwidth = bw;
            variantUri = nextLine;
          }
        }
      }

      if (variantUri) {
        streamUrl = new URL(variantUri, tsMasterUrl).href;
      }

      const variantRes = await fetch(streamUrl);
      if (!variantRes.ok) throw new Error(`Failed to fetch HLS variant playlist: HTTP ${variantRes.status}`);
      const variantText = await variantRes.text();
      const segmentUrls = [];
      for (const vline of variantText.split("\n")) {
        const t = vline.trim();
        if (t && !t.startsWith("#")) {
          segmentUrls.push(new URL(t, streamUrl).href);
        }
      }

      if (!segmentUrls.length) throw new Error("No video segments found in HLS stream");

      const Mux = typeof window !== "undefined" ? (window.muxjs || window.mux) : null;
      if (!Mux || !Mux.mp4 || !Mux.mp4.Transmuxer) {
        throw new Error("Transmuxer engine (mux.js) not available");
      }

      const outChunks = [];
      let initWritten = false;
      const transmuxer = new Mux.mp4.Transmuxer({ keepOriginalTimestamps: true });
      transmuxer.on("data", (segment) => {
        if (segment.initSegment && !initWritten) {
          outChunks.push(segment.initSegment);
          initWritten = true;
        }
        if (segment.data) {
          outChunks.push(segment.data);
        }
      });

      for (let i = 0; i < segmentUrls.length; i++) {
        if (typeof onProgress === "function") {
          onProgress(i + 1, segmentUrls.length);
        }
        const segRes = await fetch(segmentUrls[i]);
        if (!segRes.ok) throw new Error(`Failed to fetch segment ${i + 1}/${segmentUrls.length}: HTTP ${segRes.status}`);
        const segBuffer = await segRes.arrayBuffer();
        transmuxer.push(new Uint8Array(segBuffer));
        transmuxer.flush();
      }

      if (!outChunks.length) throw new Error("Transmuxing produced no video data");

      let writable = null;
      try {
        writable = await destFileHandle.createWritable({ keepExistingData: false });
        for (const chunk of outChunks) {
          await writable.write(chunk);
        }
        await writable.close();
        writable = null;
      } catch (err) {
        if (writable) {
          try { await writable.abort(); } catch (_) {}
        }
        throw err;
      }
    }
  }

  // --- UI Controller (Shadow DOM) ---
  class UIController {
    constructor() {
      this.host = document.createElement("div");
      this.host.id = "kp-backup-overlay";
      document.body.appendChild(this.host);
      this.shadow = this.host.attachShadow({ mode: "open" });
      this.render();
      this.bindElements();
    }

    render() {
      this.shadow.innerHTML = `
        <style>
          * { box-sizing: border-box; }
          .hud {
            position: fixed; top: 24px; right: 24px; width: 450px; max-width: 95vw;
            background: rgba(15, 23, 42, 0.96); backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.14); border-radius: 16px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.65); color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            z-index: 2147483647; padding: 20px; transition: transform 0.2s ease;
          }
          .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
          .title { font-size: 16px; font-weight: 700; color: #fff; margin: 0; display: flex; align-items: center; gap: 8px; }
          .badge {
            font-size: 11px; padding: 3px 8px; border-radius: 6px; font-weight: 600;
            background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);
          }
          .child-tag {
            font-size: 11px; padding: 3px 8px; border-radius: 6px; font-weight: 600;
            background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3);
          }
          .user-card {
            background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 10px; padding: 12px 14px; margin-bottom: 14px;
          }
          .ticker {
            font-size: 11px; color: #38bdf8; font-family: ui-monospace, Menlo, monospace;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 6px;
            font-weight: 500;
          }
          .progress-track { height: 10px; border-radius: 6px; background: #334155; overflow: hidden; margin: 6px 0; }
          .progress-fill {
            height: 100%; width: 0%;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            transition: width 0.25s ease;
          }
          .progress-fill.pulsing {
            background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8, #818cf8);
            background-size: 300% 100%;
            animation: indeterminate-shimmer 2s infinite linear;
          }
          @keyframes indeterminate-shimmer {
            0% { background-position: 100% 0; }
            100% { background-position: -100% 0; }
          }
          .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; margin: 12px 0; color: #94a3b8; }
          .stat-box { background: rgba(30, 41, 59, 0.6); padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); }
          .stat-lbl { font-size: 11px; color: #94a3b8; margin-bottom: 2px; }
          .stat-val { font-size: 13px; font-weight: 700; color: #f1f5f9; }
          .btn-row { display: flex; gap: 8px; margin-top: 14px; }
          .btn {
            flex: 1; padding: 10px 14px; border-radius: 8px; border: none; font-size: 13px; font-weight: 600;
            cursor: pointer; transition: all 0.15s ease;
          }
          .btn-primary { background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%); color: #fff; box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3); }
          .btn-primary:hover { opacity: 0.95; transform: translateY(-1px); }
          .btn-secondary { background: #334155; color: #e2e8f0; }
          .btn-secondary:hover { background: #475569; }
          .log-console {
            margin-top: 10px; max-height: 120px; overflow-y: auto; background: #090d16;
            padding: 8px 10px; border-radius: 8px; font-family: ui-monospace, Menlo, monospace;
            font-size: 11px; color: #cbd5e1; line-height: 1.4; border: 1px solid rgba(255, 255, 255, 0.05);
          }
          .hud.minimized {
            width: auto; max-width: 320px; padding: 8px 14px; cursor: pointer;
            border-radius: 999px; background: rgba(15, 23, 42, 0.96);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
          }
          .hud.minimized #setup-view,
          .hud.minimized #progress-view {
            display: none !important;
          }
          .hud.minimized .header {
            margin-bottom: 0;
          }
          .icon-btn {
            background: none; border: none; color: #94a3b8; font-size: 14px;
            cursor: pointer; padding: 2px 6px; border-radius: 4px; line-height: 1;
            transition: color 0.15s ease;
          }
          .icon-btn:hover { color: #f8fafc; background: rgba(255, 255, 255, 0.1); }
        </style>
        <div class="hud" id="hud">
          <div class="header">
            <h3 class="title">☁️ Kinderpedia Backup</h3>
            <div style="display:flex; align-items:center; gap:8px;">
              <span id="rate-badge" class="badge">Ready</span>
              <button id="min-btn" class="icon-btn" title="Minimize Dashboard">—</button>
            </div>
          </div>

          <div id="setup-view">
            <div class="user-card" id="user-card">
              <div id="user-display-name" style="font-size: 13px; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">
                🔍 Checking active web session...
              </div>
              <div id="children-badges-row" style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;"></div>
            </div>

            <p style="font-size: 13px; color: #cbd5e1; margin: 0 0 14px 0; line-height: 1.5;">
              Select a destination folder on your computer to save all full-resolution photos, videos, and documents directly to disk.
            </p>

            <div id="auth-prompt" style="display:none; margin-bottom: 12px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 12px;">
              <p style="font-size: 12px; color: #fca5a5; margin: 0 0 8px 0;">No active session detected. Please sign in:</p>
              <input id="email-in" type="email" placeholder="Kinderpedia Email" style="width:100%; margin-bottom:6px; padding:8px; border-radius:6px; background:#1e293b; border:1px solid #475569; color:#fff;" />
              <input id="pass-in" type="password" placeholder="Password" style="width:100%; padding:8px; border-radius:6px; background:#1e293b; border:1px solid #475569; color:#fff;" />
            </div>

            <button id="start-btn" class="btn btn-primary" style="width:100%;">📁 Select Folder & Start Backup</button>
          </div>

          <div id="progress-view" style="display:none;">
            <div class="ticker" id="activity-ticker">🚀 Initializing backup engine...</div>

            <div style="display:flex; justify-content:space-between; font-size:12px; color:#cbd5e1; margin-top:4px;">
              <span id="child-label" style="font-weight:600;">Preparing...</span>
              <span id="pct-label" style="font-weight:700; color:#38bdf8;">0.0%</span>
            </div>
            <div class="progress-track"><div id="progress-bar" class="progress-fill"></div></div>

            <div class="stat-grid">
              <div class="stat-box">
                <div class="stat-lbl" id="stat-lbl-1">Photos Downloaded</div>
                <div class="stat-val" id="stat-photos">0 / 0</div>
              </div>
              <div class="stat-box">
                <div class="stat-lbl" id="stat-lbl-2">Videos Downloaded</div>
                <div class="stat-val" id="stat-videos">0 / 0</div>
              </div>
              <div class="stat-box">
                <div class="stat-lbl" id="stat-lbl-3">Download Speed</div>
                <div class="stat-val" id="stat-speed">0 MB/s</div>
              </div>
              <div class="stat-box">
                <div class="stat-lbl" id="stat-lbl-4">ETA</div>
                <div class="stat-val" id="stat-eta">Calculating...</div>
              </div>
            </div>

            <div class="log-console" id="log-box"></div>

            <div class="btn-row">
              <button id="pause-btn" class="btn btn-secondary">Pause</button>
              <button id="stop-btn" class="btn btn-secondary" style="color:#f87171;">Stop</button>
            </div>
          </div>
        </div>
      `;
    }

    bindElements() {
      this.hud = this.shadow.getElementById("hud");
      this.rateBadge = this.shadow.getElementById("rate-badge");
      this.setupView = this.shadow.getElementById("setup-view");
      this.progressView = this.shadow.getElementById("progress-view");
      this.userCard = this.shadow.getElementById("user-card");
      this.userDisplayName = this.shadow.getElementById("user-display-name");
      this.childrenBadgesRow = this.shadow.getElementById("children-badges-row");
      this.authPrompt = this.shadow.getElementById("auth-prompt");
      this.emailIn = this.shadow.getElementById("email-in");
      this.passIn = this.shadow.getElementById("pass-in");
      this.startBtn = this.shadow.getElementById("start-btn");
      this.activityTicker = this.shadow.getElementById("activity-ticker");
      this.childLabel = this.shadow.getElementById("child-label");
      this.pctLabel = this.shadow.getElementById("pct-label");
      this.progressBar = this.shadow.getElementById("progress-bar");
      this.statLbl1 = this.shadow.getElementById("stat-lbl-1");
      this.statLbl2 = this.shadow.getElementById("stat-lbl-2");
      this.statLbl3 = this.shadow.getElementById("stat-lbl-3");
      this.statLbl4 = this.shadow.getElementById("stat-lbl-4");
      this.statPhotos = this.shadow.getElementById("stat-photos");
      this.statVideos = this.shadow.getElementById("stat-videos");
      this.statSpeed = this.shadow.getElementById("stat-speed");
      this.statEta = this.shadow.getElementById("stat-eta");
      this.logBox = this.shadow.getElementById("log-box");
      this.pauseBtn = this.shadow.getElementById("pause-btn");
      this.stopBtn = this.shadow.getElementById("stop-btn");
      this.minBtn = this.shadow.getElementById("min-btn");

      if (this.minBtn) {
        this.minBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.toggleUI();
        });
      }

      this.hud.addEventListener("click", (e) => {
        if (this.hud.classList.contains("minimized")) {
          this.toggleUI();
        }
      });
    }

    showAuthenticated(user, contexts) {
      this.rateBadge.textContent = "Connected";
      this.rateBadge.style.color = "#34d399";
      this.rateBadge.style.borderColor = "rgba(52, 211, 153, 0.4)";
      this.rateBadge.style.background = "rgba(52, 211, 153, 0.15)";

      const name = user?.name || "Parent Account";
      const email = user?.email ? ` (${user.email})` : "";
      this.userDisplayName.textContent = `👤 Logged in as ${name}${email}`;

      this.childrenBadgesRow.innerHTML = "";
      for (const ctx of contexts) {
        const tag = document.createElement("span");
        tag.className = "child-tag";
        tag.textContent = `🧒 ${ctx.child_name}`;
        this.childrenBadgesRow.appendChild(tag);
      }

      this.authPrompt.style.display = "none";
    }

    showUnauthenticated() {
      this.rateBadge.textContent = "Session: Login Needed";
      this.userDisplayName.textContent = "⚠️ No active session detected";
      this.childrenBadgesRow.innerHTML = "";
      this.authPrompt.style.display = "block";
    }

    setActivity(msg) {
      this.activityTicker.textContent = msg;
      this.activityTicker.title = msg;
    }

    log(msg) {
      const line = document.createElement("div");
      const time = new Date().toLocaleTimeString();
      line.textContent = `[${time}] ${msg}`;
      this.logBox.appendChild(line);
      this.logBox.scrollTop = this.logBox.scrollHeight;
    }

    updateDiscovery({ childName, itemType, page, foundCount, subCount, subLabel, currentItem }) {
      this.childLabel.textContent = `Finding ${itemType.toLowerCase()} for ${childName}...`;
      this.pctLabel.textContent = `Page ${page}`;
      this.progressBar.classList.add("pulsing");
      this.progressBar.style.width = "100%";

      if (itemType.toLowerCase().includes("video")) {
        this.statVideos.textContent = `${foundCount} found`;
      } else {
        this.statPhotos.textContent = subCount ? `~${subCount.toLocaleString()} found` : `${foundCount} albums found`;
      }

      const itemDesc = currentItem ? ` ("${sanitizeName(currentItem, 28)}")` : "";
      this.setActivity(`Finding ${itemType.toLowerCase()} for ${childName}... (Page ${page})${itemDesc}`);
    }

    startDownloadingUI() {
      this.progressBar.classList.remove("pulsing");
      this.progressBar.style.width = "0%";
      this.statLbl1.textContent = "Photos Downloaded";
      this.statLbl2.textContent = "Videos Downloaded";
      this.statLbl3.textContent = "Download Speed";
      this.statLbl4.textContent = "ETA";
    }

    updateStats(stats) {
      const {
        pct = 0,
        childName = "",
        photosSaved = 0,
        photosSkipped = 0,
        totalPhotos = 0,
        videosSaved = 0,
        videosSkipped = 0,
        totalVideos = 0,
        speed = "0 MB/s",
        eta = "Calculating...",
        statusBadge = "",
        currentActivity = "",
      } = stats;

      this.childLabel.textContent = childName || "Backing up...";
      this.pctLabel.textContent = `${pct.toFixed(1)}%`;
      this.progressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;

      const totalPhotosDone = photosSaved + photosSkipped;
      this.statPhotos.textContent = totalPhotos > 0
        ? `${totalPhotosDone.toLocaleString()} / ${totalPhotos.toLocaleString()}`
        : `${totalPhotosDone.toLocaleString()}`;

      const totalVideosDone = videosSaved + videosSkipped;
      this.statVideos.textContent = videosSaved > 0
        ? `${videosSaved} / ${totalVideos}`
        : `${totalVideos} cataloged`;

      this.statSpeed.textContent = speed;
      this.statEta.textContent = eta;

      if (statusBadge) this.rateBadge.textContent = statusBadge;
      if (currentActivity) this.setActivity(currentActivity);
    }

    toggleUI() {
      const isMin = this.hud.classList.toggle("minimized");
      if (this.minBtn) {
        this.minBtn.textContent = isMin ? "□" : "—";
        this.minBtn.title = isMin ? "Restore Dashboard" : "Minimize Dashboard";
      }
    }
  }

  // --- Main Orchestrator Engine ---
  class BackupEngine {
    constructor() {
      this.ui = new UIController();
      this.rate = new AdaptiveRateController();
      this.client = new KinderpediaClient(this.rate, (m) => this.ui.log(m));
      let serverBase = "http://localhost:8080";
      try {
        if (typeof document !== "undefined") {
          const scripts = Array.from(document.querySelectorAll("script"));
          for (const s of scripts) {
            if (s.src && s.src.includes("kinderpedia-backup")) {
              const u = new URL(s.src);
              if (u.hostname === "localhost" || u.hostname === "127.0.0.1") {
                serverBase = u.origin;
              }
              break;
            }
          }
        }
      } catch (_) {}
      this.vimeo = new VimeoDownloader((m) => this.ui.log(m), serverBase);
      this.isPaused = false;
      this.isCancelled = false;
      this.rootDirHandle = null;
      this.concurrency = 6;

      this.init();
      this.bindEvents();
    }

    getWorkerConfig() {
      return { url: DEFAULT_WORKER_URL, secret: DEFAULT_WORKER_SECRET };
    }

    toggleUI() {
      if (this.ui) {
        this.ui.toggleUI();
      }
    }

    async init() {
      const isAuth = await this.client.checkActiveSession();
      if (isAuth) {
        this.ui.showAuthenticated(this.client.userData, this.client.childrenContexts);
      } else {
        this.ui.showUnauthenticated();
      }
    }

    bindEvents() {
      this.ui.startBtn.addEventListener("click", async () => {
        try {
          if (!this.client.childrenContexts.length) {
            const isAuth = await this.client.checkActiveSession();
            if (!isAuth) {
              const email = this.ui.emailIn.value.trim();
              const pass = this.ui.passIn.value;
              if (!email || !pass) {
                this.ui.showUnauthenticated();
                alert("Please enter your Kinderpedia login credentials.");
                return;
              }
              await this.client.login(email, pass);
              this.ui.showAuthenticated(this.client.userData, this.client.childrenContexts);
            }
          }

          // Trigger Modern Browser File System Access API
          this.rootDirHandle = await window.showDirectoryPicker({ mode: "readwrite" });
          this.ui.setupView.style.display = "none";
          this.ui.progressView.style.display = "block";

          this.startBackup();
        } catch (err) {
          if (err.name !== "AbortError") {
            alert(`Error: ${err.message}`);
          }
        }
      });

      this.ui.pauseBtn.addEventListener("click", () => {
        this.isPaused = !this.isPaused;
        this.ui.pauseBtn.textContent = this.isPaused ? "Resume" : "Pause";
        this.ui.log(this.isPaused ? "⏸️ Backup paused" : "▶️ Backup resumed");
        this.ui.setActivity(this.isPaused ? "⏸️ Backup paused by user" : "▶️ Resuming backup operations...");
      });

      this.ui.stopBtn.addEventListener("click", () => {
        if (confirm("Stop the backup process? (Existing downloaded files will remain on disk)")) {
          this.isCancelled = true;
          this.ui.log("🛑 Cancelling backup...");
          this.ui.setActivity("🛑 Cancelling backup operations...");
        }
      });
    }

    async writeJsonAtomic(dirHandle, filename, data) {
      const fileHandle = await dirHandle.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable({ keepExistingData: false });
      await writable.write(JSON.stringify(data, null, 2));
      await writable.close();
    }

    async readJsonIfExists(dirHandle, filename) {
      try {
        const fileHandle = await dirHandle.getFileHandle(filename);
        const file = await fileHandle.getFile();
        if (file.size > 0) {
          return JSON.parse(await file.text());
        }
      } catch (_) {}
      return null;
    }

    async checkFileExists(dirHandle, filename, minBytes = 1) {
      try {
        const handle = await dirHandle.getFileHandle(filename);
        const file = await handle.getFile();
        if (file && file.size >= minBytes) {
          return true;
        }
        // File exists but is empty or too small (corrupted/interrupted): remove it so it can be re-downloaded
        try { await dirHandle.removeEntry(filename); } catch (_) {}
        return false;
      } catch (_) {
        return false;
      }
    }

    async streamBinaryDirect(url, dirHandle, filename, minBytes = 1) {
      const exists = await this.checkFileExists(dirHandle, filename, minBytes);
      if (exists) return { status: "skipped", bytes: 0 };

      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} on ${url}`);

      const cl = res.headers.get("content-length");
      const expectedBytes = cl ? parseInt(cl, 10) : minBytes;

      const fileHandle = await dirHandle.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable({ keepExistingData: false });
      try {
        await res.body.pipeTo(writable);
      } catch (err) {
        try { await writable.abort(); } catch (_) {}
        try { await dirHandle.removeEntry(filename); } catch (_) {}
        throw err;
      }

      // Strict integrity verification on disk
      try {
        const diskFile = await fileHandle.getFile();
        if (!diskFile || diskFile.size === 0 || (expectedBytes > 1 && diskFile.size < expectedBytes)) {
          try { await dirHandle.removeEntry(filename); } catch (_) {}
          throw new Error(`Incomplete file: ${filename} (got ${diskFile?.size || 0} B, expected ${expectedBytes} B)`);
        }
        return { status: "downloaded", bytes: diskFile.size };
      } catch (err) {
        try { await dirHandle.removeEntry(filename); } catch (_) {}
        throw err;
      }
    }

    refreshLiveUI(childName) {
      if (!this.stats) return;
      const elapsedS = Math.max(1, (performance.now() - this.stats.t0) / 1000);
      const mbSec = this.stats.bytesSaved / (1024 * 1024) / elapsedS;
      const photosMin = Math.round((this.stats.photosSaved / elapsedS) * 60);

      const totalProcessedPhotos = this.stats.photosSaved + this.stats.photosSkipped;
      const totalProcessedVideos = this.stats.videosSaved + this.stats.videosSkipped;
      const totalProcessed = totalProcessedPhotos + totalProcessedVideos;
      const totalItems = (this.stats.totalExpectedPhotos || 0) + (this.stats.totalVideos || 0);

      const pct = totalItems > 0 ? (totalProcessed / totalItems) * 100 : 0;
      const remainingPhotos = Math.max(0, (this.stats.totalExpectedPhotos || 0) - totalProcessedPhotos);
      const remainingVideos = Math.max(0, (this.stats.totalVideos || 0) - totalProcessedVideos);

      let eta = "Calculating...";
      if (photosMin > 0 && remainingPhotos > 0 && elapsedS > 3) {
        const remainingSec = (remainingPhotos / photosMin) * 60;
        eta = formatDuration(remainingSec);
      } else if (remainingPhotos === 0 && (this.stats.totalExpectedPhotos || 0) > 0 && remainingVideos === 0) {
        eta = "Done";
      }

      this.ui.updateStats({
        pct,
        childName,
        photosSaved: this.stats.photosSaved,
        photosSkipped: this.stats.photosSkipped,
        totalPhotos: this.stats.totalExpectedPhotos,
        videosSaved: this.stats.videosSaved,
        videosSkipped: this.stats.videosSkipped,
        totalVideos: this.stats.totalVideos,
        speed: `${mbSec.toFixed(1)} MB/s`,
        eta,
        statusBadge: this.isPaused ? "Paused" : "Downloading",
      });
    }

    async fetchGalleryPage(ctx, mode, page, orderBy) {
      const query = new URLSearchParams({
        mode,
        page: String(page),
        items_per_page: "30",
        order_by: orderBy,
        order_type: "DESC",
      });
      const resp = await this.client.httpJson(`/data/gallery?${query}`, {
        headers: {
          "X-KINDERGARTEN-ID": String(ctx.kindergarten_id),
          "X-CHILD-ID": String(ctx.child_id),
        },
      });
      const payload = resp.response || resp.result || resp;
      if (payload && Array.isArray(payload[mode])) return payload[mode];
      if (payload && Array.isArray(payload.items)) return payload.items;
      if (Array.isArray(payload)) return payload;
      return [];
    }

    /**
     * Incremental listing (mirrors Python `list_albums(known=...)` / `list_videos(known_ids=...)`): pages are
     * ordered newest-first, so scanning stops at the first page whose items are all known (and, for albums,
     * have an unchanged `(updated_at, count_images)`). Returns the fresh items merged into the cached index.
     */
    async catalogIncremental(ctx, metadataDir, cached, { mode, orderBy, indexFile, isUnchanged }) {
      const fresh = [];
      const seen = new Set();
      let page = 1;
      while (!this.isCancelled) {
        while (this.isPaused && !this.isCancelled) await sleep(500);
        if (this.isCancelled) break;
        this.ui.setActivity(`🔍 Checking ${mode} page ${page} for new items...`);
        let items;
        try {
          items = await this.fetchGalleryPage(ctx, mode, page, orderBy);
        } catch (e) {
          this.ui.log(`⚠️ Error checking ${mode} page ${page}: ${e.message}`);
          break;
        }
        if (!items.length) break;
        let added = 0;
        let pageUnchanged = true;
        for (const item of items) {
          if (!item || item.id == null) continue;
          if (!isUnchanged(item)) pageUnchanged = false;
          if (!seen.has(String(item.id))) {
            seen.add(String(item.id));
            fresh.push(item);
            added++;
          }
        }
        if (pageUnchanged || items.length < 30 || added === 0) break;
        page++;
      }
      const merged = mergeIncrementalListing(fresh, cached);
      if (!this.isCancelled) {
        await this.writeJsonAtomic(metadataDir, indexFile, merged);
      }
      return { merged, freshCount: fresh.filter((x) => !isUnchanged(x)).length };
    }

    async catalogChildAlbums(ctx, metadataDir) {
      const finalIndex = await this.readJsonIfExists(metadataDir, "albums_index.json");
      if (Array.isArray(finalIndex) && finalIndex.length > 0) {
        const known = new Map(finalIndex.filter((a) => a && a.id != null).map((a) => [String(a.id), albumListingSignature(a)]));
        const { merged, freshCount } = await this.catalogIncremental(ctx, metadataDir, finalIndex, {
          mode: "albums",
          orderBy: "updated_at",
          indexFile: "albums_index.json",
          isUnchanged: (a) => known.get(String(a.id)) === albumListingSignature(a),
        });
        this.stats.totalExpectedAlbums = merged.length;
        this.stats.totalExpectedPhotos = merged.reduce((acc, a) => acc + (parseInt(a.count_images, 10) || 0), 0);
        this.ui.log(`📸 ${freshCount} new/changed album(s) since the last backup (${merged.length} total).`);
        this.refreshLiveUI(ctx.child_name);
        return merged;
      }

      const partialState = await this.readJsonIfExists(metadataDir, "albums_index_partial.json");
      let allAlbums = [];
      const seenIds = new Set();
      let startPage = 1;

      if (partialState && Array.isArray(partialState.items) && partialState.items.length > 0) {
        allAlbums = [...partialState.items];
        for (const a of allAlbums) {
          if (a.id) seenIds.add(a.id);
        }
        startPage = partialState.next_page || 1;
        this.stats.totalExpectedAlbums = allAlbums.length;
        this.stats.totalExpectedPhotos = allAlbums.reduce((acc, a) => acc + (parseInt(a.count_images, 10) || 0), 0);
        this.ui.log(`📸 Resuming photo albums from disk...`);
      }

      let catalogingComplete = partialState?.complete === true;
      let page = startPage;

      if (!catalogingComplete) {
        while (!this.isCancelled) {
          while (this.isPaused && !this.isCancelled) await sleep(500);
          if (this.isCancelled) break;

          this.ui.setActivity(`🔍 Querying Albums Page ${page}...`);
          const query = new URLSearchParams({
            mode: "albums",
            page: String(page),
            items_per_page: "30",
            order_by: "updated_at",
            order_type: "DESC",
          });

          let resp;
          try {
            resp = await this.client.httpJson(`/data/gallery?${query}`, {
              headers: {
                "X-KINDERGARTEN-ID": String(ctx.kindergarten_id),
                "X-CHILD-ID": String(ctx.child_id),
              },
            });
          } catch (e) {
            this.ui.log(`⚠️ Error cataloging albums page ${page}: ${e.message}`);
            break;
          }

          const payload = resp.response || resp.result || resp;
          let items = [];
          if (payload && Array.isArray(payload.albums)) {
            items = payload.albums;
          } else if (payload && Array.isArray(payload.items)) {
            items = payload.items;
          } else if (Array.isArray(payload)) {
            items = payload;
          }

          if (!items.length) break;

          let addedThisPage = 0;
          for (const item of items) {
            if (item.id && !seenIds.has(item.id)) {
              seenIds.add(item.id);
              allAlbums.push(item);
              this.stats.totalExpectedAlbums++;
              this.stats.totalExpectedPhotos += parseInt(item.count_images, 10) || 0;
              addedThisPage++;
            }
          }

          this.refreshLiveUI(ctx.child_name);

          await this.writeJsonAtomic(metadataDir, "albums_index_partial.json", {
            complete: false,
            next_page: page + 1,
            total_albums: allAlbums.length,
            total_photos_est: this.stats.totalExpectedPhotos,
            updated_at: new Date().toISOString(),
            items: allAlbums,
          });

          if (page === 1) {
            this.ui.log(`📸 Discovered initial albums. Cataloging album index...`);
          }

          if (items.length < 30 || addedThisPage === 0) break;
          page++;
        }
      }

      if (!this.isCancelled) {
        await this.writeJsonAtomic(metadataDir, "albums_index.json", allAlbums);
        await this.writeJsonAtomic(metadataDir, "albums_index_partial.json", {
          complete: true,
          next_page: page,
          total_albums: allAlbums.length,
          total_photos_est: this.stats.totalExpectedPhotos,
          updated_at: new Date().toISOString(),
          items: allAlbums,
        });
        this.ui.log(`📸 Cataloged ${allAlbums.length} albums (${this.stats.totalExpectedPhotos.toLocaleString()} total photos).`);
      }

      return allAlbums;
    }

    async dirExists(baseDirHandle, relPath) {
      try {
        let cur = baseDirHandle;
        for (const part of String(relPath).split("/").filter(Boolean)) {
          cur = await cur.getDirectoryHandle(part);
        }
        return true;
      } catch (_) {
        return false;
      }
    }

    /**
     * Load Metadata/<Child>/folder_map.json, or rebuild the album part from Metadata/<Child>/Albums/*.json
     * (`album_listing.id`) for archives created before the map existed (mirrors Python `rebuild_folder_map`).
     */
    async loadFolderMap(metadataDir, metaAlbumsDir, albumsDir) {
      const parsed = parseFolderMap(await this.readJsonIfExists(metadataDir, FOLDER_MAP_FILE));
      if (parsed) return parsed;
      const albums = new Map();
      const candidates = new Map();
      try {
        for await (const entry of metaAlbumsDir.values()) {
          if (entry.kind !== "file" || !entry.name.endsWith(".json")) continue;
          const leaf = entry.name.slice(0, -5);
          let meta = null;
          try {
            meta = JSON.parse(await (await entry.getFile()).text());
          } catch (_) {
            continue;
          }
          const id = meta && meta.album_listing && meta.album_listing.id;
          if (id == null || !Number.isInteger(Number(id))) continue;
          if (!candidates.has(Number(id))) candidates.set(Number(id), []);
          candidates.get(Number(id)).push(leaf);
        }
      } catch (_) {}
      for (const [id, leaves] of candidates.entries()) {
        leaves.sort();
        let chosen = leaves[0];
        if (leaves.length > 1) {
          for (const leaf of leaves) {
            if (await this.dirExists(albumsDir, albumFolderFromLeaf(leaf))) {
              chosen = leaf;
              break;
            }
          }
        }
        albums.set(id, albumFolderFromLeaf(chosen));
      }
      if (albums.size) this.ui.log(`🗂️ Rebuilt ${albums.size} album folder identities from Metadata.`);
      return { albums, videos: new Map() };
    }

    async processChildPhotos(ctx, albumsDir, metadataDir, metaAlbumsDir, dirCache, resolveAlbumsCatalog, folderMap) {
      let allAlbums = [];
      let albumFolderMap = new Map();
      try {
        allAlbums = await this.catalogChildAlbums(ctx, metadataDir);
        // Persisted identity: known album IDs keep their folders; only unknown IDs get new names.
        albumFolderMap = assignAlbumFolderNames(allAlbums, folderMap.albums);
        for (const [id, folder] of albumFolderMap.entries()) folderMap.albums.set(id, folder);
        if (!this.isCancelled && (allAlbums.length || folderMap.albums.size)) {
          await this.writeJsonAtomic(metadataDir, FOLDER_MAP_FILE, serializeFolderMap(folderMap.albums, folderMap.videos));
        }
      } finally {
        if (typeof resolveAlbumsCatalog === "function") {
          resolveAlbumsCatalog({ allAlbums, albumFolderMap });
        }
      }

      if (this.isCancelled || !allAlbums.length) return;

      await this.downloadAlbumList(ctx, albumsDir, metadataDir, metaAlbumsDir, dirCache, allAlbums, albumFolderMap);
      if (!this.isCancelled) {
        this.ui.log(`📸 All ${allAlbums.length} albums downloaded.`);
      }
    }

    async downloadSingleAlbum(ctx, albumsDir, metadataDir, metaAlbumsDir, dirCache, album, folderRel) {
      if (this.isCancelled) return;
      while (this.isPaused && !this.isCancelled) await sleep(500);

      const aid = album.id;
      const aname = album.name || `album_${aid}`;
      const expectedCnt = parseInt(album.count_images, 10) || 0;
      const leafName = String(folderRel).split("/").pop();
      const albumMetaFileName = `${leafName}.json`;
      const albumDir = await getNestedDirectoryHandle(albumsDir, folderRel, dirCache);

      // FAST-PATH: Count only numbered album photos (`001_<id>.jpg`), ignoring co-located `.mp4` videos
      let meta = await this.readJsonIfExists(metaAlbumsDir, albumMetaFileName);
      // Reuse the cached manifest unless the listing's (updated_at, count_images) changed since it was fetched
      // or it is empty for a non-empty album (mirrors Python engine `_prepare_album`).
      const manifestFresh =
        !!meta &&
        Array.isArray(meta.images) &&
        albumListingSignature(meta.album_listing) === albumListingSignature(album) &&
        (meta.images.length > 0 || expectedCnt === 0);
      if (manifestFresh && meta.completed === true) {
        let diskPhotoCount = 0;
        try {
          if (typeof albumDir.values === "function") {
            for await (const entry of albumDir.values()) {
              if (
                entry.kind === "file" &&
                NUMBERED_PHOTO_RE.test(entry.name) &&
                !entry.name.endsWith(".json") &&
                !entry.name.endsWith(".part")
              ) {
                diskPhotoCount++;
              }
            }
          }
        } catch (_) {}
        if (diskPhotoCount >= meta.images.length && meta.images.length > 0) {
          this.stats.photosSkipped += meta.images.length;
          this.stats.albumsDone++;
          this.refreshLiveUI(ctx.child_name);
          return;
        }
      }

      // Fetch album image manifest if missing or incomplete
      let images = manifestFresh ? meta.images : [];
      if (!manifestFresh) {
        this.ui.setActivity(`🔍 Fetching photo list: "${sanitizeName(aname, 35)}"...`);
        const res = await this.client.getAlbumImages(ctx.kindergarten_id, ctx.child_id, aid, expectedCnt);
        images = res.images;
        meta = { album_listing: album, album_details: res.albumDetails, images, completed: false };
        await this.writeJsonAtomic(metaAlbumsDir, albumMetaFileName, meta);
      }

      // Build all photo tasks for this album (`001_<id>.<ext>`)
      const allTasks = images.map((img, i) => {
        let url = img.download_url || img.fullsize || img.url_original || img.url_large || img.url || img.src || img.thumb;
        if (typeof url === "string") {
          if (url.startsWith("//")) url = "https:" + url;
          else if (url.startsWith("/")) url = "https://app.kinderpedia.co" + url;
        }
        const ext = (img.extension || "jpg").replace(/^\./, "");
        const imgId = sanitizeName(String(img.id || i + 1), 32);
        const fname = `${String(i + 1).padStart(3, "0")}_${imgId}.${ext}`;
        return { url, fname, id: img.id };
      });

      // Fast directory scan: count numbered photos only (ignoring co-located .mp4 videos)
      let existingPhotosCount = 0;
      try {
        if (typeof albumDir.values === "function") {
          for await (const entry of albumDir.values()) {
            if (
              entry.kind === "file" &&
              NUMBERED_PHOTO_RE.test(entry.name) &&
              !entry.name.endsWith(".json") &&
              !entry.name.endsWith(".part")
            ) {
              existingPhotosCount++;
            }
          }
        }
      } catch (_) {}

      if (existingPhotosCount >= allTasks.length && allTasks.length > 0) {
        meta.completed = true;
        meta.completed_at = new Date().toISOString();
        meta.photo_count = allTasks.length;
        await this.writeJsonAtomic(metaAlbumsDir, albumMetaFileName, meta);

        this.stats.photosSkipped += allTasks.length;
        this.stats.albumsDone++;
        this.refreshLiveUI(ctx.child_name);
        return;
      }

      // Detect interrupted album: check which individual photos already exist on disk
      const pendingTasks = [];
      let alreadyOnDisk = 0;
      for (const task of allTasks) {
        const exists = await this.checkFileExists(albumDir, task.fname);
        if (exists) {
          alreadyOnDisk++;
          this.stats.photosSkipped++;
        } else {
          pendingTasks.push(task);
        }
      }

      if (alreadyOnDisk > 0 && pendingTasks.length > 0) {
        // Resume interrupted album silently via ticker
      } else if (pendingTasks.length === 0 && allTasks.length > 0) {
        meta.completed = true;
        meta.completed_at = new Date().toISOString();
        meta.photo_count = allTasks.length;
        await this.writeJsonAtomic(metaAlbumsDir, albumMetaFileName, meta);
        this.stats.albumsDone++;
        this.refreshLiveUI(ctx.child_name);
        return;
      }

      // Download pending tasks using 6 parallel workers
      let activeIndex = 0;
      let albumSaved = 0;

      const workers = Array.from({ length: this.concurrency }, async () => {
        while (activeIndex < pendingTasks.length && !this.isCancelled) {
          const task = pendingTasks[activeIndex++];
          if (!task?.url) continue;
          try {
            const res = await this.streamBinaryDirect(task.url, albumDir, task.fname);
            if (res.status === "downloaded") {
              this.stats.photosSaved++;
              albumSaved++;
              this.stats.bytesSaved += res.bytes;
              this.ui.setActivity(`⬇️ [Photo] Saved ${task.fname} (${formatBytes(res.bytes)})`);
            } else if (res.status === "skipped") {
              this.stats.photosSkipped++;
            }
          } catch (err) {
            console.warn(`[Photo] Failed to download ${task.fname}:`, err);
          }
          this.refreshLiveUI(ctx.child_name);
        }
      });

      await Promise.all(workers);

      if (!this.isCancelled) {
        const isComplete = (alreadyOnDisk + albumSaved) >= allTasks.length && allTasks.length > 0;
        meta.completed = isComplete;
        meta.completed_at = new Date().toISOString();
        meta.photo_count = allTasks.length;
        await this.writeJsonAtomic(metaAlbumsDir, albumMetaFileName, meta);

        await this.writeJsonAtomic(metadataDir, "backup_progress.json", {
          last_completed_album_id: aid,
          last_completed_album_name: aname,
          total_photos_processed: this.stats.photosSaved + this.stats.photosSkipped,
          updated_at: new Date().toISOString(),
        });

        this.stats.albumsDone++;
        this.refreshLiveUI(ctx.child_name);
        if (albumSaved > 0) {
          this.ui.log(`📸 Album "${sanitizeName(aname, 40)}" saved (${albumSaved} photos)`);
        }
      }
    }

    async downloadAlbumList(ctx, albumsDir, metadataDir, metaAlbumsDir, dirCache, albumList, albumFolderMap) {
      for (const album of albumList) {
        if (this.isCancelled) break;
        const folderRel = albumFolderMap.get(Number(album.id));
        if (!folderRel) continue;
        await this.downloadSingleAlbum(ctx, albumsDir, metadataDir, metaAlbumsDir, dirCache, album, folderRel);
      }
    }

    async catalogChildVideos(ctx, metadataDir) {
      const finalIndex = await this.readJsonIfExists(metadataDir, "videos_index.json");
      if (Array.isArray(finalIndex) && finalIndex.length > 0) {
        const knownIds = new Set(finalIndex.filter((v) => v && v.id != null).map((v) => String(v.id)));
        const { merged } = await this.catalogIncremental(ctx, metadataDir, finalIndex, {
          mode: "videos",
          orderBy: "dateadd",
          indexFile: "videos_index.json",
          isUnchanged: (v) => knownIds.has(String(v.id)),
        });
        this.stats.totalVideos = merged.length;
        this.refreshLiveUI(ctx.child_name);
        return merged;
      }

      const partialState = await this.readJsonIfExists(metadataDir, "videos_index_partial.json");
      let allVideos = [];
      const seenIds = new Set();
      let startPage = 1;

      if (partialState && Array.isArray(partialState.items) && partialState.items.length > 0) {
        allVideos = [...partialState.items];
        for (const v of allVideos) {
          if (v.id) seenIds.add(v.id);
        }
        startPage = partialState.next_page || 1;
        this.stats.totalVideos = allVideos.length;
        this.ui.log(`🎬 Resuming videos from disk...`);
      }

      let videoCatalogingComplete = partialState?.complete === true;
      let page = startPage;

      if (!videoCatalogingComplete) {
        while (!this.isCancelled) {
          while (this.isPaused && !this.isCancelled) await sleep(500);
          if (this.isCancelled) break;

          const query = new URLSearchParams({
            mode: "videos",
            page: String(page),
            items_per_page: "30",
            order_by: "dateadd",
            order_type: "DESC",
          });

          let resp;
          try {
            resp = await this.client.httpJson(`/data/gallery?${query}`, {
              headers: {
                "X-KINDERGARTEN-ID": String(ctx.kindergarten_id),
                "X-CHILD-ID": String(ctx.child_id),
              },
            });
          } catch (_) {
            break;
          }

          const payload = resp.response || resp.result || resp;
          let items = [];
          if (payload && Array.isArray(payload.videos)) {
            items = payload.videos;
          } else if (payload && Array.isArray(payload.items)) {
            items = payload.items;
          } else if (Array.isArray(payload)) {
            items = payload;
          }
          if (!items.length) break;

          let addedThisPage = 0;
          for (const item of items) {
            if (item.id && !seenIds.has(item.id)) {
              seenIds.add(item.id);
              allVideos.push(item);
              this.stats.totalVideos++;
              addedThisPage++;
            }
          }

          this.refreshLiveUI(ctx.child_name);

          await this.writeJsonAtomic(metadataDir, "videos_index_partial.json", {
            complete: false,
            next_page: page + 1,
            total_videos: allVideos.length,
            updated_at: new Date().toISOString(),
            items: allVideos,
          });

          if (page === 1) {
            this.ui.log(`🎬 Discovered initial videos. Cataloging video index...`);
          }

          if (items.length < 30 || addedThisPage === 0) break;
          page++;
        }
      }

      if (!this.isCancelled) {
        await this.writeJsonAtomic(metadataDir, "videos_index.json", allVideos);
        await this.writeJsonAtomic(metadataDir, "videos_index_partial.json", {
          complete: true,
          next_page: page,
          total_videos: allVideos.length,
          updated_at: new Date().toISOString(),
          items: allVideos,
        });
        this.ui.log(`🎬 Cataloged ${allVideos.length} videos (matching into day albums).`);
      }

      return allVideos;
    }

    async processChildVideos(ctx, albumsDir, metadataDir, dirCache, albumsCatalogPromise, folderMap) {
      const allVideos = await this.catalogChildVideos(ctx, metadataDir);
      const { allAlbums, albumFolderMap } = await albumsCatalogPromise;
      if (this.isCancelled || !allVideos.length) return;

      const metaVideosDir = await metadataDir.getDirectoryHandle("Videos", { create: true });
      const videoTargets = planVideoTargets(allVideos, allAlbums, albumFolderMap, folderMap.videos);
      for (const [id, target] of videoTargets.entries()) folderMap.videos.set(id, target);
      await this.writeJsonAtomic(metadataDir, FOLDER_MAP_FILE, serializeFolderMap(folderMap.albums, folderMap.videos));
      await this.downloadVideoList(ctx, albumsDir, metaVideosDir, dirCache, allVideos, videoTargets);
    }

    async downloadSingleVideo(ctx, albumsDir, metaVideosDir, dirCache, vid, target) {
      if (this.isCancelled) return;
      while (this.isPaused && !this.isCancelled) await sleep(500);

      const { folderRel, fileName: vname } = target;
      const targetAlbumDir = await getNestedDirectoryHandle(albumsDir, folderRel, dirCache);

      // Video must be at least 10 KB to be considered valid and skipped
      const exists = await this.checkFileExists(targetAlbumDir, vname, 10000);
      if (exists) {
        this.stats.videosSkipped++;
        this.refreshLiveUI(ctx.child_name);
        return;
      }

      let videoUrl = vid.url || vid.share_url || "";
      if (!videoUrl) {
        const cachedVid = await this.readJsonIfExists(metaVideosDir, `${vid.id}.json`);
        if (cachedVid) {
          videoUrl = cachedVid.url || cachedVid.share_url || "";
        }
      }
      if (!videoUrl) {
        try {
          const details = await this.client.getVideoDetails(ctx.kindergarten_id, ctx.child_id, vid.id);
          await this.writeJsonAtomic(metaVideosDir, `${vid.id}.json`, { ...vid, ...details });
          videoUrl = details.url || details.share_url || "";
        } catch (_) {}
      }

      // Check if local streaming helper server (serve.js) is available
      if (this.vimeoHelperAvailable === undefined) {
        try {
          const check = await fetch("http://localhost:8080/vimeo-config?test=1", { signal: AbortSignal.timeout(600) });
          this.vimeoHelperAvailable = check.ok;
        } catch (_) {
          this.vimeoHelperAvailable = false;
        }
      }

      // Check if Cloudflare Worker is configured and reachable
      const workerCfg = this.getWorkerConfig();
      if (this.vimeoWorkerAvailable === undefined) {
        if (workerCfg && workerCfg.url) {
          try {
            const check = await fetch(`${workerCfg.url}?ping=1&secret=${encodeURIComponent(workerCfg.secret)}`, { signal: AbortSignal.timeout(1500) });
            this.vimeoWorkerAvailable = check.ok;
          } catch (_) {
            this.vimeoWorkerAvailable = false;
          }
        } else {
          this.vimeoWorkerAvailable = false;
        }
      }

      if (videoUrl && videoUrl.includes("vimeo.com")) {
        // ON KINDERPEDIA: If neither local helper nor Cloudflare Worker is active, quiet cataloging
        if (!this.vimeoHelperAvailable && !this.vimeoWorkerAvailable) {
          this.stats.videosSkipped++;
          this.refreshLiveUI(ctx.child_name);
          return;
        }

        try {
          this.ui.setActivity(`🎬 Downloading video: "${sanitizeName(vid.name || vid.id, 30)}"...`);
          const vHandle = await targetAlbumDir.getFileHandle(vname, { create: true });
          await this.vimeo.downloadVideo(videoUrl, vHandle, workerCfg, (curr, total) => {
            this.ui.setActivity(`🎬 Downloading video: "${sanitizeName(vid.name || vid.id, 25)}" (${curr}/${total} segments)...`);
          }, !!this.vimeoHelperAvailable);

          // Post-download integrity verification
          const vFile = await vHandle.getFile();
          if (!vFile || vFile.size < 10000) {
            try { await targetAlbumDir.removeEntry(vname); } catch (_) {}
            throw new Error(`Video file is incomplete (${vFile?.size || 0} bytes)`);
          }
          this.stats.videosSaved++;
          this.stats.bytesSaved += vFile.size;
          this.ui.log(`🎬 Video downloaded: "${folderRel}/${vname}" (${(vFile.size / (1024 * 1024)).toFixed(1)} MB)`);
        } catch (err) {
          try { await targetAlbumDir.removeEntry(vname); } catch (_) {}
          console.warn("[Video] Download failed:", err);
        }
      } else if (videoUrl && (videoUrl.includes(".mp4") || videoUrl.includes("kinderpedia.co"))) {
        try {
          this.ui.setActivity(`🎬 Downloading video: "${sanitizeName(vid.name || vid.id, 30)}"...`);
          const res = await this.streamBinaryDirect(videoUrl, targetAlbumDir, vname, 10000);
          if (res.status === "downloaded") {
            this.stats.videosSaved++;
            this.stats.bytesSaved += res.bytes;
            this.ui.log(`🎬 Video downloaded: "${folderRel}/${vname}"`);
          }
        } catch (err) {
          try { await targetAlbumDir.removeEntry(vname); } catch (_) {}
          console.warn("[Video] Download failed:", err);
        }
      }
      this.refreshLiveUI(ctx.child_name);
    }

    async downloadVideoList(ctx, albumsDir, metaVideosDir, dirCache, videoList, videoTargets) {
      for (const vid of videoList) {
        if (this.isCancelled) break;
        const target = videoTargets.get(Number(vid.id));
        if (!target) continue;
        await this.downloadSingleVideo(ctx, albumsDir, metaVideosDir, dirCache, vid, target);
      }
    }

    async processChildDocs(ctx, childDir, metadataDir) {
      const docsDir = await childDir.getDirectoryHandle("Documents", { create: true });
      const docs = await this.loadOrFetchDocs(ctx, metadataDir);
      const usedDocNames = new Map();
      for (let dIdx = 0; dIdx < docs.length; dIdx++) {
        if (this.isCancelled) break;
        const doc = docs[dIdx];
        const durl = doc.url || doc.file_url || doc.document_url || doc.download_url;
        if (!durl) continue;
        const did = sanitizeName(String(doc.id || dIdx + 1), 32);
        let dname = sanitizeName(doc.name || doc.title || `doc_${did}`, 60);
        const ext = (doc.extension || "pdf").replace(/^\./, "");
        if (dname.toLowerCase().endsWith("." + ext.toLowerCase()) && dname.length > ext.length + 1) {
          dname = dname.substring(0, dname.length - (ext.length + 1)).replace(/[. ]+$/, "") || `doc_${did}`;
        }
        let fname = `${dname}.${ext}`;
        const prevDid = usedDocNames.get(fname.toLowerCase());
        if (prevDid !== undefined && prevDid !== did) {
          fname = `${dname}_${did}.${ext}`;
        }
        usedDocNames.set(fname.toLowerCase(), did);

        // Documents must match expected size or at least 100 bytes
        const expectedBytes = typeof doc.size === "number" && doc.size > 0 ? doc.size : 100;
        const exists = await this.checkFileExists(docsDir, fname, expectedBytes);
        if (exists) continue;

        try {
          this.ui.setActivity(`📄 Downloading document: ${fname}...`);
          const res = await this.streamBinaryDirect(durl, docsDir, fname, expectedBytes);
          if (res.status === "downloaded") {
            this.stats.docsSaved++;
            this.stats.bytesSaved += res.bytes;
          }
        } catch (_) {
          try { await docsDir.removeEntry(fname); } catch (_) {}
        }
        this.refreshLiveUI(ctx.child_name);
      }
      if (this.stats.docsSaved > 0) {
        this.ui.log(`📄 Saved ${this.stats.docsSaved} documents.`);
      }
    }

    async loadOrFetchDocs(ctx, metadataDir) {
      const finalIndex = await this.readJsonIfExists(metadataDir, "documents_index.json");
      if (Array.isArray(finalIndex) && finalIndex.length > 0) {
        this.ui.log(`📄 Found ${finalIndex.length} documents. Downloading...`);
        return finalIndex;
      }

      const partialState = await this.readJsonIfExists(metadataDir, "documents_index_partial.json");
      let docs = [];
      const seen = new Set();
      let completedEntities = new Set();

      if (partialState && Array.isArray(partialState.items)) {
        docs = partialState.items;
        for (const d of docs) {
          if (d.id) seen.add(d.id);
        }
        if (Array.isArray(partialState.completed_entities)) {
          completedEntities = new Set(partialState.completed_entities);
        }
        this.ui.log(`📄 Resuming documents from disk...`);
      }

      const entities = [
        ["documents", 0],
        ["child-portfolio-folders", ctx.child_id],
        ["children_public", ctx.child_id],
      ];

      for (const [entityType, entityId] of entities) {
        if (this.isCancelled) break;
        if (completedEntities.has(entityType)) continue;

        let page = 1;
        while (!this.isCancelled) {
          while (this.isPaused && !this.isCancelled) await sleep(500);
          if (this.isCancelled) break;

          const query = new URLSearchParams({
            entity_type: entityType,
            entity_id: String(entityId),
            items_per_page: "30",
            page: String(page),
          });
          let resp;
          try {
            resp = await this.client.httpJson(`/data/documents?${query}`, {
              headers: {
                "X-KINDERGARTEN-ID": String(ctx.kindergarten_id),
                "X-CHILD-ID": String(ctx.child_id),
              },
            });
          } catch (_) {
            break;
          }

          const payload = resp.response || resp.result || resp;
          let items = [];
          if (payload && Array.isArray(payload.documents)) {
            items = payload.documents;
          } else if (payload && Array.isArray(payload.files)) {
            items = payload.files;
          } else if (payload && Array.isArray(payload.items)) {
            items = payload.items;
          } else if (Array.isArray(payload)) {
            items = payload;
          }
          if (!items.length) break;

          let added = 0;
          for (const item of items) {
            const did = item.id;
            if (did && !seen.has(did)) {
              seen.add(did);
              docs.push(item);
              added++;
            }
          }

          if (items.length < 30 || added === 0) break;
          page++;
        }

        completedEntities.add(entityType);
        await this.writeJsonAtomic(metadataDir, "documents_index_partial.json", {
          complete: false,
          completed_entities: Array.from(completedEntities),
          total_docs: docs.length,
          updated_at: new Date().toISOString(),
          items: docs,
        });

        this.ui.updateDiscovery({
          childName: ctx.child_name,
          itemType: "Documents",
          page: 1,
          foundCount: docs.length,
          subCount: null,
          subLabel: null,
        });
      }

      if (!this.isCancelled) {
        await this.writeJsonAtomic(metadataDir, "documents_index.json", docs);
        await this.writeJsonAtomic(metadataDir, "documents_index_partial.json", {
          complete: true,
          completed_entities: Array.from(completedEntities),
          total_docs: docs.length,
          updated_at: new Date().toISOString(),
          items: docs,
        });
      }

      return docs;
    }

    async startBackup() {
      this.ui.setActivity("🚀 Connecting to Kinderpedia...");
      this.ui.log("🚀 Starting Kinderpedia backup...");

      let contexts = this.client.childrenContexts;
      if (!contexts || !contexts.length) {
        await this.client.checkActiveSession();
        contexts = this.client.childrenContexts;
      }
      const childFolderMap = resolveChildFolderNames(contexts);
      this.ui.log(
        `Found children: ${contexts
          .map((c) => `${c.child_name} (→ ${childFolderMap.get(`${c.child_id}_${c.kindergarten_id}`)})`)
          .join(", ")}`
      );

      this.ui.startDownloadingUI();
      const rootMetadataDir = await this.rootDirHandle.getDirectoryHandle("Metadata", { create: true });

      for (const ctx of contexts) {
        if (this.isCancelled) break;
        const childFolderName =
          childFolderMap.get(`${ctx.child_id}_${ctx.kindergarten_id}`) ||
          sanitizeName(getChildShortName(ctx), 60);
        this.ui.log(`📂 Starting backup for ${ctx.child_name} (folder "${childFolderName}")...`);
        const childDir = await this.rootDirHandle.getDirectoryHandle(childFolderName, { create: true });
        const albumsDir = await childDir.getDirectoryHandle("Albums", { create: true });
        const metadataDir = await rootMetadataDir.getDirectoryHandle(childFolderName, { create: true });
        const metaAlbumsDir = await metadataDir.getDirectoryHandle("Albums", { create: true });
        const dirCache = new Map();
        const folderMap = await this.loadFolderMap(metadataDir, metaAlbumsDir, albumsDir);

        // Avatar (downloaded in background if present)
        if (ctx.avatar_url) {
          this.streamBinaryDirect(ctx.avatar_url, childDir, "avatar.png").catch(() => {});
        }

        // Shared stats for live HUD telemetry
        this.stats = {
          photosSaved: 0,
          photosSkipped: 0,
          bytesSaved: 0,
          totalExpectedPhotos: 0,
          totalExpectedAlbums: 0,
          albumsDone: 0,
          videosSaved: 0,
          videosSkipped: 0,
          totalVideos: 0,
          docsSaved: 0,
          t0: performance.now(),
        };

        let resolveAlbumsCatalog;
        const albumsCatalogPromise = new Promise((r) => {
          resolveAlbumsCatalog = r;
        });

        // MAXIMUM CONCURRENCY: Photos, Videos (matched into day albums), and Docs run in parallel!
        await Promise.all([
          this.processChildPhotos(ctx, albumsDir, metadataDir, metaAlbumsDir, dirCache, resolveAlbumsCatalog, folderMap),
          this.processChildVideos(ctx, albumsDir, metadataDir, dirCache, albumsCatalogPromise, folderMap),
          this.processChildDocs(ctx, childDir, metadataDir),
        ]);

        this.ui.log(`🎉 Backup complete for ${childFolderName}!`);
      }

      this.ui.log(this.isCancelled ? "🛑 Backup cancelled." : "🎉 All backups complete!");
      this.ui.setActivity(this.isCancelled ? "🛑 Backup stopped." : "🎉 Backup complete! All files saved directly to disk.");
      this.refreshLiveUI("Completed");
    }
  }

  // Initialize and mount instance
  window.__kinderpedia_backup_instance = new BackupEngine();
})();
