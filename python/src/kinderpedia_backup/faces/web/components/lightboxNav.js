// Lightbox navigation: per-context item lists (batch / gallery / child-cluster / unnamed cluster), photo
// de-duplication with "seen" memory, the windowed thumbnail strip, prev/next, and the quick-decision rows.
import { clusterAction, decideCard } from "../actions.js";
import { post } from "../api.js";
import { resolvePersonName, setOverview, state, stripWindow, subscribe, wrapIndex } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml } from "../util/dom.js";
import { plural, thumbSrc } from "../util/format.js";
import { childFaceStatusMeta, renderChildClusters } from "../views/childClusters.js";
import { assignCluster } from "../views/clusters.js";
import { closePhotoModal, openPhotoModal } from "./lightbox.js";
import { openReassignModal } from "./reassignModal.js";
import { showToast } from "./toasts.js";

export const MODAL_STRIP_RADIUS = 30; // the strip shows ±30 items around the current one
const SEEN_STORAGE_KEY = "kpSeenClusterPhotoIds";
const BATCH_SYM = { approved: "✓", unsure: "?", reassigned: "↪", disproved: "✗" };

export function getModalNavItems(ctx) {
  if (!ctx) return [];
  if (ctx.type === "cluster") return ctx.faces || [];
  if (ctx.type === "batch") return state.batchCards;
  if (ctx.type === "gallery") return ctx.items || [];
  if (ctx.type === "album") return ctx.photos || [];
  return [];
}

// ---------------------------------------------------------------- seen photos (one stop per photo)
export function loadSeenPhotoIds() {
  try {
    const arr = JSON.parse(window.localStorage.getItem(SEEN_STORAGE_KEY) || "[]");
    return new Set(Array.isArray(arr) ? arr.map(Number).filter(Number.isFinite) : []);
  } catch (_e) {
    return new Set();
  }
}

let seenSaveTimer = null;
function saveSeenPhotoIds() {
  if (seenSaveTimer) clearTimeout(seenSaveTimer);
  seenSaveTimer = setTimeout(() => {
    try {
      window.localStorage.setItem(SEEN_STORAGE_KEY, JSON.stringify(Array.from(state.seenPhotoIds)));
    } catch (_e) {
      // Storage full / disabled: de-dup still works for this session.
    }
  }, 300);
}

export function markPhotoSeen(photoId) {
  const pid = Number(photoId);
  if (!Number.isFinite(pid) || state.seenPhotoIds.has(pid)) return;
  state.seenPhotoIds.add(pid);
  saveSeenPhotoIds();
}

export const isPhotoSeen = (photoId) => state.seenPhotoIds.has(Number(photoId));

/**
 * Navigable view of a de-duplicated context (pure): one item per photo (preferring the anchor face),
 * skipping photos in `seen` unless `showSeen`. The anchor photo and `keepPhotoIds` are always kept.
 */
export function buildDedupedNavView(rawItems, anchorFaceId, anchorPhotoId, { seen, showSeen, keepPhotoIds = null }) {
  const byPhoto = new Map();
  for (const it of rawItems || []) {
    const ph = Number(it.photo_id);
    if (!byPhoto.has(ph) || (anchorFaceId && it.face_id === anchorFaceId)) byPhoto.set(ph, it);
  }
  const view = [];
  let hiddenSeen = 0;
  for (const [ph, it] of byPhoto) {
    const kept = ph === Number(anchorPhotoId) || (keepPhotoIds && keepPhotoIds.has(ph));
    if (!kept && !showSeen && seen.has(ph)) hiddenSeen += 1;
    else view.push(it);
  }
  return { view, hiddenSeen, duplicates: (rawItems || []).length - byPhoto.size };
}

export function applyDedupedNav(ctx, anchorFaceId, anchorPhotoId, keepCurrentView = false) {
  const keepPhotoIds = keepCurrentView ? new Set(getModalNavItems(ctx).map((it) => Number(it.photo_id))) : null;
  const raw = ctx.type === "cluster" ? ctx.rawFaces : ctx.rawItems;
  const res = buildDedupedNavView(raw, anchorFaceId, anchorPhotoId, {
    seen: state.seenPhotoIds,
    showSeen: state.showSeenPhotos,
    keepPhotoIds,
  });
  if (ctx.type === "cluster") ctx.faces = res.view;
  else ctx.items = res.view;
  ctx.hiddenSeen = res.hiddenSeen;
  ctx.duplicates = res.duplicates;
  ctx.index = Math.max(0, res.view.findIndex((it) => Number(it.photo_id) === Number(anchorPhotoId)));
}

function dedupeNavNote(ctx) {
  if (!ctx || !ctx.dedupe) return "";
  const parts = [];
  if (ctx.hiddenSeen > 0) parts.push(`${ctx.hiddenSeen} seen skipped`);
  if (ctx.duplicates > 0) parts.push(`${plural(ctx.duplicates, "repeat")} merged`);
  return parts.length ? ` · ${parts.join(" · ")}` : "";
}

function syncSeenToggleButtons(ctx) {
  const toggleBtn = el("btnToggleSeenPhotos");
  const resetBtn = el("btnResetSeenPhotos");
  if (!toggleBtn || !resetBtn) return;
  const active = Boolean(ctx && ctx.dedupe);
  toggleBtn.hidden = !active || (!state.showSeenPhotos && !(ctx.hiddenSeen > 0));
  toggleBtn.textContent = state.showSeenPhotos ? "Hide seen" : `Show ${ctx && ctx.hiddenSeen ? ctx.hiddenSeen : ""} seen`;
  resetBtn.hidden = !active || state.seenPhotoIds.size === 0;
}

function rebuildCurrentDedupedNav() {
  const ctx = state.modalNavContext;
  if (!ctx || !ctx.dedupe) return;
  const cur = getModalNavItems(ctx)[ctx.index];
  applyDedupedNav(ctx, cur ? cur.face_id : null, cur ? cur.photo_id : state.modalPhoto && state.modalPhoto.photo_id);
  renderModalNavControls();
}

export async function jumpToModalNavIndex(newIndex) {
  const ctx = state.modalNavContext;
  const items = getModalNavItems(ctx);
  if (!ctx || !items.length) return;
  ctx.index = wrapIndex(newIndex, items.length);
  const item = items[ctx.index];
  let highlightFaceId = item.face_id || null;
  if (!highlightFaceId && ctx.filterPersonId && Array.isArray(item.faces)) {
    const match = item.faces.find((f) => f.person_id === ctx.filterPersonId);
    if (match) highlightFaceId = match.face_id;
  }
  await openPhotoModal(item.photo_id, highlightFaceId, true);
}

// ---------------------------------------------------------------- rendering
function stripItemHtml(item, idx, ctx) {
  let stateCls = "";
  let badge = "";
  if (ctx.type === "batch") {
    stateCls = `state-${item.decision || "approved"}`;
    badge = BATCH_SYM[item.decision] || "✗";
  } else if (ctx.ccKey && item.pressure) {
    stateCls = item.pressure > 0 ? "state-approved" : "state-disproved";
    badge = item.pressure > 0 ? "▲" : "▼";
  }
  return html`
    <button type="button" class="modal-cluster-thumb-btn ${stateCls} ${idx === ctx.index ? "active" : ""}" data-cidx="${idx}" title="${item.album_date || item.year || ""} · ${item.album_title || ""}">
      <img src="${thumbSrc(item)}" alt="" loading="lazy" />
      ${badge && html`<span class="modal-strip-badge">${badge}</span>`}
    </button>`;
}

function childClusterQuickRowHtml(cur, who) {
  const pr = Number(cur.pressure || 0);
  return html`
    <span class="modal-quick-label">This face for ${who || "this child"}: <strong>${childFaceStatusMeta(cur.status).label}</strong>${pr > 0 ? " · ▲ boosted" : pr < 0 ? " · ▼ penalized" : ""}</span>
    <button type="button" class="btn btn-tonal btn-xs ${pr > 0 ? "is-selected" : ""}" data-ccact="boost" title="Good example: nudge this face and look-alikes up (B)">▲ Boost <kbd>B</kbd></button>
    <button type="button" class="btn btn-tonal btn-xs ${pr < 0 ? "is-selected" : ""}" data-ccact="penalize" title="Doubtful: nudge this face and look-alikes down (N)">▼ Penalize <kbd>N</kbd></button>
    ${pr !== 0 && html`<button type="button" class="btn btn-ghost btn-xs" data-ccact="clear" title="Remove boost / penalize">Clear</button>`}
    <button type="button" class="btn btn-ghost btn-xs" data-ccact="exclude" title="Not ${who || "this child"}: remove and use as a hard negative">⊘ Exclude</button>`;
}

function batchQuickRowHtml(card, who) {
  const sel = (d) => (card.decision === d ? "is-selected" : "");
  return html`
    <button type="button" class="btn btn-tonal btn-xs ${sel("approved")}" data-bdec="approved">✓ ${who || "Approve"}</button>
    <button type="button" class="btn btn-tonal btn-xs ${sel("disproved")}" data-bdec="disproved">✗ Not ${who || "match"}</button>
    <button type="button" class="btn btn-ghost btn-xs ${sel("unsure")}" data-bdec="unsure">? Unsure</button>
    <button type="button" class="btn btn-ghost btn-xs ${sel("reassigned")}" data-bdec="reassign">↪ Another…</button>`;
}

export function renderModalNavControls() {
  const ctx = state.modalNavContext;
  const navEl = el("modalClusterNav");
  const boxEl = el("modalClusterBox");
  const quickEl = el("modalBatchQuickRow");
  const items = getModalNavItems(ctx);
  if (!ctx || !items.length) {
    navEl.hidden = true;
    boxEl.hidden = true;
    syncSeenToggleButtons(null);
    return;
  }
  syncSeenToggleButtons(ctx);
  navEl.hidden = false;
  boxEl.hidden = false;
  el("btnClusterPrev").disabled = items.length <= 1;
  el("btnClusterNext").disabled = items.length <= 1;
  const isCluster = ctx.type === "cluster";
  const isBatch = ctx.type === "batch";
  const isAlbum = ctx.type === "album";
  const isChildCluster = ctx.type === "gallery" && Boolean(ctx.ccKey);
  el("btnClusterExcludeFace").hidden = !isCluster;
  el("modalClusterAssignForm").hidden = !isCluster;
  quickEl.hidden = !(isBatch || isChildCluster);
  const who = ctx.personId ? resolvePersonName(ctx.personId, ctx.personName || "") : ctx.personName || "";
  const pos = `${ctx.index + 1} of ${items.length}${who ? ` (${who})` : ""}`;
  if (isCluster) {
    el("modalClusterPos").textContent = `${ctx.index + 1} / ${items.length} photos in cluster${dedupeNavNote(ctx)}`;
    el("modalClusterTitle").textContent = `${ctx.archiveChild} · ${ctx.year} (${ctx.photoCount} photos)`;
  } else if (isAlbum) {
    const filterNote = ctx.filterPersonName ? ` · Filter: ${ctx.filterPersonName}` : "";
    const label = `Album Photo ${ctx.index + 1} of ${items.length}${filterNote}`;
    el("modalClusterPos").textContent = label;
    el("modalClusterTitle").textContent = ctx.albumTitle || label;
  } else {
    const label = `${isBatch ? "Batch Photo" : ctx.navLabel || "Gallery Photo"} ${pos}`;
    el("modalClusterPos").textContent = `${label}${dedupeNavNote(ctx)}`;
    el("modalClusterTitle").textContent = label;
  }

  // Window the strip: rebuilding one <img> per item on every arrow key made long contexts sluggish.
  const { start, end } = stripWindow(ctx.index, items.length, MODAL_STRIP_RADIUS);
  const thumbsEl = el("modalClusterThumbs");
  setHtml(
    thumbsEl,
    html`${start > 0 && html`<span class="modal-strip-more" title="Earlier photos">… ${start} more</span>`}${items
      .slice(start, end)
      .map((item, rel) => stripItemHtml(item, start + rel, ctx))}${end < items.length &&
    html`<span class="modal-strip-more" title="Later photos">${items.length - end} more …</span>`}`
  );
  const cur = items[ctx.index];
  if (isChildCluster) setHtml(quickEl, cur ? childClusterQuickRowHtml(cur, who) : html``);
  if (isBatch) setHtml(quickEl, cur ? batchQuickRowHtml(cur, who) : html``);
  const activeBtn = thumbsEl.querySelector(".modal-cluster-thumb-btn.active");
  if (activeBtn && typeof activeBtn.scrollIntoView === "function") activeBtn.scrollIntoView({ block: "nearest", inline: "center" });
}

// ---------------------------------------------------------------- quick actions
/** Child-cluster lightbox: boost / penalize (toggle) / clear / exclude the current face. */
export function applyModalChildClusterAction(action, btn = null) {
  const ctx = state.modalNavContext;
  if (!ctx || ctx.type !== "gallery" || !ctx.ccKey || !ctx.personId) return;
  const cur = getModalNavItems(ctx)[ctx.index];
  if (!cur) return;
  const curPr = Number(cur.pressure || 0);
  // Pressing Boost on a boosted face (or Penalize on a penalized one) toggles it off:
  const act = (action === "boost" && curPr > 0) || (action === "penalize" && curPr < 0) ? "clear" : action;
  runBusy(
    btn,
    async () => {
      await clusterAction(ctx.personId, act, [{ photo_id: cur.photo_id, face_id: cur.face_id }]);
      state.modalDirty = true;
      if (act === "exclude") {
        for (const arr of [ctx.rawItems, ctx.items]) {
          const i = Array.isArray(arr) ? arr.findIndex((f) => f.face_id === cur.face_id) : -1;
          if (i >= 0) arr.splice(i, 1);
        }
        if (!ctx.items.length) return closePhotoModal();
        return jumpToModalNavIndex(Math.min(ctx.index, ctx.items.length - 1));
      }
      cur.pressure = act === "boost" ? 1 : act === "penalize" ? -1 : 0;
      renderModalNavControls();
      return undefined;
    },
    { context: "Update face pressure" }
  );
}

function applyBatchQuickDecision(act) {
  const ctx = state.modalNavContext;
  const card = state.batchCards[ctx ? ctx.index : -1];
  if (!ctx || ctx.type !== "batch" || !card) return;
  if (act === "reassign") {
    openReassignModal({
      photoId: card.photo_id,
      faceId: card.face_id || null,
      fromPersonId: ctx.personId || null,
      fromPersonName: resolvePersonName(ctx.personId, ctx.personName || ""),
      initialName: resolvePersonName(card.targetPersonId, card.targetName || ""),
    });
    return;
  }
  decideCard(ctx.index, act);
}

/** Unnamed-cluster lightbox: drop the current photo from the cluster. */
async function excludeCurrentClusterFace() {
  const ctx = state.modalNavContext;
  const curItem = ctx && ctx.type === "cluster" ? ctx.faces[ctx.index] : null;
  if (!curItem) return;
  const res = await post(`/api/clusters/${ctx.clusterId}/remove-face`, { face_id: curItem.face_id });
  state.modalDirty = true;
  setOverview(res.overview);
  if (!res.cluster || !res.cluster.faces || !res.cluster.faces.length) {
    await closePhotoModal();
    showToast("Excluded photo — cluster is now empty.");
    return;
  }
  ctx.rawFaces = res.cluster.faces;
  ctx.photoCount = res.cluster.photo_count;
  applyDedupedNav(ctx, null, curItem.photo_id, true);
  if (!ctx.faces.length) {
    await closePhotoModal();
    showToast("Excluded photo — no unseen photos left in this cluster.");
    return;
  }
  const stillThere = ctx.faces.findIndex((it) => Number(it.photo_id) === Number(curItem.photo_id));
  await jumpToModalNavIndex(stillThere >= 0 ? stillThere : Math.min(ctx.index, ctx.faces.length - 1));
  showToast("Excluded photo from cluster — child will stay uncategorized when confirming.");
}

export function stepModal(delta) {
  if (!state.modalNavContext) return;
  runBusy(el(delta < 0 ? "btnClusterPrev" : "btnClusterNext"), () => jumpToModalNavIndex(state.modalNavContext.index + delta), {
    context: delta < 0 ? "Previous photo" : "Next photo",
  });
}

export function bindLightboxNav() {
  subscribe("batch-cards", () => {
    const ctx = state.modalNavContext;
    if (ctx && ctx.type === "batch" && !el("photoModal").hidden) renderModalNavControls();
  });
  delegate("modalClusterThumbs", "click", ".modal-cluster-thumb-btn", (_ev, btn) =>
    runBusy(btn, () => jumpToModalNavIndex(Number(btn.getAttribute("data-cidx"))), { context: "Load photo" })
  );
  delegate("modalBatchQuickRow", "click", "[data-ccact]", (_ev, btn) => applyModalChildClusterAction(btn.getAttribute("data-ccact"), btn));
  delegate("modalBatchQuickRow", "click", "[data-bdec]", (_ev, btn) => applyBatchQuickDecision(btn.getAttribute("data-bdec")));
  on("btnClusterPrev", "click", () => stepModal(-1));
  on("btnClusterNext", "click", () => stepModal(1));
  on("btnToggleSeenPhotos", "click", () => {
    state.showSeenPhotos = !state.showSeenPhotos;
    rebuildCurrentDedupedNav();
  });
  on("btnResetSeenPhotos", "click", () => {
    const n = state.seenPhotoIds.size;
    if (!n || !window.confirm(`Forget ${plural(n, "reviewed photo")}? They will show up again in clusters.`)) return;
    state.seenPhotoIds = new Set();
    saveSeenPhotoIds();
    rebuildCurrentDedupedNav();
    if (state.mode === "childclusters") renderChildClusters();
    showToast("Reviewed-photo history cleared.");
  });
  on("btnClusterExcludeFace", "click", (_ev, btn) =>
    runBusy(btn, () => excludeCurrentClusterFace(), { busyText: "Excluding…", context: "Exclude photo from cluster" })
  );
  on("modalClusterAssignForm", "submit", (ev, form) => {
    ev.preventDefault();
    const ctx = state.modalNavContext;
    if (!ctx || ctx.type !== "cluster") return;
    const inp = el("modalClusterNameInput");
    const name = inp.value.trim();
    if (!name) return;
    runBusy(
      form.querySelector('button[type="submit"]'),
      async () => {
        await assignCluster(ctx.clusterId, name, "classmate", true);
        state.modalDirty = false;
        await closePhotoModal();
      },
      { busyText: "Confirming…", alsoDisable: [inp, el("btnClusterExcludeFace")], context: "Confirm cluster" }
    );
  });
}

