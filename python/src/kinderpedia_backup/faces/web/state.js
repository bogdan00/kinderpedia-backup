// Shared UI state: one store object, a tiny topic pub/sub, selectors, and pure reducers (no DOM here, so
// everything below is unit-tested under `node --test`).

export const state = {
  overview: null,
  mode: "batch", // "batch" | "childclusters" | "clusters" | "gallery"
  selectedPersonId: null,
  batchMeta: null,
  batchCards: [], // { ...card, decision, targetName, targetPersonId, alreadySavedInModal }
  lastClickedBatchIdx: null,
  batchUserTouched: false,
  reassignContext: null,
  seenPhotoIds: new Set(),
  showSeenPhotos: false,
  bgStatus: null,
  bgStatusAt: 0,
  galleryOffset: 0,
  galleryLimit: 96,
  galleryItems: [], // items of the rendered gallery page (lightbox nav context)
  gallerySig: "", // signature of the rendered gallery page (skip no-op re-renders)
  pendingJobs: new Map(), // job_id -> job started from this tab
  handledJobIds: new Set(),
  modalPhoto: null,
  modalHighlightFaceId: null,
  modalNavContext: null,
  modalDirty: false,
  manualDrawMode: false,
  manualDraftBox: null,
  renameTargetPersonId: null,
  backendOffline: false,
  rematchPending: false,
  rematchRunning: false,
  sseSource: null,
  childClusters: null, // last GET /api/people/<pid>/clusters payload
  childClustersPersonId: null,
  childClustersExpanded: new Set(), // cluster_key strings
  childClustersSelected: new Map(), // cluster_key -> Set(face_id)
  classesMatrix: null,
  classesMembers: new Map(), // class_id -> Set(person_id) (local working copy)
  classesDirty: new Set(), // class_ids with unsaved membership edits
  albums: [],
  albumsTotal: 0,
  albumsOffset: 0,
  albumsLimit: 48,
  selectedAlbumKey: null,
  currentAlbumDetail: null,
  albumFilterChild: "",
  albumFilterYear: "",
  albumSearchQuery: "",
  albumPersonFilter: null,
};

// ---------------------------------------------------------------- pub/sub
// Topics: "overview", "batch-cards" (payload: changed indices or null = all), "modal-photo",
// "faces-changed" (a face label was persisted), "backend" (online/offline), "sync" (re-match / jobs / bg status).
const listeners = new Map();

export function subscribe(topic, fn) {
  if (!listeners.has(topic)) listeners.set(topic, new Set());
  listeners.get(topic).add(fn);
  return () => listeners.get(topic).delete(fn);
}

export function emit(topic, payload) {
  for (const fn of Array.from(listeners.get(topic) || [])) fn(payload);
}

export function select(selector) {
  return selector(state);
}

export function setOverview(overview) {
  if (!overview) return;
  state.overview = overview;
  emit("overview");
}

/** Replace the batch (fresh payload) or commit updated cards; listeners re-render `changed` (null = all). */
export function setBatchCards(cards, changed = null) {
  state.batchCards = cards;
  emit("batch-cards", changed);
}

// ---------------------------------------------------------------- selectors
export const people = (s = state) => (s.overview && s.overview.people) || [];

export function getSelectedPerson(s = state) {
  if (!s.selectedPersonId) return null;
  return people(s).find((p) => p.person_id === s.selectedPersonId) || null;
}

export function resolvePersonName(personId, fallbackName = "", s = state) {
  const found = personId ? people(s).find((p) => p.person_id === personId) : null;
  return found ? found.display_name : fallbackName;
}

/** Case-insensitive display-name lookup (optionally ignoring `excludeId`). */
export function findPersonByName(name, excludeId = null, s = state) {
  const key = String(name || "").trim().toLowerCase();
  if (!key) return null;
  return people(s).find((p) => p.person_id !== excludeId && p.display_name.toLowerCase() === key) || null;
}

// ---------------------------------------------------------------- batch reducers (pure)
export const DECISIONS = ["approved", "disproved", "unsure", "reassigned"];

/** Server card -> UI card: the server's `default_decision` (approved / unsure / disproved) is the start state. */
export function cardFromServer(c) {
  const decision = DECISIONS.includes(c.default_decision) ? c.default_decision : "disproved";
  return { ...c, decision, targetName: "", targetPersonId: null, alreadySavedInModal: false };
}

/** A local (not yet submitted) decision: clears any reassignment target / modal-saved flag. */
export function withDecision(card, decision) {
  return { ...card, decision, targetName: "", targetPersonId: null, alreadySavedInModal: false };
}

/** Cards `[a..b]` (either order) set to `decision`; returns { cards, changed }. */
export function decideRange(cards, a, b, decision) {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  const changed = [];
  const next = cards.map((c, i) => {
    if (i < lo || i > hi) return c;
    changed.push(i);
    return withDecision(c, decision);
  });
  return { cards: next, changed };
}

/** "Toggle all ✓/✗" ("approve") or "all ?" ("unsure") over cards not already saved from the lightbox. */
export function toggleAll(cards, kind) {
  const mutable = cards.filter((c) => !c.alreadySavedInModal);
  if (!mutable.length) return cards;
  let next;
  if (kind === "unsure") next = mutable.every((c) => c.decision === "unsure") ? "approved" : "unsure";
  else next = mutable.some((c) => c.decision === "approved") ? "disproved" : "approved";
  return cards.map((c) => (c.alreadySavedInModal ? c : withDecision(c, next)));
}

/**
 * A label persisted on the server (lightbox tag / untag / unsure, reassign) -> matching cards (same face,
 * or same photo for face-less cards) take the outcome and are marked already saved. Returns { cards, changed }.
 */
export function applyLabelToCards(cards, change, selectedPersonId) {
  const { photoId, faceId, personId, personName, status } = change;
  const changed = [];
  const next = cards.map((c, i) => {
    const hit = (faceId && c.face_id === faceId) || (!c.face_id && c.photo_id === photoId);
    if (!hit) return c;
    let out;
    if (status === "unsure") out = withDecision(c, "unsure");
    else if (status === "rejected") out = withDecision(c, "disproved");
    else if (status === "confirmed" && personId === selectedPersonId) out = withDecision(c, "approved");
    else if (status === "confirmed") out = { ...withDecision(c, "reassigned"), targetPersonId: personId, targetName: personName };
    else return c;
    changed.push(i);
    return { ...out, alreadySavedInModal: true, face_id: c.face_id || faceId || null };
  });
  return { cards: next, changed };
}

/** A deleted person can no longer be a reassignment target: such unsaved cards fall back to "unsure". */
export function dropPersonFromCards(cards, personId) {
  return cards.map((c) =>
    c.decision === "reassigned" && c.targetPersonId === personId && !c.alreadySavedInModal
      ? withDecision(c, "unsure")
      : c
  );
}

/** Rename of a reassignment target (matched by id, or by old name for not-yet-created people). */
export function renamePersonInCards(cards, personId, oldName, newName) {
  const old = String(oldName || "").toLowerCase();
  return cards.map((c) =>
    c.decision === "reassigned" &&
    (c.targetPersonId === personId || (c.targetName && c.targetName.toLowerCase() === old))
      ? { ...c, targetPersonId: personId, targetName: newName }
      : c
  );
}

export function batchCounts(cards) {
  const n = { approved: 0, disproved: 0, unsure: 0, reassigned: 0 };
  for (const c of cards) n[DECISIONS.includes(c.decision) ? c.decision : "disproved"] += 1;
  return n;
}

/** POST /api/batch/submit lists (cards already saved from the lightbox are skipped). */
export function batchSubmitLists(cards, resolveName = (pid, name) => name) {
  const lists = { approved: [], disproved: [], unsure: [], reassigned: [] };
  for (const c of cards) {
    if (c.alreadySavedInModal) continue;
    const ref = { photo_id: c.photo_id, face_id: c.face_id };
    if (c.decision === "approved") lists.approved.push(ref);
    else if (c.decision === "unsure") lists.unsure.push(ref);
    else if (c.decision === "reassigned" && c.targetPersonId) lists.reassigned.push({ ...ref, target_person_id: c.targetPersonId });
    else if (c.decision === "reassigned" && c.targetName)
      lists.reassigned.push({ ...ref, new_target_name: resolveName(c.targetPersonId, c.targetName) });
    else lists.disproved.push(ref);
  }
  return lists;
}

// ---------------------------------------------------------------- gallery / lightbox reducers (pure)
/**
 * Removals / re-matches can shrink the list under the current page: an empty page past the end clamps to
 * the last page that still has items. Returns the new offset, or null when no clamp is needed.
 */
export function clampGalleryOffset({ offset, limit, total, itemsLen }) {
  if (itemsLen > 0 || total <= 0 || offset <= 0) return null;
  const lim = Math.max(1, limit);
  return Math.max(0, Math.floor((total - 1) / lim) * lim);
}

/** Visible slice [start, end) of the lightbox strip: ±radius around `index`. */
export function stripWindow(index, length, radius) {
  return { start: Math.max(0, index - radius), end: Math.min(length, index + radius + 1) };
}

/** Wrap-around navigation index. */
export function wrapIndex(i, length) {
  return length ? ((i % length) + length) % length : 0;
}
