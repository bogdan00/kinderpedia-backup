// THE implementation of every face-label action, used by all views (batch grid, lightbox, gallery,
// child clusters, reassign picker):
//   persisted:  tagFace / reassignFace / markFaceUnsure / rejectFace / untagFace / tagManualBox / clusterAction
//               (boost · penalize · clear · exclude)
//   local:      decideCard / decideCardClick / toggleAllCards (approve · disprove · unsure, sent on batch submit)
// After a persisted change the shared fan-out updates the overview, every matching batch card, the open
// lightbox photo, and notifies views ("faces-changed") — no view re-implements any of it.
import { post } from "./api.js";
import { showToast } from "./components/toasts.js";
import {
  applyLabelToCards,
  decideRange,
  emit,
  findPersonByName,
  resolvePersonName,
  setBatchCards,
  setOverview,
  state,
  toggleAll,
  withDecision,
} from "./state.js";
import { plural } from "./util/format.js";

/** Shared fan-out after the server persisted `change` = { photoId, faceId, personId, personName, status }. */
function afterLabelChange(res, change) {
  setOverview(res.overview);
  const personName = resolvePersonName(change.personId, change.personName || "");
  const full = { ...change, faceId: change.faceId || res.face_id || null, personName };
  const { cards, changed } = applyLabelToCards(state.batchCards, full, state.selectedPersonId);
  if (changed.length) setBatchCards(cards, changed);
  if (res.photo && state.modalPhoto && state.modalPhoto.photo_id === change.photoId) {
    state.modalPhoto = res.photo;
    state.modalDirty = true;
    emit("modal-photo");
  }
  emit("faces-changed", full);
  return personName;
}

/** POST /api/faces/tag. `personId` wins; else `name` resolves to an existing person or creates one. */
async function postTag({ photoId, faceId = null, personId = null, name = "", replacePersonId = null, status }) {
  const clean = String(name || "").trim();
  const existing = personId ? null : findPersonByName(clean);
  const body = { photo_id: photoId, face_id: faceId || null, replace_person_id: replacePersonId || null, status };
  const pid = personId || (existing && existing.person_id) || null;
  if (pid) body.person_id = pid;
  else body.new_name = clean;
  const res = await post("/api/faces/tag", body);
  const personName = afterLabelChange(res, { photoId, faceId, personId: res.person_id || pid, personName: clean, status });
  return { res, personName };
}

/** Confirm a face as a person (lightbox name field). */
export async function tagFace({ photoId, faceId, personId = null, name = "" }) {
  const { res, personName } = await postTag({ photoId, faceId, personId, name, status: "confirmed" });
  showToast(`Tagged ${personName} and synced Batch Trainer & archive!`);
  return res;
}

/** Move a face from `fromPersonId` to another (existing or new) person — batch grid, gallery, lightbox. */
export async function reassignFace({ photoId, faceId, fromPersonId = null, targetPersonId = null, targetName = "" }) {
  const { res, personName } = await postTag({
    photoId,
    faceId,
    personId: targetPersonId,
    name: targetName,
    replacePersonId: fromPersonId,
    status: "confirmed",
  });
  showToast(`Reassigned to ${personName} — face tag & gallery updated!`);
  return res;
}

/** Mark a child face to ignore (don't tag across archive, acts as background veto in matcher). */
export async function ignoreFace({ photoId, faceId, fromPersonId = null }) {
  const res = await post("/api/faces/ignore", {
    photo_id: photoId,
    face_id: faceId || null,
    from_person_id: fromPersonId || null,
  });
  afterLabelChange(res, {
    photoId,
    faceId: res.face_id || faceId || null,
    personId: res.person_id,
    personName: "Ignored",
    status: "confirmed",
  });
  showToast("Marked child face to ignore — muted across archive.");
  return res;
}

/** "I'm not sure" for `personId` (excluded from training, no hard negative). */
export async function markFaceUnsure({ photoId, faceId, personId, personName = "" }) {
  const { res } = await postTag({ photoId, faceId, personId, name: personName, status: "unsure" });
  showToast(`Marked as "I'm not sure" (?) for ${resolvePersonName(personId, personName)}.`);
  return res;
}

/** Not `personId`: remove and learn as a hard negative. */
export async function rejectFace({ photoId, faceId, personId, personName = "" }) {
  const { res } = await postTag({ photoId, faceId, personId, name: personName, status: "rejected" });
  showToast(`Removed from ${resolvePersonName(personId, personName)} and added to hard negatives.`);
  return res;
}

/** Drop whatever label the face has (it becomes uncategorized; manual boxes are deleted). */
export async function untagFace({ photoId, faceId, prevName = "" }) {
  const res = await post("/api/faces/untag", { photo_id: photoId, face_id: faceId });
  afterLabelChange(res, { photoId, faceId, personId: null, personName: "", status: "rejected" });
  showToast(`Removed tag${prevName ? ` (${prevName})` : ""} — child is now uncategorized.`);
  return res;
}

/** Tag a hand-drawn box (lightbox) as an existing or new person. */
export async function tagManualBox({ photoId, box, personId = null, name = "" }) {
  const clean = String(name || "").trim();
  const existing = personId ? null : findPersonByName(clean);
  const pid = personId || (existing && existing.person_id) || null;
  const body = { photo_id: photoId, box };
  if (pid) body.person_id = pid;
  else body.new_name = clean;
  const res = await post("/api/faces/manual", body);
  state.manualDrawMode = false;
  state.manualDraftBox = null;
  state.modalHighlightFaceId = res.face_id;
  const personName = afterLabelChange(res, {
    photoId,
    faceId: res.face_id,
    personId: res.person_id,
    personName: clean,
    status: "confirmed",
  });
  showToast(`Tagged ${personName} (manual box) — crop & ByChild/ export ready!`);
  return res;
}

export const CLUSTER_ACTIONS = ["boost", "penalize", "clear", "exclude"];

/** Toast for a child-cluster action on `n` faces of `name` (pure). */
export function clusterActionMessage(action, n, name) {
  const faces = plural(n, "face");
  return (
    {
      boost: `▲ Boosted ${faces} for ${name} — look-alikes will rank higher after the background rematch.`,
      penalize: `▼ Penalized ${faces} for ${name} — look-alikes will rank lower after the background rematch.`,
      clear: `Cleared pressure on ${faces} for ${name}.`,
      exclude: `⊘ Excluded ${faces} from ${name} (hard negatives).`,
    }[action] || `Applied to ${faces}.`
  );
}

/** Boost / penalize / clear pressure / exclude `faces` ([{photo_id, face_id}]) for `personId`. Returns applied count. */
export async function clusterAction(personId, action, faces) {
  if (!CLUSTER_ACTIONS.includes(action)) throw new Error(`Unknown cluster action ${action}`);
  const res = await post(`/api/people/${personId}/cluster-action`, { action, faces });
  setOverview(res.overview);
  const n = typeof res.applied === "number" ? res.applied : faces.length;
  emit("faces-changed", { personId, action, faces });
  showToast(clusterActionMessage(action, n, resolvePersonName(personId, "")));
  return n;
}

// ---------------------------------------------------------------- local batch decisions (sent on submit)
/** Set one card's decision (approve / disprove / unsure). */
export function decideCard(idx, decision) {
  if (!state.batchCards[idx]) return;
  const cards = state.batchCards.slice();
  cards[idx] = withDecision(cards[idx], decision);
  state.batchUserTouched = true;
  setBatchCards(cards, [idx]);
}

/** Card click: toggle ✓/✗ (shift: the whole range since the last click takes the new decision). */
export function decideCardClick(idx, shiftKey) {
  const card = state.batchCards[idx];
  if (!card) return;
  const decision = card.decision === "approved" ? "disproved" : "approved";
  const from = shiftKey && state.lastClickedBatchIdx !== null ? state.lastClickedBatchIdx : idx;
  const { cards, changed } = decideRange(state.batchCards, from, idx, decision);
  state.batchUserTouched = true;
  state.lastClickedBatchIdx = idx;
  setBatchCards(cards, changed);
}

/** "Toggle all ✓/✗" (kind "approve") or "all ?" (kind "unsure"). */
export function toggleAllCards(kind) {
  const next = toggleAll(state.batchCards, kind);
  if (next !== state.batchCards) setBatchCards(next, null);
}
