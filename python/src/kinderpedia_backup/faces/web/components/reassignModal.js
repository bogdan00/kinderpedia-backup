// Reassign picker: choose an existing child (chip) or type a name; the move itself is actions.reassignFace,
// shared by the batch grid, the gallery and the lightbox.
import { reassignFace } from "../actions.js";
import { people, state } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml } from "../util/dom.js";
import { compareByDisplayName, thumbSrc } from "../util/format.js";

/** ctx = { photoId, faceId, fromPersonId, fromPersonName, initialName } */
export function openReassignModal(ctx) {
  if (!ctx || !ctx.photoId) return;
  state.reassignContext = ctx;
  el("reassignThumb").src = thumbSrc({ face_id: ctx.faceId, photo_id: ctx.photoId });
  const subEl = el("reassignSub");
  if (subEl) {
    subEl.textContent = ctx.fromPersonName
      ? `Move out of ${ctx.fromPersonName} and update this face's tag immediately:`
      : "Pick an existing label below or type a new child name:";
  }
  const choices = people()
    .filter((p) => p.role !== "ignore" && (!ctx.fromPersonId || p.person_id !== ctx.fromPersonId))
    .slice()
    .sort(compareByDisplayName);
  setHtml(
    el("reassignChips"),
    choices.length
      ? html`${choices.map(
          (p) => html`<button type="button" class="reassign-chip" data-pid="${p.person_id}" data-name="${p.display_name}">${p.display_name}</button>`
        )}`
      : html`<span class="reassign-sub">No other children enrolled yet — type a name below!</span>`
  );
  const inp = el("reassignInput");
  inp.value = ctx.initialName || "";
  el("reassignModal").hidden = false;
  setTimeout(() => {
    inp.focus();
    inp.select();
  }, 20);
}

export function closeReassignModal() {
  el("reassignModal").hidden = true;
  state.reassignContext = null;
}

async function applyReassignChoice(targetPid, targetName, triggerBtn) {
  const ctx = state.reassignContext;
  if (!ctx || !ctx.photoId) return closeReassignModal();
  const clean = String(targetName || "").trim();
  if (!clean) return undefined;
  const others = [
    el("reassignInput"),
    el("btnSubmitReassign"),
    el("btnCancelReassign"),
    ...el("reassignChips").querySelectorAll("button"),
  ].filter((node) => node && node !== triggerBtn);
  return runBusy(
    triggerBtn,
    async () => {
      await reassignFace({
        photoId: ctx.photoId,
        faceId: ctx.faceId || null,
        fromPersonId: ctx.fromPersonId || null,
        targetPersonId: targetPid || null,
        targetName: clean,
      });
      closeReassignModal();
    },
    { busyText: "Assigning…", alsoDisable: others, context: "Reassign face" }
  );
}

export function bindReassignModal() {
  delegate("reassignChips", "click", ".reassign-chip", (_ev, chip) =>
    applyReassignChoice(Number(chip.getAttribute("data-pid")), chip.getAttribute("data-name") || "", chip)
  );
  on("reassignForm", "submit", (ev) => {
    ev.preventDefault();
    applyReassignChoice(null, el("reassignInput").value, el("btnSubmitReassign"));
  });
  on("btnCancelReassign", "click", closeReassignModal);
  on("reassignModal", "click", (ev, modal) => {
    if (ev.target === modal) closeReassignModal();
  });
}
