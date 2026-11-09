// Lightbox manual tagging: drag a box around an undetected face, then pick / type who it is.
import { tagManualBox } from "../actions.js";
import { people, state } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml, siblingControls, toElement } from "../util/dom.js";
import { compareByDisplayName } from "../util/format.js";
import { renderModalFaceOverlays } from "./lightbox.js";
import { showToast } from "./toasts.js";

const MIN_BOX_PX = 8;

export function syncManualDrawUi() {
  const stage = el("photoStage");
  const btn = el("btnDrawManualTag");
  const banner = el("photoDrawBanner");
  if (stage) stage.classList.toggle("is-drawing", Boolean(state.manualDrawMode));
  if (btn) {
    btn.classList.toggle("active", Boolean(state.manualDrawMode));
    setHtml(btn, state.manualDrawMode ? html`Cancel Draw <kbd>Esc</kbd>` : html`+ Tag Child <kbd>T</kbd>`);
  }
  if (banner) banner.hidden = !(state.manualDrawMode && !state.manualDraftBox);
}

export function setManualDrawMode(enabled) {
  state.manualDrawMode = Boolean(enabled);
  if (!state.manualDrawMode) state.manualDraftBox = null;
  syncManualDrawUi();
  if (state.modalPhoto && !el("photoModal").hidden) renderModalFaceOverlays();
}

export function cancelManualDraft() {
  state.manualDraftBox = null;
  syncManualDrawUi();
  renderModalFaceOverlays();
}

/** Chip order: the selected child first (unless already confirmed in this photo), people already in it last. */
export function manualChipOrder(allPeople, selectedPersonId, alreadyInPhoto) {
  const rank = (p) => (p.person_id === selectedPersonId && !alreadyInPhoto.has(p.person_id) ? 0 : alreadyInPhoto.has(p.person_id) ? 2 : 1);
  return allPeople
    .filter((p) => p.role !== "ignore")
    .slice()
    .sort((a, b) => rank(a) - rank(b) || compareByDisplayName(a, b));
}

/** Draft box + "who is this?" popover (listeners are delegated on the layer, see bindManualBox). */
export function appendManualDraftPopover(layer) {
  const draft = state.manualDraftBox;
  const detail = state.modalPhoto;
  if (!draft || !detail) return;
  const alreadyInPhoto = new Set(
    (detail.faces || []).filter((f) => f.person_id != null && f.status === "confirmed").map((f) => f.person_id)
  );
  const chips = manualChipOrder(people(), state.selectedPersonId, alreadyInPhoto);
  const boxStyle = `left:${draft.leftPct}%;top:${draft.topPct}%;width:${draft.widthPct}%;height:${draft.heightPct}%`;
  layer.appendChild(toElement(html`<div class="bbox-overlay bbox-draft" style="${boxStyle}"></div>`));
  const pop = toElement(html`
    <div class="manual-tag-popover">
      <div class="manual-popover-head">
        <span>Who is in this box?</span>
        <button type="button" class="manual-popover-close" title="Cancel box (Esc)">✕</button>
      </div>
      ${chips.length &&
      html`<div class="manual-popover-chips">${chips.map(
        (p) =>
          html`<button type="button" class="manual-chip ${p.person_id === state.selectedPersonId && !alreadyInPhoto.has(p.person_id) ? "is-primary" : ""}" data-pid="${p.person_id}" data-name="${p.display_name}">${p.display_name}</button>`
      )}</div>`}
      <form class="manual-popover-form">
        <input type="text" list="knownPeopleDatalist" placeholder="Or type child name…" autocomplete="off" />
        <button type="submit" class="btn btn-primary btn-xs">Tag ✓</button>
      </form>
    </div>`);
  // Below the box (or above it near the bottom), clamped horizontally inside the stage:
  if (draft.heightPct > 70) {
    pop.style.top = "10px";
  } else if (draft.topPct + draft.heightPct > 64) {
    pop.style.bottom = `${Math.max(2, 100 - draft.topPct)}%`;
    pop.style.marginBottom = "6px";
  } else {
    pop.style.top = `${Math.min(92, draft.topPct + draft.heightPct)}%`;
    pop.style.marginTop = "6px";
  }
  layer.appendChild(pop);
  const layerW = layer.clientWidth || 320;
  const popW = pop.offsetWidth || 280;
  const desiredLeftPx = (draft.leftPct / 100) * layerW;
  pop.style.left = `${Math.max(6, Math.min(Math.max(6, layerW - popW - 6), desiredLeftPx))}px`;
  const inp = pop.querySelector("input");
  setTimeout(() => inp.focus({ preventScroll: true }), 15);
}

function submitDraft(personId, name, triggerEl) {
  const draft = state.manualDraftBox;
  const detail = state.modalPhoto;
  if (!draft || !detail || (!personId && !String(name || "").trim())) return;
  runBusy(triggerEl, () => tagManualBox({ photoId: detail.photo_id, box: draft.bbox, personId, name }), {
    busyText: "Tagging…",
    alsoDisable: siblingControls(triggerEl.closest(".manual-tag-popover"), triggerEl, "button, input"),
    context: "Manual face tag",
  });
}

/** Pixel rectangle (layer coords) -> natural-image bbox + percentage draft geometry (pure). */
export function draftFromRect({ minX, maxX, minY, maxY }, layerRect, scale, nat) {
  return {
    bbox: {
      left: Math.max(0, Math.round(minX * scale.x)),
      top: Math.max(0, Math.round(minY * scale.y)),
      right: Math.min(nat.w, Math.round(maxX * scale.x)),
      bottom: Math.min(nat.h, Math.round(maxY * scale.y)),
    },
    leftPct: (100 * minX) / layerRect.width,
    topPct: (100 * minY) / layerRect.height,
    widthPct: (100 * (maxX - minX)) / layerRect.width,
    heightPct: (100 * (maxY - minY)) / layerRect.height,
  };
}

function startDrag(ev, layer, img) {
  ev.preventDefault();
  ev.stopPropagation();
  state.manualDraftBox = null;
  syncManualDrawUi();
  renderModalFaceOverlays();
  const rect = layer.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const startX = clamp(ev.clientX - rect.left, 0, rect.width);
  const startY = clamp(ev.clientY - rect.top, 0, rect.height);
  const draftEl = toElement(html`<div class="bbox-overlay bbox-draft"></div>`);
  layer.appendChild(draftEl);
  const track = (clientX, clientY) => {
    const curX = clamp(clientX - rect.left, 0, rect.width);
    const curY = clamp(clientY - rect.top, 0, rect.height);
    const r = { minX: Math.min(startX, curX), maxX: Math.max(startX, curX), minY: Math.min(startY, curY), maxY: Math.max(startY, curY) };
    draftEl.style.left = `${(100 * r.minX) / rect.width}%`;
    draftEl.style.top = `${(100 * r.minY) / rect.height}%`;
    draftEl.style.width = `${(100 * (r.maxX - r.minX)) / rect.width}%`;
    draftEl.style.height = `${(100 * (r.maxY - r.minY)) / rect.height}%`;
    return r;
  };
  track(ev.clientX, ev.clientY);
  // Drag-session listeners: added on mousedown, removed on mouseup (never accumulate).
  const onMove = (mev) => track(mev.clientX, mev.clientY);
  const onUp = (uev) => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    const r = track(uev.clientX, uev.clientY);
    draftEl.remove();
    if (r.maxX - r.minX < MIN_BOX_PX || r.maxY - r.minY < MIN_BOX_PX) {
      showToast("Drag a rectangle around the child's head/face to tag.");
      return;
    }
    const natW = (state.modalPhoto && state.modalPhoto.width) || img.naturalWidth || img.clientWidth;
    const natH = (state.modalPhoto && state.modalPhoto.height) || img.naturalHeight || img.clientHeight;
    state.manualDraftBox = draftFromRect(r, rect, { x: natW / img.clientWidth, y: natH / img.clientHeight }, { w: natW, h: natH });
    syncManualDrawUi();
    renderModalFaceOverlays();
  };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}

export function bindManualBox() {
  const layer = el("faceBoxesLayer");
  layer.addEventListener("mousedown", (ev) => {
    if (!state.manualDrawMode || !state.modalPhoto || ev.button !== 0) return;
    if (ev.target.closest(".manual-tag-popover")) return;
    const img = el("modalPhotoImg");
    if (img && img.clientWidth && img.clientHeight) startDrag(ev, layer, img);
  });
  delegate(layer, "click", ".manual-popover-close", () => cancelManualDraft());
  delegate(layer, "click", ".manual-chip", (_ev, chip) =>
    submitDraft(Number(chip.getAttribute("data-pid")), chip.getAttribute("data-name") || "", chip)
  );
  delegate(layer, "submit", ".manual-popover-form", (ev, form) => {
    ev.preventDefault();
    submitDraft(null, form.querySelector("input").value, form.querySelector('button[type="submit"]'));
  });
  on("btnDrawManualTag", "click", () => setManualDrawMode(!state.manualDrawMode));
}
