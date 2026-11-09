// Mode 1: Active-learning batch trainer ("approve all except 1-2").
import { decideCard, decideCardClick, toggleAllCards } from "../actions.js";
import { api, apiUrl, fetchQuietJson, post } from "../api.js";
import { openPhotoModal } from "../components/lightbox.js";
import { openReassignModal } from "../components/reassignModal.js";
import { showToast } from "../components/toasts.js";
import {
  batchCounts,
  batchSubmitLists,
  cardFromServer,
  getSelectedPerson,
  resolvePersonName,
  setBatchCards,
  setOverview,
  state,
  subscribe,
} from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml, toElement } from "../util/dom.js";
import { thumbSrc } from "../util/format.js";

const AUDIT_TOOLTIP = "random sample of auto-accepted matches, to keep them honest";

export function batchMode() {
  const sel = el("batchModeSelect");
  return sel && sel.value === "confident" ? "confident" : "impact";
}

const batchSize = () => Number(el("batchSizeSelect").value || 24);

export function applyBatchPayload(batchData) {
  state.batchMeta = batchData;
  state.batchUserTouched = false;
  state.lastClickedBatchIdx = null;
  setBatchCards((batchData.cards || []).map(cardFromServer), null);
}

export async function loadBatchForSelected() {
  const selP = getSelectedPerson();
  if (!selP) {
    el("batchControls").hidden = true;
    el("batchHeading").textContent = "Select a child on the left (or name an Unnamed Cluster)";
    setHtml(
      el("batchGrid"),
      html`<div class="empty-state">Pick any child in the left sidebar or open <strong>Unnamed Kids &amp; Years</strong> to start categorizing.</div>`
    );
    return;
  }
  applyBatchPayload(await api(apiUrl("/api/batch", { person_id: selP.person_id, size: batchSize(), mode: batchMode() })));
}

/** After a background re-match: fetch a batch if the queue was empty, else refresh the header counts. */
export async function refreshBatchAfterRematch() {
  const targetPid = state.selectedPersonId;
  if (!state.batchCards.length) {
    const data = await fetchQuietJson(apiUrl("/api/batch", { person_id: targetPid, size: batchSize(), mode: batchMode() }));
    const stillWanted = state.mode === "batch" && state.selectedPersonId === targetPid && !state.batchCards.length;
    if (data && Array.isArray(data.cards) && data.cards.length && stillWanted) applyBatchPayload(data);
    return;
  }
  const selP = getSelectedPerson();
  if (selP && state.batchMeta) {
    state.batchMeta.remaining_unreviewed = selP.review_ready_count;
    state.batchMeta.album_cohesion_count = selP.album_cohesion_count;
    updateBatchHeaderCopy();
  }
}

function updateBatchButtonLabel() {
  if (!getSelectedPerson()) return;
  const n = batchCounts(state.batchCards);
  const parts = [`${n.approved} ✓`, `${n.disproved} ✗`];
  if (n.unsure > 0) parts.push(`${n.unsure} ?`);
  if (n.reassigned > 0) parts.push(`${n.reassigned} ↪`);
  el("btnSubmitBatch").textContent = `Submit Batch (${parts.join(", ")})`;
}

function updateBatchHeaderCopy() {
  const selP = getSelectedPerson();
  const cards = state.batchCards;
  if (!selP || !cards.length) return;
  const meta = state.batchMeta;
  const hiSim = cards[0].similarity.toFixed(2);
  const loSim = cards[cards.length - 1].similarity.toFixed(2);
  const remCount = Math.max(cards.length, meta ? meta.remaining_unreviewed : cards.length);
  el("batchHeading").textContent = `Are these ${selP.display_name}? (${cards.length} shown of ${remCount} queued · sim ${hiSim} → ${loSim})`;
  const topPeers = ((meta && meta.top_peers) || []).slice(0, 4);
  const peersNote = topPeers.length
    ? html`<br />Learned peers: ${topPeers.map(
        (p, i) => html`${i ? ", " : ""}<strong>${p.display_name}</strong> (${p.shared_photos})`
      )} — small-photo peer boost &amp; cross-classroom guard active.`
    : "";
  setHtml(
    el("batchSubheading"),
    html`Click any wrong face for <strong>✗ Not ${selP.display_name}</strong> (hard-negative), click <strong>?</strong> if you're not sure (skips safely), <strong>↪</strong> for another child, or press <kbd>Enter</kbd> to submit.${peersNote}`
  );
}

export function renderBatchGrid() {
  const selP = getSelectedPerson();
  const grid = el("batchGrid");
  if (!selP) return;
  if (!state.batchCards.length) {
    el("batchControls").hidden = true;
    const refreshing = state.rematchPending || state.rematchRunning;
    el("batchHeading").textContent = refreshing
      ? `Updating matches for ${selP.display_name} in the background…`
      : `All candidate matches reviewed for ${selP.display_name}!`;
    el("batchSubheading").textContent = refreshing
      ? "Scanning archive embeddings with your latest tags — new candidates (if any) will appear automatically."
      : "Want to catch earlier toddler years? Check Unnamed Kids & Years to link a 2022–2024 cluster, or switch to Confirmed Gallery & Export.";
    setHtml(
      grid,
      refreshing
        ? html`<div class="empty-state">Background matcher is refreshing candidates for <strong>${selP.display_name}</strong>…</div>`
        : html`<div class="empty-state">0 unreviewed candidates remaining for <strong>${selP.display_name}</strong>.</div>`
    );
    return;
  }
  el("batchControls").hidden = false;
  updateBatchHeaderCopy();
  setHtml(grid, html`${state.batchCards.map((c, idx) => batchCardHtml(c, idx, selP))}`);
  updateBatchButtonLabel();
}

function badgeText(c, selP) {
  if (c.decision === "approved") return `✓ ${selP.display_name}`;
  if (c.decision === "unsure") return "? Not sure";
  if (c.decision === "reassigned") {
    return `↪ ${resolvePersonName(c.targetPersonId, c.targetName)}${c.alreadySavedInModal ? " ✓" : ""}`;
  }
  return `✗ Not ${selP.display_name}`;
}

export function batchCardHtml(c, idx, selP) {
  const companions = Array.isArray(c.companions) ? c.companions : [];
  const peerSuffix = companions.length ? ` · with ${companions.slice(0, 2).join(", ")}` : "";
  const dateLine = `${c.album_date || c.year} · ${c.archive_child}${peerSuffix}`;
  const sim = Number(c.similarity || 0);
  return html`
    <article class="face-card state-${c.decision}${c.audit ? " is-audit" : ""}" data-idx="${idx}">
      <div class="face-thumb-wrap">
        <img src="${thumbSrc(c)}" alt="" loading="lazy" />
        <span class="decision-badge">${badgeText(c, selP)}</span>
        ${c.audit && html`<span class="audit-pill" title="${AUDIT_TOOLTIP}">audit</span>`}
        <span class="sim-pill">${sim.toFixed(2)} · ${c.year}</span>
        ${c.impact > 0 &&
        html`<span class="impact-pill" title="Answering this photo also settles ~${c.impact} other open photos of this child">⚡ ${c.impact}</span>`}
        <div class="card-quick-tools">
          <button type="button" class="icon-btn btn-card-unsure ${c.decision === "unsure" ? "active-unsure" : ""}" data-idx="${idx}" title="I'm not sure — skip without teaching positive or negative">?</button>
          <button type="button" class="icon-btn btn-card-reassign" data-idx="${idx}" title="Actually another child…">↪</button>
          <button type="button" class="icon-btn btn-card-zoom" data-idx="${idx}" data-photo="${c.photo_id}" data-face="${c.face_id || ""}" title="Inspect full classroom group photo (←/→ to step through batch)">🔍</button>
        </div>
      </div>
      <div class="face-card-body">
        <div class="card-date" title="${dateLine}">${dateLine}</div>
        <div class="card-album" title="${c.album_title}">${c.album_title}</div>
      </div>
    </article>`;
}

/** Re-render only the given cards in place (other card nodes survive). */
function updateBatchCardEls(indices) {
  const selP = getSelectedPerson();
  const grid = el("batchGrid");
  if (!selP || !grid || state.mode !== "batch") return;
  for (const idx of new Set(indices)) {
    const c = state.batchCards[idx];
    const cardEl = grid.querySelector(`.face-card[data-idx="${idx}"]`);
    if (!c || !cardEl) return renderBatchGrid();
    cardEl.replaceWith(toElement(batchCardHtml(c, idx, selP)));
  }
  updateBatchButtonLabel();
}

export function openBatchCardInLightbox(idx) {
  const selP = getSelectedPerson();
  const card = state.batchCards[idx];
  if (!selP || !card) return undefined;
  state.modalNavContext = { type: "batch", index: idx, personId: selP.person_id, personName: selP.display_name };
  state.modalDirty = false;
  return openPhotoModal(card.photo_id, card.face_id || null, true);
}

export function openBatchCardReassign(idx) {
  const selP = getSelectedPerson();
  const card = state.batchCards[idx];
  if (!selP || !card) return;
  openReassignModal({
    photoId: card.photo_id,
    faceId: card.face_id || null,
    fromPersonId: selP.person_id,
    fromPersonName: selP.display_name,
    initialName: resolvePersonName(card.targetPersonId, card.targetName || ""),
  });
}

export async function submitCurrentBatch() {
  const selP = getSelectedPerson();
  if (!selP || !state.batchCards.length) return;
  await runBusy(
    el("btnSubmitBatch"),
    async () => {
      const prevCohesion = selP.album_cohesion_count || 0;
      const res = await post("/api/batch/submit", {
        person_id: selP.person_id,
        batch_size: batchSize(),
        mode: batchMode(),
        ...batchSubmitLists(state.batchCards, resolvePersonName),
      });
      setOverview(res.overview);
      applyBatchPayload(res.next_batch);
      const updP = getSelectedPerson();
      const sweptDelta = Math.max(0, ((updP && updP.album_cohesion_count) || 0) - prevCohesion);
      const a = res.applied;
      const cohNote = sweptDelta > 0 ? ` (+${sweptDelta} same-album burst photos auto-matched!)` : "";
      const unsNote = a.unsure > 0 ? `, ${a.unsure} marked unsure` : "";
      const reNote = a.reassigned > 0 ? `, ${a.reassigned} reassigned` : "";
      const conflictNote =
        a.skipped_conflicts > 0 ? ` (${a.skipped_conflicts} skipped: already confirmed for another child)` : "";
      showToast(
        `Saved: +${a.approved} confirmed for ${selP.display_name}${cohNote}, ${a.disproved} hard-negatives${unsNote}${reNote}${conflictNote}.`
      );
    },
    { busyText: "Updating anchors…", keepUpdatedLabel: true, context: "Submit batch", onFinally: updateBatchButtonLabel }
  );
}

export function bindBatch() {
  subscribe("batch-cards", (changed) => (changed ? updateBatchCardEls(changed) : renderBatchGrid()));
  subscribe("sync", () => {
    if (state.mode === "batch" && state.selectedPersonId && !state.batchCards.length) renderBatchGrid();
  });

  delegate("batchGrid", "click", ".face-card[data-idx]", (ev, cardEl) => {
    const idx = Number(cardEl.getAttribute("data-idx"));
    if (!getSelectedPerson() || !state.batchCards[idx]) return;
    const iconBtn = ev.target.closest(".icon-btn");
    if (!iconBtn) return decideCardClick(idx, ev.shiftKey);
    ev.stopPropagation();
    if (iconBtn.classList.contains("btn-card-unsure")) {
      decideCard(idx, state.batchCards[idx].decision === "unsure" ? "approved" : "unsure");
      state.lastClickedBatchIdx = idx;
    } else if (iconBtn.classList.contains("btn-card-zoom")) {
      runBusy(iconBtn, () => openBatchCardInLightbox(idx), { context: "Load full photo" });
    } else if (iconBtn.classList.contains("btn-card-reassign")) {
      openBatchCardReassign(idx);
    }
  });
  on("btnToggleSelectAll", "click", () => toggleAllCards("approve"));
  on("btnMarkAllUnsure", "click", () => toggleAllCards("unsure"));
  for (const id of ["batchModeSelect", "batchSizeSelect"]) {
    on(id, "change", (_ev, node) => runBusy(node, () => loadBatchForSelected(), { context: "Reload batch" }));
  }
  on("btnSubmitBatch", "click", () => submitCurrentBatch());
}
