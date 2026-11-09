// Mode 3: Confirmed gallery (pagination, per-photo reassign / remove, lightbox).
import { rejectFace } from "../actions.js";
import { api, apiUrl, fetchQuietJson } from "../api.js";
import { openPhotoModal } from "../components/lightbox.js";
import { openReassignModal } from "../components/reassignModal.js";
import { reportError } from "../components/toasts.js";
import { clampGalleryOffset, getSelectedPerson, state, subscribe } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml } from "../util/dom.js";
import { thumbSrc } from "../util/format.js";

/** Gallery badge for an item (pure). */
export function galleryBadge(item) {
  const sim = Number(item.similarity || 0).toFixed(2);
  if (item.status === "confirmed") return "✓ Confirmed";
  if (item.status === "unsure") return "? Unsure";
  if (item.source === "album_cohesion") return `✓ Same-Album (${sim})`;
  if (item.source === "model_trusted") return `✓ Learned (${sim})`;
  if (item.status === "auto_high") return `Auto (${sim})`;
  return `Cand (${sim})`;
}

function galleryCardHtml(item, idx, selP) {
  const face = item.face_id || "";
  return html`
    <article class="face-card ${item.status === "unsure" ? "state-unsure" : "state-approved"}" data-gidx="${idx}" data-photo="${item.photo_id}" data-face="${face}">
      <div class="face-thumb-wrap">
        <img src="${thumbSrc(item)}" alt="" loading="lazy" />
        <span class="decision-badge">${galleryBadge(item)}</span>
        <span class="sim-pill">${item.year}</span>
        <div class="card-quick-tools">
          <button type="button" class="icon-btn btn-gal-reassign" title="Actually another child… (re-tag face &amp; move)">↪</button>
          <button type="button" class="icon-btn btn-gal-remove" title="Not ${selP.display_name} (remove)">✗</button>
          <button type="button" class="icon-btn btn-gal-zoom" title="Open full classroom photo">🔍</button>
        </div>
      </div>
      <div class="face-card-body">
        <div class="card-date">${item.album_date || item.year} · ${item.archive_child}</div>
        <div class="card-album" title="${item.album_title}">${item.album_title}</div>
      </div>
    </article>`;
}

export async function loadGallery(opts = {}) {
  const { quiet = false } = opts;
  const selP = getSelectedPerson();
  const grid = el("galleryGrid");
  const pag = el("galleryPagination");
  if (!selP) {
    state.galleryItems = [];
    state.gallerySig = "";
    setHtml(grid, html`<div class="empty-state">Select a child in the left roster to inspect their photo gallery.</div>`);
    setHtml(pag, html``);
    return;
  }
  const status = el("galleryStatusFilter").value;
  const year = el("galleryYearFilter").value;
  const fetchPage = () => {
    const path = apiUrl("/api/gallery", {
      person_id: selP.person_id,
      status,
      year,
      offset: state.galleryOffset,
      limit: state.galleryLimit,
    });
    return quiet ? fetchQuietJson(path) : api(path);
  };
  let data = await fetchPage();
  if (!data || state.selectedPersonId !== selP.person_id) return;
  const clamped = clampGalleryOffset({
    offset: state.galleryOffset,
    limit: state.galleryLimit,
    total: data.total,
    itemsLen: data.items.length,
  });
  if (clamped !== null) {
    state.galleryOffset = clamped;
    data = await fetchPage();
    if (!data || state.selectedPersonId !== selP.person_id) return;
  }

  el("galleryHeading").textContent = `${selP.display_name} — ${data.total.toLocaleString()} photos`;
  const sig = JSON.stringify([
    selP.person_id,
    status,
    year,
    state.galleryOffset,
    data.total,
    data.items.map((it) => [it.photo_id, it.face_id, it.status, it.source, it.similarity]),
  ]);
  state.galleryItems = data.items;
  if (sig === state.gallerySig && grid.querySelector(".face-card")) return; // no visible change: keep DOM
  state.gallerySig = sig;

  if (!data.items.length) {
    setHtml(grid, html`<div class="empty-state">No photos match this filter yet.</div>`);
    setHtml(pag, html``);
    return;
  }
  setHtml(grid, html`${data.items.map((item, idx) => galleryCardHtml(item, idx, selP))}`);
  const hasPrev = state.galleryOffset > 0;
  const hasNext = state.galleryOffset + state.galleryLimit < data.total;
  const last = Math.min(data.total, state.galleryOffset + data.items.length);
  setHtml(
    pag,
    html`
      <button type="button" class="btn btn-secondary btn-sm" id="galPrev" data-gal-page="prev" ${hasPrev ? "" : html`disabled`}>← Newer</button>
      <span>${state.galleryOffset + 1}–${last} of ${data.total}</span>
      <button type="button" class="btn btn-secondary btn-sm" id="galNext" data-gal-page="next" ${hasNext ? "" : html`disabled`}>Older →</button>`
  );
}

function openGalleryItemInModal(gidx, photoId, faceId) {
  const selP = getSelectedPerson();
  if (!selP) return undefined;
  state.modalNavContext = {
    type: "gallery",
    items: state.galleryItems || [],
    index: Number.isFinite(gidx) ? gidx : 0,
    personId: selP.person_id,
    personName: selP.display_name,
  };
  state.modalDirty = false;
  return openPhotoModal(photoId, faceId, true);
}

export function bindGallery() {
  // Any persisted label change reloads the visible page (while the lightbox is open it reloads on close).
  subscribe("faces-changed", () => {
    if (state.mode === "gallery" && el("photoModal").hidden) loadGallery().catch((e) => reportError(e, "Refresh gallery"));
  });

  delegate("galleryGrid", "click", ".face-card[data-gidx]", (ev, card) => {
    const selP = getSelectedPerson();
    if (!selP) return;
    const btn = ev.target.closest(".icon-btn");
    const photoId = Number(card.getAttribute("data-photo"));
    const faceId = Number(card.getAttribute("data-face")) || null;
    if (btn) ev.stopPropagation();
    if (!btn || btn.classList.contains("btn-gal-zoom")) {
      const gidx = Number(card.getAttribute("data-gidx"));
      runBusy(btn || card, () => openGalleryItemInModal(gidx, photoId, faceId), { context: "Load full photo" });
    } else if (btn.classList.contains("btn-gal-reassign")) {
      openReassignModal({
        photoId,
        faceId,
        fromPersonId: selP.person_id,
        fromPersonName: selP.display_name,
        initialName: "",
      });
    } else if (btn.classList.contains("btn-gal-remove")) {
      runBusy(btn, () => rejectFace({ photoId, faceId, personId: selP.person_id, personName: selP.display_name }), {
        context: "Remove photo",
      });
    }
  });

  delegate("galleryPagination", "click", "[data-gal-page]", (_ev, btn) => {
    if (btn.disabled) return;
    const prev = btn.getAttribute("data-gal-page") === "prev";
    runBusy(
      btn,
      async () => {
        state.galleryOffset = prev
          ? Math.max(0, state.galleryOffset - state.galleryLimit)
          : state.galleryOffset + state.galleryLimit;
        await loadGallery();
      },
      { busyText: "Loading…", context: prev ? "Load newer photos" : "Load older photos" }
    );
  });

  for (const id of ["galleryStatusFilter", "galleryYearFilter"]) {
    on(id, "change", (_ev, node) => {
      state.galleryOffset = 0;
      runBusy(node, () => loadGallery(), { context: "Filter gallery" });
    });
  }
}
