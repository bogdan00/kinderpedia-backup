// Mode 5: Album Browser — navigate chronological albums, spot-check matches photo-by-photo,
// filter photos within an album by child, and approve album matches in bulk.
import { api, post } from "../api.js";
import { openPhotoModal } from "../components/lightbox.js";
import { showToast } from "../components/toasts.js";
import { state, subscribe } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml } from "../util/dom.js";
import { plural, thumbSrc } from "../util/format.js";

let albumSearchTimer = null;

// ---------------------------------------------------------------- album listing
export async function loadAlbums(resetOffset = false) {
  if (resetOffset) state.albumsOffset = 0;
  const qs = new URLSearchParams();
  if (state.albumFilterChild) qs.set("child", state.albumFilterChild);
  if (state.albumFilterYear) qs.set("year", state.albumFilterYear);
  if (state.albumSearchQuery) qs.set("q", state.albumSearchQuery);
  qs.set("limit", String(state.albumsLimit));
  qs.set("offset", String(state.albumsOffset));

  const data = await api(`/api/albums?${qs.toString()}`);
  state.albums = data.albums || [];
  state.albumsTotal = data.total || 0;

  syncAlbumFilterDropdowns(data.available_children || [], data.available_years || []);
  renderAlbumsList();
}

function syncAlbumFilterDropdowns(children, years) {
  const childSelect = el("albumChildFilter");
  const yearSelect = el("albumYearFilter");
  if (!childSelect || !yearSelect) return;

  if (childSelect.options.length <= 1 && children.length) {
    const curVal = state.albumFilterChild;
    setHtml(
      childSelect,
      html`
        <option value="">All Children</option>
        ${children.map((c) => html`<option value="${c}" ${c === curVal ? "selected" : ""}>${c}</option>`)}
      `
    );
  }

  if (yearSelect.options.length <= 1 && years.length) {
    const curVal = state.albumFilterYear;
    setHtml(
      yearSelect,
      html`
        <option value="">All Years</option>
        ${years.map((y) => html`<option value="${y}" ${y === curVal ? "selected" : ""}>${y}</option>`)}
      `
    );
  }
}

function albumCardHtml(alb) {
  const topPills = (alb.top_people || []).map(
    (p) => html`
      <span class="album-tag-chip ${p.confirmed_count > 0 ? "is-confirmed" : ""}">
        ${p.name} <strong>${p.count}</strong>
      </span>
    `
  );

  return html`
    <div class="album-card" data-album-key="${alb.album_key}">
      <div class="album-card-cover-wrap">
        <img
          class="album-card-cover"
          src="/media/photo/${alb.cover_photo_id}"
          loading="lazy"
          alt="${alb.album_title}"
        />
        <div class="album-card-badges">
          <span class="badge badge-child">${alb.archive_child}</span>
          ${alb.year ? html`<span class="badge badge-year">${alb.year}</span>` : ""}
        </div>
      </div>
      <div class="album-card-body">
        <h3 class="album-card-title" title="${alb.album_title}">${alb.album_title}</h3>
        <div class="album-card-meta">
          <span>${plural(alb.photo_count, "photo")}</span>
          <span>·</span>
          <span>${plural(alb.face_count, "face")}</span>
        </div>
        <div class="album-card-status-bar">
          ${alb.confirmed_count > 0
            ? html`<span class="status-pill status-pill-confirmed" title="${alb.confirmed_count} confirmed">
                ✓ ${alb.confirmed_count}
              </span>`
            : ""}
          ${alb.auto_count > 0
            ? html`<span class="status-pill status-pill-auto" title="${alb.auto_count} auto-matches">
                ⚡ ${alb.auto_count}
              </span>`
            : ""}
          ${alb.candidate_count > 0
            ? html`<span class="status-pill status-pill-candidate" title="${alb.candidate_count} candidates">
                ? ${alb.candidate_count}
              </span>`
            : ""}
          ${alb.unassigned_count > 0
            ? html`<span class="status-pill status-pill-none" title="${alb.unassigned_count} unassigned">
                ∅ ${alb.unassigned_count}
              </span>`
            : ""}
        </div>
        ${topPills.length ? html`<div class="album-card-people">${topPills}</div>` : ""}
      </div>
    </div>
  `;
}

function renderAlbumsList() {
  el("albumListView").hidden = false;
  el("albumDetailView").hidden = true;

  const grid = el("albumsGrid");
  const empty = el("albumsEmptyState");
  const countLabel = el("albumsCountLabel");

  const total = state.albumsTotal;
  const offset = state.albumsOffset;
  const limit = state.albumsLimit;
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(total, offset + limit);

  if (countLabel) {
    countLabel.textContent = total > 0 ? `Showing albums ${start}–${end} of ${total}` : "No albums found";
  }

  const prevBtn = el("btnAlbumsPrev");
  const nextBtn = el("btnAlbumsNext");
  if (prevBtn) prevBtn.disabled = offset <= 0;
  if (nextBtn) nextBtn.disabled = end >= total;

  if (!state.albums.length) {
    setHtml(grid, html``);
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;

  setHtml(grid, html`${state.albums.map(albumCardHtml)}`);
}

// ---------------------------------------------------------------- album detail
export async function openAlbumByKey(albumKey) {
  state.selectedAlbumKey = albumKey;
  state.albumPersonFilter = null;
  el("albumListView").hidden = true;
  el("albumDetailView").hidden = false;
  setHtml(el("albumPhotosGrid"), html`<div class="empty-state">Loading album photos…</div>`);

  const detail = await api(`/api/albums/detail?key=${encodeURIComponent(albumKey)}`);
  state.currentAlbumDetail = detail;
  renderAlbumDetail();
}

export async function openAlbumByPhotoId(photoId) {
  el("albumListView").hidden = true;
  el("albumDetailView").hidden = false;
  setHtml(el("albumPhotosGrid"), html`<div class="empty-state">Loading album photos…</div>`);

  const detail = await api(`/api/albums/detail?photo_id=${encodeURIComponent(photoId)}`);
  state.selectedAlbumKey = detail.album_key;
  state.currentAlbumDetail = detail;
  state.albumPersonFilter = null;
  renderAlbumDetail();
}

export function renderAlbumDetail() {
  const detail = state.currentAlbumDetail;
  if (!detail) return;

  el("albumDetailTitle").textContent = detail.album_title;
  el("albumDetailMeta").textContent = `${detail.archive_child} · ${detail.album_date || detail.year} · ${plural(
    detail.photo_count,
    "photo"
  )} · ${plural(detail.face_count, "face")}`;

  const approveBtn = el("btnApproveAlbum");
  if (approveBtn) {
    approveBtn.hidden = detail.auto_count <= 0;
    approveBtn.textContent = `✓ Approve ${plural(detail.auto_count, "Auto-Match")}`;
  }

  // Filter chips by person
  const personChipsWrap = el("albumPersonFilterBar");
  if (personChipsWrap) {
    const isAll = state.albumPersonFilter === null;
    const isUnass = state.albumPersonFilter === "unassigned";

    const personPills = (detail.people || []).map((p) => {
      const active = state.albumPersonFilter === p.person_id;
      return html`
        <button
          type="button"
          class="album-filter-pill ${active ? "active" : ""}"
          data-album-filter-person="${p.person_id}"
        >
          ${p.display_name} <strong>${p.total_count}</strong>
          ${p.confirmed_count > 0 ? html`<span class="pill-badge">✓${p.confirmed_count}</span>` : ""}
        </button>
      `;
    });

    setHtml(
      personChipsWrap,
      html`
        <button
          type="button"
          class="album-filter-pill ${isAll ? "active" : ""}"
          data-album-filter-person=""
        >
          All Photos (${detail.photo_count})
        </button>
        <button
          type="button"
          class="album-filter-pill ${isUnass ? "active" : ""}"
          data-album-filter-person="unassigned"
        >
          Unassigned Faces (${detail.unassigned_count})
        </button>
        ${personPills}
      `
    );
  }

  renderAlbumPhotos();
}

function photoCardHtml(photo, pIdx) {
  const faceChips = (photo.faces || []).map((face) => {
    let statusLabel = "∅";
    let statusCls = "status-none";
    if (face.status === "confirmed") {
      statusLabel = "✓";
      statusCls = "status-confirmed";
    } else if (face.status === "auto_high") {
      const pct = face.similarity ? Math.round(face.similarity * 100) : "";
      statusLabel = `⚡${pct ? ` ${pct}%` : ""}`;
      statusCls = "status-auto";
    } else if (face.status === "candidate") {
      const pct = face.similarity ? Math.round(face.similarity * 100) : "";
      statusLabel = `?${pct ? ` ${pct}%` : ""}`;
      statusCls = "status-candidate";
    }

    const displayName = face.person_name || "Unassigned";

    return html`
      <button
        type="button"
        class="photo-face-chip ${statusCls}"
        data-open-face-photo="${photo.photo_id}"
        data-open-face-id="${face.face_id}"
        data-photo-index="${pIdx}"
        title="${displayName} (${face.status || "unassigned"})"
      >
        <img class="photo-face-avatar" src="${thumbSrc(face)}" alt="" loading="lazy" />
        <span class="photo-face-name">${displayName}</span>
        <span class="photo-face-badge">${statusLabel}</span>
      </button>
    `;
  });

  return html`
    <div class="album-photo-card" data-album-photo-id="${photo.photo_id}" data-photo-index="${pIdx}">
      <div class="album-photo-thumb-wrap" data-open-photo-idx="${pIdx}">
        <img
          class="album-photo-thumb"
          src="/media/photo/${photo.photo_id}"
          loading="lazy"
          alt="${photo.filename}"
        />
        <div class="album-photo-badges">
          <span class="badge badge-seq">#${photo.seq_in_album || pIdx + 1}</span>
          <span class="badge badge-faces">${plural(photo.face_count, "face")}</span>
        </div>
      </div>
      <div class="album-photo-footer">
        <div class="album-photo-meta">${photo.filename}</div>
        <div class="album-photo-faces-strip">
          ${faceChips.length ? faceChips : html`<span class="album-photo-nofaces">No faces detected</span>`}
        </div>
      </div>
    </div>
  `;
}

function renderAlbumPhotos() {
  const detail = state.currentAlbumDetail;
  if (!detail) return;

  const grid = el("albumPhotosGrid");
  const filterPid = state.albumPersonFilter;

  let photos = detail.photos || [];
  if (filterPid === "unassigned") {
    photos = photos.filter((p) => (p.faces || []).some((f) => !f.person_id));
  } else if (typeof filterPid === "number") {
    photos = photos.filter((p) => (p.faces || []).some((f) => f.person_id === filterPid));
  }

  if (!photos.length) {
    setHtml(grid, html`<div class="empty-state">No photos match the selected person filter.</div>`);
    return;
  }

  setHtml(grid, html`${photos.map((photo, pIdx) => photoCardHtml(photo, pIdx))}`);
}

// ---------------------------------------------------------------- event wiring
export function bindAlbumsEvents() {
  on("btnBackToAlbums", "click", () => {
    state.selectedAlbumKey = null;
    state.currentAlbumDetail = null;
    state.albumPersonFilter = null;
    loadAlbums();
  });

  on("albumChildFilter", "change", (ev) => {
    state.albumFilterChild = ev.target.value;
    loadAlbums(true);
  });

  on("albumYearFilter", "change", (ev) => {
    state.albumFilterYear = ev.target.value;
    loadAlbums(true);
  });

  on("albumSearchInput", "input", (ev) => {
    if (albumSearchTimer) clearTimeout(albumSearchTimer);
    albumSearchTimer = setTimeout(() => {
      state.albumSearchQuery = ev.target.value.trim();
      loadAlbums(true);
    }, 250);
  });

  on("btnAlbumsPrev", "click", () => {
    state.albumsOffset = Math.max(0, state.albumsOffset - state.albumsLimit);
    loadAlbums();
  });

  on("btnAlbumsNext", "click", () => {
    if (state.albumsOffset + state.albumsLimit < state.albumsTotal) {
      state.albumsOffset += state.albumsLimit;
      loadAlbums();
    }
  });

  // Clicking an album card
  delegate(document.body, "click", ".album-card[data-album-key]", (_ev, card) => {
    const key = card.getAttribute("data-album-key");
    if (key) openAlbumByKey(key);
  });

  // Clicking a person filter pill inside album detail
  delegate(document.body, "click", "[data-album-filter-person]", (_ev, btn) => {
    const val = btn.getAttribute("data-album-filter-person");
    if (!val) {
      state.albumPersonFilter = null;
    } else if (val === "unassigned") {
      state.albumPersonFilter = "unassigned";
    } else {
      state.albumPersonFilter = Number(val);
    }
    renderAlbumDetail();
  });

  // Clicking a photo thumbnail in album detail -> opens Lightbox with album nav context
  delegate(document.body, "click", ".album-photo-thumb-wrap[data-open-photo-idx]", (_ev, wrap) => {
    const pIdx = Number(wrap.getAttribute("data-open-photo-idx"));
    const detail = state.currentAlbumDetail;
    if (!detail || !detail.photos || !detail.photos[pIdx]) return;
    const photo = detail.photos[pIdx];

    state.modalNavContext = {
      type: "album",
      albumKey: detail.album_key,
      photos: detail.photos,
      index: pIdx,
    };
    openPhotoModal(photo.photo_id, null, true);
  });

  // Clicking a face chip -> opens Lightbox focused on that face with album nav context
  delegate(document.body, "click", ".photo-face-chip[data-open-face-photo]", (_ev, chip) => {
    const photoId = Number(chip.getAttribute("data-open-face-photo"));
    const faceId = Number(chip.getAttribute("data-open-face-id"));
    const pIdx = Number(chip.getAttribute("data-photo-index"));
    const detail = state.currentAlbumDetail;
    if (!detail) return;

    state.modalNavContext = {
      type: "album",
      albumKey: detail.album_key,
      photos: detail.photos,
      index: Number.isFinite(pIdx) ? pIdx : 0,
    };
    openPhotoModal(photoId, faceId, true);
  });

  // Approve all auto-matches in current album
  on("btnApproveAlbum", "click", (ev, btn) => {
    const detail = state.currentAlbumDetail;
    if (!detail || !detail.album_key) return;

    runBusy(
      btn,
      async () => {
        const res = await post("/api/albums/approve-all", { album_key: detail.album_key });
        showToast(`Approved ${plural(res.approved_count, "match")} in album! Rematch scheduled.`);
        const updated = await api(`/api/albums/detail?key=${encodeURIComponent(detail.album_key)}`);
        state.currentAlbumDetail = updated;
        renderAlbumDetail();
      },
      { busyText: "Approving…", context: "Approve Album Matches" }
    );
  });

  // When face tags change, refresh album detail if open
  subscribe("faces-changed", async () => {
    if (state.mode === "albums" && state.selectedAlbumKey && !el("albumDetailView").hidden) {
      try {
        const updated = await api(`/api/albums/detail?key=${encodeURIComponent(state.selectedAlbumKey)}`);
        state.currentAlbumDetail = updated;
        renderAlbumDetail();
      } catch (_e) {}
    }
  });
}
