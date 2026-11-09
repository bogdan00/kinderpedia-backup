// Full classroom-photo lightbox: photo + face boxes, the side list of faces (tag / untag / unsure), quiet
// refresh after re-matches. Navigation (strip, prev/next, quick rows) lives in lightboxNav.js and manual
// box drawing in manualBox.js. All face-label changes go through actions.js.
import { markFaceUnsure, tagFace, untagFace } from "../actions.js";
import { api, fetchQuietJson } from "../api.js";
import { getSelectedPerson, state, subscribe } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml, siblingControls } from "../util/dom.js";
import { reloadChildClustersPreservingUi } from "../views/childClusters.js";
import { loadClusters } from "../views/clusters.js";
import { loadGallery } from "../views/gallery.js";
import { applyDedupedNav, markPhotoSeen, renderModalNavControls } from "./lightboxNav.js";
import { appendManualDraftPopover, syncManualDrawUi } from "./manualBox.js";
import { reportError, showToast } from "./toasts.js";

const isOpen = () => !el("photoModal").hidden;

function setStageStatus(text, isErr = false) {
  const stageStatus = el("photoStageStatus");
  if (!stageStatus) return;
  stageStatus.hidden = !text;
  stageStatus.classList.toggle("is-error", Boolean(text) && isErr);
  if (text) stageStatus.textContent = text;
}

function applyPhotoModalDetail(detail, highlightFaceId = null) {
  state.modalPhoto = detail;
  state.modalHighlightFaceId = highlightFaceId;
  el("modalMetaLine").textContent = `${detail.archive_child} · ${detail.album_date || detail.year} · ${detail.filename}`;
  el("modalAlbumTitle").textContent = detail.album_title;
  const cdnBtn = el("modalCdnLink");
  cdnBtn.hidden = !detail.best_live_url;
  if (detail.best_live_url) cdnBtn.href = detail.best_live_url;

  const imgEl = el("modalPhotoImg");
  setHtml(el("faceBoxesLayer"), html``);
  const targetSrc = `/media/photo/${detail.photo_id}`;
  if (imgEl.getAttribute("src") !== targetSrc || !imgEl.complete || !imgEl.naturalWidth) {
    setStageStatus("Loading photo…");
    imgEl.src = targetSrc;
  }
  if (imgEl.complete && imgEl.naturalWidth > 0) {
    setStageStatus("");
    renderModalFaceOverlays();
  }
  renderModalNavControls();
  renderModalSideFaces();
  syncManualDrawUi();
  el("photoModal").hidden = false;
}

export async function openPhotoModal(photoId, highlightFaceId = null, keepNavContext = false) {
  state.manualDrawMode = false;
  state.manualDraftBox = null;
  if (!keepNavContext) {
    state.modalNavContext = null;
    state.modalDirty = false;
  }
  const detail = await api(`/api/photos/${photoId}`);
  if (state.modalNavContext && state.modalNavContext.dedupe) markPhotoSeen(detail.photo_id);
  applyPhotoModalDetail(detail, highlightFaceId);
}

export async function openClusterFaceModal(clusterId, faceId) {
  const [clusterData, photoDetail] = await Promise.all([
    api(`/api/clusters/${clusterId}/faces`),
    api(`/api/faces/${faceId}/photo`),
  ]);
  const faces = clusterData.faces || [];
  if (!faces.some((item) => item.face_id === faceId)) {
    faces.unshift({
      face_id: faceId,
      photo_id: photoDetail.photo_id,
      album_date: photoDetail.album_date || photoDetail.year,
      album_title: photoDetail.album_title,
    });
  }
  state.manualDrawMode = false;
  state.manualDraftBox = null;
  state.modalNavContext = {
    type: "cluster",
    dedupe: true,
    clusterId: clusterData.cluster_id,
    archiveChild: clusterData.archive_child,
    year: clusterData.year,
    photoCount: clusterData.photo_count,
    personName: clusterData.person_name || "",
    rawFaces: faces,
    faces,
    index: 0,
  };
  applyDedupedNav(state.modalNavContext, faceId, photoDetail.photo_id);
  markPhotoSeen(photoDetail.photo_id);
  el("modalClusterNameInput").value = clusterData.person_name || "";
  applyPhotoModalDetail(photoDetail, faceId);
}

export function renderModalFaceOverlays() {
  const detail = state.modalPhoto;
  if (!detail) return;
  const layer = el("faceBoxesLayer");
  const natW = detail.width || el("modalPhotoImg").naturalWidth || 1;
  const natH = detail.height || el("modalPhotoImg").naturalHeight || 1;
  setHtml(
    layer,
    html`${detail.faces.map((f) => {
      const cls = ["bbox-overlay", f.person_name && "named", f.is_manual && "manual", f.face_id === state.modalHighlightFaceId && "highlight"]
        .filter(Boolean)
        .join(" ");
      const style = `left:${(100 * f.box.left) / natW}%;top:${(100 * f.box.top) / natH}%;width:${(100 * (f.box.right - f.box.left)) / natW}%;height:${(100 * (f.box.bottom - f.box.top)) / natH}%`;
      const label = f.person_name ? `${f.is_manual ? "✎ " : ""}${f.person_name}` : `#${f.face_idx + 1} (name…)`;
      return html`<div class="${cls}" data-fid="${f.face_id}" style="${style}"><span class="bbox-label">${label}</span></div>`;
    })}`
  );
  if (state.manualDraftBox) appendManualDraftPopover(layer);
}

function faceStatusHtml(f) {
  const sim = f.similarity != null ? ` · ${f.similarity.toFixed(2)}` : "";
  if (f.person_id == null) return html`<span class="modal-face-status uncat">Uncategorized</span>`;
  if (f.status === "confirmed") return html`<span class="modal-face-status confirmed">✓ Confirmed${f.is_manual ? " · ✎ Manual" : ""}</span>`;
  if (f.status === "auto_high") return html`<span class="modal-face-status auto">Auto match${sim}</span>`;
  return html`<span class="modal-face-status guess">Model guess${sim}</span>`;
}

/** Batch mode offers "? Not sure" for the selected child's unconfirmed (or highlighted untagged) face. */
function unsureTargetFor(f, selP) {
  const inClusterNav = state.modalNavContext && state.modalNavContext.type === "cluster";
  if (inClusterNav || f.is_manual || state.mode !== "batch" || !selP) return null;
  const hasTag = f.person_id != null;
  if (hasTag && f.person_id === selP.person_id && f.status !== "confirmed") return selP;
  if (!hasTag && f.face_id === state.modalHighlightFaceId) return selP;
  return null;
}

function sideFaceHtml(f, selP) {
  const hasTag = f.person_id != null;
  const unsure = unsureTargetFor(f, selP);
  const untagTitle = f.is_manual
    ? `Remove manual box (${f.person_name || ""})`
    : `Remove tag (${f.person_name || ""}) and leave this child uncategorized`;
  return html`
    <form class="modal-face-row ${f.face_id === state.modalHighlightFaceId ? "highlight" : ""}" data-fid="${f.face_id}">
      <img src="/media/face/${Number(f.face_id)}.jpg" alt="" title="Highlight this face on the photo" />
      <div class="modal-face-controls">
        <div class="modal-face-status-row">${faceStatusHtml(f)}</div>
        <input type="text" class="modal-face-input" data-fid="${f.face_id}" list="knownPeopleDatalist" placeholder="Name this child…" value="${f.person_name || ""}" />
        <div class="modal-face-btn-row">
          <button type="submit" class="btn btn-tonal btn-sm">Save Tag ✓</button>
          ${(hasTag || f.is_manual) &&
          html`<button type="button" class="btn btn-untag btn-sm btn-modal-untag" data-fid="${f.face_id}" title="${untagTitle}">✗ Remove Tag</button>`}
          ${unsure &&
          html`<button type="button" class="btn btn-ghost btn-sm btn-modal-unsure ${hasTag ? "btn-modal-unsure-compact" : ""}" data-fid="${f.face_id}" data-pid="${unsure.person_id}" title="Mark as I'm not sure for ${unsure.display_name}">${hasTag ? "?" : "? Not sure"}</button>`}
        </div>
      </div>
    </form>`;
}

export function renderModalSideFaces() {
  const detail = state.modalPhoto;
  if (!detail) return;
  const listEl = el("modalFacesList");
  if (!detail.faces.length) {
    setHtml(
      listEl,
      html`<p class="modal-hint">No faces auto-detected in this photo — click <strong>+ Tag Child (T)</strong> above to draw a box around any child!</p>`
    );
    return;
  }
  const selP = getSelectedPerson();
  setHtml(listEl, html`${detail.faces.map((f) => sideFaceHtml(f, selP))}`);
}

function highlightFace(fid, focusInput = false) {
  state.modalHighlightFaceId = fid;
  renderModalFaceOverlays();
  renderModalSideFaces();
  const input = focusInput && document.querySelector(`.modal-face-input[data-fid="${fid}"]`);
  if (input) {
    input.focus();
    input.select();
  }
}

const modalFace = (fid) => ((state.modalPhoto && state.modalPhoto.faces) || []).find((x) => x.face_id === fid) || null;

function untagFromRow(btn, fid) {
  const detail = state.modalPhoto;
  const face = modalFace(fid);
  return runBusy(btn, () => untagFace({ photoId: detail.photo_id, faceId: fid, prevName: (face && face.person_name) || "" }), {
    busyText: "Removing…",
    alsoDisable: siblingControls(btn.closest(".modal-face-row"), btn),
    context: "Remove face tag",
  });
}

export async function closePhotoModal() {
  const wasDirty = state.modalDirty;
  state.manualDrawMode = false;
  state.manualDraftBox = null;
  syncManualDrawUi();
  el("photoModal").hidden = true;
  state.modalPhoto = null;
  state.modalNavContext = null;
  state.modalDirty = false;
  if (!wasDirty) return;
  try {
    if (state.mode === "gallery") await loadGallery();
    else if (state.mode === "clusters") await loadClusters();
    else if (state.mode === "childclusters") await reloadChildClustersPreservingUi();
  } catch (err) {
    reportError(err, "Refresh view");
  }
}

/** Re-fetch the open photo after a re-match, preserving in-progress typing / focus in the name fields. */
export async function refreshOpenPhotoModalQuietly() {
  if (!state.modalPhoto || !isOpen() || state.manualDrawMode || state.manualDraftBox) return;
  const targetPhotoId = state.modalPhoto.photo_id;
  const prevNames = new Map((state.modalPhoto.faces || []).map((f) => [f.face_id, f.person_name || ""]));
  const dirtyInputs = new Map();
  let focused = null;
  document.querySelectorAll(".modal-face-input").forEach((inp) => {
    const fid = Number(inp.getAttribute("data-fid"));
    if (inp === document.activeElement) {
      focused = { fid, start: inp.selectionStart, end: inp.selectionEnd };
      dirtyInputs.set(fid, inp.value);
    } else if (inp.value !== (prevNames.get(fid) || "")) {
      dirtyInputs.set(fid, inp.value);
    }
  });
  const fresh = await fetchQuietJson(`/api/photos/${targetPhotoId}`);
  if (!fresh || !state.modalPhoto || state.modalPhoto.photo_id !== targetPhotoId || !isOpen()) return;
  state.modalPhoto = fresh;
  renderModalFaceOverlays();
  renderModalSideFaces();
  const inputFor = (fid) => document.querySelector(`.modal-face-input[data-fid="${fid}"]`);
  dirtyInputs.forEach((val, fid) => {
    const inp = inputFor(fid);
    if (inp) inp.value = val;
  });
  const focusInp = focused && inputFor(focused.fid);
  if (focusInp) {
    focusInp.focus({ preventScroll: true });
    if (focused.start !== null && focused.end !== null) {
      try {
        focusInp.setSelectionRange(focused.start, focused.end);
      } catch (_e) {}
    }
  }
}

export function bindLightbox() {
  subscribe("modal-photo", () => {
    if (!isOpen()) return;
    renderModalNavControls();
    renderModalFaceOverlays();
    renderModalSideFaces();
  });

  const imgEl = el("modalPhotoImg");
  imgEl.addEventListener("load", () => {
    setStageStatus("");
    renderModalFaceOverlays();
  });
  imgEl.addEventListener("error", () => {
    if (state.modalPhoto) setStageStatus("⚠ Could not load full photo (backend may be restarting) — click to retry", true);
  });
  on("photoStageStatus", "click", (_ev, node) => {
    if (!node.classList.contains("is-error") || !state.modalPhoto) return;
    setStageStatus("Loading photo…");
    imgEl.src = `/media/photo/${state.modalPhoto.photo_id}?t=${Date.now()}`;
  });

  delegate("faceBoxesLayer", "click", ".bbox-overlay[data-fid]", (_ev, box) => {
    if (!state.manualDrawMode) highlightFace(Number(box.getAttribute("data-fid")), true);
  });

  const list = el("modalFacesList");
  delegate(list, "click", ".modal-face-row > img", (_ev, img) =>
    highlightFace(Number(img.closest(".modal-face-row").getAttribute("data-fid")))
  );
  delegate(list, "submit", ".modal-face-row", (ev, form) => {
    ev.preventDefault();
    const fid = Number(form.getAttribute("data-fid"));
    const name = form.querySelector("input").value.trim();
    const submitBtn = form.querySelector('button[type="submit"]');
    const face = modalFace(fid);
    if (!name) {
      if (face && face.person_id != null) untagFromRow(submitBtn, fid);
      else showToast("Face is already uncategorized — type a name to tag.");
      return;
    }
    runBusy(submitBtn, () => tagFace({ photoId: state.modalPhoto.photo_id, faceId: fid, name }), {
      busyText: "Saving…",
      alsoDisable: siblingControls(form, submitBtn),
      context: "Save face tag",
    });
  });
  delegate(list, "click", ".btn-modal-untag", (_ev, btn) => untagFromRow(btn, Number(btn.getAttribute("data-fid"))));
  delegate(list, "click", ".btn-modal-unsure", (_ev, btn) => {
    const personId = Number(btn.getAttribute("data-pid"));
    if (!personId) return;
    runBusy(btn, () => markFaceUnsure({ photoId: state.modalPhoto.photo_id, faceId: Number(btn.getAttribute("data-fid")), personId }), {
      busyText: btn.classList.contains("btn-modal-unsure-compact") ? "…" : "Saving…",
      alsoDisable: siblingControls(btn.closest(".modal-face-row"), btn),
      context: "Mark face unsure",
    });
  });

  on("btnCloseModal", "click", (_ev, btn) => runBusy(btn, () => closePhotoModal(), { context: "Close modal" }));
  on("photoModal", "click", (ev, modal) => {
    if (ev.target === modal) closePhotoModal();
  });
}
