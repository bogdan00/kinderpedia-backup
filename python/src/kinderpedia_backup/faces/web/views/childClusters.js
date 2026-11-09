// Mode 1b: per-child look-alike clusters (boost / penalize / clear / exclude, per card or per selection).
import { clusterAction } from "../actions.js";
import { api, apiUrl, fetchQuietJson } from "../api.js";
import { openPhotoModal } from "../components/lightbox.js";
import { applyDedupedNav, isPhotoSeen } from "../components/lightboxNav.js";
import { getSelectedPerson, state } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml, siblingControls, toElement } from "../util/dom.js";
import { plural } from "../util/format.js";

const CHILD_CLUSTER_STRIP_MAX = 12;
const childClusterFetches = new Map(); // cluster_key -> in-flight Promise

export function childClusterIsFullyConfirmed(c) {
  const ct = c.counts || {};
  return Number(ct.confirmed || 0) >= Number(c.size || 0) && !Number(ct.auto_high || 0) && !Number(ct.candidate || 0) && !Number(ct.none || 0);
}

export function childFaceStatusMeta(status) {
  if (status === "confirmed") return { sym: "✓", label: "Confirmed" };
  if (status === "auto_high") return { sym: "A", label: "Auto match" };
  if (status === "candidate") return { sym: "?", label: "Candidate" };
  return { sym: "∅", label: "Not attributed (pressure only)" };
}

function findChildCluster(ckey) {
  const data = state.childClusters;
  if (!data || !Array.isArray(data.clusters)) return null;
  return data.clusters.find((c) => String(c.cluster_key) === String(ckey)) || null;
}

function childClusterSelection(ckey) {
  const sel = state.childClustersSelected.get(String(ckey));
  return sel && sel.size ? sel : null;
}

const childClusterIsTruncated = (c) => Boolean(c && c.faces_truncated && (c.faces || []).length < Number(c.size || 0));
const thresholdValue = () => el("childClustersThreshold").value || "0.50";

/**
 * The clusters payload ships only the first faces of each card; fetch the remainder (served from the
 * server-side cache) before expanding, opening the lightbox or acting on "all".
 */
async function ensureFullChildCluster(ckey, opts = {}) {
  const { quiet = false, noReload = false } = opts;
  const key = String(ckey);
  const data = state.childClusters;
  const c = findChildCluster(key);
  if (!c || !childClusterIsTruncated(c)) return c;
  if (childClusterFetches.has(key)) return childClusterFetches.get(key);
  const p = (async () => {
    const path = apiUrl(`/api/people/${state.childClustersPersonId}/clusters/faces`, {
      key,
      threshold: thresholdValue(),
      min_size: 3,
      offset: (c.faces || []).length,
    });
    const res = quiet
      ? await fetchQuietJson(path)
      : await api(path).catch(async (err) => {
          if (err && err.status === 404 && !noReload) {
            await reloadChildClustersPreservingUi({ quiet: true });
            return null;
          }
          throw err;
        });
    if (!res || state.childClusters !== data) return findChildCluster(key);
    if (res.signature && data.signature && res.signature !== data.signature) {
      // Data moved on server-side since this view was loaded: refresh everything instead.
      if (noReload) return c;
      await reloadChildClustersPreservingUi({ quiet: true });
      return findChildCluster(key);
    }
    c.faces = (c.faces || []).concat(res.faces || []);
    c.faces_truncated = false;
    return c;
  })();
  childClusterFetches.set(key, p);
  try {
    return await p;
  } finally {
    childClusterFetches.delete(key);
  }
}

/** Like ensureFullChildCluster, but never returns a still-truncated card (retries once after a reload). */
async function ensureFullChildClusterStrict(ckey) {
  for (let attempt = 0; attempt < 2; attempt++) {
    const c = await ensureFullChildCluster(ckey);
    if (!c) return null;
    if (!childClusterIsTruncated(c)) return c;
  }
  throw new Error("This child's clusters changed while loading — please try again");
}

export async function loadChildClusters(opts = {}) {
  const { quiet = false } = opts;
  const selP = getSelectedPerson();
  const listEl = el("childClustersList");
  if (!selP) {
    state.childClusters = null;
    state.childClustersPersonId = null;
    state.childClustersExpanded.clear();
    state.childClustersSelected.clear();
    el("childClustersHeading").textContent = "Child Clusters";
    setHtml(listEl, html`<div class="empty-state">Select a child in the left roster to see their faces grouped into look-alike clusters.</div>`);
    return;
  }
  const pid = selP.person_id;
  if (state.childClustersPersonId !== pid) {
    state.childClustersExpanded.clear();
    state.childClustersSelected.clear();
    if (!quiet) {
      state.childClusters = null;
      setHtml(listEl, html`<div class="empty-state">Grouping ${selP.display_name}'s faces…</div>`);
    }
  }
  const path = apiUrl(`/api/people/${pid}/clusters`, { threshold: thresholdValue(), min_size: 3 });
  const data = quiet ? await fetchQuietJson(path) : await api(path);
  if (!data) return;
  if (state.selectedPersonId !== pid || state.mode !== "childclusters") return; // stale response

  const prev = state.childClusters;
  if (
    prev &&
    state.childClustersPersonId === pid &&
    data.signature &&
    prev.signature === data.signature &&
    listEl.querySelector(".cc-card, .empty-state")
  ) {
    return; // nothing changed server-side: keep DOM + already-expanded faces as they are
  }
  state.childClusters = data;
  state.childClustersPersonId = pid;

  // Keep expanded cards & per-face selections whose cluster still exists:
  const byKey = new Map((data.clusters || []).map((c) => [String(c.cluster_key), c]));
  for (const key of Array.from(state.childClustersExpanded)) {
    if (!byKey.has(key)) state.childClustersExpanded.delete(key);
  }
  // Expanded cards need their full face list again (the payload only ships a preview):
  await Promise.all(
    Array.from(state.childClustersExpanded).map((key) =>
      ensureFullChildCluster(key, { quiet: true, noReload: true }).catch(() => null)
    )
  );
  if (state.childClusters !== data) return; // superseded by a newer load meanwhile
  for (const [key, sel] of Array.from(state.childClustersSelected.entries())) {
    const c = byKey.get(key);
    const alive = new Set(((c && c.faces) || []).map((f) => f.face_id));
    for (const fid of Array.from(sel)) if (!alive.has(fid)) sel.delete(fid);
    if (!sel.size) state.childClustersSelected.delete(key);
  }
  renderChildClusters();
}

export async function reloadChildClustersPreservingUi(opts = {}) {
  const panel = el("viewChildClusters");
  const prevScroll = panel ? panel.scrollTop : 0;
  await loadChildClusters(opts);
  if (panel) panel.scrollTop = prevScroll;
}

const fmt2 = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "–");

function thumbHtml(f, idx, key, sel) {
  const st = childFaceStatusMeta(f.status);
  const prCls = f.pressure > 0 ? "pr-boost" : f.pressure < 0 ? "pr-penalize" : "";
  const seen = isPhotoSeen(f.photo_id);
  const simTxt = typeof f.similarity === "number" ? ` · sim ${f.similarity.toFixed(2)}` : "";
  const prTxt = f.pressure > 0 ? " · ▲ boosted" : f.pressure < 0 ? " · ▼ penalized" : "";
  const tip = `${f.album_date || f.year || ""} · ${f.album_title || ""} — ${st.label}${simTxt}${prTxt}${seen ? " · photo already reviewed" : ""}`;
  return html`
    <button type="button" class="cluster-thumb-btn cc-thumb st-${f.status || "none"} ${prCls} ${sel && sel.has(f.face_id) ? "is-selected" : ""} ${seen ? "is-seen" : ""}" data-ckey="${key}" data-fidx="${idx}" title="${tip}">
      <img src="/media/face/${Number(f.face_id)}.jpg" alt="" loading="lazy" />
      <span class="cc-thumb-status">${st.sym}</span>
      ${f.pressure ? html`<span class="cc-thumb-pressure">${f.pressure > 0 ? "▲" : "▼"}</span>` : ""}
    </button>`;
}

function childClusterCardHtml(c, personName) {
  const key = String(c.cluster_key);
  const expanded = state.childClustersExpanded.has(key);
  const sel = childClusterSelection(key);
  const faces = c.faces || [];
  const shown = expanded ? faces : faces.slice(0, CHILD_CLUSTER_STRIP_MAX);
  const ct = c.counts || {};
  const nBoost = Number((c.pressure || {}).boost || 0);
  const nPen = Number((c.pressure || {}).penalize || 0);
  const chips = [
    html`<span class="cc-chip cc-chip-confirmed" title="Human confirmed">✓ ${Number(ct.confirmed || 0)}</span>`,
    html`<span class="cc-chip cc-chip-auto" title="Auto matched (included)">auto ${Number(ct.auto_high || 0)}</span>`,
    html`<span class="cc-chip cc-chip-cand" title="Unreviewed candidates">cand ${Number(ct.candidate || 0)}</span>`,
    Number(ct.none || 0) > 0 &&
      html`<span class="cc-chip cc-chip-none" title="Not currently attributed (only carrying pressure)">none ${Number(ct.none)}</span>`,
    nBoost > 0 && html`<span class="cc-chip cc-chip-boost" title="Faces with boost pressure">▲ boosted ${nBoost}</span>`,
    nPen > 0 && html`<span class="cc-chip cc-chip-penalize" title="Faces with penalize pressure">▼ penalized ${nPen}</span>`,
  ];
  const totalFaces = Math.max(faces.length, Number(c.size || 0));
  const selSuffix = sel ? ` (${sel.size} selected)` : "";
  const actionBtn = (action, cls, title, label) =>
    html`<button type="button" class="btn ${cls} btn-xs" data-cc-action="${action}" data-ckey="${key}" title="${title}">${label}</button>`;
  return html`
    <div class="cluster-card cc-card ${c.is_loose ? "cc-loose" : ""} ${expanded ? "cc-expanded" : ""} ${sel ? "cc-has-selection" : ""}" data-ckey="${key}">
      <div class="cc-head">
        <span class="cluster-meta">${c.year || ""} · ${Number(c.size || faces.length)} faces${c.is_loose ? html` · <span class="cc-loose-label">Loose / ungrouped faces</span>` : ""}</span>
        <span class="cc-stats" title="Cohesion = how tightly the faces agree with each other; mean similarity = average match score to ${personName}">coh ${fmt2(c.cohesion)} · sim ${fmt2(c.mean_similarity)}</span>
      </div>
      <div class="cc-chips">${chips}</div>
      <div class="cc-thumbs ${expanded ? "cc-thumbs-all" : ""}">${shown.map((f, idx) => thumbHtml(f, idx, key, sel))}</div>
      <div class="cc-toggle-row">${
        totalFaces > CHILD_CLUSTER_STRIP_MAX &&
        html`<button type="button" class="btn btn-ghost btn-xs cc-toggle-all" data-ckey="${key}">${expanded ? "Show fewer" : `Show all ${totalFaces}`}</button>`
      }${
        sel &&
        html`<span class="cc-sel-note">${sel.size} selected <button type="button" class="btn btn-ghost btn-xs cc-clear-sel" data-ckey="${key}">Clear</button></span>`
      }</div>
      <div class="cc-actions">
        ${actionBtn("boost", "btn-tonal", "Soft positive pressure on these faces and look-alikes (not confirmed)", `▲ Boost${selSuffix}`)}
        ${actionBtn("penalize", "btn-tonal", "Soft negative pressure on these faces and look-alikes (not rejected)", `▼ Penalize${selSuffix}`)}
        ${nBoost + nPen > 0 && actionBtn("clear", "btn-ghost", "Remove boost / penalize pressure", `Clear pressure${selSuffix}`)}
        ${actionBtn("exclude", "btn-danger", `Remove from ${personName} as hard negatives`, `⊘ Exclude ${sel ? `(${sel.size} selected)` : "all"}`)}
      </div>
    </div>`;
}

export function renderChildClusters() {
  const data = state.childClusters;
  const listEl = el("childClustersList");
  const selP = getSelectedPerson();
  if (!data || !selP) return;
  const personName = selP.display_name;
  const clusters = data.clusters || [];
  const hide = el("childClustersHideConfirmed").checked;
  const visible = clusters.filter((c) => !(hide && childClusterIsFullyConfirmed(c)));
  const hiddenCount = clusters.length - visible.length;
  el("childClustersHeading").textContent = `${personName} — ${Number(data.total_faces || 0).toLocaleString()} faces in ${clusters.length} clusters`;
  if (!visible.length) {
    setHtml(
      listEl,
      clusters.length
        ? html`<div class="empty-state">All ${clusters.length} clusters of <strong>${personName}</strong> are fully confirmed. Untick <em>Hide fully confirmed clusters</em> to see them.</div>`
        : html`<div class="empty-state">No faces attributed to <strong>${personName}</strong> yet.</div>`
    );
    return;
  }
  const groups = [];
  for (const c of visible) {
    const yr = String(c.year || "Unknown year");
    if (!groups.length || groups[groups.length - 1].year !== yr) groups.push({ year: yr, items: [] });
    groups[groups.length - 1].items.push(c);
  }
  setHtml(
    listEl,
    html`${hiddenCount > 0 && html`<div class="cc-hidden-note">${plural(hiddenCount, "fully confirmed cluster")} hidden.</div>`}${groups.map(
      (g) => html`
        <div class="cc-year-group">
          <h3 class="cc-year-heading">${g.year} <span>${plural(g.items.length, "cluster")} · ${g.items.reduce((acc, c) => acc + Number(c.size || 0), 0)} faces</span></h3>
          <div class="cc-grid">${g.items.map((c) => childClusterCardHtml(c, personName))}</div>
        </div>`
    )}`
  );
}

function rerenderChildClusterCard(ckey) {
  const c = findChildCluster(ckey);
  const selP = getSelectedPerson();
  const cardEl = Array.from(el("childClustersList").querySelectorAll(".cc-card")).find(
    (node) => node.getAttribute("data-ckey") === String(ckey)
  );
  if (c && cardEl && selP) cardEl.replaceWith(toElement(childClusterCardHtml(c, selP.display_name)));
}

function openChildClusterFace(ckey, fidx, triggerEl) {
  const selP = getSelectedPerson();
  const c0 = findChildCluster(ckey);
  const anchor = c0 && (c0.faces || [])[fidx];
  if (!anchor || !selP) return;
  runBusy(
    triggerEl,
    async () => {
      const c = await ensureFullChildClusterStrict(ckey);
      if (!c) return undefined;
      const faces = c.faces || [];
      const item = faces.find((f) => f.face_id === anchor.face_id);
      if (!item) return undefined;
      state.modalNavContext = {
        type: "gallery",
        dedupe: true,
        navLabel: c.is_loose ? `${c.year} loose face` : `${c.year} cluster face`,
        ccKey: ckey,
        rawItems: faces,
        items: faces,
        index: 0,
        personId: selP.person_id,
        personName: selP.display_name,
      };
      applyDedupedNav(state.modalNavContext, item.face_id, item.photo_id);
      state.modalDirty = false;
      return openPhotoModal(item.photo_id, item.face_id || null, true);
    },
    { context: "Load full photo" }
  );
}

const BUSY_LABELS = { boost: "Boosting…", penalize: "Penalizing…", clear: "Clearing…", exclude: "Excluding…" };

function runChildClusterAction(ckey, action, btn) {
  const c0 = findChildCluster(ckey);
  const selP = getSelectedPerson();
  if (!c0 || !selP) return;
  const sel = childClusterSelection(ckey);
  const nPlanned = sel ? sel.size : Math.max((c0.faces || []).length, Number(c0.size || 0));
  if (!nPlanned) return;
  if (action === "exclude" && !window.confirm(`Remove ${plural(nPlanned, "face")} from ${selP.display_name}? They become hard negatives.`)) {
    return;
  }
  runBusy(
    btn,
    async () => {
      // Acting on "all" needs every face of the card, not just the preview strip:
      const c = sel ? c0 : await ensureFullChildClusterStrict(ckey);
      if (!c) return;
      const faces = (c.faces || [])
        .filter((f) => !sel || sel.has(f.face_id))
        .map((f) => ({ photo_id: f.photo_id, face_id: f.face_id }));
      if (!faces.length) return;
      await clusterAction(selP.person_id, action, faces);
      state.childClustersSelected.delete(String(ckey));
      await reloadChildClustersPreservingUi();
    },
    {
      busyText: BUSY_LABELS[action] || "Working…",
      alsoDisable: siblingControls(btn.closest(".cc-card"), btn, "button[data-cc-action]"),
      context: "Cluster action",
    }
  );
}

function toggleFaceSelection(ckey, fidx) {
  const face = ((findChildCluster(ckey) || {}).faces || [])[fidx];
  if (!face) return;
  const key = String(ckey);
  const sel = state.childClustersSelected.get(key) || new Set();
  if (sel.has(face.face_id)) sel.delete(face.face_id);
  else sel.add(face.face_id);
  if (sel.size) state.childClustersSelected.set(key, sel);
  else state.childClustersSelected.delete(key);
  rerenderChildClusterCard(ckey);
}

export function bindChildClusters() {
  const listEl = el("childClustersList");
  delegate(listEl, "click", ".cc-thumb", (ev, thumb) => {
    const ckey = thumb.getAttribute("data-ckey");
    const fidx = Number(thumb.getAttribute("data-fidx"));
    if (ev.shiftKey || ev.metaKey || ev.ctrlKey) {
      ev.preventDefault();
      toggleFaceSelection(ckey, fidx);
    } else {
      openChildClusterFace(ckey, fidx, thumb);
    }
  });
  delegate(listEl, "click", ".cc-toggle-all", (_ev, toggle) => {
    const ckey = String(toggle.getAttribute("data-ckey"));
    if (state.childClustersExpanded.has(ckey)) {
      state.childClustersExpanded.delete(ckey);
      rerenderChildClusterCard(ckey);
      return;
    }
    runBusy(
      toggle,
      async () => {
        await ensureFullChildCluster(ckey);
        state.childClustersExpanded.add(ckey);
        rerenderChildClusterCard(ckey);
      },
      { busyText: "Loading…", context: "Show all faces" }
    );
  });
  delegate(listEl, "click", ".cc-clear-sel", (_ev, btn) => {
    const ckey = String(btn.getAttribute("data-ckey"));
    state.childClustersSelected.delete(ckey);
    rerenderChildClusterCard(ckey);
  });
  delegate(listEl, "click", "button[data-cc-action]", (_ev, btn) =>
    runChildClusterAction(btn.getAttribute("data-ckey"), btn.getAttribute("data-cc-action"), btn)
  );
  on("childClustersThreshold", "change", (_ev, node) =>
    runBusy(node, () => reloadChildClustersPreservingUi(), { context: "Regroup clusters" })
  );
  on("childClustersHideConfirmed", "change", () => renderChildClusters());
  on("btnReloadChildClusters", "click", (_ev, node) =>
    runBusy(node, () => reloadChildClustersPreservingUi(), { context: "Reload clusters" })
  );
}
