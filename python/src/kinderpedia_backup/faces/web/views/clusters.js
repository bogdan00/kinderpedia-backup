// Mode 2: Unnamed classroom kids & multi-year clusters (name / confirm / ignore a cluster).
import { api, apiUrl, post } from "../api.js";
import { openClusterFaceModal } from "../components/lightbox.js";
import { showToast } from "../components/toasts.js";
import { syncModeTabs } from "../router.js";
import { setOverview, state } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml, siblingControls } from "../util/dom.js";
import { applyBatchPayload } from "./batch.js";

export async function loadClusters() {
  const data = await api(
    apiUrl("/api/clusters", {
      year: el("clusterYearFilter").value,
      archive_child: el("clusterChildFilter").value,
      include_assigned: el("clusterShowAssigned").checked ? "1" : "0",
      limit: 180,
    })
  );
  renderClusters(data.clusters || []);
}

function clusterCardHtml(c) {
  const strip = (c.sample_face_ids || []).slice(0, 6).map(
    (fid) => html`
      <button type="button" class="cluster-thumb-btn" data-cid="${c.cluster_id}" data-fid="${fid}" title="Click to view full classroom photo in context">
        <img src="/media/face/${Number(fid)}.jpg" alt="" loading="lazy" />
        <span class="cluster-thumb-zoom">🔍</span>
      </button>`
  );
  let hint = "";
  if (c.suggested_person_name && !c.person_id) {
    hint = html`<button type="button" class="cluster-hint" data-cid="${c.cluster_id}" data-name="${c.suggested_person_name}">
        Looks like ${c.suggested_person_name} (${c.suggested_similarity.toFixed(2)}) ✓
      </button>`;
  } else if (c.person_name) {
    hint = html`<span class="cluster-hint">Named: ${c.person_name}</span>`;
  }
  return html`
    <div class="cluster-card" data-cid="${c.cluster_id}">
      <div class="cluster-head">
        <span class="cluster-meta">${c.archive_child} · ${c.year} · ${c.photo_count} photos</span>
        ${hint}
      </div>
      <div class="cluster-strip">${strip}</div>
      <form class="cluster-assign-row" data-cid="${c.cluster_id}">
        <input type="text" list="knownPeopleDatalist" placeholder="This IS… (e.g. child name)" value="${c.person_name || ""}" required />
        <button type="submit" class="btn btn-approve btn-sm">Confirm &amp; Train</button>
        <button type="button" class="btn btn-ghost btn-sm btn-mute-cluster" data-cid="${c.cluster_id}" title="Hide adult / wall poster cluster">Ignore</button>
      </form>
    </div>`;
}

function renderClusters(clusters) {
  setHtml(
    el("clustersGrid"),
    clusters.length
      ? html`${clusters.map(clusterCardHtml)}`
      : html`<div class="empty-state">No unassigned clusters match these filters!</div>`
  );
}

export async function assignCluster(clusterId, personName, role, jumpToBatch) {
  const res = await post(`/api/clusters/${clusterId}/assign`, {
    new_name: personName,
    role,
    batch_size: Number(el("batchSizeSelect").value || 24),
  });
  if (jumpToBatch && role !== "ignore") {
    state.selectedPersonId = res.person_id;
    state.mode = "batch";
    setOverview(res.overview);
    syncModeTabs();
    applyBatchPayload(res.next_batch);
    showToast(`Confirmed ${res.confirmed_from_cluster} cluster photos for ${personName}! Here are similar matches to review.`);
  } else {
    setOverview(res.overview);
    await loadClusters();
    showToast("Cluster muted.");
  }
}

export function bindClusters() {
  const grid = el("clustersGrid");
  delegate(grid, "click", ".cluster-thumb-btn", (_ev, btn) => {
    const cid = Number(btn.getAttribute("data-cid"));
    const fid = Number(btn.getAttribute("data-fid"));
    runBusy(btn, () => openClusterFaceModal(cid, fid), { context: "Load cluster photo" });
  });
  delegate(grid, "click", ".cluster-hint[data-name]", (_ev, btn) => {
    const name = btn.getAttribute("data-name");
    runBusy(btn, () => assignCluster(Number(btn.getAttribute("data-cid")), name, "classmate", true), {
      busyText: `Confirming ${name}…`,
      context: "Confirm cluster",
    });
  });
  delegate(grid, "submit", ".cluster-assign-row", (ev, form) => {
    ev.preventDefault();
    const name = form.querySelector("input").value.trim();
    if (!name) return;
    const submitBtn = form.querySelector('button[type="submit"]');
    runBusy(submitBtn, () => assignCluster(Number(form.getAttribute("data-cid")), name, "classmate", true), {
      busyText: "Confirming…",
      alsoDisable: siblingControls(form, submitBtn),
      context: "Confirm cluster",
    });
  });
  delegate(grid, "click", ".btn-mute-cluster", (_ev, btn) => {
    runBusy(btn, () => assignCluster(Number(btn.getAttribute("data-cid")), "_Ignored_Background", "ignore", false), {
      busyText: "Ignoring…",
      alsoDisable: siblingControls(btn.closest(".cluster-assign-row"), btn),
      context: "Ignore cluster",
    });
  });
  for (const id of ["clusterChildFilter", "clusterYearFilter", "clusterShowAssigned"]) {
    on(id, "change", (_ev, node) => runBusy(node, () => loadClusters(), { context: "Filter clusters" }));
  }
}
