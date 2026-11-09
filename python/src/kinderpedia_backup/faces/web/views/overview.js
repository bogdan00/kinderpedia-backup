// Header / toolbar: archive coverage, filter options, selected-child toolbar, and the global actions
// (refresh scan, discover clusters, add / merge child, ByChild sync, URL export, reconnect, view tabs).
import { api, apiUrl, downloadApiFile, post, startBackgroundJob } from "../api.js";
import { syncJobButtons } from "../components/statusChip.js";
import { showToast } from "../components/toasts.js";
import { loadActiveView, selectPerson, switchMode } from "../router.js";
import { findPersonByName, getSelectedPerson, setOverview, state, subscribe } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml } from "../util/dom.js";

function syncOptions(selectEl, items, allLabel) {
  const prev = selectEl.value;
  setHtml(selectEl, html`<option value="">${allLabel}</option>${items.map((v) => html`<option value="${v}">${v}</option>`)}`);
  if (items.includes(prev)) selectEl.value = prev;
}

export function renderOverview() {
  const ov = state.overview;
  if (!ov) return;
  el("coverageText").textContent = `${ov.coverage_pct}%`;
  el("coverageFill").style.width = `${Math.min(100, ov.coverage_pct)}%`;
  el("coverageSub").textContent = `${ov.faces_categorized.toLocaleString()} / ${ov.faces_total.toLocaleString()} faces across ${ov.photos_total.toLocaleString()} photos`;
  el("unassignedBadge").textContent = String(ov.unassigned_clusters);
  const classesBadge = el("classesBadge");
  if (classesBadge) classesBadge.textContent = String((ov.classes || []).length);
  syncOptions(el("clusterChildFilter"), ov.archive_children || [], "All Classrooms");
  syncOptions(el("clusterYearFilter"), ov.years || [], "All Years");
  syncOptions(el("galleryYearFilter"), ov.years || [], "All Years");
  const selP = getSelectedPerson();
  el("personToolbar").hidden = !selP;
  if (selP) el("activePersonChip").textContent = `${selP.display_name} (${selP.total_included_count} photos)`;
}

/** Toast for a background-job start ({ job, started } from startBackgroundJob). */
function announceJob({ job, started }) {
  if (started === true) showToast(`${job.label || "Background job"} started — keep working, you'll be notified when it's done.`);
  else if (started === false) showToast(`${job.label || "That job"} is already running — you'll be notified when it finishes.`);
}

function runJob(btn, path, body, context) {
  runBusy(btn, async () => announceJob(await startBackgroundJob(path, body)), { context, onFinally: syncJobButtons });
}

function addPerson(btn) {
  const inp = el("newPersonInput");
  const name = inp.value.trim();
  if (!name) return;
  runBusy(
    btn,
    async () => {
      const res = await post("/api/people", { name, role: "classmate" });
      inp.value = "";
      state.overview = res.overview;
      await selectPerson(res.person_id, "clusters");
      showToast(`Added ${name}. Pick a cluster or photo face to seed ${name}!`);
    },
    { busyText: "Adding…", alsoDisable: [inp], context: "Add child" }
  );
}

function mergeSelectedPerson(btn) {
  const selP = getSelectedPerson();
  if (!selP) return;
  const targetName = window.prompt(`Merge "${selP.display_name}" into which existing child name?`);
  if (!targetName) return;
  const target = findPersonByName(targetName, selP.person_id);
  if (!target) {
    showToast(`No existing child named "${targetName.trim()}".`, "error");
    return;
  }
  runBusy(
    btn,
    async () => {
      const res = await post(`/api/people/${selP.person_id}`, { merge_into_person_id: target.person_id }, "PATCH");
      state.overview = res.overview;
      await selectPerson(target.person_id, "gallery");
      showToast(`Merged into ${target.display_name}!`);
    },
    { busyText: "Merging…", context: "Merge child" }
  );
}

function exportUrls(btn) {
  const selP = getSelectedPerson();
  if (!selP) return;
  const fmt = btn.getAttribute("data-export") || "txt";
  const fallback = `${selP.display_name.replace(/\s+/g, "_")}_urls.${fmt}`;
  runBusy(
    btn,
    async () => {
      const fname = await downloadApiFile(apiUrl("/api/export-urls", { person: selP.display_name, format: fmt }), fallback);
      showToast(`Downloaded ${fname}`);
    },
    { busyText: `Exporting .${fmt}…`, context: "Export URLs" }
  );
}

export function bindOverview() {
  subscribe("overview", renderOverview);
  subscribe("backend-overview", (ov) => setOverview(ov));
  on("btnRetryBackend", "click", (_ev, btn) =>
    runBusy(
      btn,
      async () => {
        setOverview(await api("/api/overview"));
        await loadActiveView();
      },
      { busyText: "Checking…", context: "Reconnect" }
    )
  );
  delegate(document.body, "click", ".tab-btn[data-mode]", (_ev, btn) =>
    runBusy(btn, () => switchMode(btn.getAttribute("data-mode")), { context: "Switch view" })
  );
  delegate(document.body, "click", "[data-export]", (_ev, btn) => exportUrls(btn));
  on("btnRecluster", "click", (_ev, btn) => runJob(btn, "/api/recluster", {}, "Refresh scan"));
  on("btnMaterializeAll", "click", (_ev, btn) => runJob(btn, "/api/materialize", {}, "Sync all folders"));
  on("btnSyncPersonFolder", "click", (_ev, btn) => {
    const selP = getSelectedPerson();
    if (selP) runJob(btn, "/api/materialize", { person: selP.display_name }, "Sync folder");
  });
  on("btnDiscoverClusters", "click", (_ev, btn) => runBusy(btn, () => switchMode("clusters"), { context: "Load clusters" }));
  on("btnAddPerson", "click", (_ev, btn) => addPerson(btn));
  on("newPersonInput", "keydown", (ev) => {
    if (ev.key !== "Enter") return;
    ev.preventDefault();
    el("btnAddPerson").click();
  });
  on("btnMergePerson", "click", (_ev, btn) => mergeSelectedPerson(btn));
}
