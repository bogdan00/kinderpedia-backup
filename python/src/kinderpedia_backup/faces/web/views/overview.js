// Header / toolbar: archive coverage, filter options, selected-child toolbar, and the global actions
// (refresh scan, discover clusters, add / merge child, ByChild sync, URL export, reconnect, view tabs).
import { api, apiUrl, downloadApiFile, post, startBackgroundJob } from "../api.js";
import { syncJobButtons } from "../components/statusChip.js";
import { showToast } from "../components/toasts.js";
import { loadActiveView, selectPerson, switchMode } from "../router.js";
import { findPersonByName, getSelectedPerson, setOverview, state, subscribe } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml } from "../util/dom.js";
import { plural } from "../util/format.js";

const benchmarkCache = new Map();
let benchmarkOverviewRef = null;
let benchmarkReqSeq = 0;

function syncOptions(selectEl, items, allLabel) {
  const prev = selectEl.value;
  setHtml(selectEl, html`<option value="">${allLabel}</option>${items.map((v) => html`<option value="${v}">${v}</option>`)}`);
  if (items.includes(prev)) selectEl.value = prev;
}

function tierForPct(pct, role) {
  if (role === "educator") return { cls: "tier-good", note: "educator" };
  if (pct == null) return { cls: "tier-low", note: "no class baseline" };
  if (pct >= 70) return { cls: "tier-good", note: "close to benchmark" };
  if (pct >= 40) return { cls: "tier-mid", note: "gap — may need tagging" };
  return { cls: "tier-low", note: "big gap — likely untagged" };
}

function renderBenchmarkData(data) {
  const box = el("exportBenchmarkBox");
  if (!box) return;
  const selP = getSelectedPerson();
  if (!selP || (data && data.person_id !== selP.person_id)) return;
  if (!data) {
    box.hidden = false;
    setHtml(box, html`<div class="benchmark-sub">Checking class benchmark…</div>`);
    return;
  }
  box.hidden = false;
  if (data.role === "ignore") {
    setHtml(box, html`<div class="benchmark-sub">Ignored group — excluded from benchmarks &amp; exports.</div>`);
    return;
  }

  const fmt = (n) => Number(n || 0).toLocaleString();
  const classes = Array.isArray(data.classes) ? data.classes : [];
  if (!classes.length) {
    const fam = (data.family_names || []).join(" / ") || "your family child";
    setHtml(
      box,
      html`<div class="benchmark-sub">
        ${fmt(data.total_exportable)} photos ready · assign to a class in <strong>🏫 Classes</strong> to compare vs ${fam}.
      </div>`
    );
    return;
  }

  if (data.is_self_benchmark) {
    setHtml(
      box,
      html`<div class="benchmark-header">
          <span>★ Family Benchmark</span>
          <strong class="benchmark-pct tier-good">${fmt(data.total_exportable)} photos</strong>
        </div>
        <div class="benchmark-track"><div class="benchmark-fill is-benchmark" style="width: 100%"></div></div>
        <div class="benchmark-sub">
          Baseline across ${plural(classes.length, "class", "classes")} (${fmt(data.class_archive_total)} archive photos)
        </div>
        <div class="benchmark-rows">
          ${classes.map(
            (c) => html`<div class="benchmark-row" title="${fmt(c.child_count)} of ${fmt(c.class_total_photos)} photos in ${c.class_name}">
              <span class="benchmark-row-name">${c.class_name}</span>
              <span class="benchmark-row-vals">${fmt(c.child_count)} (${c.pct_of_class != null ? `${Math.round(c.pct_of_class)}%` : "—"})</span>
            </div>`
          )}
        </div>`
    );
    return;
  }

  const benchNames = Array.isArray(data.benchmark_names) && data.benchmark_names.length ? data.benchmark_names : ["benchmark"];
  const benchTitle = benchNames.join(" & ");
  const multiBench = benchNames.length > 1;
  const pct = data.pct_of_benchmark != null ? Math.round(data.pct_of_benchmark) : null;
  const tier = tierForPct(pct, data.role);
  const scopeLabel = data.scope === "detected" ? "active years" : plural(classes.length, "class", "classes");
  const fillW = pct != null ? Math.min(100, Math.max(0, pct)) : 0;

  setHtml(
    box,
    html`<div class="benchmark-header" title="Compares exportable photos in each shared class against ${benchTitle} (teacher distributes photos roughly equally)">
        <span>vs ${benchTitle} (${scopeLabel})</span>
        <strong class="benchmark-pct ${tier.cls}">${pct != null ? `${pct}%` : "—"}</strong>
      </div>
      <div class="benchmark-track"><div class="benchmark-fill ${tier.cls}" style="width: ${fillW}%"></div></div>
      <div class="benchmark-sub">${fmt(data.in_classes_count)} / ${fmt(data.benchmark_count)} photos · ${tier.note}</div>
      <div class="benchmark-rows">
        ${classes.map((c) => {
          const rp = c.pct_of_benchmark != null ? Math.round(c.pct_of_benchmark) : null;
          const rt = tierForPct(rp, data.role);
          const who = multiBench && c.benchmark_name ? ` ${c.benchmark_name}` : "";
          return html`<div class="benchmark-row" title="${c.class_name}: ${fmt(c.child_count)} vs ${c.benchmark_name || "benchmark"} (${fmt(c.benchmark_count)})">
            <span class="benchmark-row-name">${c.class_name}</span>
            <span class="benchmark-row-vals">${fmt(c.child_count)} / ${fmt(c.benchmark_count)}${who} <span class="benchmark-row-pct ${rt.cls}">(${rp != null ? `${rp}%` : "—"})</span></span>
          </div>`;
        })}
      </div>
      ${data.scope === "detected" ? html`<div class="benchmark-footnote">Not ticked in <strong>🏫 Classes</strong> yet — matched by year.</div>` : ""}
      ${data.outside_classes_count > 0 ? html`<div class="benchmark-footnote">+${fmt(data.outside_classes_count)} outside enrolled classes (${fmt(data.total_exportable)} exported total)</div>` : ""}`
  );
}

async function refreshSelectedBenchmark(selP) {
  const box = el("exportBenchmarkBox");
  if (!box || !selP) {
    if (box) box.hidden = true;
    return;
  }
  if (selP.role === "ignore") {
    renderBenchmarkData({ person_id: selP.person_id, role: "ignore" });
    return;
  }
  const pid = selP.person_id;
  if (benchmarkCache.has(pid)) {
    renderBenchmarkData(benchmarkCache.get(pid));
    return;
  }
  renderBenchmarkData(null);
  const seq = ++benchmarkReqSeq;
  try {
    const data = await api(`/api/people/${pid}/benchmark`);
    if (benchmarkOverviewRef !== state.overview) return;
    benchmarkCache.set(pid, data);
    if (seq === benchmarkReqSeq || (getSelectedPerson() && getSelectedPerson().person_id === pid)) {
      renderBenchmarkData(data);
    }
  } catch {
    if (seq === benchmarkReqSeq && box && getSelectedPerson()?.person_id === pid) {
      box.hidden = false;
      setHtml(box, html`<div class="benchmark-sub">Could not load class benchmark.</div>`);
    }
  }
}

export function renderOverview() {
  const ov = state.overview;
  if (!ov) return;
  if (benchmarkOverviewRef !== ov) {
    benchmarkOverviewRef = ov;
    benchmarkCache.clear();
  }
  el("coverageText").textContent = `${ov.coverage_pct}%`;
  el("coverageFill").style.width = `${Math.min(100, ov.coverage_pct)}%`;
  const known = ov.faces_known != null ? ov.faces_known : ov.faces_categorized;
  const ignored = ov.faces_ignored != null ? ov.faces_ignored : 0;
  const subDetail = ignored > 0 ? ` (${known.toLocaleString()} tagged · ${ignored.toLocaleString()} ignored)` : "";
  el("coverageSub").textContent = `${ov.faces_categorized.toLocaleString()} / ${ov.faces_total.toLocaleString()} faces triaged${subDetail}`;
  const meterEl = el("coverageMeter");
  if (meterEl) {
    meterEl.title = `Triage progress: ${ov.faces_categorized.toLocaleString()} of ${ov.faces_total.toLocaleString()} faces settled (${known.toLocaleString()} confirmed/auto-tagged, ${ignored.toLocaleString()} ignored) across ${ov.photos_total.toLocaleString()} photos`;
  }
  el("unassignedBadge").textContent = String(ov.unassigned_clusters);
  const classesBadge = el("classesBadge");
  if (classesBadge) classesBadge.textContent = String((ov.classes || []).length);
  syncOptions(el("clusterChildFilter"), ov.archive_children || [], "All Classrooms");
  syncOptions(el("clusterYearFilter"), ov.years || [], "All Years");
  syncOptions(el("galleryYearFilter"), ov.years || [], "All Years");
  const selP = getSelectedPerson();
  el("personToolbar").hidden = !selP;
  if (selP) el("activePersonChip").textContent = `${selP.display_name} (${selP.total_included_count} photos)`;
  const topExportBtn = el("btnSyncPersonFolder");
  if (topExportBtn) topExportBtn.hidden = !selP || selP.role === "ignore";

  const exportChildBtn = el("btnExportChild");
  const exportAllBtn = el("btnMaterializeAll");
  const exportablePeopleCount = (ov.people || []).filter((p) => p.role === "family" || p.role === "classmate").length;
  if (exportChildBtn && exportAllBtn) {
    if (selP) {
      const canExportSingle = selP.role !== "ignore";
      exportChildBtn.hidden = !canExportSingle;
      if (canExportSingle && exportChildBtn.dataset.busy !== "1") {
        exportChildBtn.textContent = `Export ${selP.display_name}'s Photos`;
      }
      exportAllBtn.className = "btn btn-ghost btn-xs rail-export-all-btn";
      if (exportAllBtn.dataset.busy !== "1") {
        exportAllBtn.textContent =
          exportablePeopleCount > 0 ? `or export all ${exportablePeopleCount} children…` : "or export all children…";
      }
    } else {
      exportChildBtn.hidden = true;
      exportAllBtn.className = "btn btn-primary btn-full";
      if (exportAllBtn.dataset.busy !== "1") {
        exportAllBtn.textContent = "Export All Children";
      }
    }
  }
  refreshSelectedBenchmark(selP);
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
  on("btnExportChild", "click", (_ev, btn) => {
    const selP = getSelectedPerson();
    if (selP && selP.role !== "ignore") {
      runJob(btn, "/api/materialize", { person: selP.display_name }, `Export ${selP.display_name}'s photos`);
    }
  });
  on("btnMaterializeAll", "click", (_ev, btn) => runJob(btn, "/api/materialize", {}, "Export all children"));
  on("btnSyncPersonFolder", "click", (_ev, btn) => {
    const selP = getSelectedPerson();
    if (selP && selP.role !== "ignore") {
      runJob(btn, "/api/materialize", { person: selP.display_name }, "Export photo folder");
    }
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
