// Classes (feed + date range) dialog and the class × child membership matrix.
import { api, post } from "../api.js";
import { showToast } from "../components/toasts.js";
import { setOverview, state } from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml, siblingControls } from "../util/dom.js";
import { plural, setsEqual } from "../util/format.js";

const CLASS_ROLE_ORDER = { family: 0, classmate: 1, educator: 2 };
const CLASS_ROLE_TITLES = { family: "Family Children", classmate: "Classroom Colleagues", educator: "Educators" };
const CLASS_SUGGEST_MIN = 3;

const overviewClasses = () => (state.overview && state.overview.classes) || [];

function classFeedOptionsHtml(selected) {
  const feeds = ((state.overview && state.overview.archive_children) || []).slice();
  if (selected && !feeds.includes(selected)) feeds.push(selected);
  return html`<option value="" ${selected ? "" : html`selected`}>Any feed</option>${feeds.map(
    (f) => html`<option value="${f}" ${f === selected ? html`selected` : ""}>${f}</option>`
  )}`;
}

function classDateRangeLabel(c) {
  if (!c.start_date && !c.end_date) return "any date";
  return `${c.start_date || "…"} → ${c.end_date || "…"}`;
}

function classOriginalMembers(cid) {
  const src = (state.classesMatrix && state.classesMatrix.classes) || overviewClasses();
  const c = src.find((x) => x.class_id === cid);
  return new Set((c && c.member_ids) || []);
}

const classMembers = (cid) => {
  if (!state.classesMembers.has(cid)) state.classesMembers.set(cid, new Set());
  return state.classesMembers.get(cid);
};

async function openClassesModal() {
  setHtml(el("newClassFeed"), classFeedOptionsHtml(""));
  renderClassesList();
  setHtml(el("classesMatrixWrap"), html`<div class="empty-state">Loading membership matrix…</div>`);
  el("classesModal").hidden = false;
  await loadClassesMatrix({ keepDirty: false });
}

export function closeClassesModal(force = false) {
  if (el("classesModal").hidden) return;
  const n = state.classesDirty.size;
  if (!force && n > 0 && !window.confirm(`You have unsaved membership changes in ${plural(n, "class", "classes")}. Discard them?`)) {
    return;
  }
  el("classesModal").hidden = true;
  state.classesDirty.clear();
  state.classesMembers.clear();
  state.classesMatrix = null;
}

async function loadClassesMatrix({ keepDirty = true } = {}) {
  const data = await api("/api/classes/matrix");
  const prevMembers = state.classesMembers;
  const prevDirty = keepDirty ? state.classesDirty : new Set();
  state.classesMatrix = data;
  const nextMembers = new Map();
  const nextDirty = new Set();
  for (const c of data.classes || []) {
    if (prevDirty.has(c.class_id) && prevMembers.has(c.class_id)) {
      nextMembers.set(c.class_id, new Set(prevMembers.get(c.class_id)));
      nextDirty.add(c.class_id);
    } else {
      nextMembers.set(c.class_id, new Set(c.member_ids || []));
    }
  }
  state.classesMembers = nextMembers;
  state.classesDirty = nextDirty;
  renderClassesList();
  renderClassesMatrix();
}

function renderClassesList() {
  const body = el("classesListBody");
  const classes = overviewClasses();
  const photoCounts = new Map(((state.classesMatrix && state.classesMatrix.classes) || []).map((c) => [c.class_id, c.photo_count]));
  if (!classes.length) {
    setHtml(body, html`<tr><td colspan="6" class="classes-empty">No classes yet — add one below or use <strong>✨ Auto-create</strong>.</td></tr>`);
    return;
  }
  setHtml(
    body,
    html`${classes.map((c) => {
      const pc = photoCounts.get(c.class_id);
      return html`
        <tr data-cid="${c.class_id}">
          <td><input type="text" class="cls-name" value="${c.name}" autocomplete="off" /></td>
          <td><select class="input-select cls-feed">${classFeedOptionsHtml(c.archive_child || "")}</select></td>
          <td><input type="date" class="cls-start" value="${c.start_date || ""}" /></td>
          <td><input type="date" class="cls-end" value="${c.end_date || ""}" /></td>
          <td class="cls-stats">${(c.member_ids || []).length} kids${typeof pc === "number" ? ` · ${pc.toLocaleString()} photos` : ""}</td>
          <td class="cls-actions">
            <button type="button" class="btn btn-tonal btn-xs cls-save" data-cid="${c.class_id}">Save</button>
            <button type="button" class="btn btn-danger btn-xs cls-delete" data-cid="${c.class_id}">Delete</button>
          </td>
        </tr>`;
    })}`
  );
}

function readClassRowFields(rowEl) {
  return {
    name: rowEl.querySelector(".cls-name").value.trim(),
    archive_child: rowEl.querySelector(".cls-feed").value || "",
    start_date: rowEl.querySelector(".cls-start").value || "",
    end_date: rowEl.querySelector(".cls-end").value || "",
  };
}

export function validateClassFields(f) {
  if (!f.name) return "Class name is required.";
  if (f.start_date && f.end_date && f.start_date > f.end_date) return "Start date must be before end date.";
  return "";
}

async function afterClassesMutation(res, toastMsg) {
  if (res) setOverview(res.overview);
  await loadClassesMatrix({ keepDirty: true });
  if (toastMsg) showToast(toastMsg);
}

function matrixHeadHtml(c) {
  const sub = `${c.archive_child || "Any feed"} · ${classDateRangeLabel(c)}`;
  return html`
    <th class="cm-col-head ${state.classesDirty.has(c.class_id) ? "is-dirty" : ""}" data-cid="${c.class_id}">
      <div class="cm-col-name" title="${c.name}">${c.name}</div>
      <div class="cm-col-sub" title="${sub}">${sub}</div>
      <div class="cm-col-tools">
        <span class="cm-col-count" data-count-cid="${c.class_id}">${classMembers(c.class_id).size} ✓</span>
        <button type="button" class="btn btn-ghost btn-xs cm-tick-suggested" data-cid="${c.class_id}" title="Tick everyone already confirmed / auto-matched on at least ${CLASS_SUGGEST_MIN} photos of this class">Tick suggested</button>
      </div>
    </th>`;
}

function matrixCellHtml(c, p, counts) {
  const hint = Number((counts[String(c.class_id)] || {})[String(p.person_id)] || 0);
  return html`
    <td class="cm-cell ${hint >= CLASS_SUGGEST_MIN ? "is-suggested" : ""}">
      <label><input type="checkbox" class="cm-check" data-cid="${c.class_id}" data-pid="${p.person_id}" ${classMembers(c.class_id).has(p.person_id) ? html`checked` : ""} />${
        hint > 0 && html`<span class="cm-hint" title="${hint} photos in this class">${hint}</span>`
      }</label>
    </td>`;
}

function renderClassesMatrix() {
  const wrap = el("classesMatrixWrap");
  const data = state.classesMatrix;
  if (!data) return;
  const prevLeft = wrap.scrollLeft;
  const prevTop = wrap.scrollTop;
  const classes = data.classes || [];
  const roleRank = (p) => (CLASS_ROLE_ORDER[p.role] !== undefined ? CLASS_ROLE_ORDER[p.role] : 9);
  const kids = (data.people || [])
    .filter((p) => p.role !== "ignore")
    .slice()
    .sort((a, b) => roleRank(a) - roleRank(b) || String(a.display_name).localeCompare(String(b.display_name)));
  if (!classes.length || !kids.length) {
    setHtml(
      wrap,
      !classes.length
        ? html`<div class="empty-state">Create a class above to assign kids to it.</div>`
        : html`<div class="empty-state">No children enrolled yet.</div>`
    );
    updateClassesDirtyUi();
    return;
  }
  const rows = [];
  let lastRole = null;
  for (const p of kids) {
    if (p.role !== lastRole) {
      lastRole = p.role;
      rows.push(
        html`<tr class="cm-group-row"><th class="cm-row-head cm-group-label">${CLASS_ROLE_TITLES[p.role] || p.role || "Other"}</th><td colspan="${classes.length}"></td></tr>`
      );
    }
    rows.push(
      html`<tr><th class="cm-row-head" title="${p.display_name}">${p.display_name}</th>${classes.map((c) => matrixCellHtml(c, p, data.counts || {}))}</tr>`
    );
  }
  setHtml(
    wrap,
    html`
      <table class="classes-matrix">
        <thead><tr><th class="cm-corner">Child \\ Class</th>${classes.map(matrixHeadHtml)}</tr></thead>
        <tbody>${rows}</tbody>
      </table>`
  );
  wrap.scrollLeft = prevLeft;
  wrap.scrollTop = prevTop;
  updateClassesDirtyUi();
}

function markClassDirtyState(cid) {
  if (setsEqual(classMembers(cid), classOriginalMembers(cid))) state.classesDirty.delete(cid);
  else state.classesDirty.add(cid);
}

function updateClassesDirtyUi() {
  const n = state.classesDirty.size;
  const saveBtn = el("btnSaveMemberships");
  const note = el("classesDirtyNote");
  if (saveBtn && saveBtn.dataset.busy !== "1") {
    saveBtn.disabled = n === 0;
    saveBtn.textContent = n ? `Save memberships (${plural(n, "class", "classes")})` : "Save memberships";
  }
  if (note) {
    note.hidden = n === 0;
    note.textContent = n ? "Unsaved changes" : "";
  }
  const wrap = el("classesMatrixWrap");
  wrap.querySelectorAll(".cm-col-head").forEach((th) => th.classList.toggle("is-dirty", state.classesDirty.has(Number(th.getAttribute("data-cid")))));
  wrap.querySelectorAll("[data-count-cid]").forEach((node) => {
    node.textContent = `${classMembers(Number(node.getAttribute("data-count-cid"))).size} ✓`;
  });
}

async function saveClassMemberships(btn) {
  const dirty = Array.from(state.classesDirty);
  if (!dirty.length) return;
  await runBusy(
    btn,
    async () => {
      let done = 0;
      for (const cid of dirty) {
        btn.textContent = `Saving ${done + 1}/${dirty.length}…`;
        const members = Array.from(classMembers(cid));
        const res = await post(`/api/classes/${cid}/members`, { person_ids: members });
        done++;
        state.classesDirty.delete(cid);
        if (res.overview) state.overview = res.overview;
        const idx = state.classesMatrix && res.class ? (state.classesMatrix.classes || []).findIndex((c) => c.class_id === cid) : -1;
        if (idx >= 0) {
          state.classesMatrix.classes[idx] = { ...state.classesMatrix.classes[idx], ...res.class, member_ids: res.class.member_ids || members };
        }
      }
      setOverview(state.overview);
      await loadClassesMatrix({ keepDirty: true });
      showToast(`Saved memberships for ${plural(done, "class", "classes")} — background rematch scheduled.`);
    },
    {
      context: "Save memberships",
      alsoDisable: [el("btnAutoCreateClasses"), el("btnAddClass")],
      onFinally: () => {
        if (btn.isConnected) updateClassesDirtyUi();
      },
    }
  );
}

function addClass(btn) {
  const fields = {
    name: el("newClassName").value.trim(),
    archive_child: el("newClassFeed").value || "",
    start_date: el("newClassStart").value || "",
    end_date: el("newClassEnd").value || "",
  };
  const err = validateClassFields(fields);
  if (err) return showToast(err, "error");
  return runBusy(
    btn,
    async () => {
      const res = await post("/api/classes", fields);
      for (const id of ["newClassName", "newClassStart", "newClassEnd"]) el(id).value = "";
      await afterClassesMutation(res, `Added class "${fields.name}".`);
    },
    {
      busyText: "Adding…",
      alsoDisable: ["newClassName", "newClassFeed", "newClassStart", "newClassEnd"].map(el),
      context: "Add class",
    }
  );
}

function saveClassRow(btn, rowEl, cid) {
  const fields = readClassRowFields(rowEl);
  const err = validateClassFields(fields);
  if (err) return showToast(err, "error");
  return runBusy(
    btn,
    async () => afterClassesMutation(await post(`/api/classes/${cid}`, fields, "PATCH"), `Saved class "${fields.name}".`),
    { busyText: "Saving…", alsoDisable: siblingControls(rowEl, btn), context: "Save class" }
  );
}

function deleteClassRow(btn, rowEl, cid) {
  const cls = overviewClasses().find((c) => c.class_id === cid);
  const clsName = cls ? cls.name : "this class";
  const nMembers = cls ? (cls.member_ids || []).length : 0;
  if (!window.confirm(`Delete class "${clsName}"? Its ${plural(nMembers, "membership")} will be removed.`)) return undefined;
  return runBusy(
    btn,
    async () => {
      const res = await api(`/api/classes/${cid}`, { method: "DELETE" });
      state.classesDirty.delete(cid);
      state.classesMembers.delete(cid);
      await afterClassesMutation(res, `Deleted class "${clsName}".`);
    },
    { busyText: "Deleting…", alsoDisable: siblingControls(rowEl, btn), context: "Delete class" }
  );
}

function tickSuggested(cid) {
  if (!state.classesMatrix) return;
  const colCounts = (state.classesMatrix.counts || {})[String(cid)] || {};
  const members = classMembers(cid);
  const eligible = new Set((state.classesMatrix.people || []).filter((p) => p.role !== "ignore").map((p) => p.person_id));
  let added = 0;
  for (const [pidStr, cnt] of Object.entries(colCounts)) {
    const pid = Number(pidStr);
    if (Number(cnt) >= CLASS_SUGGEST_MIN && eligible.has(pid) && !members.has(pid)) {
      members.add(pid);
      added++;
    }
  }
  markClassDirtyState(cid);
  renderClassesMatrix();
  showToast(
    added ? `Ticked ${plural(added, "suggested kid")} — click Save memberships to apply.` : "No additional suggestions for this class."
  );
}

export function bindClasses() {
  on("btnOpenClasses", "click", (_ev, btn) => runBusy(btn, () => openClassesModal(), { context: "Load classes" }));
  on("btnCloseClasses", "click", () => closeClassesModal());
  on("classesModal", "click", (ev, modal) => {
    if (ev.target === modal) closeClassesModal();
  });
  on("btnAutoCreateClasses", "click", (_ev, btn) =>
    runBusy(
      btn,
      async () => {
        const res = await post("/api/classes/auto", {});
        const n = Number(res.created || 0);
        await afterClassesMutation(
          res,
          n > 0 ? `✨ Created ${plural(n, "class", "classes")} by feed & school year.` : "No new classes — every feed / school year already has one."
        );
      },
      { busyText: "Creating…", context: "Auto-create classes" }
    )
  );
  on("btnAddClass", "click", (_ev, btn) => addClass(btn));
  on("newClassName", "keydown", (ev) => {
    if (ev.key !== "Enter") return;
    ev.preventDefault();
    el("btnAddClass").click();
  });
  delegate("classesListBody", "click", ".cls-save, .cls-delete", (_ev, btn) => {
    const rowEl = btn.closest("tr");
    const cid = Number(btn.getAttribute("data-cid"));
    if (!rowEl || !cid) return;
    if (btn.classList.contains("cls-save")) saveClassRow(btn, rowEl, cid);
    else deleteClassRow(btn, rowEl, cid);
  });
  delegate("classesMatrixWrap", "change", ".cm-check", (_ev, cb) => {
    const cid = Number(cb.getAttribute("data-cid"));
    const pid = Number(cb.getAttribute("data-pid"));
    if (cb.checked) classMembers(cid).add(pid);
    else classMembers(cid).delete(pid);
    markClassDirtyState(cid);
    updateClassesDirtyUi();
  });
  delegate("classesMatrixWrap", "click", ".cm-tick-suggested", (_ev, btn) => tickSuggested(Number(btn.getAttribute("data-cid"))));
  on("btnSaveMemberships", "click", (_ev, btn) => saveClassMemberships(btn));
}
