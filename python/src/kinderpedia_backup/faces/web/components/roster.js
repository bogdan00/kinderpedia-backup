// Sidebar roster (grouped by role), the known-people datalist, and the rename / merge / birthday / delete dialog.
import { api, post } from "../api.js";
import { loadActiveView, selectPerson } from "../router.js";
import {
  dropPersonFromCards,
  findPersonByName,
  people,
  renamePersonInCards,
  setBatchCards,
  setOverview,
  state,
  subscribe,
} from "../state.js";
import { runBusy } from "../util/busy.js";
import { delegate, el, html, on, setHtml } from "../util/dom.js";
import { compareByDisplayName } from "../util/format.js";
import { loadBatchForSelected } from "../views/batch.js";
import { reloadChildClustersPreservingUi } from "../views/childClusters.js";
import { showToast } from "./toasts.js";

const ROLE_GROUPS = [
  { key: "family", title: "Family Children" },
  { key: "classmate", title: "Classroom Colleagues" },
  { key: "educator", title: "Educators" },
  { key: "ignore", title: "Ignored / Muted" },
];

function renderDatalist() {
  let dl = el("knownPeopleDatalist");
  if (!dl) {
    dl = document.createElement("datalist");
    dl.id = "knownPeopleDatalist";
    document.body.appendChild(dl);
  }
  const named = people().filter((p) => p.role !== "ignore").slice().sort(compareByDisplayName);
  setHtml(dl, html`${named.map((p) => html`<option value="${p.display_name}"></option>`)}`);
}

function rosterItemHtml(p, groupKey, classNameById) {
  const classNames = (p.class_ids || []).map((cid) => classNameById.get(cid)).filter(Boolean).join(", ");
  return html`
    <div role="button" tabindex="0" class="roster-item ${p.person_id === state.selectedPersonId ? "active" : ""}" data-pid="${p.person_id}">
      ${p.cover_face_id
        ? html`<img class="roster-avatar" src="/media/face/${Number(p.cover_face_id)}.jpg" alt="" loading="lazy" />`
        : html`<div class="roster-avatar-placeholder">${p.display_name.slice(0, 1).toUpperCase()}</div>`}
      <div class="roster-meta">
        <div class="roster-name">${p.display_name}</div>
        <div class="roster-counts">${p.confirmed_count} confirmed · ${p.total_included_count} total${
          p.gender && p.gender !== "unknown"
            ? html` · <span class="roster-gender" title="Gender: ${p.gender}">${p.gender === "girl" ? "👧 girl" : p.gender === "boy" ? "👦 boy" : p.gender}</span>`
            : ""
        }${
          p.birth_date && html` · <span class="roster-birth" title="Birthday — no tags before this date">🎂 ${p.birth_date}</span>`
        }</div>
        ${classNames && html`<div class="roster-classes" title="Classes: ${classNames}">🏫 ${classNames}</div>`}
      </div>
      ${p.review_ready_count > 0 &&
      groupKey !== "ignore" &&
      html`<span class="roster-queue-pill" title="Unconfirmed batch matches ready">${p.review_ready_count}</span>`}
      <button type="button" class="roster-edit-btn" data-edit-pid="${p.person_id}" title="Rename ${p.display_name}">✎</button>
      <button type="button" class="roster-delete-btn" data-delete-pid="${p.person_id}" title="Delete ${p.display_name}">🗑</button>
    </div>`;
}

export function renderRoster() {
  const ov = state.overview;
  if (!ov) return;
  renderDatalist();
  const classNameById = new Map((ov.classes || []).map((c) => [c.class_id, c.name]));
  const groups = ROLE_GROUPS.map((g) => ({ ...g, members: ov.people.filter((p) => p.role === g.key) })).filter(
    (g) => g.members.length
  );
  setHtml(
    el("rosterList"),
    groups.length
      ? html`${groups.map(
          (g) => html`<div class="roster-section-label">${g.title} (${g.members.length})</div>${g.members.map((p) =>
            rosterItemHtml(p, g.key, classNameById)
          )}`
        )}`
      : html`<div class="empty-state">No children enrolled yet. Click <strong>Discover Unnamed Kids</strong> above to start!</div>`
  );
}

export function openRenameModal(personId) {
  const target = people().find((p) => p.person_id === personId);
  if (!target) return;
  state.renameTargetPersonId = target.person_id;
  el("renamePersonTitle").textContent = `Edit "${target.display_name}"`;
  const inp = el("renamePersonInput");
  inp.value = target.display_name;
  el("renamePersonRoleSelect").value = target.role || "classmate";
  const genderSel = el("renamePersonGenderSelect");
  if (genderSel) genderSel.value = target.gender || "unknown";
  el("renamePersonBirthInput").value = target.birth_date || "";
  el("renamePersonModal").hidden = false;
  setTimeout(() => {
    inp.focus();
    inp.select();
  }, 10);
}

export function closeRenameModal() {
  el("renamePersonModal").hidden = true;
  state.renameTargetPersonId = null;
}

const tagCount = (p) => Number(p.confirmed_count || 0) + Number(p.auto_high_count || 0) + Number(p.candidate_count || 0);

/** Refresh after a person-level change without discarding in-progress batch clicks. */
async function refreshAfterPersonChange({ reloadBatch = false } = {}) {
  if (state.mode === "batch" && state.selectedPersonId && state.batchUserTouched && !reloadBatch) {
    setBatchCards(state.batchCards, null);
  } else {
    await loadActiveView();
  }
}

async function deletePerson(personId, triggerBtn = null, extraDisable = []) {
  const target = people().find((p) => p.person_id === personId);
  if (!target) return;
  const ok = window.confirm(
    `Delete "${target.display_name}"? All ${tagCount(target)} tags will be removed and those faces become uncategorized. This cannot be undone.`
  );
  if (!ok) return;
  await runBusy(
    triggerBtn,
    async () => {
      const res = await api(`/api/people/${personId}`, { method: "DELETE" });
      if (state.renameTargetPersonId === personId) closeRenameModal();
      state.batchCards = dropPersonFromCards(state.batchCards, personId);
      if (state.selectedPersonId === personId) {
        state.selectedPersonId = null;
        state.batchCards = [];
        state.batchMeta = null;
        state.lastClickedBatchIdx = null;
        state.childClusters = null;
        state.childClustersPersonId = null;
      }
      setOverview(res.overview);
      await refreshAfterPersonChange();
      const removedNote = typeof res.removed_assignments === "number" ? ` (${res.removed_assignments} tags removed)` : "";
      showToast(`Deleted "${target.display_name}"${removedNote}.`);
    },
    { context: "Delete person", alsoDisable: extraDisable }
  );
}

const renameControls = () =>
  ["renamePersonInput", "renamePersonRoleSelect", "renamePersonBirthInput", "btnClearRenameBirth", "btnDeleteRenamePerson", "btnCancelRenamePerson"].map(el);

function mergeInto(source, target) {
  if (!window.confirm(`"${target.display_name}" already exists. Merge "${source.display_name}" into "${target.display_name}"?`)) return;
  runBusy(
    el("btnSaveRenamePerson"),
    async () => {
      const res = await post(`/api/people/${source.person_id}`, { merge_into_person_id: target.person_id }, "PATCH");
      closeRenameModal();
      if (state.selectedPersonId === source.person_id) state.selectedPersonId = target.person_id;
      setOverview(res.overview);
      await loadActiveView();
      showToast(`Merged "${source.display_name}" into ${target.display_name}!`);
    },
    { busyText: "Merging…", alsoDisable: renameControls(), context: "Merge child" }
  );
}

function submitRename() {
  const target = people().find((p) => p.person_id === state.renameTargetPersonId);
  if (!target) return;
  const oldName = target.display_name;
  const newName = el("renamePersonInput").value.trim();
  if (!newName) return;
  const existing = findPersonByName(newName, target.person_id);
  if (existing) return mergeInto(target, existing);
  const oldBirth = target.birth_date || "";
  const newBirth = (el("renamePersonBirthInput").value || "").trim();
  const birthChanged = newBirth !== oldBirth;
  if (
    birthChanged &&
    newBirth &&
    (!oldBirth || newBirth > oldBirth) &&
    !window.confirm(
      `Set birthday of "${oldName}" to ${newBirth}? Existing tags of ${oldName} on photos dated before ${newBirth} will be removed.`
    )
  ) {
    return undefined;
  }
  return runBusy(
    el("btnSaveRenamePerson"),
    async () => {
      const body = { display_name: newName, role: el("renamePersonRoleSelect").value || target.role };
      if (birthChanged) body.birth_date = newBirth;
      const res = await post(`/api/people/${target.person_id}`, body, "PATCH");
      const genderSel = el("renamePersonGenderSelect");
      const newGender = genderSel ? genderSel.value : null;
      if (newGender && newGender !== (target.gender || "unknown")) {
        await post(`/api/people/${target.person_id}/gender`, { gender: newGender });
      }
      closeRenameModal();
      const updatedName = (res.person && res.person.display_name) || newName;
      state.batchCards = renamePersonInCards(state.batchCards, target.person_id, oldName, updatedName);
      setOverview(res.overview);
      const birthRemoved = Number(res.birthday_removed || 0);
      const isSelected = state.selectedPersonId === target.person_id;
      if (state.mode === "batch") {
        if (birthRemoved > 0 && isSelected && !state.batchUserTouched) await loadBatchForSelected();
        else setBatchCards(state.batchCards, null);
      } else if (state.mode === "childclusters") {
        if (isSelected) await reloadChildClustersPreservingUi();
      } else {
        await loadActiveView();
      }
      let birthNote = "";
      if (birthRemoved > 0) birthNote = ` · birthday removed ${birthRemoved} earlier tag${birthRemoved === 1 ? "" : "s"}`;
      else if (birthChanged) birthNote = newBirth ? ` · birthday ${newBirth}` : " · birthday cleared";
      showToast(`Updated "${oldName}" → ${updatedName}${birthNote}`);
    },
    { busyText: "Saving…", alsoDisable: renameControls(), context: "Rename child" }
  );
}

export function bindRoster() {
  subscribe("overview", renderRoster);
  const rosterEl = el("rosterList");
  const pick = (rowEl) =>
    runBusy(rowEl, () => selectPerson(Number(rowEl.getAttribute("data-pid"))), { context: "Load child" });
  delegate(rosterEl, "click", ".roster-item", (ev, rowEl) => {
    const editBtn = ev.target.closest(".roster-edit-btn");
    const delBtn = ev.target.closest(".roster-delete-btn");
    if (editBtn) openRenameModal(Number(editBtn.getAttribute("data-edit-pid")));
    else if (delBtn) deletePerson(Number(delBtn.getAttribute("data-delete-pid")), delBtn);
    else pick(rowEl);
  });
  delegate(rosterEl, "keydown", ".roster-item", (ev, rowEl) => {
    if ((ev.key !== "Enter" && ev.key !== " ") || ev.target.closest(".roster-edit-btn, .roster-delete-btn")) return;
    ev.preventDefault();
    pick(rowEl);
  });

  on("renamePersonForm", "submit", (ev) => {
    ev.preventDefault();
    submitRename();
  });
  on("btnClearRenameBirth", "click", () => (el("renamePersonBirthInput").value = ""));
  on("btnDeleteRenamePerson", "click", (_ev, btn) => {
    const pid = state.renameTargetPersonId;
    if (!pid) return;
    deletePerson(
      pid,
      btn,
      ["btnSaveRenamePerson", "btnCancelRenamePerson", "renamePersonInput", "renamePersonRoleSelect", "renamePersonBirthInput", "btnClearRenameBirth"].map(el)
    );
  });
  on("btnCancelRenamePerson", "click", closeRenameModal);
  on("renamePersonModal", "click", (ev, modal) => {
    if (ev.target === modal) closeRenameModal();
  });
  on("btnRenamePerson", "click", () => openRenameModal(state.selectedPersonId));
  on("btnRenamePersonTop", "click", () => openRenameModal(state.selectedPersonId));
}
