(() => {
  "use strict";

  const csrfMeta = document.querySelector('meta[name="kp-csrf"]');
  const CSRF_TOKEN = csrfMeta ? csrfMeta.getAttribute("content") || "" : "";

  const state = {
    overview: null,
    mode: "batch", // "batch" | "childclusters" | "clusters" | "gallery"
    selectedPersonId: null,
    batchMeta: null,
    batchCards: [], // { ...card, decision: "approved"|"disproved"|"unsure"|"reassigned", targetName?: string, targetPersonId?: number, alreadySavedInModal?: boolean }
    lastClickedBatchIdx: null,
    reassignContext: null,
    seenPhotoIds: new Set(),
    bgStatus: null,
    bgStatusAt: 0,
    showSeenPhotos: false,
    galleryOffset: 0,
    galleryLimit: 96,
    modalPhoto: null,
    modalHighlightFaceId: null,
    modalNavContext: null,
    modalDirty: false,
    manualDrawMode: false,
    manualDraftBox: null,
    renameTargetPersonId: null,
    backendOffline: false,
    rematchPending: false,
    rematchRunning: false,
    batchUserTouched: false,
    sseSource: null,
    // Per-child look-alike clusters view ("childclusters" mode):
    childClusters: null, // last GET /api/people/<pid>/clusters payload
    childClustersPersonId: null,
    childClustersExpanded: new Set(), // cluster_key strings
    childClustersSelected: new Map(), // cluster_key -> Set(face_id)
    // Classes modal:
    classesMatrix: null, // GET /api/classes/matrix payload
    classesMembers: new Map(), // class_id -> Set(person_id) (local working copy)
    classesDirty: new Set(), // class_ids with unsaved membership edits
  };

  const el = (id) => document.getElementById(id);

  let inFlightRequests = 0;
  let reconnectTimer = null;

  function adjustInFlight(delta) {
    inFlightRequests = Math.max(0, inFlightRequests + delta);
    const bar = el("topProgressBar");
    if (bar) {
      bar.hidden = inFlightRequests === 0;
    }
  }

  let toastTimer = null;
  function showToast(msg, kind = "info") {
    const node = el("toast");
    if (!node) return;
    node.textContent = kind === "error" && !String(msg).startsWith("⚠") ? `⚠ ${msg}` : msg;
    node.classList.toggle("toast-error", kind === "error");
    node.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(
      () => {
        node.hidden = true;
      },
      kind === "error" ? 6800 : 3400
    );
  }

  function scheduleAutoReconnectCheck() {
    if (reconnectTimer) return;
    reconnectTimer = setInterval(async () => {
      if (!state.backendOffline) {
        clearInterval(reconnectTimer);
        reconnectTimer = null;
        return;
      }
      try {
        const resp = await fetch("/api/overview", { cache: "no-store" });
        if (resp.ok) {
          const ov = await resp.json();
          state.overview = ov;
          clearBackendDisconnected(true);
          renderOverview();
        }
      } catch (_ignored) {
        // Still restarting; keep waiting quietly.
      }
    }, 3000);
  }

  function setBackendDisconnected(detailMsg) {
    state.backendOffline = true;
    const banner = el("backendAlertBanner");
    const msgEl = el("backendAlertMsg");
    if (msgEl && detailMsg) {
      msgEl.textContent = `${detailMsg} — your current selections on screen are preserved.`;
    }
    if (banner) banner.hidden = false;
    renderBgStatus();
    scheduleAutoReconnectCheck();
  }

  function clearBackendDisconnected(announce = false) {
    const wasOffline = state.backendOffline;
    state.backendOffline = false;
    const banner = el("backendAlertBanner");
    if (banner) banner.hidden = true;
    if (reconnectTimer) {
      clearInterval(reconnectTimer);
      reconnectTimer = null;
    }
    renderBgStatus();
    if (wasOffline && announce) {
      showToast("✓ Backend reconnected! You can continue right where you left off.");
    }
  }

  function reportError(err, context = "") {
    const raw = (err && err.message) || String(err || "Unknown error");
    const prefix = context ? `${context}: ` : "";
    showToast(`${prefix}${raw}`, "error");
  }

  async function runBusy(controlEl, actionFn, opts = {}) {
    const {
      busyText = null,
      context = "",
      alsoDisable = [],
      keepUpdatedLabel = false,
      onFinally = null,
    } = opts;

    if (controlEl && controlEl.dataset && controlEl.dataset.busy === "1") {
      return;
    }

    const canDisable = (node) =>
      node &&
      (node.tagName === "BUTTON" ||
        node.tagName === "INPUT" ||
        node.tagName === "SELECT" ||
        node.tagName === "TEXTAREA");

    let origHtml = null;
    let origDisabled = false;
    const extraStates = alsoDisable.map((node) => ({
      node,
      disabled: canDisable(node) ? node.disabled : false,
    }));

    if (controlEl) {
      if (controlEl.dataset) controlEl.dataset.busy = "1";
      controlEl.classList.add("is-loading");
      controlEl.setAttribute("aria-busy", "true");
      if (canDisable(controlEl)) {
        origDisabled = controlEl.disabled;
        controlEl.disabled = true;
      }
      if (busyText !== null) {
        origHtml = controlEl.innerHTML;
        controlEl.textContent = busyText;
      }
    }
    for (const st of extraStates) {
      if (canDisable(st.node)) st.node.disabled = true;
    }

    try {
      return await actionFn();
    } catch (err) {
      reportError(err, context);
      return undefined;
    } finally {
      if (controlEl) {
        if (controlEl.dataset) delete controlEl.dataset.busy;
        controlEl.classList.remove("is-loading");
        controlEl.removeAttribute("aria-busy");
        if (canDisable(controlEl) && controlEl.isConnected) {
          controlEl.disabled = origDisabled;
        }
        if (busyText !== null && origHtml !== null && !keepUpdatedLabel && controlEl.isConnected) {
          controlEl.innerHTML = origHtml;
        }
      }
      for (const st of extraStates) {
        if (canDisable(st.node) && st.node.isConnected) {
          st.node.disabled = st.disabled;
        }
      }
      if (typeof onFinally === "function") {
        onFinally();
      }
    }
  }

  async function api(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    if (opts.method && opts.method !== "GET") {
      headers["X-KP-Token"] = CSRF_TOKEN;
      headers["Content-Type"] = "application/json";
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), opts.timeoutMs || 45000);
    adjustInFlight(1);
    try {
      let resp;
      try {
        resp = await fetch(
          path,
          Object.assign({}, opts, { headers, signal: controller.signal })
        );
      } catch (netErr) {
        const msg =
          netErr && netErr.name === "AbortError"
            ? "Request timed out waiting for backend server"
            : "Cannot reach backend on 127.0.0.1:8765 (server may be restarting)";
        setBackendDisconnected(msg);
        throw new Error(msg);
      }

      if (resp.status === 502 || resp.status === 503 || resp.status === 504) {
        const msg = `Backend temporarily unavailable (HTTP ${resp.status})`;
        setBackendDisconnected(msg);
        throw new Error(msg);
      }

      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.error || `Request failed (HTTP ${resp.status})`);
      }
      clearBackendDisconnected(true);
      if (data && data.rematch_scheduled) {
        state.rematchPending = true;
        updateSyncIndicator();
      }
      return data;
    } finally {
      clearTimeout(timeoutId);
      adjustInFlight(-1);
    }
  }

  async function downloadApiFile(path, fallbackFilename) {
    adjustInFlight(1);
    try {
      let resp;
      try {
        resp = await fetch(path, { cache: "no-store" });
      } catch (_netErr) {
        const msg = "Cannot reach backend on 127.0.0.1:8765 (server may be restarting)";
        setBackendDisconnected(msg);
        throw new Error(msg);
      }
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.error || `Export failed (HTTP ${resp.status})`);
      }
      clearBackendDisconnected(true);
      const disp = resp.headers.get("Content-Disposition") || "";
      const match = disp.match(/filename="?([^";]+)"?/i);
      const filename = (match && match[1]) || fallbackFilename;
      const blob = await resp.blob();
      const objUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(objUrl), 5000);
      return filename;
    } finally {
      adjustInFlight(-1);
    }
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function batchMode() {
    const sel = el("batchModeSelect");
    return sel && sel.value === "confident" ? "confident" : "impact";
  }

  function getSelectedPerson() {
    if (!state.overview || !state.selectedPersonId) return null;
    return state.overview.people.find((p) => p.person_id === state.selectedPersonId) || null;
  }

  function resolvePersonName(personId, fallbackName = "") {
    if (personId && state.overview && state.overview.people) {
      const found = state.overview.people.find((p) => p.person_id === personId);
      if (found) return found.display_name;
    }
    return fallbackName;
  }

  function compareByDisplayName(a, b) {
    return String(a.display_name || "").localeCompare(String(b.display_name || ""), undefined, {
      sensitivity: "base",
      numeric: true,
    });
  }

  function buildPersonDatalist() {
    let dl = document.getElementById("knownPeopleDatalist");
    if (!dl) {
      dl = document.createElement("datalist");
      dl.id = "knownPeopleDatalist";
      document.body.appendChild(dl);
    }
    const people = (state.overview && state.overview.people) || [];
    dl.innerHTML = people
      .filter((p) => p.role !== "ignore")
      .slice()
      .sort(compareByDisplayName)
      .map((p) => `<option value="${escapeHtml(p.display_name)}"></option>`)
      .join("");
  }

  function renderOverview() {
    const ov = state.overview;
    if (!ov) return;
    buildPersonDatalist();

    el("coverageText").textContent = `${ov.coverage_pct}%`;
    el("coverageFill").style.width = `${Math.min(100, ov.coverage_pct)}%`;
    el("coverageSub").textContent = `${ov.faces_categorized.toLocaleString()} / ${ov.faces_total.toLocaleString()} faces across ${ov.photos_total.toLocaleString()} photos`;
    el("unassignedBadge").textContent = String(ov.unassigned_clusters);
    const classesBadge = el("classesBadge");
    if (classesBadge) classesBadge.textContent = String((ov.classes || []).length);
    const classNameById = new Map((ov.classes || []).map((c) => [c.class_id, c.name]));

    // Populate year & classroom dropdowns once or keep selection:
    const syncOptions = (selectEl, items, allLabel) => {
      const prev = selectEl.value;
      selectEl.innerHTML =
        `<option value="">${escapeHtml(allLabel)}</option>` +
        items.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
      if (items.includes(prev)) selectEl.value = prev;
    };
    syncOptions(el("clusterChildFilter"), ov.archive_children || [], "All Classrooms");
    syncOptions(el("clusterYearFilter"), ov.years || [], "All Years");
    syncOptions(el("galleryYearFilter"), ov.years || [], "All Years");

    // Roster list grouped by role:
    const groups = [
      { key: "family", title: "Family Children" },
      { key: "classmate", title: "Classroom Colleagues" },
      { key: "educator", title: "Educators" },
      { key: "ignore", title: "Ignored / Muted" },
    ];
    const rosterEl = el("rosterList");
    let html = "";
    for (const grp of groups) {
      const members = ov.people.filter((p) => p.role === grp.key);
      if (!members.length) continue;
      html += `<div class="roster-section-label">${escapeHtml(grp.title)} (${members.length})</div>`;
      for (const p of members) {
        const isAct = p.person_id === state.selectedPersonId ? "active" : "";
        const avatarHtml = p.cover_face_id
          ? `<img class="roster-avatar" src="/media/face/${p.cover_face_id}.jpg" alt="" loading="lazy" />`
          : `<div class="roster-avatar-placeholder">${escapeHtml(p.display_name.slice(0, 1).toUpperCase())}</div>`;
        const queueHtml =
          p.review_ready_count > 0 && grp.key !== "ignore"
            ? `<span class="roster-queue-pill" title="Unconfirmed batch matches ready">${p.review_ready_count}</span>`
            : "";
        const classNames = (p.class_ids || [])
          .map((cid) => classNameById.get(cid))
          .filter(Boolean);
        const birthHtml = p.birth_date
          ? ` · <span class="roster-birth" title="Birthday — no tags before this date">🎂 ${escapeHtml(p.birth_date)}</span>`
          : "";
        const classHtml = classNames.length
          ? `<div class="roster-classes" title="Classes: ${escapeHtml(classNames.join(", "))}">🏫 ${escapeHtml(classNames.join(", "))}</div>`
          : "";
        html += `
          <div role="button" tabindex="0" class="roster-item ${isAct}" data-pid="${p.person_id}">
            ${avatarHtml}
            <div class="roster-meta">
              <div class="roster-name">${escapeHtml(p.display_name)}</div>
              <div class="roster-counts">${p.confirmed_count} confirmed · ${p.total_included_count} total${birthHtml}</div>
              ${classHtml}
            </div>
            ${queueHtml}
            <button type="button" class="roster-edit-btn" data-edit-pid="${p.person_id}" title="Rename ${escapeHtml(p.display_name)}">✎</button>
            <button type="button" class="roster-delete-btn" data-delete-pid="${p.person_id}" title="Delete ${escapeHtml(p.display_name)}">🗑</button>
          </div>`;
      }
    }
    if (!html) {
      html = `<div class="empty-state">No children enrolled yet. Click <strong>Discover Unnamed Kids</strong> above to start!</div>`;
    }
    rosterEl.innerHTML = html;

    rosterEl.querySelectorAll(".roster-edit-btn").forEach((editBtn) => {
      editBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const pid = Number(editBtn.getAttribute("data-edit-pid"));
        openRenameModal(pid);
      });
    });

    rosterEl.querySelectorAll(".roster-delete-btn").forEach((delBtn) => {
      delBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const pid = Number(delBtn.getAttribute("data-delete-pid"));
        deletePerson(pid, delBtn);
      });
    });

    rosterEl.querySelectorAll(".roster-item").forEach((rowEl) => {
      rowEl.addEventListener("click", () => {
        const pid = Number(rowEl.getAttribute("data-pid"));
        runBusy(rowEl, () => selectPerson(pid), { context: "Load child" });
      });
      rowEl.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          if (
            ev.target &&
            ev.target.classList &&
            (ev.target.classList.contains("roster-edit-btn") ||
              ev.target.classList.contains("roster-delete-btn"))
          ) {
            return;
          }
          ev.preventDefault();
          const pid = Number(rowEl.getAttribute("data-pid"));
          runBusy(rowEl, () => selectPerson(pid), { context: "Load child" });
        }
      });
    });

    const selP = getSelectedPerson();
    const toolbar = el("personToolbar");
    if (selP) {
      toolbar.hidden = false;
      el("activePersonChip").textContent = `${selP.display_name} (${selP.total_included_count} photos)`;
    } else {
      toolbar.hidden = true;
    }
  }

  function openRenameModal(personId) {
    if (!state.overview || !personId) return;
    const target = (state.overview.people || []).find((p) => p.person_id === personId);
    if (!target) return;
    state.renameTargetPersonId = target.person_id;
    el("renamePersonTitle").textContent = `Edit "${target.display_name}"`;
    const inp = el("renamePersonInput");
    inp.value = target.display_name;
    el("renamePersonRoleSelect").value = target.role || "classmate";
    el("renamePersonBirthInput").value = target.birth_date || "";
    el("renamePersonModal").hidden = false;
    setTimeout(() => {
      inp.focus();
      inp.select();
    }, 10);
  }

  function closeRenameModal() {
    el("renamePersonModal").hidden = true;
    state.renameTargetPersonId = null;
  }

  function personTagCount(p) {
    if (!p) return 0;
    return (
      Number(p.confirmed_count || 0) +
      Number(p.auto_high_count || 0) +
      Number(p.candidate_count || 0)
    );
  }

  async function deletePerson(personId, triggerBtn = null, extraDisable = []) {
    if (!state.overview || !personId) return false;
    const target = (state.overview.people || []).find((p) => p.person_id === personId);
    if (!target) return false;
    const n = personTagCount(target);
    const ok = window.confirm(
      `Delete "${target.display_name}"? All ${n} tags will be removed and those faces become uncategorized. This cannot be undone.`
    );
    if (!ok) return false;
    let deleted = false;
    await runBusy(
      triggerBtn,
      async () => {
        const res = await api(`/api/people/${personId}`, { method: "DELETE" });
        deleted = true;
        if (res.overview) state.overview = res.overview;
        if (state.renameTargetPersonId === personId) closeRenameModal();
        for (const c of state.batchCards) {
          if (c.decision === "reassigned" && c.targetPersonId === personId && !c.alreadySavedInModal) {
            c.decision = "unsure";
            c.targetPersonId = null;
            c.targetName = "";
          }
        }
        if (state.selectedPersonId === personId) {
          state.selectedPersonId = null;
          state.batchCards = [];
          state.batchMeta = null;
          state.lastClickedBatchIdx = null;
          state.childClusters = null;
          state.childClustersPersonId = null;
        }
        renderOverview();
        if (state.mode === "batch" && state.selectedPersonId && state.batchUserTouched) {
          // Keep the user's in-progress batch decisions; just refresh labels.
          renderBatchGrid(state.batchMeta);
        } else {
          await loadActiveView();
        }
        const removedNote =
          typeof res.removed_assignments === "number" ? ` (${res.removed_assignments} tags removed)` : "";
        showToast(`Deleted "${target.display_name}"${removedNote}.`);
      },
      { context: "Delete person", alsoDisable: extraDisable }
    );
    return deleted;
  }

  async function selectPerson(personId, switchMode = null) {
    state.selectedPersonId = personId;
    state.galleryOffset = 0;
    if (switchMode) state.mode = switchMode;
    else if (state.mode === "clusters") state.mode = "batch";
    renderOverview();
    syncModeTabs();
    if (state.mode === "batch") {
      await loadBatchForSelected();
    } else if (state.mode === "childclusters") {
      await loadChildClusters();
    } else if (state.mode === "gallery") {
      await loadGallery();
    }
  }

  function syncModeTabs() {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-mode") === state.mode);
    });
    el("viewBatch").hidden = state.mode !== "batch";
    el("viewChildClusters").hidden = state.mode !== "childclusters";
    el("viewClusters").hidden = state.mode !== "clusters";
    el("viewGallery").hidden = state.mode !== "gallery";
  }

  async function loadActiveView() {
    if (state.mode === "batch") await loadBatchForSelected();
    else if (state.mode === "childclusters") await loadChildClusters();
    else if (state.mode === "clusters") await loadClusters();
    else await loadGallery();
  }

  async function switchMode(newMode) {
    state.mode = newMode;
    syncModeTabs();
    await loadActiveView();
  }

  // ------------------------------------------------------------------
  // Mode 1: Active-Learning Batch Trainer ("Approve All Except 1-2")
  // ------------------------------------------------------------------

  function applyBatchPayload(batchData) {
    state.batchMeta = batchData;
    state.batchCards = (batchData.cards || []).map((c) => ({
      ...c,
      decision: c.default_approved ? "approved" : "disproved",
      targetName: "",
      targetPersonId: null,
      alreadySavedInModal: false,
    }));
    state.batchUserTouched = false;
    state.lastClickedBatchIdx = null;
    renderBatchGrid(batchData);
  }

  async function loadBatchForSelected() {
    const selP = getSelectedPerson();
    if (!selP) {
      el("batchControls").hidden = true;
      el("batchHeading").textContent = "Select a child on the left (or name an Unnamed Cluster)";
      el("batchGrid").innerHTML = `<div class="empty-state">Pick any child in the left sidebar or open <strong>Unnamed Kids &amp; Years</strong> to start categorizing.</div>`;
      return;
    }
    const size = Number(el("batchSizeSelect").value || 24);
    const data = await api(`/api/batch?person_id=${selP.person_id}&size=${size}&mode=${batchMode()}`);
    applyBatchPayload(data);
  }

  function updateBatchButtonLabel() {
    const selP = getSelectedPerson();
    if (!selP) return;
    let nApp = 0;
    let nDis = 0;
    let nUns = 0;
    let nRe = 0;
    for (const c of state.batchCards) {
      if (c.decision === "approved") nApp++;
      else if (c.decision === "unsure") nUns++;
      else if (c.decision === "reassigned") nRe++;
      else nDis++;
    }
    const parts = [`${nApp} ✓`, `${nDis} ✗`];
    if (nUns > 0) parts.push(`${nUns} ?`);
    if (nRe > 0) parts.push(`${nRe} ↪`);
    el("btnSubmitBatch").textContent = `Submit Batch (${parts.join(", ")})`;
  }

  function openReassignModal(ctx) {
    if (!ctx || !ctx.photoId) return;
    state.reassignContext = ctx;
    const thumbSrc = ctx.faceId ? `/media/face/${ctx.faceId}.jpg` : `/media/photo/${ctx.photoId}`;
    el("reassignThumb").src = thumbSrc;

    const subEl = el("reassignSub");
    if (subEl) {
      subEl.textContent = ctx.fromPersonName
        ? `Move out of ${ctx.fromPersonName} and update this face's tag immediately:`
        : "Pick an existing label below or type a new child name:";
    }

    const people = ((state.overview && state.overview.people) || [])
      .filter((p) => p.role !== "ignore" && (!ctx.fromPersonId || p.person_id !== ctx.fromPersonId))
      .slice()
      .sort(compareByDisplayName);
    const chipsEl = el("reassignChips");
    if (people.length > 0) {
      chipsEl.innerHTML = people
        .map(
          (p) =>
            `<button type="button" class="reassign-chip" data-pid="${p.person_id}" data-name="${escapeHtml(p.display_name)}">${escapeHtml(p.display_name)}</button>`
        )
        .join("");
      chipsEl.querySelectorAll(".reassign-chip").forEach((btn) => {
        btn.addEventListener("click", () => {
          const pid = Number(btn.getAttribute("data-pid"));
          const name = btn.getAttribute("data-name") || "";
          applyReassignChoice(pid, name, btn);
        });
      });
    } else {
      chipsEl.innerHTML = `<span class="reassign-sub">No other children enrolled yet — type a name below!</span>`;
    }

    const inp = el("reassignInput");
    inp.value = ctx.initialName || "";
    el("reassignModal").hidden = false;
    setTimeout(() => {
      inp.focus();
      inp.select();
    }, 20);
  }

  function closeReassignModal() {
    el("reassignModal").hidden = true;
    state.reassignContext = null;
  }

  async function applyReassignChoice(targetPid, targetName, triggerBtn = null) {
    const ctx = state.reassignContext;
    if (!ctx || !ctx.photoId) {
      closeReassignModal();
      return;
    }
    const clean = (targetName || "").trim();
    if (!clean) return;
    let resolvedPid = targetPid || null;
    if (!resolvedPid && state.overview && state.overview.people) {
      const existing = state.overview.people.find(
        (p) => p.display_name.toLowerCase() === clean.toLowerCase()
      );
      if (existing) {
        resolvedPid = existing.person_id;
      }
    }

    const activeBtn = triggerBtn || el("btnSubmitReassign");
    const allModalControls = [
      el("reassignInput"),
      el("btnSubmitReassign"),
      el("btnCancelReassign"),
      ...Array.from(el("reassignChips").querySelectorAll("button")),
    ].filter((node) => node && node !== activeBtn);

    await runBusy(
      activeBtn,
      async () => {
        const payload = {
          photo_id: ctx.photoId,
          face_id: ctx.faceId || null,
          replace_person_id: ctx.fromPersonId || null,
          status: "confirmed",
        };
        if (resolvedPid) {
          payload.person_id = resolvedPid;
        } else {
          payload.new_name = clean;
        }

        const res = await api("/api/faces/tag", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        state.overview = res.overview;
        renderOverview();

        const savedFaceId = res.face_id || ctx.faceId || null;
        const finalName = resolvePersonName(res.person_id, clean);
        closeReassignModal();

        if (ctx.source === "batch" && ctx.cardIdx !== null && state.batchCards[ctx.cardIdx]) {
          const card = state.batchCards[ctx.cardIdx];
          if (savedFaceId && !card.face_id) {
            card.face_id = savedFaceId;
          }
          card.decision = "reassigned";
          card.targetPersonId = res.person_id;
          card.targetName = finalName;
          card.alreadySavedInModal = true;
        }
        syncBatchCardFromModalTag(ctx.photoId, savedFaceId, res.person_id, finalName, "confirmed");

        if (!el("photoModal").hidden && state.modalPhoto && state.modalPhoto.photo_id === ctx.photoId) {
          state.modalDirty = true;
          state.modalPhoto = res.photo;
          renderModalNavControls();
          renderModalFaceOverlays();
          renderModalSideFaces();
        }

        if (ctx.source === "gallery") {
          await loadGallery();
        } else if (state.mode === "batch") {
          renderBatchGrid(state.batchMeta);
        }

        showToast(`Reassigned to ${finalName} — face tag & gallery updated!`);
      },
      {
        busyText: "Assigning…",
        alsoDisable: allModalControls,
        context: "Reassign face",
      }
    );
  }

  function updateBatchHeaderCopy(effMeta = null) {
    const selP = getSelectedPerson();
    if (!selP || !state.batchCards.length) return;
    const meta = effMeta || state.batchMeta;
    const hiSim = state.batchCards[0].similarity.toFixed(2);
    const loSim = state.batchCards[state.batchCards.length - 1].similarity.toFixed(2);
    const remCount = Math.max(
      state.batchCards.length,
      meta ? meta.remaining_unreviewed : state.batchCards.length
    );
    el("batchHeading").textContent = `Are these ${selP.display_name}? (${state.batchCards.length} shown of ${remCount} queued · sim ${hiSim} → ${loSim})`;
    const topPeers = (meta && meta.top_peers) || [];
    const peersNote = topPeers.length
      ? `<br />Learned peers: ${topPeers
          .slice(0, 4)
          .map((p) => `<strong>${escapeHtml(p.display_name)}</strong> (${p.shared_photos})`)
          .join(", ")} — small-photo peer boost &amp; cross-classroom guard active.`
      : "";
    el("batchSubheading").innerHTML =
      `Click any wrong face for <strong>✗ Not ${escapeHtml(selP.display_name)}</strong> (hard-negative), ` +
      `click <strong>?</strong> if you're not sure (skips safely), <strong>↪</strong> for another child, or press <kbd>Enter</kbd> to submit.` +
      peersNote;
  }

  function renderBatchGrid(meta = null) {
    if (meta) state.batchMeta = meta;
    const effMeta = meta || state.batchMeta;
    const selP = getSelectedPerson();
    const grid = el("batchGrid");
    if (!selP) return;

    if (!state.batchCards.length) {
      el("batchControls").hidden = true;
      if (state.rematchPending || state.rematchRunning) {
        el("batchHeading").textContent = `Updating matches for ${selP.display_name} in the background…`;
        el("batchSubheading").textContent =
          "Scanning archive embeddings with your latest tags — new candidates (if any) will appear automatically.";
        grid.innerHTML = `<div class="empty-state">Background matcher is refreshing candidates for <strong>${escapeHtml(selP.display_name)}</strong>…</div>`;
      } else {
        el("batchHeading").textContent = `All candidate matches reviewed for ${selP.display_name}!`;
        el("batchSubheading").textContent =
          "Want to catch earlier toddler years? Check Unnamed Kids & Years to link a 2022–2024 cluster, or switch to Confirmed Gallery & Export.";
        grid.innerHTML = `<div class="empty-state">0 unreviewed candidates remaining for <strong>${escapeHtml(selP.display_name)}</strong>.</div>`;
      }
      return;
    }

    el("batchControls").hidden = false;
    updateBatchHeaderCopy(effMeta);

    grid.innerHTML = state.batchCards
      .map((c, idx) => {
        const stClass = `state-${c.decision}`;
        const liveTargetName = resolvePersonName(c.targetPersonId, c.targetName);
        const badgeText =
          c.decision === "approved"
            ? `✓ ${escapeHtml(selP.display_name)}`
            : c.decision === "unsure"
            ? `? Not sure`
            : c.decision === "reassigned"
            ? `↪ ${escapeHtml(liveTargetName)}${c.alreadySavedInModal ? " ✓" : ""}`
            : `✗ Not ${escapeHtml(selP.display_name)}`;
        const thumbSrc = c.face_id ? `/media/face/${c.face_id}.jpg` : `/media/photo/${c.photo_id}`;
        const unsureActive = c.decision === "unsure" ? "active-unsure" : "";
        const compList = Array.isArray(c.companions) ? c.companions : [];
        const peerSuffix = compList.length ? ` · with ${compList.slice(0, 2).join(", ")}` : "";
        const dateLine = `${c.album_date || c.year} · ${c.archive_child}${peerSuffix}`;
        return `
          <article class="face-card ${stClass}" data-idx="${idx}">
            <div class="face-thumb-wrap">
              <img src="${thumbSrc}" alt="" loading="lazy" />
              <span class="decision-badge">${badgeText}</span>
              <span class="sim-pill">${c.similarity.toFixed(2)} · ${escapeHtml(c.year)}</span>
              ${c.impact > 0 ? `<span class="impact-pill" title="Answering this photo also settles ~${c.impact} other open photos of this child">⚡ ${c.impact}</span>` : ""}
              <div class="card-quick-tools">
                <button type="button" class="icon-btn btn-card-unsure ${unsureActive}" data-idx="${idx}" title="I'm not sure — skip without teaching positive or negative">?</button>
                <button type="button" class="icon-btn btn-card-reassign" data-idx="${idx}" title="Actually another child…">↪</button>
                <button type="button" class="icon-btn btn-card-zoom" data-idx="${idx}" data-photo="${c.photo_id}" data-face="${c.face_id || ""}" title="Inspect full classroom group photo (←/→ to step through batch)">🔍</button>
              </div>
            </div>
            <div class="face-card-body">
              <div class="card-date" title="${escapeHtml(dateLine)}">${escapeHtml(dateLine)}</div>
              <div class="card-album" title="${escapeHtml(c.album_title)}">${escapeHtml(c.album_title)}</div>
            </div>
          </article>`;
      })
      .join("");

    grid.querySelectorAll(".face-card").forEach((cardEl) => {
      cardEl.addEventListener("click", (ev) => {
        if (ev.target.closest(".icon-btn")) return;
        const idx = Number(cardEl.getAttribute("data-idx"));
        const targetDecision =
          state.batchCards[idx].decision === "approved" ? "disproved" : "approved";
        if (ev.shiftKey && state.lastClickedBatchIdx !== null) {
          const a = Math.min(state.lastClickedBatchIdx, idx);
          const b = Math.max(state.lastClickedBatchIdx, idx);
          for (let i = a; i <= b; i++) {
            state.batchCards[i].decision = targetDecision;
            state.batchCards[i].targetName = "";
            state.batchCards[i].targetPersonId = null;
            state.batchCards[i].alreadySavedInModal = false;
          }
        } else {
          state.batchCards[idx].decision = targetDecision;
          state.batchCards[idx].targetName = "";
          state.batchCards[idx].targetPersonId = null;
          state.batchCards[idx].alreadySavedInModal = false;
        }
        state.batchUserTouched = true;
        state.lastClickedBatchIdx = idx;
        renderBatchGrid(effMeta);
      });
    });

    grid.querySelectorAll(".btn-card-unsure").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const idx = Number(btn.getAttribute("data-idx"));
        const cur = state.batchCards[idx].decision;
        state.batchCards[idx].decision = cur === "unsure" ? "approved" : "unsure";
        state.batchCards[idx].targetName = "";
        state.batchCards[idx].targetPersonId = null;
        state.batchCards[idx].alreadySavedInModal = false;
        state.batchUserTouched = true;
        state.lastClickedBatchIdx = idx;
        renderBatchGrid(effMeta);
      });
    });

    grid.querySelectorAll(".btn-card-zoom").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const idx = Number(btn.getAttribute("data-idx"));
        runBusy(
          btn,
          () => {
            state.modalNavContext = {
              type: "batch",
              cards: state.batchCards,
              index: Number.isFinite(idx) ? idx : 0,
              personId: selP.person_id,
              personName: selP.display_name,
            };
            state.modalDirty = false;
            return openPhotoModal(
              Number(btn.getAttribute("data-photo")),
              Number(btn.getAttribute("data-face")) || null,
              true
            );
          },
          { context: "Load full photo" }
        );
      });
    });

    grid.querySelectorAll(".btn-card-reassign").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const idx = Number(btn.getAttribute("data-idx"));
        const c = state.batchCards[idx];
        if (!c) return;
        openReassignModal({
          source: "batch",
          cardIdx: idx,
          photoId: c.photo_id,
          faceId: c.face_id || null,
          fromPersonId: selP ? selP.person_id : null,
          fromPersonName: selP ? selP.display_name : "",
          initialName: resolvePersonName(c.targetPersonId, c.targetName || ""),
        });
      });
    });

    updateBatchButtonLabel();
  }

  async function submitCurrentBatch() {
    const selP = getSelectedPerson();
    if (!selP || !state.batchCards.length) return;
    const btn = el("btnSubmitBatch");
    await runBusy(
      btn,
      async () => {
        const approved = [];
        const disproved = [];
        const unsure = [];
        const reassigned = [];
        for (const c of state.batchCards) {
          if (c.alreadySavedInModal) {
            // Already persisted immediately via the full-photo modal; don't overwrite!
            continue;
          }
          if (c.decision === "approved") {
            approved.push({ photo_id: c.photo_id, face_id: c.face_id });
          } else if (c.decision === "unsure") {
            unsure.push({ photo_id: c.photo_id, face_id: c.face_id });
          } else if (c.decision === "reassigned" && (c.targetPersonId || c.targetName)) {
            const payload = { photo_id: c.photo_id, face_id: c.face_id };
            if (c.targetPersonId) {
              payload.target_person_id = c.targetPersonId;
            } else {
              payload.new_target_name = resolvePersonName(c.targetPersonId, c.targetName);
            }
            reassigned.push(payload);
          } else {
            disproved.push({ photo_id: c.photo_id, face_id: c.face_id });
          }
        }
        const prevCohesion = selP.album_cohesion_count || 0;
        const size = Number(el("batchSizeSelect").value || 24);
        const res = await api("/api/batch/submit", {
          method: "POST",
          body: JSON.stringify({
            person_id: selP.person_id,
            batch_size: size,
            mode: batchMode(),
            approved,
            disproved,
            unsure,
            reassigned,
          }),
        });
        state.overview = res.overview;
        renderOverview();
        applyBatchPayload(res.next_batch);
        const updP = getSelectedPerson();
        const sweptDelta = Math.max(0, ((updP && updP.album_cohesion_count) || 0) - prevCohesion);
        const cohNote = sweptDelta > 0 ? ` (+${sweptDelta} same-album burst photos auto-matched!)` : "";
        const unsNote = res.applied.unsure > 0 ? `, ${res.applied.unsure} marked unsure` : "";
        const reNote = res.applied.reassigned > 0 ? `, ${res.applied.reassigned} reassigned` : "";
        showToast(
          `Saved: +${res.applied.approved} confirmed for ${selP.display_name}${cohNote}, ${res.applied.disproved} hard-negatives${unsNote}${reNote}.`
        );
      },
      {
        busyText: "Updating anchors…",
        keepUpdatedLabel: true,
        context: "Submit batch",
        onFinally: updateBatchButtonLabel,
      }
    );
  }

  // ------------------------------------------------------------------
  // Mode 1b: Per-Child Look-Alike Clusters (Exclude / Boost / Penalize)
  // ------------------------------------------------------------------

  const CHILD_CLUSTER_STRIP_MAX = 12;

  function childClusterIsFullyConfirmed(c) {
    const ct = c.counts || {};
    return (
      Number(ct.confirmed || 0) >= Number(c.size || 0) &&
      !Number(ct.auto_high || 0) &&
      !Number(ct.candidate || 0) &&
      !Number(ct.none || 0)
    );
  }

  function findChildCluster(ckey) {
    const data = state.childClusters;
    if (!data || !Array.isArray(data.clusters)) return null;
    return data.clusters.find((c) => String(c.cluster_key) === String(ckey)) || null;
  }

  function findChildClusterCardEl(ckey) {
    const listEl = el("childClustersList");
    if (!listEl) return null;
    return (
      Array.from(listEl.querySelectorAll(".cc-card")).find(
        (node) => node.getAttribute("data-ckey") === String(ckey)
      ) || null
    );
  }

  function childClusterSelection(ckey) {
    const sel = state.childClustersSelected.get(String(ckey));
    return sel && sel.size ? sel : null;
  }

  async function loadChildClusters(opts = {}) {
    const { quiet = false } = opts;
    const selP = getSelectedPerson();
    const listEl = el("childClustersList");
    if (!selP) {
      state.childClusters = null;
      state.childClustersPersonId = null;
      state.childClustersExpanded.clear();
      state.childClustersSelected.clear();
      el("childClustersHeading").textContent = "Child Clusters";
      listEl.innerHTML = `<div class="empty-state">Select a child in the left roster to see their faces grouped into look-alike clusters.</div>`;
      return;
    }
    const pid = selP.person_id;
    if (state.childClustersPersonId !== pid) {
      state.childClustersExpanded.clear();
      state.childClustersSelected.clear();
      if (!quiet) {
        state.childClusters = null;
        listEl.innerHTML = `<div class="empty-state">Grouping ${escapeHtml(selP.display_name)}'s faces…</div>`;
      }
    }
    const threshold = el("childClustersThreshold").value || "0.50";
    const qs = new URLSearchParams({ threshold, min_size: "3" });
    const path = `/api/people/${pid}/clusters?${qs.toString()}`;
    const data = quiet ? await fetchQuietJson(path) : await api(path);
    if (!data) return;
    if (state.selectedPersonId !== pid || state.mode !== "childclusters") return; // stale response

    state.childClusters = data;
    state.childClustersPersonId = pid;

    // Keep expanded cards & per-face selections whose cluster still exists:
    const byKey = new Map((data.clusters || []).map((c) => [String(c.cluster_key), c]));
    for (const key of Array.from(state.childClustersExpanded)) {
      if (!byKey.has(key)) state.childClustersExpanded.delete(key);
    }
    for (const [key, sel] of Array.from(state.childClustersSelected.entries())) {
      const c = byKey.get(key);
      if (!c) {
        state.childClustersSelected.delete(key);
        continue;
      }
      const alive = new Set((c.faces || []).map((f) => f.face_id));
      for (const fid of Array.from(sel)) {
        if (!alive.has(fid)) sel.delete(fid);
      }
      if (!sel.size) state.childClustersSelected.delete(key);
    }
    renderChildClusters();
  }

  async function reloadChildClustersPreservingUi(opts = {}) {
    const panel = el("viewChildClusters");
    const prevScroll = panel ? panel.scrollTop : 0;
    await loadChildClusters(opts);
    if (panel) panel.scrollTop = prevScroll;
  }

  function childFaceStatusMeta(status) {
    if (status === "confirmed") return { sym: "✓", label: "Confirmed" };
    if (status === "auto_high") return { sym: "A", label: "Auto match" };
    if (status === "candidate") return { sym: "?", label: "Candidate" };
    return { sym: "∅", label: "Not attributed (pressure only)" };
  }

  function renderChildClusterCardHtml(c, personName) {
    const key = String(c.cluster_key);
    const expanded = state.childClustersExpanded.has(key);
    const sel = childClusterSelection(key);
    const faces = c.faces || [];
    const shown = expanded ? faces : faces.slice(0, CHILD_CLUSTER_STRIP_MAX);
    const ct = c.counts || {};
    const pr = c.pressure || {};
    const nBoost = Number(pr.boost || 0);
    const nPen = Number(pr.penalize || 0);
    const hasPressure = nBoost + nPen > 0;

    const fmt = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "–");
    const chips = [
      `<span class="cc-chip cc-chip-confirmed" title="Human confirmed">✓ ${Number(ct.confirmed || 0)}</span>`,
      `<span class="cc-chip cc-chip-auto" title="Auto matched (included)">auto ${Number(ct.auto_high || 0)}</span>`,
      `<span class="cc-chip cc-chip-cand" title="Unreviewed candidates">cand ${Number(ct.candidate || 0)}</span>`,
    ];
    if (Number(ct.none || 0) > 0) {
      chips.push(
        `<span class="cc-chip cc-chip-none" title="Not currently attributed (only carrying pressure)">none ${Number(ct.none)}</span>`
      );
    }
    if (nBoost > 0) chips.push(`<span class="cc-chip cc-chip-boost" title="Faces with boost pressure">▲ boosted ${nBoost}</span>`);
    if (nPen > 0) chips.push(`<span class="cc-chip cc-chip-penalize" title="Faces with penalize pressure">▼ penalized ${nPen}</span>`);

    const thumbs = shown
      .map((f, idx) => {
        const st = childFaceStatusMeta(f.status);
        const prCls = f.pressure > 0 ? "pr-boost" : f.pressure < 0 ? "pr-penalize" : "";
        const selCls = sel && sel.has(f.face_id) ? "is-selected" : "";
        const seenCls = isPhotoSeen(f.photo_id) ? "is-seen" : "";
        const simTxt = typeof f.similarity === "number" ? ` · sim ${f.similarity.toFixed(2)}` : "";
        const prTxt = f.pressure > 0 ? " · ▲ boosted" : f.pressure < 0 ? " · ▼ penalized" : "";
        const tip = `${f.album_date || f.year || ""} · ${f.album_title || ""} — ${st.label}${simTxt}${prTxt}${seenCls ? " · photo already reviewed" : ""}`;
        return `
          <button type="button" class="cluster-thumb-btn cc-thumb st-${escapeHtml(f.status || "none")} ${prCls} ${selCls} ${seenCls}" data-ckey="${escapeHtml(key)}" data-fidx="${idx}" title="${escapeHtml(tip)}">
            <img src="/media/face/${Number(f.face_id)}.jpg" alt="" loading="lazy" />
            <span class="cc-thumb-status">${st.sym}</span>
            ${f.pressure ? `<span class="cc-thumb-pressure">${f.pressure > 0 ? "▲" : "▼"}</span>` : ""}
          </button>`;
      })
      .join("");

    const toggleHtml =
      faces.length > CHILD_CLUSTER_STRIP_MAX
        ? `<button type="button" class="btn btn-ghost btn-xs cc-toggle-all" data-ckey="${escapeHtml(key)}">${
            expanded ? "Show fewer" : `Show all ${faces.length}`
          }</button>`
        : "";

    const selSuffix = sel ? ` (${sel.size} selected)` : "";
    const selNote = sel
      ? `<span class="cc-sel-note">${sel.size} selected <button type="button" class="btn btn-ghost btn-xs cc-clear-sel" data-ckey="${escapeHtml(key)}">Clear</button></span>`
      : "";

    const looseLabel = c.is_loose ? ` · <span class="cc-loose-label">Loose / ungrouped faces</span>` : "";
    return `
      <div class="cluster-card cc-card ${c.is_loose ? "cc-loose" : ""} ${expanded ? "cc-expanded" : ""} ${sel ? "cc-has-selection" : ""}" data-ckey="${escapeHtml(key)}">
        <div class="cc-head">
          <span class="cluster-meta">${escapeHtml(c.year || "")} · ${Number(c.size || faces.length)} faces${looseLabel}</span>
          <span class="cc-stats" title="Cohesion = how tightly the faces agree with each other; mean similarity = average match score to ${escapeHtml(personName)}">coh ${fmt(c.cohesion)} · sim ${fmt(c.mean_similarity)}</span>
        </div>
        <div class="cc-chips">${chips.join("")}</div>
        <div class="cc-thumbs ${expanded ? "cc-thumbs-all" : ""}">${thumbs}</div>
        <div class="cc-toggle-row">${toggleHtml}${selNote}</div>
        <div class="cc-actions">
          <button type="button" class="btn btn-tonal btn-xs" data-cc-action="boost" data-ckey="${escapeHtml(key)}" title="Soft positive pressure on these faces and look-alikes (not confirmed)">▲ Boost${selSuffix}</button>
          <button type="button" class="btn btn-tonal btn-xs" data-cc-action="penalize" data-ckey="${escapeHtml(key)}" title="Soft negative pressure on these faces and look-alikes (not rejected)">▼ Penalize${selSuffix}</button>
          ${
            hasPressure
              ? `<button type="button" class="btn btn-ghost btn-xs" data-cc-action="clear" data-ckey="${escapeHtml(key)}" title="Remove boost / penalize pressure">Clear pressure${selSuffix}</button>`
              : ""
          }
          <button type="button" class="btn btn-danger btn-xs" data-cc-action="exclude" data-ckey="${escapeHtml(key)}" title="Remove from ${escapeHtml(personName)} as hard negatives">⊘ Exclude ${sel ? `(${sel.size} selected)` : "all"}</button>
        </div>
      </div>`;
  }

  function renderChildClusters() {
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
      listEl.innerHTML = `<div class="empty-state">${
        clusters.length
          ? `All ${clusters.length} clusters of <strong>${escapeHtml(personName)}</strong> are fully confirmed. Untick <em>Hide fully confirmed clusters</em> to see them.`
          : `No faces attributed to <strong>${escapeHtml(personName)}</strong> yet.`
      }</div>`;
      return;
    }

    const groups = [];
    let cur = null;
    for (const c of visible) {
      const yr = String(c.year || "Unknown year");
      if (!cur || cur.year !== yr) {
        cur = { year: yr, items: [] };
        groups.push(cur);
      }
      cur.items.push(c);
    }

    let html = "";
    if (hiddenCount > 0) {
      html += `<div class="cc-hidden-note">${hiddenCount} fully confirmed cluster${hiddenCount === 1 ? "" : "s"} hidden.</div>`;
    }
    for (const g of groups) {
      const nFaces = g.items.reduce((acc, c) => acc + Number(c.size || 0), 0);
      html += `
        <div class="cc-year-group">
          <h3 class="cc-year-heading">${escapeHtml(g.year)} <span>${g.items.length} cluster${g.items.length === 1 ? "" : "s"} · ${nFaces} faces</span></h3>
          <div class="cc-grid">${g.items.map((c) => renderChildClusterCardHtml(c, personName)).join("")}</div>
        </div>`;
    }
    listEl.innerHTML = html;
  }

  function rerenderChildClusterCard(ckey) {
    const c = findChildCluster(ckey);
    const cardEl = findChildClusterCardEl(ckey);
    const selP = getSelectedPerson();
    if (!c || !cardEl || !selP) return;
    const tmp = document.createElement("div");
    tmp.innerHTML = renderChildClusterCardHtml(c, selP.display_name).trim();
    const fresh = tmp.firstElementChild;
    if (fresh) cardEl.replaceWith(fresh);
  }

  function openChildClusterFace(ckey, fidx, triggerEl) {
    const c = findChildCluster(ckey);
    const selP = getSelectedPerson();
    if (!c || !selP) return;
    const faces = c.faces || [];
    const item = faces[fidx];
    if (!item) return;
    runBusy(
      triggerEl,
      () => {
        state.modalNavContext = {
          type: "gallery",
          dedupe: true,
          navLabel: c.is_loose ? `${c.year} loose face` : `${c.year} cluster face`,
          ccKey: ckey,
          rawItems: faces,
          items: faces,
          index: fidx,
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

  function runChildClusterAction(ckey, action, btn) {
    const c = findChildCluster(ckey);
    const selP = getSelectedPerson();
    if (!c || !selP) return;
    const sel = childClusterSelection(ckey);
    const chosen = (c.faces || []).filter((f) => !sel || sel.has(f.face_id));
    const faces = chosen.map((f) => ({ photo_id: f.photo_id, face_id: f.face_id }));
    if (!faces.length) return;
    const name = selP.display_name;
    if (action === "exclude") {
      const ok = window.confirm(
        `Remove ${faces.length} face${faces.length === 1 ? "" : "s"} from ${name}? They become hard negatives.`
      );
      if (!ok) return;
    }
    const cardEl = btn.closest(".cc-card");
    const others = cardEl
      ? Array.from(cardEl.querySelectorAll("button[data-cc-action]")).filter((b) => b !== btn)
      : [];
    const busyLabels = {
      boost: "Boosting…",
      penalize: "Penalizing…",
      clear: "Clearing…",
      exclude: "Excluding…",
    };
    runBusy(
      btn,
      async () => {
        const res = await api(`/api/people/${selP.person_id}/cluster-action`, {
          method: "POST",
          body: JSON.stringify({ action, faces }),
        });
        if (res.overview) {
          state.overview = res.overview;
          renderOverview();
        }
        state.childClustersSelected.delete(String(ckey));
        const n = typeof res.applied === "number" ? res.applied : faces.length;
        const plural = n === 1 ? "" : "s";
        const msgs = {
          boost: `▲ Boosted ${n} face${plural} for ${name} — look-alikes will rank higher after the background rematch.`,
          penalize: `▼ Penalized ${n} face${plural} for ${name} — look-alikes will rank lower after the background rematch.`,
          clear: `Cleared pressure on ${n} face${plural} for ${name}.`,
          exclude: `⊘ Excluded ${n} face${plural} from ${name} (hard negatives).`,
        };
        await reloadChildClustersPreservingUi();
        showToast(msgs[action] || `Applied to ${n} faces.`);
      },
      { busyText: busyLabels[action] || "Working…", alsoDisable: others, context: "Cluster action" }
    );
  }

  function bindChildClusterEvents() {
    const listEl = el("childClustersList");
    listEl.addEventListener("click", (ev) => {
      const thumb = ev.target.closest(".cc-thumb");
      if (thumb && listEl.contains(thumb)) {
        const ckey = thumb.getAttribute("data-ckey");
        const fidx = Number(thumb.getAttribute("data-fidx"));
        if (ev.shiftKey || ev.metaKey || ev.ctrlKey) {
          ev.preventDefault();
          const c = findChildCluster(ckey);
          const face = c && (c.faces || [])[fidx];
          if (!face) return;
          let sel = state.childClustersSelected.get(String(ckey));
          if (!sel) {
            sel = new Set();
            state.childClustersSelected.set(String(ckey), sel);
          }
          if (sel.has(face.face_id)) sel.delete(face.face_id);
          else sel.add(face.face_id);
          if (!sel.size) state.childClustersSelected.delete(String(ckey));
          rerenderChildClusterCard(ckey);
          return;
        }
        openChildClusterFace(ckey, fidx, thumb);
        return;
      }
      const toggle = ev.target.closest(".cc-toggle-all");
      if (toggle) {
        const ckey = String(toggle.getAttribute("data-ckey"));
        if (state.childClustersExpanded.has(ckey)) state.childClustersExpanded.delete(ckey);
        else state.childClustersExpanded.add(ckey);
        rerenderChildClusterCard(ckey);
        return;
      }
      const clearSel = ev.target.closest(".cc-clear-sel");
      if (clearSel) {
        const ckey = String(clearSel.getAttribute("data-ckey"));
        state.childClustersSelected.delete(ckey);
        rerenderChildClusterCard(ckey);
        return;
      }
      const actBtn = ev.target.closest("button[data-cc-action]");
      if (actBtn) {
        runChildClusterAction(actBtn.getAttribute("data-ckey"), actBtn.getAttribute("data-cc-action"), actBtn);
      }
    });

    el("childClustersThreshold").addEventListener("change", (ev) => {
      runBusy(ev.currentTarget, () => reloadChildClustersPreservingUi(), { context: "Regroup clusters" });
    });
    el("childClustersHideConfirmed").addEventListener("change", () => {
      renderChildClusters();
    });
    el("btnReloadChildClusters").addEventListener("click", (ev) => {
      runBusy(ev.currentTarget, () => reloadChildClustersPreservingUi(), { context: "Reload clusters" });
    });
  }

  // ------------------------------------------------------------------
  // Mode 2: Unnamed Classroom Kids & Multi-Year Clusters
  // ------------------------------------------------------------------

  async function loadClusters() {
    const year = el("clusterYearFilter").value;
    const arch = el("clusterChildFilter").value;
    const inc = el("clusterShowAssigned").checked ? "1" : "0";
    const qs = new URLSearchParams({
      year,
      archive_child: arch,
      include_assigned: inc,
      limit: "180",
    });
    const data = await api(`/api/clusters?${qs.toString()}`);
    renderClusters(data.clusters || []);
  }

  function renderClusters(clusters) {
    const grid = el("clustersGrid");
    if (!clusters.length) {
      grid.innerHTML = `<div class="empty-state">No unassigned clusters match these filters!</div>`;
      return;
    }
    grid.innerHTML = clusters
      .map((c) => {
        const samples = (c.sample_face_ids || []).slice(0, 6);
        const stripHtml = samples
          .map(
            (fid) => `
              <button type="button" class="cluster-thumb-btn" data-cid="${c.cluster_id}" data-fid="${fid}" title="Click to view full classroom photo in context">
                <img src="/media/face/${fid}.jpg" alt="" loading="lazy" />
                <span class="cluster-thumb-zoom">🔍</span>
              </button>`
          )
          .join("");
        const hintHtml =
          c.suggested_person_name && !c.person_id
            ? `<button type="button" class="cluster-hint" data-cid="${c.cluster_id}" data-name="${escapeHtml(c.suggested_person_name)}">
                Looks like ${escapeHtml(c.suggested_person_name)} (${c.suggested_similarity.toFixed(2)}) ✓
              </button>`
            : c.person_name
            ? `<span class="cluster-hint">Named: ${escapeHtml(c.person_name)}</span>`
            : "";
        return `
          <div class="cluster-card" data-cid="${c.cluster_id}">
            <div class="cluster-head">
              <span class="cluster-meta">${escapeHtml(c.archive_child)} · ${escapeHtml(c.year)} · ${c.photo_count} photos</span>
              ${hintHtml}
            </div>
            <div class="cluster-strip">${stripHtml}</div>
            <form class="cluster-assign-row" data-cid="${c.cluster_id}">
              <input type="text" list="knownPeopleDatalist" placeholder="This IS… (e.g. child name)" value="${escapeHtml(c.person_name || "")}" required />
              <button type="submit" class="btn btn-approve btn-sm">Confirm &amp; Train</button>
              <button type="button" class="btn btn-ghost btn-sm btn-mute-cluster" data-cid="${c.cluster_id}" title="Hide adult / wall poster cluster">Ignore</button>
            </form>
          </div>`;
      })
      .join("");

    grid.querySelectorAll(".cluster-thumb-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cid = Number(btn.getAttribute("data-cid"));
        const fid = Number(btn.getAttribute("data-fid"));
        runBusy(btn, () => openClusterFaceModal(cid, fid), {
          context: "Load cluster photo",
        });
      });
    });

    grid.querySelectorAll(".cluster-hint[data-name]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cid = Number(btn.getAttribute("data-cid"));
        const name = btn.getAttribute("data-name");
        runBusy(btn, () => assignCluster(cid, name, "classmate", true), {
          busyText: `Confirming ${name}…`,
          context: "Confirm cluster",
        });
      });
    });

    grid.querySelectorAll(".cluster-assign-row").forEach((form) => {
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const cid = Number(form.getAttribute("data-cid"));
        const input = form.querySelector("input");
        const submitBtn = form.querySelector('button[type="submit"]');
        const muteBtn = form.querySelector(".btn-mute-cluster");
        const name = input.value.trim();
        if (!name) return;
        runBusy(submitBtn, () => assignCluster(cid, name, "classmate", true), {
          busyText: "Confirming…",
          alsoDisable: [input, muteBtn].filter(Boolean),
          context: "Confirm cluster",
        });
      });
    });

    grid.querySelectorAll(".btn-mute-cluster").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cid = Number(btn.getAttribute("data-cid"));
        const form = btn.closest(".cluster-assign-row");
        const extra = form
          ? [form.querySelector("input"), form.querySelector('button[type="submit"]')].filter(Boolean)
          : [];
        runBusy(btn, () => assignCluster(cid, "_Ignored_Background", "ignore", false), {
          busyText: "Ignoring…",
          alsoDisable: extra,
          context: "Ignore cluster",
        });
      });
    });
  }

  async function assignCluster(clusterId, personName, role, jumpToBatch) {
    const size = Number(el("batchSizeSelect").value || 24);
    const res = await api(`/api/clusters/${clusterId}/assign`, {
      method: "POST",
      body: JSON.stringify({
        new_name: personName,
        role,
        batch_size: size,
      }),
    });
    state.overview = res.overview;
    if (jumpToBatch && role !== "ignore") {
      state.selectedPersonId = res.person_id;
      state.mode = "batch";
      renderOverview();
      syncModeTabs();
      applyBatchPayload(res.next_batch);
      showToast(
        `Confirmed ${res.confirmed_from_cluster} cluster photos for ${personName}! Here are similar matches to review.`
      );
    } else {
      renderOverview();
      await loadClusters();
      showToast("Cluster muted.");
    }
  }

  // ------------------------------------------------------------------
  // Mode 3: Confirmed Gallery, Rename/Merge & Export
  // ------------------------------------------------------------------

  async function loadGallery() {
    const selP = getSelectedPerson();
    const grid = el("galleryGrid");
    if (!selP) {
      grid.innerHTML = `<div class="empty-state">Select a child in the left roster to inspect their photo gallery.</div>`;
      el("galleryPagination").innerHTML = "";
      return;
    }
    const st = el("galleryStatusFilter").value;
    const yr = el("galleryYearFilter").value;
    const qs = new URLSearchParams({
      person_id: String(selP.person_id),
      status: st,
      year: yr,
      offset: String(state.galleryOffset),
      limit: String(state.galleryLimit),
    });
    const data = await api(`/api/gallery?${qs.toString()}`);
    el("galleryHeading").textContent = `${selP.display_name} — ${data.total.toLocaleString()} photos`;

    if (!data.items.length) {
      grid.innerHTML = `<div class="empty-state">No photos match this filter yet.</div>`;
      el("galleryPagination").innerHTML = "";
      return;
    }

    grid.innerHTML = data.items
      .map((item, idx) => {
        const thumbSrc = item.face_id
          ? `/media/face/${item.face_id}.jpg`
          : `/media/photo/${item.photo_id}`;
        const galStateClass = item.status === "unsure" ? "state-unsure" : "state-approved";
        const badgeLabel =
          item.status === "confirmed"
            ? "✓ Confirmed"
            : item.status === "unsure"
            ? "? Unsure"
            : item.source === "album_cohesion"
            ? `✓ Same-Album (${item.similarity.toFixed(2)})`
            : item.source === "model_trusted"
            ? `✓ Learned (${item.similarity.toFixed(2)})`
            : item.status === "auto_high"
            ? `Auto (${item.similarity.toFixed(2)})`
            : `Cand (${item.similarity.toFixed(2)})`;
        return `
          <article class="face-card ${galStateClass}" data-gidx="${idx}" data-photo="${item.photo_id}" data-face="${item.face_id || ""}">
            <div class="face-thumb-wrap">
              <img src="${thumbSrc}" alt="" loading="lazy" />
              <span class="decision-badge">${badgeLabel}</span>
              <span class="sim-pill">${escapeHtml(item.year)}</span>
              <div class="card-quick-tools">
                <button type="button" class="icon-btn btn-gal-reassign" data-photo="${item.photo_id}" data-face="${item.face_id || ""}" title="Actually another child… (re-tag face &amp; move)">↪</button>
                <button type="button" class="icon-btn btn-gal-remove" data-photo="${item.photo_id}" data-face="${item.face_id || ""}" title="Not ${escapeHtml(selP.display_name)} (remove)">✗</button>
                <button type="button" class="icon-btn btn-gal-zoom" data-gidx="${idx}" data-photo="${item.photo_id}" data-face="${item.face_id || ""}" title="Open full classroom photo">🔍</button>
              </div>
            </div>
            <div class="face-card-body">
              <div class="card-date">${escapeHtml(item.album_date || item.year)} · ${escapeHtml(item.archive_child)}</div>
              <div class="card-album" title="${escapeHtml(item.album_title)}">${escapeHtml(item.album_title)}</div>
            </div>
          </article>`;
      })
      .join("");

    const openGalleryItemInModal = (gidx, photoId, faceId) => {
      state.modalNavContext = {
        type: "gallery",
        items: data.items,
        index: Number.isFinite(gidx) ? gidx : 0,
        personId: selP.person_id,
        personName: selP.display_name,
      };
      state.modalDirty = false;
      return openPhotoModal(photoId, faceId, true);
    };

    grid.querySelectorAll(".face-card").forEach((card) => {
      card.addEventListener("click", (ev) => {
        if (ev.target.closest(".icon-btn")) return;
        runBusy(
          card,
          () =>
            openGalleryItemInModal(
              Number(card.getAttribute("data-gidx")),
              Number(card.getAttribute("data-photo")),
              Number(card.getAttribute("data-face")) || null
            ),
          { context: "Load full photo" }
        );
      });
    });

    grid.querySelectorAll(".btn-gal-zoom").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        runBusy(
          btn,
          () =>
            openGalleryItemInModal(
              Number(btn.getAttribute("data-gidx")),
              Number(btn.getAttribute("data-photo")),
              Number(btn.getAttribute("data-face")) || null
            ),
          { context: "Load full photo" }
        );
      });
    });

    grid.querySelectorAll(".btn-gal-reassign").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const photoId = Number(btn.getAttribute("data-photo"));
        const faceId = Number(btn.getAttribute("data-face")) || null;
        openReassignModal({
          source: "gallery",
          cardIdx: null,
          photoId,
          faceId,
          fromPersonId: selP ? selP.person_id : null,
          fromPersonName: selP ? selP.display_name : "",
          initialName: "",
        });
      });
    });

    grid.querySelectorAll(".btn-gal-remove").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const photoId = Number(btn.getAttribute("data-photo"));
        const faceId = Number(btn.getAttribute("data-face")) || null;
        runBusy(
          btn,
          async () => {
            const res = await api("/api/faces/tag", {
              method: "POST",
              body: JSON.stringify({
                photo_id: photoId,
                face_id: faceId,
                person_id: selP.person_id,
                status: "rejected",
              }),
            });
            state.overview = res.overview;
            renderOverview();
            await loadGallery();
            showToast(`Removed from ${selP.display_name} and added to hard negatives.`);
          },
          { context: "Remove photo" }
        );
      });
    });

    const pag = el("galleryPagination");
    const hasPrev = state.galleryOffset > 0;
    const hasNext = state.galleryOffset + state.galleryLimit < data.total;
    pag.innerHTML = `
      <button type="button" class="btn btn-secondary btn-sm" id="galPrev" ${hasPrev ? "" : "disabled"}>← Newer</button>
      <span>${state.galleryOffset + 1}–${Math.min(data.total, state.galleryOffset + data.items.length)} of ${data.total}</span>
      <button type="button" class="btn btn-secondary btn-sm" id="galNext" ${hasNext ? "" : "disabled"}>Older →</button>
    `;
    if (hasPrev) {
      el("galPrev").addEventListener("click", (ev) => {
        runBusy(
          ev.currentTarget,
          async () => {
            state.galleryOffset = Math.max(0, state.galleryOffset - state.galleryLimit);
            await loadGallery();
          },
          { busyText: "Loading…", context: "Load newer photos" }
        );
      });
    }
    if (hasNext) {
      el("galNext").addEventListener("click", (ev) => {
        runBusy(
          ev.currentTarget,
          async () => {
            state.galleryOffset += state.galleryLimit;
            await loadGallery();
          },
          { busyText: "Loading…", context: "Load older photos" }
        );
      });
    }
  }

  // ------------------------------------------------------------------
  // Full Classroom Photo Lightbox & Direct Any-Child Tagging
  // ------------------------------------------------------------------

  function applyPhotoModalDetail(detail, highlightFaceId = null) {
    state.modalPhoto = detail;
    state.modalHighlightFaceId = highlightFaceId;

    el("modalMetaLine").textContent = `${detail.archive_child} · ${detail.album_date || detail.year} · ${detail.filename}`;
    el("modalAlbumTitle").textContent = detail.album_title;
    const cdnBtn = el("modalCdnLink");
    if (detail.best_live_url) {
      cdnBtn.href = detail.best_live_url;
      cdnBtn.hidden = false;
    } else {
      cdnBtn.hidden = true;
    }

    const imgEl = el("modalPhotoImg");
    const stageStatus = el("photoStageStatus");
    el("faceBoxesLayer").innerHTML = "";
    const targetSrc = `/media/photo/${detail.photo_id}`;

    const setStageStatus = (text, isErr = false) => {
      if (!stageStatus) return;
      if (!text) {
        stageStatus.hidden = true;
        stageStatus.classList.remove("is-error");
        stageStatus.onclick = null;
      } else {
        stageStatus.textContent = text;
        stageStatus.classList.toggle("is-error", isErr);
        stageStatus.hidden = false;
      }
    };

    imgEl.onload = () => {
      setStageStatus("");
      renderModalFaceOverlays();
    };
    imgEl.onerror = () => {
      setStageStatus("⚠ Could not load full photo (backend may be restarting) — click to retry", true);
      if (stageStatus) {
        stageStatus.onclick = () => {
          setStageStatus("Loading photo…", false);
          imgEl.src = `${targetSrc}?t=${Date.now()}`;
        };
      }
    };

    if (imgEl.getAttribute("src") === targetSrc && imgEl.complete && imgEl.naturalWidth > 0) {
      setStageStatus("");
      renderModalFaceOverlays();
    } else {
      setStageStatus("Loading photo…", false);
      imgEl.src = targetSrc;
      if (imgEl.complete && imgEl.naturalWidth > 0) {
        setStageStatus("");
        renderModalFaceOverlays();
      }
    }

    renderModalNavControls();
    renderModalSideFaces();
    syncManualDrawUi();
    el("photoModal").hidden = false;
  }

  function setManualDrawMode(enabled) {
    state.manualDrawMode = Boolean(enabled);
    if (!state.manualDrawMode) {
      state.manualDraftBox = null;
    }
    syncManualDrawUi();
    if (state.modalPhoto && !el("photoModal").hidden) {
      renderModalFaceOverlays();
    }
  }

  function syncManualDrawUi() {
    const stage = el("photoStage");
    const btn = el("btnDrawManualTag");
    const banner = el("photoDrawBanner");
    const active = Boolean(state.manualDrawMode && !state.manualDraftBox);
    if (stage) {
      stage.classList.toggle("is-drawing", Boolean(state.manualDrawMode));
    }
    if (btn) {
      btn.classList.toggle("active", Boolean(state.manualDrawMode));
      btn.innerHTML = state.manualDrawMode ? `Cancel Draw <kbd>Esc</kbd>` : `+ Tag Child <kbd>T</kbd>`;
    }
    if (banner) {
      banner.hidden = !active;
    }
  }

  async function submitManualDrawnBox(bbox, targetPid, targetName, triggerEl = null) {
    const detail = state.modalPhoto;
    if (!detail || !bbox) return;
    const clean = (targetName || "").trim();
    if (!targetPid && !clean) return;

    let resolvedPid = targetPid || null;
    if (!resolvedPid && state.overview && state.overview.people) {
      const existing = state.overview.people.find(
        (p) => p.display_name.toLowerCase() === clean.toLowerCase()
      );
      if (existing) resolvedPid = existing.person_id;
    }

    const pop = el("faceBoxesLayer").querySelector(".manual-tag-popover");
    const allControls = pop
      ? Array.from(pop.querySelectorAll("button, input")).filter((n) => n !== triggerEl)
      : [];

    await runBusy(
      triggerEl,
      async () => {
        const body = {
          photo_id: detail.photo_id,
          box: bbox,
        };
        if (resolvedPid) {
          body.person_id = resolvedPid;
        } else {
          body.new_name = clean;
        }
        const res = await api("/api/faces/manual", {
          method: "POST",
          body: JSON.stringify(body),
        });
        state.manualDrawMode = false;
        state.manualDraftBox = null;
        state.modalDirty = true;
        state.modalPhoto = res.photo;
        state.modalHighlightFaceId = res.face_id;
        state.overview = res.overview;
        const finalName = resolvePersonName(res.person_id, clean);
        syncManualDrawUi();
        renderOverview();
        syncBatchCardFromModalTag(detail.photo_id, res.face_id, res.person_id, finalName, "confirmed");
        renderModalNavControls();
        renderModalFaceOverlays();
        renderModalSideFaces();
        showToast(`Tagged ${finalName} (manual box) — crop & ByChild/ export ready!`);
      },
      {
        busyText: "Tagging…",
        alsoDisable: allControls,
        context: "Manual face tag",
      }
    );
  }

  function appendManualDraftPopover(layer) {
    const draft = state.manualDraftBox;
    const detail = state.modalPhoto;
    if (!draft || !detail) return;

    const draftBoxEl = document.createElement("div");
    draftBoxEl.className = "bbox-overlay bbox-draft";
    draftBoxEl.style.left = `${draft.leftPct}%`;
    draftBoxEl.style.top = `${draft.topPct}%`;
    draftBoxEl.style.width = `${draft.widthPct}%`;
    draftBoxEl.style.height = `${draft.heightPct}%`;
    layer.appendChild(draftBoxEl);

    const alreadyInPhoto = new Set(
      (detail.faces || [])
        .filter((f) => f.person_id != null && f.status === "confirmed")
        .map((f) => f.person_id)
    );
    const people = ((state.overview && state.overview.people) || [])
      .filter((p) => p.role !== "ignore")
      .slice()
      .sort((a, b) => {
        const aSel = a.person_id === state.selectedPersonId && !alreadyInPhoto.has(a.person_id) ? 0 : 1;
        const bSel = b.person_id === state.selectedPersonId && !alreadyInPhoto.has(b.person_id) ? 0 : 1;
        if (aSel !== bSel) return aSel - bSel;
        const aIn = alreadyInPhoto.has(a.person_id) ? 1 : 0;
        const bIn = alreadyInPhoto.has(b.person_id) ? 1 : 0;
        if (aIn !== bIn) return aIn - bIn;
        return compareByDisplayName(a, b);
      });

    const pop = document.createElement("div");
    pop.className = "manual-tag-popover";
    const chipsHtml = people.length
      ? `<div class="manual-popover-chips">${people
          .map((p) => {
            const isPrim = p.person_id === state.selectedPersonId && !alreadyInPhoto.has(p.person_id);
            return `<button type="button" class="manual-chip ${isPrim ? "is-primary" : ""}" data-pid="${p.person_id}" data-name="${escapeHtml(p.display_name)}">${escapeHtml(p.display_name)}</button>`;
          })
          .join("")}</div>`
      : "";

    pop.innerHTML = `
      <div class="manual-popover-head">
        <span>Who is in this box?</span>
        <button type="button" class="manual-popover-close" title="Cancel box (Esc)">✕</button>
      </div>
      ${chipsHtml}
      <form class="manual-popover-form">
        <input type="text" list="knownPeopleDatalist" placeholder="Or type child name…" autocomplete="off" />
        <button type="submit" class="btn btn-primary btn-xs">Tag ✓</button>
      </form>
    `;

    // Position right below (or above if near bottom) and clamp horizontally inside stage:
    if (draft.heightPct > 70) {
      pop.style.top = "10px";
    } else if (draft.topPct + draft.heightPct > 64) {
      pop.style.bottom = `${Math.max(2, 100 - draft.topPct)}%`;
      pop.style.marginBottom = "6px";
    } else {
      pop.style.top = `${Math.min(92, draft.topPct + draft.heightPct)}%`;
      pop.style.marginTop = "6px";
    }
    layer.appendChild(pop);

    const layerW = layer.clientWidth || 320;
    const popW = pop.offsetWidth || 280;
    const desiredLeftPx = (draft.leftPct / 100) * layerW;
    const clampedLeftPx = Math.max(6, Math.min(Math.max(6, layerW - popW - 6), desiredLeftPx));
    pop.style.left = `${clampedLeftPx}px`;

    pop.addEventListener("mousedown", (ev) => ev.stopPropagation());
    pop.addEventListener("click", (ev) => ev.stopPropagation());

    const closeBtn = pop.querySelector(".manual-popover-close");
    if (closeBtn) {
      closeBtn.addEventListener("click", () => {
        state.manualDraftBox = null;
        syncManualDrawUi();
        renderModalFaceOverlays();
      });
    }

    pop.querySelectorAll(".manual-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const pid = Number(chip.getAttribute("data-pid"));
        const name = chip.getAttribute("data-name") || "";
        submitManualDrawnBox(draft.bbox, pid, name, chip);
      });
    });

    const form = pop.querySelector(".manual-popover-form");
    const inp = pop.querySelector("input");
    if (form && inp) {
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const subBtn = form.querySelector('button[type="submit"]');
        submitManualDrawnBox(draft.bbox, null, inp.value, subBtn);
      });
      setTimeout(() => inp.focus({ preventScroll: true }), 15);
    }
  }

  function bindManualBoxDrawing() {
    const layer = el("faceBoxesLayer");
    if (!layer) return;
    layer.addEventListener("mousedown", (ev) => {
      if (!state.manualDrawMode || !state.modalPhoto || ev.button !== 0) return;
      if (ev.target.closest(".manual-tag-popover")) return;

      const img = el("modalPhotoImg");
      if (!img || !img.clientWidth || !img.clientHeight) return;

      ev.preventDefault();
      ev.stopPropagation();

      state.manualDraftBox = null;
      syncManualDrawUi();
      renderModalFaceOverlays();

      const rect = layer.getBoundingClientRect();
      if (rect.width < 10 || rect.height < 10) return;
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      const startX = clamp(ev.clientX - rect.left, 0, rect.width);
      const startY = clamp(ev.clientY - rect.top, 0, rect.height);

      const draftEl = document.createElement("div");
      draftEl.className = "bbox-overlay bbox-draft";
      layer.appendChild(draftEl);

      const updateDraftRect = (clientX, clientY) => {
        const curX = clamp(clientX - rect.left, 0, rect.width);
        const curY = clamp(clientY - rect.top, 0, rect.height);
        const minX = Math.min(startX, curX);
        const maxX = Math.max(startX, curX);
        const minY = Math.min(startY, curY);
        const maxY = Math.max(startY, curY);
        draftEl.style.left = `${(100 * minX) / rect.width}%`;
        draftEl.style.top = `${(100 * minY) / rect.height}%`;
        draftEl.style.width = `${(100 * (maxX - minX)) / rect.width}%`;
        draftEl.style.height = `${(100 * (maxY - minY)) / rect.height}%`;
        return { minX, maxX, minY, maxY };
      };

      updateDraftRect(ev.clientX, ev.clientY);

      const onMove = (mev) => {
        updateDraftRect(mev.clientX, mev.clientY);
      };

      const onUp = (uev) => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        const { minX, maxX, minY, maxY } = updateDraftRect(uev.clientX, uev.clientY);
        draftEl.remove();

        if (maxX - minX < 8 || maxY - minY < 8) {
          showToast("Drag a rectangle around the child's head/face to tag.");
          return;
        }

        const natW = (state.modalPhoto && state.modalPhoto.width) || img.naturalWidth || img.clientWidth;
        const natH = (state.modalPhoto && state.modalPhoto.height) || img.naturalHeight || img.clientHeight;
        const scaleX = natW / img.clientWidth;
        const scaleY = natH / img.clientHeight;
        const bbox = {
          left: Math.max(0, Math.round(minX * scaleX)),
          top: Math.max(0, Math.round(minY * scaleY)),
          right: Math.min(natW, Math.round(maxX * scaleX)),
          bottom: Math.min(natH, Math.round(maxY * scaleY)),
        };

        state.manualDraftBox = {
          bbox,
          leftPct: (100 * minX) / rect.width,
          topPct: (100 * minY) / rect.height,
          widthPct: (100 * (maxX - minX)) / rect.width,
          heightPct: (100 * (maxY - minY)) / rect.height,
        };
        syncManualDrawUi();
        renderModalFaceOverlays();
      };

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });
  }

  async function openPhotoModal(photoId, highlightFaceId = null, keepNavContext = false) {
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

  async function openClusterFaceModal(clusterId, faceId) {
    const [clusterData, photoDetail] = await Promise.all([
      api(`/api/clusters/${clusterId}/faces`),
      api(`/api/faces/${faceId}/photo`),
    ]);
    const faces = clusterData.faces || [];
    let idx = faces.findIndex((item) => item.face_id === faceId);
    if (idx < 0) {
      faces.unshift({
        face_id: faceId,
        photo_id: photoDetail.photo_id,
        album_date: photoDetail.album_date || photoDetail.year,
        album_title: photoDetail.album_title,
      });
      idx = 0;
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
      index: idx,
    };
    applyDedupedNav(state.modalNavContext, faceId, photoDetail.photo_id);
    markPhotoSeen(photoDetail.photo_id);
    el("modalClusterNameInput").value = clusterData.person_name || "";
    applyPhotoModalDetail(photoDetail, faceId);
  }

  function getModalNavItems(ctx) {
    if (!ctx) return [];
    if (ctx.type === "cluster") return ctx.faces || [];
    if (ctx.type === "batch") return ctx.cards || [];
    if (ctx.type === "gallery") return ctx.items || [];
    return [];
  }

  // ------------------------------------------------------------------
  // Cluster photo de-duplication: one stop per photo, and photos already
  // reviewed (while going through any other cluster) are skipped.
  // ------------------------------------------------------------------

  const SEEN_STORAGE_KEY = "kpSeenClusterPhotoIds";

  function loadSeenPhotoIds() {
    try {
      const raw = window.localStorage.getItem(SEEN_STORAGE_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return new Set(Array.isArray(arr) ? arr.map(Number).filter(Number.isFinite) : []);
    } catch (_e) {
      return new Set();
    }
  }

  let seenSaveTimer = null;
  function saveSeenPhotoIds() {
    if (seenSaveTimer) clearTimeout(seenSaveTimer);
    seenSaveTimer = setTimeout(() => {
      try {
        window.localStorage.setItem(SEEN_STORAGE_KEY, JSON.stringify(Array.from(state.seenPhotoIds)));
      } catch (_e) {
        // Storage full / disabled: de-dup still works for this session.
      }
    }, 300);
  }

  function markPhotoSeen(photoId) {
    const pid = Number(photoId);
    if (!Number.isFinite(pid) || state.seenPhotoIds.has(pid)) return;
    state.seenPhotoIds.add(pid);
    saveSeenPhotoIds();
  }

  function isPhotoSeen(photoId) {
    return state.seenPhotoIds.has(Number(photoId));
  }

  /**
   * Build the navigable view for a de-duplicated context: one item per photo (preferring the
   * anchor face), skipping photos already seen elsewhere unless `state.showSeenPhotos` is on.
   * The anchor photo (the one the user clicked) is always kept.
   */
  function buildDedupedNavView(rawItems, anchorFaceId, anchorPhotoId, keepPhotoIds = null) {
    const byPhoto = new Map();
    for (const it of rawItems || []) {
      const ph = Number(it.photo_id);
      if (!byPhoto.has(ph)) {
        byPhoto.set(ph, it);
      } else if (anchorFaceId && it.face_id === anchorFaceId) {
        byPhoto.set(ph, it);
      }
    }
    const view = [];
    let hiddenSeen = 0;
    for (const [ph, it] of byPhoto) {
      const isKept = ph === Number(anchorPhotoId) || (keepPhotoIds && keepPhotoIds.has(ph));
      if (!isKept && !state.showSeenPhotos && isPhotoSeen(ph)) {
        hiddenSeen += 1;
        continue;
      }
      view.push(it);
    }
    return {
      view,
      hiddenSeen,
      duplicates: (rawItems || []).length - byPhoto.size,
    };
  }

  function applyDedupedNav(ctx, anchorFaceId, anchorPhotoId, keepCurrentView = false) {
    const raw = ctx.type === "cluster" ? ctx.rawFaces : ctx.rawItems;
    const keep = keepCurrentView
      ? new Set(getModalNavItems(ctx).map((it) => Number(it.photo_id)))
      : null;
    const res = buildDedupedNavView(raw, anchorFaceId, anchorPhotoId, keep);
    if (ctx.type === "cluster") ctx.faces = res.view;
    else ctx.items = res.view;
    ctx.hiddenSeen = res.hiddenSeen;
    ctx.duplicates = res.duplicates;
    let idx = res.view.findIndex((it) => Number(it.photo_id) === Number(anchorPhotoId));
    if (idx < 0) idx = 0;
    ctx.index = idx;
  }

  function dedupeNavNote(ctx) {
    if (!ctx || !ctx.dedupe) return "";
    const parts = [];
    if (ctx.hiddenSeen > 0) parts.push(`${ctx.hiddenSeen} seen skipped`);
    if (ctx.duplicates > 0) parts.push(`${ctx.duplicates} repeat${ctx.duplicates === 1 ? "" : "s"} merged`);
    return parts.length ? ` · ${parts.join(" · ")}` : "";
  }

  function syncSeenToggleButtons(ctx) {
    const toggleBtn = el("btnToggleSeenPhotos");
    const resetBtn = el("btnResetSeenPhotos");
    if (!toggleBtn || !resetBtn) return;
    const active = Boolean(ctx && ctx.dedupe);
    toggleBtn.hidden = !active || (!state.showSeenPhotos && !(ctx.hiddenSeen > 0));
    toggleBtn.textContent = state.showSeenPhotos ? "Hide seen" : `Show ${ctx && ctx.hiddenSeen ? ctx.hiddenSeen : ""} seen`;
    resetBtn.hidden = !active || state.seenPhotoIds.size === 0;
  }

  function rebuildCurrentDedupedNav() {
    const ctx = state.modalNavContext;
    if (!ctx || !ctx.dedupe) return;
    const items = getModalNavItems(ctx);
    const cur = items[ctx.index];
    const anchorPhoto = cur ? cur.photo_id : state.modalPhoto && state.modalPhoto.photo_id;
    applyDedupedNav(ctx, cur ? cur.face_id : null, anchorPhoto);
    renderModalNavControls();
  }

  function renderChildClusterQuickRow(ctx, items, rowEl, personName) {
    const cur = items[ctx.index];
    if (!cur) {
      rowEl.innerHTML = "";
      return;
    }
    const pr = Number(cur.pressure || 0);
    const who = escapeHtml(personName || "this child");
    const stLabel = childFaceStatusMeta(cur.status).label;
    rowEl.innerHTML = `
      <span class="modal-quick-label">This face for ${who}: <strong>${escapeHtml(stLabel)}</strong>${
        pr > 0 ? " · ▲ boosted" : pr < 0 ? " · ▼ penalized" : ""
      }</span>
      <button type="button" class="btn btn-tonal btn-xs ${pr > 0 ? "is-selected" : ""}" data-ccact="boost" title="Good example: nudge this face and look-alikes up (B)">▲ Boost <kbd>B</kbd></button>
      <button type="button" class="btn btn-tonal btn-xs ${pr < 0 ? "is-selected" : ""}" data-ccact="penalize" title="Doubtful: nudge this face and look-alikes down (N)">▼ Penalize <kbd>N</kbd></button>
      ${pr ? `<button type="button" class="btn btn-ghost btn-xs" data-ccact="clear" title="Remove boost / penalize">Clear</button>` : ""}
      <button type="button" class="btn btn-ghost btn-xs" data-ccact="exclude" title="Not ${who}: remove and use as a hard negative">⊘ Exclude</button>
    `;
    rowEl.querySelectorAll("[data-ccact]").forEach((b) => {
      b.addEventListener("click", () => applyModalChildClusterAction(b.getAttribute("data-ccact"), b));
    });
  }

  function applyModalChildClusterAction(action, btn = null) {
    const ctx = state.modalNavContext;
    if (!ctx || ctx.type !== "gallery" || !ctx.ccKey || !ctx.personId) return;
    const items = getModalNavItems(ctx);
    const cur = items[ctx.index];
    if (!cur) return;
    const curPr = Number(cur.pressure || 0);
    let act = action;
    // Pressing Boost on a boosted face (or Penalize on a penalized one) toggles it off:
    if ((act === "boost" && curPr > 0) || (act === "penalize" && curPr < 0)) act = "clear";
    runBusy(
      btn,
      async () => {
        const res = await api(`/api/people/${ctx.personId}/cluster-action`, {
          method: "POST",
          body: JSON.stringify({ action: act, faces: [{ photo_id: cur.photo_id, face_id: cur.face_id }] }),
        });
        state.modalDirty = true;
        if (res.overview) {
          state.overview = res.overview;
          renderOverview();
        }
        const name = resolvePersonName(ctx.personId, ctx.personName || "");
        if (act === "exclude") {
          for (const arr of [ctx.rawItems, ctx.items]) {
            if (!Array.isArray(arr)) continue;
            const i = arr.findIndex((f) => f.face_id === cur.face_id);
            if (i >= 0) arr.splice(i, 1);
          }
          showToast(`Removed from ${name} — learned as a hard negative.`);
          if (!ctx.items.length) {
            await closePhotoModal();
            return;
          }
          await jumpToModalNavIndex(Math.min(ctx.index, ctx.items.length - 1));
          return;
        }
        cur.pressure = act === "boost" ? 1 : act === "penalize" ? -1 : 0;
        showToast(
          act === "boost"
            ? `▲ Boosted as a good example of ${name}.`
            : act === "penalize"
            ? `▼ Penalized for ${name} (not rejected).`
            : "Pressure cleared."
        );
        renderModalNavControls();
      },
      { context: "Update face pressure" }
    );
  }

  async function jumpToModalNavIndex(newIndex) {
    const ctx = state.modalNavContext;
    const items = getModalNavItems(ctx);
    if (!ctx || !items.length) return;
    const len = items.length;
    const wrapped = ((newIndex % len) + len) % len;
    ctx.index = wrapped;
    const item = items[wrapped];
    await openPhotoModal(item.photo_id, item.face_id || null, true);
  }

  function renderModalNavControls() {
    const ctx = state.modalNavContext;
    const navEl = el("modalClusterNav");
    const boxEl = el("modalClusterBox");
    const excludeBtn = el("btnClusterExcludeFace");
    const assignForm = el("modalClusterAssignForm");
    const batchQuickEl = el("modalBatchQuickRow");
    const items = getModalNavItems(ctx);

    if (!ctx || !items.length) {
      navEl.hidden = true;
      boxEl.hidden = true;
      syncSeenToggleButtons(null);
      return;
    }
    syncSeenToggleButtons(ctx);
    const navNote = dedupeNavNote(ctx);

    navEl.hidden = false;
    boxEl.hidden = false;
    el("btnClusterPrev").disabled = items.length <= 1;
    el("btnClusterNext").disabled = items.length <= 1;

    const isCluster = ctx.type === "cluster";
    const isBatch = ctx.type === "batch";
    const isChildCluster = ctx.type === "gallery" && Boolean(ctx.ccKey);
    excludeBtn.hidden = !isCluster;
    assignForm.hidden = !isCluster;
    batchQuickEl.hidden = !(isBatch || isChildCluster);

    const livePersonName = ctx.personId
      ? resolvePersonName(ctx.personId, ctx.personName || "")
      : ctx.personName || "";

    if (isCluster) {
      el("modalClusterPos").textContent = `${ctx.index + 1} / ${items.length} photos in cluster${navNote}`;
      el("modalClusterTitle").textContent = `${ctx.archiveChild} · ${ctx.year} (${ctx.photoCount} photos)`;
    } else if (isBatch) {
      const label = `Batch Photo ${ctx.index + 1} of ${items.length}${livePersonName ? ` (${livePersonName})` : ""}`;
      el("modalClusterPos").textContent = label;
      el("modalClusterTitle").textContent = label;
    } else {
      const label = `${ctx.navLabel || "Gallery Photo"} ${ctx.index + 1} of ${items.length}${livePersonName ? ` (${livePersonName})` : ""}`;
      el("modalClusterPos").textContent = `${label}${navNote}`;
      el("modalClusterTitle").textContent = label;
    }

    const thumbsEl = el("modalClusterThumbs");
    thumbsEl.innerHTML = items
      .map((item, idx) => {
        const thumbSrc = item.face_id
          ? `/media/face/${item.face_id}.jpg`
          : `/media/photo/${item.photo_id}`;
        const activeCls = idx === ctx.index ? "active" : "";
        let stateCls = "";
        let badgeHtml = "";
        if (isBatch) {
          stateCls = `state-${item.decision || "approved"}`;
          const sym =
            item.decision === "approved"
              ? "✓"
              : item.decision === "unsure"
              ? "?"
              : item.decision === "reassigned"
              ? "↪"
              : "✗";
          badgeHtml = `<span class="modal-strip-badge">${sym}</span>`;
        } else if (isChildCluster && item.pressure) {
          stateCls = item.pressure > 0 ? "state-approved" : "state-disproved";
          badgeHtml = `<span class="modal-strip-badge">${item.pressure > 0 ? "▲" : "▼"}</span>`;
        }
        const tip = `${item.album_date || item.year || ""} · ${item.album_title || ""}`;
        return `
        <button type="button" class="modal-cluster-thumb-btn ${stateCls} ${activeCls}" data-cidx="${idx}" title="${escapeHtml(tip)}">
          <img src="${thumbSrc}" alt="" loading="lazy" />
          ${badgeHtml}
        </button>`;
      })
      .join("");

    thumbsEl.querySelectorAll(".modal-cluster-thumb-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.getAttribute("data-cidx"));
        runBusy(btn, () => jumpToModalNavIndex(idx), {
          context: "Load photo",
        });
      });
    });

    if (isChildCluster) {
      renderChildClusterQuickRow(ctx, items, batchQuickEl, livePersonName);
    }

    if (isBatch) {
      const curCard = items[ctx.index];
      if (curCard) {
        batchQuickEl.innerHTML = `
          <button type="button" class="btn btn-tonal btn-xs ${curCard.decision === "approved" ? "is-selected" : ""}" data-bdec="approved">✓ ${escapeHtml(livePersonName || "Approve")}</button>
          <button type="button" class="btn btn-tonal btn-xs ${curCard.decision === "disproved" ? "is-selected" : ""}" data-bdec="disproved">✗ Not ${escapeHtml(livePersonName || "match")}</button>
          <button type="button" class="btn btn-ghost btn-xs ${curCard.decision === "unsure" ? "is-selected" : ""}" data-bdec="unsure">? Unsure</button>
          <button type="button" class="btn btn-ghost btn-xs ${curCard.decision === "reassigned" ? "is-selected" : ""}" data-bdec="reassign">↪ Another…</button>
        `;
        batchQuickEl.querySelectorAll("[data-bdec]").forEach((qbtn) => {
          qbtn.addEventListener("click", () => {
            const act = qbtn.getAttribute("data-bdec");
            if (act === "reassign") {
              openReassignModal({
                source: "batch",
                cardIdx: ctx.index,
                photoId: curCard.photo_id,
                faceId: curCard.face_id || null,
                fromPersonId: ctx.personId || null,
                fromPersonName: livePersonName,
                initialName: resolvePersonName(curCard.targetPersonId, curCard.targetName || ""),
              });
              return;
            }
            curCard.decision = act;
            curCard.targetName = "";
            curCard.targetPersonId = null;
            curCard.alreadySavedInModal = false;
            state.batchUserTouched = true;
            if (state.mode === "batch") {
              renderBatchGrid(state.batchMeta);
            }
            renderModalNavControls();
          });
        });
      }
    }

    const activeBtn = thumbsEl.querySelector(".modal-cluster-thumb-btn.active");
    if (activeBtn && typeof activeBtn.scrollIntoView === "function") {
      activeBtn.scrollIntoView({ block: "nearest", inline: "center" });
    }
  }

  function renderModalFaceOverlays() {
    const detail = state.modalPhoto;
    if (!detail) return;
    const layer = el("faceBoxesLayer");
    const natW = detail.width || el("modalPhotoImg").naturalWidth || 1;
    const natH = detail.height || el("modalPhotoImg").naturalHeight || 1;

    layer.innerHTML = detail.faces
      .map((f) => {
        const leftPct = (100 * f.box.left) / natW;
        const topPct = (100 * f.box.top) / natH;
        const wPct = (100 * (f.box.right - f.box.left)) / natW;
        const hPct = (100 * (f.box.bottom - f.box.top)) / natH;
        const cls = [
          "bbox-overlay",
          f.person_name ? "named" : "",
          f.is_manual ? "manual" : "",
          f.face_id === state.modalHighlightFaceId ? "highlight" : "",
        ]
          .filter(Boolean)
          .join(" ");
        const prefix = f.is_manual ? "✎ " : "";
        const lbl = f.person_name ? `${prefix}${escapeHtml(f.person_name)}` : `#${f.face_idx + 1} (name…)`;
        return `
          <div class="${cls}" data-fid="${f.face_id}" style="left:${leftPct}%;top:${topPct}%;width:${wPct}%;height:${hPct}%">
            <span class="bbox-label">${lbl}</span>
          </div>`;
      })
      .join("");

    layer.querySelectorAll(".bbox-overlay").forEach((boxEl) => {
      boxEl.addEventListener("click", () => {
        if (state.manualDrawMode) return;
        const fid = Number(boxEl.getAttribute("data-fid"));
        state.modalHighlightFaceId = fid;
        renderModalFaceOverlays();
        renderModalSideFaces();
        const input = document.querySelector(`.modal-face-input[data-fid="${fid}"]`);
        if (input) {
          input.focus();
          input.select();
        }
      });
    });

    if (state.manualDraftBox) {
      appendManualDraftPopover(layer);
    }
  }

  function syncBatchCardFromModalTag(photoId, faceId, taggedPersonId, taggedName, status) {
    if (!state.batchCards.length) return;
    let changed = false;
    for (const c of state.batchCards) {
      const matchesFace = faceId && c.face_id === faceId;
      const matchesPhotoFallback = !c.face_id && c.photo_id === photoId;
      if (!matchesFace && !matchesPhotoFallback) continue;

      if (status === "unsure") {
        c.decision = "unsure";
        c.targetPersonId = null;
        c.targetName = "";
        c.alreadySavedInModal = true;
        changed = true;
      } else if (status === "rejected") {
        c.decision = "disproved";
        c.targetPersonId = null;
        c.targetName = "";
        c.alreadySavedInModal = true;
        changed = true;
      } else if (status === "confirmed") {
        if (taggedPersonId === state.selectedPersonId) {
          c.decision = "approved";
          c.targetPersonId = null;
          c.targetName = "";
        } else {
          c.decision = "reassigned";
          c.targetPersonId = taggedPersonId;
          c.targetName = resolvePersonName(taggedPersonId, taggedName);
        }
        c.alreadySavedInModal = true;
        changed = true;
      }
    }
    if (changed) {
      if (state.mode === "batch") {
        renderBatchGrid(state.batchMeta);
      }
      if (!el("photoModal").hidden && state.modalNavContext && state.modalNavContext.type === "batch") {
        renderModalNavControls();
      }
    }
  }

  async function untagModalFace(photoId, faceId, prevName) {
    const res = await api("/api/faces/untag", {
      method: "POST",
      body: JSON.stringify({
        photo_id: photoId,
        face_id: faceId,
      }),
    });
    state.modalDirty = true;
    state.modalPhoto = res.photo;
    state.overview = res.overview;
    renderOverview();
    syncBatchCardFromModalTag(photoId, faceId, null, "", "rejected");
    renderModalFaceOverlays();
    renderModalSideFaces();
    const who = prevName ? ` (${prevName})` : "";
    showToast(`Removed tag${who} — child is now uncategorized.`);
  }

  function renderModalSideFaces() {
    const detail = state.modalPhoto;
    if (!detail) return;
    const selP = getSelectedPerson();
    const listEl = el("modalFacesList");
    if (!detail.faces.length) {
      listEl.innerHTML = `<p class="modal-hint">No faces auto-detected in this photo — click <strong>+ Tag Child (T)</strong> above to draw a box around any child!</p>`;
      return;
    }
    listEl.innerHTML = detail.faces
      .map((f) => {
        const hasTag = f.person_id != null;
        const simSuffix = f.similarity != null ? ` · ${f.similarity.toFixed(2)}` : "";
        const manualSuffix = f.is_manual ? " · ✎ Manual" : "";
        let statusHtml = `<span class="modal-face-status uncat">Uncategorized</span>`;
        if (hasTag && f.status === "confirmed") {
          statusHtml = `<span class="modal-face-status confirmed">✓ Confirmed${manualSuffix}</span>`;
        } else if (hasTag && f.status === "auto_high") {
          statusHtml = `<span class="modal-face-status auto">Auto match${simSuffix}</span>`;
        } else if (hasTag) {
          statusHtml = `<span class="modal-face-status guess">Model guess${simSuffix}</span>`;
        }

        const untagTitle = f.is_manual
          ? `Remove manual box (${escapeHtml(f.person_name || "")})`
          : `Remove tag (${escapeHtml(f.person_name || "")}) and leave this child uncategorized`;
        const untagBtnHtml = hasTag || f.is_manual
          ? `<button type="button" class="btn btn-untag btn-sm btn-modal-untag" data-fid="${f.face_id}" title="${untagTitle}">✗ Remove Tag</button>`
          : "";

        let unsureTargetId = null;
        let unsureTargetName = "";
        const inClusterNav = state.modalNavContext && state.modalNavContext.type === "cluster";
        if (!inClusterNav && !f.is_manual && state.mode === "batch" && selP) {
          if (hasTag && f.person_id === selP.person_id && f.status !== "confirmed") {
            unsureTargetId = selP.person_id;
            unsureTargetName = selP.display_name;
          } else if (!hasTag && f.face_id === state.modalHighlightFaceId) {
            unsureTargetId = selP.person_id;
            unsureTargetName = selP.display_name;
          }
        }
        const unsureBtnHtml = unsureTargetId
          ? `<button type="button" class="btn btn-ghost btn-sm btn-modal-unsure ${hasTag ? "btn-modal-unsure-compact" : ""}" data-fid="${f.face_id}" data-pid="${unsureTargetId}" data-pname="${escapeHtml(unsureTargetName)}" title="Mark as I'm not sure for ${escapeHtml(unsureTargetName)}">${hasTag ? "?" : "? Not sure"}</button>`
          : "";

        return `
        <form class="modal-face-row ${f.face_id === state.modalHighlightFaceId ? "highlight" : ""}" data-fid="${f.face_id}">
          <img src="/media/face/${f.face_id}.jpg" alt="" title="Highlight this face on the photo" />
          <div class="modal-face-controls">
            <div class="modal-face-status-row">${statusHtml}</div>
            <input type="text" class="modal-face-input" data-fid="${f.face_id}" list="knownPeopleDatalist"
              placeholder="Name this child…" value="${escapeHtml(f.person_name || "")}" />
            <div class="modal-face-btn-row">
              <button type="submit" class="btn btn-tonal btn-sm">Save Tag ✓</button>
              ${untagBtnHtml}
              ${unsureBtnHtml}
            </div>
          </div>
        </form>`;
      })
      .join("");

    listEl.querySelectorAll(".modal-face-row").forEach((form) => {
      const fid = Number(form.getAttribute("data-fid"));
      const img = form.querySelector("img");
      if (img) {
        img.addEventListener("click", () => {
          state.modalHighlightFaceId = fid;
          renderModalFaceOverlays();
          renderModalSideFaces();
        });
      }
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const input = form.querySelector("input");
        const submitBtn = form.querySelector('button[type="submit"]');
        const extraBtns = Array.from(form.querySelectorAll('button:not([type="submit"])'));
        const name = input.value.trim();
        const faceObj = detail.faces.find((x) => x.face_id === fid);
        if (!name) {
          if (faceObj && faceObj.person_id != null) {
            runBusy(
              submitBtn,
              () => untagModalFace(detail.photo_id, fid, faceObj.person_name || ""),
              {
                busyText: "Removing…",
                alsoDisable: [input, ...extraBtns],
                context: "Remove face tag",
              }
            );
          } else {
            showToast("Face is already uncategorized — type a name to tag.");
          }
          return;
        }
        runBusy(
          submitBtn,
          async () => {
            const res = await api("/api/faces/tag", {
              method: "POST",
              body: JSON.stringify({
                photo_id: detail.photo_id,
                face_id: fid,
                new_name: name,
                status: "confirmed",
              }),
            });
            state.modalDirty = true;
            state.modalPhoto = res.photo;
            state.overview = res.overview;
            const finalName = resolvePersonName(res.person_id, name);
            renderOverview();
            syncBatchCardFromModalTag(detail.photo_id, fid, res.person_id, finalName, "confirmed");
            renderModalFaceOverlays();
            renderModalSideFaces();
            showToast(`Tagged ${finalName} and synced Batch Trainer & archive!`);
          },
          {
            busyText: "Saving…",
            alsoDisable: [input, ...extraBtns],
            context: "Save face tag",
          }
        );
      });
    });

    listEl.querySelectorAll(".btn-modal-untag").forEach((btn) => {
      btn.addEventListener("click", () => {
        const fid = Number(btn.getAttribute("data-fid"));
        const form = btn.closest(".modal-face-row");
        const extra = form
          ? Array.from(form.querySelectorAll("input, button")).filter((n) => n !== btn)
          : [];
        const faceObj = detail.faces.find((x) => x.face_id === fid);
        runBusy(
          btn,
          () => untagModalFace(detail.photo_id, fid, (faceObj && faceObj.person_name) || ""),
          {
            busyText: "Removing…",
            alsoDisable: extra,
            context: "Remove face tag",
          }
        );
      });
    });

    listEl.querySelectorAll(".btn-modal-unsure").forEach((btn) => {
      btn.addEventListener("click", () => {
        const fid = Number(btn.getAttribute("data-fid"));
        const pid = Number(btn.getAttribute("data-pid"));
        const pname = btn.getAttribute("data-pname") || "";
        if (!pid) return;
        const form = btn.closest(".modal-face-row");
        const extra = form
          ? Array.from(form.querySelectorAll("input, button")).filter((n) => n !== btn)
          : [];
        runBusy(
          btn,
          async () => {
            const res = await api("/api/faces/tag", {
              method: "POST",
              body: JSON.stringify({
                photo_id: detail.photo_id,
                face_id: fid,
                person_id: pid,
                status: "unsure",
              }),
            });
            state.modalDirty = true;
            state.modalPhoto = res.photo;
            state.overview = res.overview;
            renderOverview();
            syncBatchCardFromModalTag(detail.photo_id, fid, pid, "", "unsure");
            renderModalFaceOverlays();
            renderModalSideFaces();
            showToast(`Marked as "I'm not sure" (?) for ${pname}.`);
          },
          {
            busyText: btn.classList.contains("btn-modal-unsure-compact") ? "…" : "Saving…",
            alsoDisable: extra,
            context: "Mark face unsure",
          }
        );
      });
    });
  }

  async function closePhotoModal() {
    const wasDirty = state.modalDirty;
    state.manualDrawMode = false;
    state.manualDraftBox = null;
    syncManualDrawUi();
    el("photoModal").hidden = true;
    state.modalPhoto = null;
    state.modalNavContext = null;
    state.modalDirty = false;
    if (wasDirty) {
      try {
        if (state.mode === "gallery") await loadGallery();
        else if (state.mode === "clusters") await loadClusters();
        else if (state.mode === "childclusters") await reloadChildClustersPreservingUi();
      } catch (err) {
        reportError(err, "Refresh view");
      }
    }
  }

  // ------------------------------------------------------------------
  // Classes (feed + date range) & Membership Matrix
  // ------------------------------------------------------------------

  const CLASS_ROLE_ORDER = { family: 0, classmate: 1, educator: 2 };
  const CLASS_ROLE_TITLES = { family: "Family Children", classmate: "Classroom Colleagues", educator: "Educators" };
  const CLASS_SUGGEST_MIN = 3;

  function classFeedOptionsHtml(selected) {
    const feeds = ((state.overview && state.overview.archive_children) || []).slice();
    if (selected && !feeds.includes(selected)) feeds.push(selected);
    return (
      `<option value=""${selected ? "" : " selected"}>Any feed</option>` +
      feeds
        .map(
          (f) =>
            `<option value="${escapeHtml(f)}"${f === selected ? " selected" : ""}>${escapeHtml(f)}</option>`
        )
        .join("")
    );
  }

  function classDateRangeLabel(c) {
    if (!c.start_date && !c.end_date) return "any date";
    return `${c.start_date || "…"} → ${c.end_date || "…"}`;
  }

  function classOriginalMembers(cid) {
    const src = (state.classesMatrix && state.classesMatrix.classes) || (state.overview && state.overview.classes) || [];
    const c = src.find((x) => x.class_id === cid);
    return new Set((c && c.member_ids) || []);
  }

  function setsEqual(a, b) {
    if (a.size !== b.size) return false;
    for (const v of a) if (!b.has(v)) return false;
    return true;
  }

  function classesHaveUnsavedChanges() {
    return state.classesDirty.size > 0;
  }

  async function openClassesModal() {
    el("newClassFeed").innerHTML = classFeedOptionsHtml("");
    renderClassesList();
    el("classesMatrixWrap").innerHTML = `<div class="empty-state">Loading membership matrix…</div>`;
    el("classesModal").hidden = false;
    await loadClassesMatrix({ keepDirty: false });
  }

  function closeClassesModal(force = false) {
    if (el("classesModal").hidden) return;
    if (!force && classesHaveUnsavedChanges()) {
      const ok = window.confirm(
        `You have unsaved membership changes in ${state.classesDirty.size} class${state.classesDirty.size === 1 ? "" : "es"}. Discard them?`
      );
      if (!ok) return;
    }
    el("classesModal").hidden = true;
    state.classesDirty.clear();
    state.classesMembers.clear();
    state.classesMatrix = null;
  }

  async function loadClassesMatrix(opts = {}) {
    const { keepDirty = true } = opts;
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
    const classes = (state.overview && state.overview.classes) || [];
    const photoCounts = new Map(
      ((state.classesMatrix && state.classesMatrix.classes) || []).map((c) => [c.class_id, c.photo_count])
    );
    if (!classes.length) {
      body.innerHTML = `<tr><td colspan="6" class="classes-empty">No classes yet — add one below or use <strong>✨ Auto-create</strong>.</td></tr>`;
      return;
    }
    body.innerHTML = classes
      .map((c) => {
        const pc = photoCounts.get(c.class_id);
        const pcTxt = typeof pc === "number" ? ` · ${pc.toLocaleString()} photos` : "";
        return `
          <tr data-cid="${c.class_id}">
            <td><input type="text" class="cls-name" value="${escapeHtml(c.name)}" autocomplete="off" /></td>
            <td><select class="input-select cls-feed">${classFeedOptionsHtml(c.archive_child || "")}</select></td>
            <td><input type="date" class="cls-start" value="${escapeHtml(c.start_date || "")}" /></td>
            <td><input type="date" class="cls-end" value="${escapeHtml(c.end_date || "")}" /></td>
            <td class="cls-stats">${(c.member_ids || []).length} kids${pcTxt}</td>
            <td class="cls-actions">
              <button type="button" class="btn btn-tonal btn-xs cls-save" data-cid="${c.class_id}">Save</button>
              <button type="button" class="btn btn-danger btn-xs cls-delete" data-cid="${c.class_id}">Delete</button>
            </td>
          </tr>`;
      })
      .join("");
  }

  function readClassRowFields(rowEl) {
    return {
      name: rowEl.querySelector(".cls-name").value.trim(),
      archive_child: rowEl.querySelector(".cls-feed").value || "",
      start_date: rowEl.querySelector(".cls-start").value || "",
      end_date: rowEl.querySelector(".cls-end").value || "",
    };
  }

  function validateClassFields(f) {
    if (!f.name) return "Class name is required.";
    if (f.start_date && f.end_date && f.start_date > f.end_date) return "Start date must be before end date.";
    return "";
  }

  async function afterClassesMutation(res, toastMsg) {
    if (res && res.overview) {
      state.overview = res.overview;
      renderOverview();
    }
    await loadClassesMatrix({ keepDirty: true });
    if (toastMsg) showToast(toastMsg);
  }

  function renderClassesMatrix() {
    const wrap = el("classesMatrixWrap");
    const data = state.classesMatrix;
    if (!data) return;
    const prevLeft = wrap.scrollLeft;
    const prevTop = wrap.scrollTop;
    const classes = data.classes || [];
    const counts = data.counts || {};
    const people = (data.people || [])
      .filter((p) => p.role !== "ignore")
      .slice()
      .sort((a, b) => {
        const ra = CLASS_ROLE_ORDER[a.role] !== undefined ? CLASS_ROLE_ORDER[a.role] : 9;
        const rb = CLASS_ROLE_ORDER[b.role] !== undefined ? CLASS_ROLE_ORDER[b.role] : 9;
        if (ra !== rb) return ra - rb;
        return String(a.display_name).localeCompare(String(b.display_name));
      });

    if (!classes.length) {
      wrap.innerHTML = `<div class="empty-state">Create a class above to assign kids to it.</div>`;
      updateClassesDirtyUi();
      return;
    }
    if (!people.length) {
      wrap.innerHTML = `<div class="empty-state">No children enrolled yet.</div>`;
      updateClassesDirtyUi();
      return;
    }

    const headCells = classes
      .map((c) => {
        const members = state.classesMembers.get(c.class_id) || new Set();
        const dirtyCls = state.classesDirty.has(c.class_id) ? "is-dirty" : "";
        const sub = `${c.archive_child || "Any feed"} · ${classDateRangeLabel(c)}`;
        return `
          <th class="cm-col-head ${dirtyCls}" data-cid="${c.class_id}">
            <div class="cm-col-name" title="${escapeHtml(c.name)}">${escapeHtml(c.name)}</div>
            <div class="cm-col-sub" title="${escapeHtml(sub)}">${escapeHtml(sub)}</div>
            <div class="cm-col-tools">
              <span class="cm-col-count" data-count-cid="${c.class_id}">${members.size} ✓</span>
              <button type="button" class="btn btn-ghost btn-xs cm-tick-suggested" data-cid="${c.class_id}" title="Tick everyone already confirmed / auto-matched on at least ${CLASS_SUGGEST_MIN} photos of this class">Tick suggested</button>
            </div>
          </th>`;
      })
      .join("");

    let rowsHtml = "";
    let lastRole = null;
    for (const p of people) {
      if (p.role !== lastRole) {
        lastRole = p.role;
        rowsHtml += `<tr class="cm-group-row"><th class="cm-row-head cm-group-label">${escapeHtml(
          CLASS_ROLE_TITLES[p.role] || p.role || "Other"
        )}</th><td colspan="${classes.length}"></td></tr>`;
      }
      const cells = classes
        .map((c) => {
          const members = state.classesMembers.get(c.class_id) || new Set();
          const hint = Number((counts[String(c.class_id)] || {})[String(p.person_id)] || 0);
          const checked = members.has(p.person_id) ? "checked" : "";
          const hintHtml = hint > 0 ? `<span class="cm-hint" title="${hint} photos in this class">${hint}</span>` : "";
          return `
            <td class="cm-cell ${hint >= CLASS_SUGGEST_MIN ? "is-suggested" : ""}">
              <label><input type="checkbox" class="cm-check" data-cid="${c.class_id}" data-pid="${p.person_id}" ${checked} />${hintHtml}</label>
            </td>`;
        })
        .join("");
      rowsHtml += `<tr><th class="cm-row-head" title="${escapeHtml(p.display_name)}">${escapeHtml(p.display_name)}</th>${cells}</tr>`;
    }

    wrap.innerHTML = `
      <table class="classes-matrix">
        <thead><tr><th class="cm-corner">Child \\ Class</th>${headCells}</tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>`;
    wrap.scrollLeft = prevLeft;
    wrap.scrollTop = prevTop;
    updateClassesDirtyUi();
  }

  function markClassDirtyState(cid) {
    const cur = state.classesMembers.get(cid) || new Set();
    if (setsEqual(cur, classOriginalMembers(cid))) state.classesDirty.delete(cid);
    else state.classesDirty.add(cid);
  }

  function updateClassesDirtyUi() {
    const n = state.classesDirty.size;
    const saveBtn = el("btnSaveMemberships");
    const note = el("classesDirtyNote");
    if (saveBtn && saveBtn.dataset.busy !== "1") {
      saveBtn.disabled = n === 0;
      saveBtn.textContent = n ? `Save memberships (${n} class${n === 1 ? "" : "es"})` : "Save memberships";
    }
    if (note) {
      note.hidden = n === 0;
      note.textContent = n ? "Unsaved changes" : "";
    }
    const wrap = el("classesMatrixWrap");
    wrap.querySelectorAll(".cm-col-head").forEach((th) => {
      const cid = Number(th.getAttribute("data-cid"));
      th.classList.toggle("is-dirty", state.classesDirty.has(cid));
    });
    wrap.querySelectorAll("[data-count-cid]").forEach((node) => {
      const cid = Number(node.getAttribute("data-count-cid"));
      const members = state.classesMembers.get(cid) || new Set();
      node.textContent = `${members.size} ✓`;
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
          const members = Array.from(state.classesMembers.get(cid) || []);
          const res = await api(`/api/classes/${cid}/members`, {
            method: "POST",
            body: JSON.stringify({ person_ids: members }),
          });
          done++;
          state.classesDirty.delete(cid);
          if (res.overview) state.overview = res.overview;
          if (state.classesMatrix && res.class) {
            const idx = (state.classesMatrix.classes || []).findIndex((c) => c.class_id === cid);
            if (idx >= 0) {
              state.classesMatrix.classes[idx] = Object.assign({}, state.classesMatrix.classes[idx], res.class, {
                member_ids: res.class.member_ids || members,
              });
            }
          }
        }
        renderOverview();
        await loadClassesMatrix({ keepDirty: true });
        showToast(`Saved memberships for ${done} class${done === 1 ? "" : "es"} — background rematch scheduled.`);
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

  function bindClassesEvents() {
    el("btnOpenClasses").addEventListener("click", (ev) => {
      runBusy(ev.currentTarget, () => openClassesModal(), { context: "Load classes" });
    });
    el("btnCloseClasses").addEventListener("click", () => closeClassesModal());
    el("classesModal").addEventListener("click", (ev) => {
      if (ev.target === el("classesModal")) closeClassesModal();
    });

    el("btnAutoCreateClasses").addEventListener("click", (ev) => {
      const btn = ev.currentTarget;
      runBusy(
        btn,
        async () => {
          const res = await api("/api/classes/auto", { method: "POST", body: "{}" });
          const n = Number(res.created || 0);
          await afterClassesMutation(
            res,
            n > 0
              ? `✨ Created ${n} class${n === 1 ? "" : "es"} by feed & school year.`
              : "No new classes — every feed / school year already has one."
          );
        },
        { busyText: "Creating…", context: "Auto-create classes" }
      );
    });

    el("btnAddClass").addEventListener("click", (ev) => {
      const btn = ev.currentTarget;
      const fields = {
        name: el("newClassName").value.trim(),
        archive_child: el("newClassFeed").value || "",
        start_date: el("newClassStart").value || "",
        end_date: el("newClassEnd").value || "",
      };
      const err = validateClassFields(fields);
      if (err) {
        showToast(err, "error");
        return;
      }
      runBusy(
        btn,
        async () => {
          const res = await api("/api/classes", { method: "POST", body: JSON.stringify(fields) });
          el("newClassName").value = "";
          el("newClassStart").value = "";
          el("newClassEnd").value = "";
          await afterClassesMutation(res, `Added class "${fields.name}".`);
        },
        {
          busyText: "Adding…",
          alsoDisable: [el("newClassName"), el("newClassFeed"), el("newClassStart"), el("newClassEnd")],
          context: "Add class",
        }
      );
    });

    el("newClassName").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        el("btnAddClass").click();
      }
    });

    el("classesListBody").addEventListener("click", (ev) => {
      const saveBtn = ev.target.closest(".cls-save");
      const delBtn = ev.target.closest(".cls-delete");
      const btn = saveBtn || delBtn;
      if (!btn) return;
      const rowEl = btn.closest("tr");
      const cid = Number(btn.getAttribute("data-cid"));
      if (!rowEl || !cid) return;
      const rowControls = Array.from(rowEl.querySelectorAll("input, select, button")).filter((n) => n !== btn);
      if (saveBtn) {
        const fields = readClassRowFields(rowEl);
        const err = validateClassFields(fields);
        if (err) {
          showToast(err, "error");
          return;
        }
        runBusy(
          btn,
          async () => {
            const res = await api(`/api/classes/${cid}`, { method: "PATCH", body: JSON.stringify(fields) });
            await afterClassesMutation(res, `Saved class "${fields.name}".`);
          },
          { busyText: "Saving…", alsoDisable: rowControls, context: "Save class" }
        );
        return;
      }
      const cls = ((state.overview && state.overview.classes) || []).find((c) => c.class_id === cid);
      const clsName = cls ? cls.name : "this class";
      const nMembers = cls ? (cls.member_ids || []).length : 0;
      const ok = window.confirm(
        `Delete class "${clsName}"? Its ${nMembers} membership${nMembers === 1 ? "" : "s"} will be removed.`
      );
      if (!ok) return;
      runBusy(
        btn,
        async () => {
          const res = await api(`/api/classes/${cid}`, { method: "DELETE" });
          state.classesDirty.delete(cid);
          state.classesMembers.delete(cid);
          await afterClassesMutation(res, `Deleted class "${clsName}".`);
        },
        { busyText: "Deleting…", alsoDisable: rowControls, context: "Delete class" }
      );
    });

    const wrap = el("classesMatrixWrap");
    wrap.addEventListener("change", (ev) => {
      const cb = ev.target.closest(".cm-check");
      if (!cb) return;
      const cid = Number(cb.getAttribute("data-cid"));
      const pid = Number(cb.getAttribute("data-pid"));
      let members = state.classesMembers.get(cid);
      if (!members) {
        members = new Set();
        state.classesMembers.set(cid, members);
      }
      if (cb.checked) members.add(pid);
      else members.delete(pid);
      markClassDirtyState(cid);
      updateClassesDirtyUi();
    });
    wrap.addEventListener("click", (ev) => {
      const tickBtn = ev.target.closest(".cm-tick-suggested");
      if (!tickBtn || !state.classesMatrix) return;
      const cid = Number(tickBtn.getAttribute("data-cid"));
      const colCounts = (state.classesMatrix.counts || {})[String(cid)] || {};
      let members = state.classesMembers.get(cid);
      if (!members) {
        members = new Set();
        state.classesMembers.set(cid, members);
      }
      const eligible = new Set(
        (state.classesMatrix.people || []).filter((p) => p.role !== "ignore").map((p) => p.person_id)
      );
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
        added
          ? `Ticked ${added} suggested kid${added === 1 ? "" : "s"} — click Save memberships to apply.`
          : "No additional suggestions for this class."
      );
    });

    el("btnSaveMemberships").addEventListener("click", (ev) => {
      saveClassMemberships(ev.currentTarget);
    });
  }

  // ------------------------------------------------------------------
  // Live Server-Sent Events (`/api/events`) Background Rematch Sync
  // ------------------------------------------------------------------

  function updateSyncIndicator() {
    const pill = el("rematchStatusPill");
    const active = Boolean(state.rematchPending || state.rematchRunning);
    if (pill) {
      pill.hidden = !active;
      pill.textContent = state.rematchRunning ? "↻ syncing" : "↻ queued";
    }
    if (state.mode === "batch" && state.selectedPersonId && !state.batchCards.length) {
      renderBatchGrid(state.batchMeta);
    }
  }

  // ------------------------------------------------------------------
  // Backend background-work indicator (re-match, index snapshot, warm-up, long sync requests)
  // ------------------------------------------------------------------

  let bgStatusTicker = null;
  let bgStatusPoller = null;

  function fmtSecs(ms) {
    const s = Math.max(0, ms) / 1000;
    return s < 10 ? s.toFixed(1) : String(Math.round(s));
  }

  function applyBgStatus(st) {
    if (!st || typeof st !== "object") return;
    state.bgStatus = st;
    state.bgStatusAt = Date.now();
    state.rematchRunning = st.rematch === "running";
    state.rematchPending = st.rematch !== "idle";
    renderBgStatus();
    const busy = Boolean(st.busy) || st.snapshot === "pending";
    if (busy && !bgStatusTicker) {
      bgStatusTicker = setInterval(renderBgStatus, 500);
    } else if (!busy && bgStatusTicker) {
      clearInterval(bgStatusTicker);
      bgStatusTicker = null;
    }
    // Fallback poll while busy, in case the SSE stream is interrupted:
    if (st.busy && !bgStatusPoller) {
      bgStatusPoller = setInterval(async () => {
        const fresh = await fetchQuietJson("/api/status");
        if (fresh) applyBgStatus(fresh);
      }, 4000);
    } else if (!st.busy && bgStatusPoller) {
      clearInterval(bgStatusPoller);
      bgStatusPoller = null;
    }
  }

  function renderBgStatus() {
    const chip = el("bgStatusChip");
    const textEl = el("bgStatusText");
    if (!chip || !textEl) return;
    const st = state.bgStatus;
    if (state.backendOffline) {
      chip.dataset.state = "error";
      textEl.textContent = "Backend offline";
      chip.title = "Cannot reach the backend — it may be restarting.";
      return;
    }
    if (!st) {
      chip.dataset.state = "idle";
      textEl.textContent = "Connecting…";
      return;
    }
    const since = Date.now() - state.bgStatusAt;
    const lastTxt =
      st.last_rematch_ms != null
        ? `Last re-match took ${fmtSecs(st.last_rematch_ms)}s` +
          (st.last_rematch_at ? ` at ${new Date(st.last_rematch_at * 1000).toLocaleTimeString()}` : "")
        : "No re-match yet in this session";
    let stateName = "idle";
    let label = "✓ Up to date";
    if (st.sync_op) {
      stateName = "running";
      label = st.sync_op;
    } else if (st.rematch === "running") {
      stateName = "running";
      const elapsed = (st.rematch_running_ms || 0) + since;
      const eta = st.last_rematch_ms ? ` / ~${fmtSecs(st.last_rematch_ms)}s` : "";
      label = `${st.warming ? "Warming up matcher" : "Re-matching"}… ${fmtSecs(elapsed)}s${eta}`;
    } else if (st.rematch === "queued") {
      stateName = "queued";
      const due = Math.max(0, (st.rematch_due_in_ms || 0) - since);
      label = due > 0 ? `Update queued · starts in ${fmtSecs(due)}s` : "Update starting…";
    } else if (st.warming) {
      stateName = "running";
      label = "Warming up matcher…";
    } else if (st.snapshot === "saving") {
      stateName = "running";
      label = "Saving categorization index…";
    } else if (st.last_error) {
      stateName = "error";
      label = `⚠ ${st.last_error}`;
    } else if (st.snapshot === "pending") {
      const due = Math.max(0, (st.snapshot_due_in_ms || 0) - since);
      label = `✓ Up to date · index save in ${fmtSecs(due)}s`;
    }
    chip.dataset.state = stateName;
    textEl.textContent = label;
    chip.title = `${lastTxt}.${st.last_error ? ` Last error: ${st.last_error}` : ""}`;
  }

  async function fetchQuietJson(path) {
    try {
      const resp = await fetch(path, { cache: "no-store" });
      if (!resp.ok) return null;
      return await resp.json();
    } catch (_e) {
      return null;
    }
  }

  async function refreshOpenPhotoModalQuietly() {
    if (!state.modalPhoto || el("photoModal").hidden) return;
    if (state.manualDrawMode || state.manualDraftBox) return;
    const targetPhotoId = state.modalPhoto.photo_id;
    const prevFacesById = new Map(
      (state.modalPhoto.faces || []).map((f) => [f.face_id, f.person_name || ""])
    );

    // Preserve any in-progress typing or focus inside modal face inputs:
    const dirtyInputs = new Map();
    let focusedFid = null;
    let selStart = null;
    let selEnd = null;
    const activeEl = document.activeElement;
    document.querySelectorAll(".modal-face-input").forEach((inp) => {
      const fid = Number(inp.getAttribute("data-fid"));
      const prevVal = prevFacesById.get(fid) || "";
      if (inp === activeEl) {
        focusedFid = fid;
        selStart = inp.selectionStart;
        selEnd = inp.selectionEnd;
        dirtyInputs.set(fid, inp.value);
      } else if (inp.value !== prevVal) {
        dirtyInputs.set(fid, inp.value);
      }
    });

    const fresh = await fetchQuietJson(`/api/photos/${targetPhotoId}`);
    if (!fresh || !state.modalPhoto || state.modalPhoto.photo_id !== targetPhotoId || el("photoModal").hidden) {
      return;
    }

    state.modalPhoto = fresh;
    renderModalFaceOverlays();
    renderModalSideFaces();

    dirtyInputs.forEach((val, fid) => {
      const inp = document.querySelector(`.modal-face-input[data-fid="${fid}"]`);
      if (inp) {
        inp.value = val;
      }
    });
    if (focusedFid !== null) {
      const focusInp = document.querySelector(`.modal-face-input[data-fid="${focusedFid}"]`);
      if (focusInp) {
        focusInp.focus({ preventScroll: true });
        if (selStart !== null && selEnd !== null) {
          try {
            focusInp.setSelectionRange(selStart, selEnd);
          } catch (_e) {}
        }
      }
    }
  }

  function connectEventStream() {
    if (typeof window.EventSource !== "function") return;
    if (state.sseSource) {
      try {
        state.sseSource.close();
      } catch (_e) {}
    }
    const es = new EventSource("/api/events");
    state.sseSource = es;

    es.addEventListener("connected", (ev) => {
      clearBackendDisconnected(true);
      try {
        const data = JSON.parse(ev.data || "{}");
        state.rematchPending = Boolean(data.pending);
        if (data.status) applyBgStatus(data.status);
        updateSyncIndicator();
      } catch (_e) {}
    });

    es.addEventListener("rematch_scheduled", (ev) => {
      try {
        const data = JSON.parse(ev.data || "{}");
        state.rematchPending = true;
        state.rematchRunning = Boolean(data.running);
        updateSyncIndicator();
      } catch (_e) {}
    });

    es.addEventListener("rematch_status", (ev) => {
      try {
        const data = JSON.parse(ev.data || "{}");
        state.rematchRunning = Boolean(data.running);
        if (data.running) state.rematchPending = true;
        updateSyncIndicator();
      } catch (_e) {}
    });

    es.addEventListener("bg_status", (ev) => {
      try {
        applyBgStatus(JSON.parse(ev.data || "{}"));
        updateSyncIndicator();
      } catch (_e) {}
    });

    es.addEventListener("overview_updated", (ev) => {
      try {
        const data = JSON.parse(ev.data || "{}");
        if (data.overview) {
          state.overview = data.overview;
          renderOverview();
        }
      } catch (_e) {}
    });

    es.addEventListener("rematch_complete", async (ev) => {
      let payload = null;
      try {
        payload = JSON.parse(ev.data || "{}");
      } catch (_e) {
        return;
      }
      state.rematchRunning = false;
      state.rematchPending = Boolean(payload.pending);
      if (payload.overview) {
        const prevSel = getSelectedPerson();
        const prevQueue = prevSel ? prevSel.review_ready_count : null;
        state.overview = payload.overview;
        renderOverview();
        const nowSel = getSelectedPerson();
        if (prevSel && nowSel && prevQueue !== null && nowSel.person_id === prevSel.person_id) {
          const delta = nowSel.review_ready_count - prevQueue;
          if (delta > 0) {
            showToast(
              `↻ Re-match: ${delta} new possible match${delta === 1 ? "" : "es"} for ${nowSel.display_name} (look-alikes of faces you just confirmed). ${nowSel.review_ready_count} left to review.`
            );
          } else if (delta < 0) {
            showToast(`↻ Re-match settled ${-delta} more for ${nowSel.display_name}. ${nowSel.review_ready_count} left to review.`);
          }
        }
      }
      updateSyncIndicator();

      if (state.modalPhoto && !el("photoModal").hidden && inFlightRequests === 0) {
        await refreshOpenPhotoModalQuietly();
      }

      if (state.mode === "batch" && state.selectedPersonId && inFlightRequests === 0) {
        const targetPid = state.selectedPersonId;
        if (!state.batchCards.length) {
          const size = Number(el("batchSizeSelect").value || 24);
          const batchData = await fetchQuietJson(`/api/batch?person_id=${targetPid}&size=${size}&mode=${batchMode()}`);
          if (
            batchData &&
            Array.isArray(batchData.cards) &&
            batchData.cards.length > 0 &&
            state.mode === "batch" &&
            state.selectedPersonId === targetPid &&
            !state.batchCards.length
          ) {
            applyBatchPayload(batchData);
          }
        } else {
          const selP = getSelectedPerson();
          if (selP && state.batchMeta) {
            state.batchMeta.remaining_unreviewed = selP.review_ready_count;
            state.batchMeta.album_cohesion_count = selP.album_cohesion_count;
            updateBatchHeaderCopy(state.batchMeta);
          }
        }
      } else if (
        state.mode === "gallery" &&
        state.selectedPersonId &&
        el("photoModal").hidden &&
        el("reassignModal").hidden &&
        inFlightRequests === 0 &&
        el("galleryStatusFilter").value !== "confirmed"
      ) {
        try {
          await loadGallery();
        } catch (_e) {}
      } else if (
        state.mode === "childclusters" &&
        state.selectedPersonId &&
        el("photoModal").hidden &&
        el("reassignModal").hidden &&
        el("renamePersonModal").hidden &&
        inFlightRequests === 0
      ) {
        try {
          await reloadChildClustersPreservingUi({ quiet: true });
        } catch (_e) {}
      }
    });
  }

  // ------------------------------------------------------------------
  // Event Wiring & Startup
  // ------------------------------------------------------------------

  function bindEvents() {
    const toastEl = el("toast");
    if (toastEl) {
      toastEl.addEventListener("click", () => {
        toastEl.hidden = true;
      });
    }

    const retryBtn = el("btnRetryBackend");
    if (retryBtn) {
      retryBtn.addEventListener("click", () => {
        runBusy(
          retryBtn,
          async () => {
            state.overview = await api("/api/overview");
            renderOverview();
            await loadActiveView();
          },
          { busyText: "Checking…", context: "Reconnect" }
        );
      });
    }

    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () =>
        runBusy(btn, () => switchMode(btn.getAttribute("data-mode")), {
          context: "Switch view",
        })
      );
    });

    el("btnRecluster").addEventListener("click", () => {
      const btn = el("btnRecluster");
      runBusy(
        btn,
        async () => {
          const res = await api("/api/recluster", { method: "POST", body: "{}" });
          state.overview = res.overview;
          renderOverview();
          await loadActiveView();
          showToast(
            `Refreshed index: ${res.overview.photos_total.toLocaleString()} photos, ${res.clusters_created} recurring clusters.`
          );
        },
        { busyText: "Refreshing…", context: "Refresh scan" }
      );
    });

    el("btnDiscoverClusters").addEventListener("click", () => {
      const btn = el("btnDiscoverClusters");
      runBusy(btn, () => switchMode("clusters"), { context: "Load clusters" });
    });

    bindChildClusterEvents();
    bindClassesEvents();

    el("btnAddPerson").addEventListener("click", () => {
      const btn = el("btnAddPerson");
      const inp = el("newPersonInput");
      const name = inp.value.trim();
      if (!name) return;
      runBusy(
        btn,
        async () => {
          const res = await api("/api/people", {
            method: "POST",
            body: JSON.stringify({ name, role: "classmate" }),
          });
          inp.value = "";
          state.overview = res.overview;
          await selectPerson(res.person_id, "clusters");
          showToast(`Added ${name}. Pick a cluster or photo face to seed ${name}!`);
        },
        { busyText: "Adding…", alsoDisable: [inp], context: "Add child" }
      );
    });

    el("newPersonInput").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        el("btnAddPerson").click();
      }
    });

    el("btnToggleSelectAll").addEventListener("click", () => {
      if (!state.batchCards.length) return;
      const mutableCards = state.batchCards.filter((c) => !c.alreadySavedInModal);
      if (!mutableCards.length) return;
      const anyApproved = mutableCards.some((c) => c.decision === "approved");
      const next = anyApproved ? "disproved" : "approved";
      mutableCards.forEach((c) => {
        c.decision = next;
        c.targetName = "";
        c.targetPersonId = null;
        c.alreadySavedInModal = false;
      });
      renderBatchGrid(state.batchMeta);
    });

    el("btnMarkAllUnsure").addEventListener("click", () => {
      if (!state.batchCards.length) return;
      const mutableCards = state.batchCards.filter((c) => !c.alreadySavedInModal);
      if (!mutableCards.length) return;
      const allUnsure = mutableCards.every((c) => c.decision === "unsure");
      const next = allUnsure ? "approved" : "unsure";
      mutableCards.forEach((c) => {
        c.decision = next;
        c.targetName = "";
        c.targetPersonId = null;
        c.alreadySavedInModal = false;
      });
      renderBatchGrid(state.batchMeta);
    });

    el("reassignForm").addEventListener("submit", (ev) => {
      ev.preventDefault();
      applyReassignChoice(null, el("reassignInput").value, el("btnSubmitReassign"));
    });
    el("btnCancelReassign").addEventListener("click", closeReassignModal);
    el("reassignModal").addEventListener("click", (ev) => {
      if (ev.target === el("reassignModal")) closeReassignModal();
    });

    el("batchModeSelect").addEventListener("change", (ev) => {
      runBusy(ev.currentTarget, () => loadBatchForSelected(), { context: "Reload batch" });
    });
    el("batchSizeSelect").addEventListener("change", (ev) => {
      runBusy(ev.currentTarget, () => loadBatchForSelected(), { context: "Reload batch" });
    });
    el("btnSubmitBatch").addEventListener("click", () => submitCurrentBatch());

    el("clusterChildFilter").addEventListener("change", (ev) => {
      runBusy(ev.currentTarget, () => loadClusters(), { context: "Filter clusters" });
    });
    el("clusterYearFilter").addEventListener("change", (ev) => {
      runBusy(ev.currentTarget, () => loadClusters(), { context: "Filter clusters" });
    });
    el("clusterShowAssigned").addEventListener("change", (ev) => {
      runBusy(ev.currentTarget, () => loadClusters(), { context: "Filter clusters" });
    });

    el("galleryStatusFilter").addEventListener("change", (ev) => {
      state.galleryOffset = 0;
      runBusy(ev.currentTarget, () => loadGallery(), { context: "Filter gallery" });
    });
    el("galleryYearFilter").addEventListener("change", (ev) => {
      state.galleryOffset = 0;
      runBusy(ev.currentTarget, () => loadGallery(), { context: "Filter gallery" });
    });

    el("renamePersonForm").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const targetPid = state.renameTargetPersonId;
      if (!targetPid || !state.overview) return;
      const targetP = (state.overview.people || []).find((p) => p.person_id === targetPid);
      if (!targetP) return;

      const oldName = targetP.display_name;
      const trimmedNew = el("renamePersonInput").value.trim();
      const chosenRole = el("renamePersonRoleSelect").value || targetP.role;
      if (!trimmedNew) return;

      const existingTarget = (state.overview.people || []).find(
        (p) =>
          p.person_id !== targetP.person_id &&
          p.display_name.toLowerCase() === trimmedNew.toLowerCase()
      );
      if (existingTarget) {
        const ok = window.confirm(
          `"${existingTarget.display_name}" already exists. Merge "${oldName}" into "${existingTarget.display_name}"?`
        );
        if (!ok) return;
        runBusy(
          el("btnSaveRenamePerson"),
          async () => {
            const mRes = await api(`/api/people/${targetP.person_id}`, {
              method: "PATCH",
              body: JSON.stringify({ merge_into_person_id: existingTarget.person_id }),
            });
            closeRenameModal();
            state.overview = mRes.overview;
            if (state.selectedPersonId === targetP.person_id) {
              state.selectedPersonId = existingTarget.person_id;
            }
            renderOverview();
            await loadActiveView();
            showToast(`Merged "${oldName}" into ${existingTarget.display_name}!`);
          },
          {
            busyText: "Merging…",
            alsoDisable: [el("renamePersonInput"), el("renamePersonRoleSelect"), el("btnCancelRenamePerson")],
            context: "Merge child",
          }
        );
        return;
      }

      const oldBirth = targetP.birth_date || "";
      const newBirth = (el("renamePersonBirthInput").value || "").trim();
      const birthChanged = newBirth !== oldBirth;
      if (birthChanged && newBirth && (!oldBirth || newBirth > oldBirth)) {
        const okBirth = window.confirm(
          `Set birthday of "${oldName}" to ${newBirth}? Existing tags of ${oldName} on photos dated before ${newBirth} will be removed.`
        );
        if (!okBirth) return;
      }

      runBusy(
        el("btnSaveRenamePerson"),
        async () => {
          const body = {
            display_name: trimmedNew,
            role: chosenRole,
          };
          if (birthChanged) body.birth_date = newBirth;
          const res = await api(`/api/people/${targetP.person_id}`, {
            method: "PATCH",
            body: JSON.stringify(body),
          });
          closeRenameModal();
          const updatedName = (res.person && res.person.display_name) || trimmedNew;
          for (const c of state.batchCards) {
            if (
              c.decision === "reassigned" &&
              (c.targetPersonId === targetP.person_id ||
                (c.targetName && c.targetName.toLowerCase() === oldName.toLowerCase()))
            ) {
              c.targetPersonId = targetP.person_id;
              c.targetName = updatedName;
            }
          }
          state.overview = res.overview;
          renderOverview();
          const birthRemoved = Number(res.birthday_removed || 0);
          const affectsSelected = birthRemoved > 0 && state.selectedPersonId === targetP.person_id;
          if (state.mode === "batch") {
            if (affectsSelected && !state.batchUserTouched) await loadBatchForSelected();
            else renderBatchGrid(state.batchMeta);
          } else if (state.mode === "childclusters") {
            if (state.selectedPersonId === targetP.person_id) await reloadChildClustersPreservingUi();
          } else {
            await loadActiveView();
          }
          const birthNote =
            birthRemoved > 0
              ? ` · birthday removed ${birthRemoved} earlier tag${birthRemoved === 1 ? "" : "s"}`
              : birthChanged
              ? newBirth
                ? ` · birthday ${newBirth}`
                : " · birthday cleared"
              : "";
          showToast(`Updated "${oldName}" → ${updatedName}${birthNote}`);
        },
        {
          busyText: "Saving…",
          alsoDisable: [
            el("renamePersonInput"),
            el("renamePersonRoleSelect"),
            el("renamePersonBirthInput"),
            el("btnClearRenameBirth"),
            el("btnDeleteRenamePerson"),
            el("btnCancelRenamePerson"),
          ],
          context: "Rename child",
        }
      );
    });

    el("btnClearRenameBirth").addEventListener("click", () => {
      el("renamePersonBirthInput").value = "";
    });

    el("btnDeleteRenamePerson").addEventListener("click", () => {
      const pid = state.renameTargetPersonId;
      if (!pid) return;
      deletePerson(pid, el("btnDeleteRenamePerson"), [
        el("btnSaveRenamePerson"),
        el("btnCancelRenamePerson"),
        el("renamePersonInput"),
        el("renamePersonRoleSelect"),
        el("renamePersonBirthInput"),
        el("btnClearRenameBirth"),
      ]);
    });

    el("btnCancelRenamePerson").addEventListener("click", closeRenameModal);
    el("renamePersonModal").addEventListener("click", (ev) => {
      if (ev.target === el("renamePersonModal")) closeRenameModal();
    });

    el("btnRenamePerson").addEventListener("click", () => openRenameModal(state.selectedPersonId));
    el("btnRenamePersonTop").addEventListener("click", () => openRenameModal(state.selectedPersonId));

    el("btnMergePerson").addEventListener("click", () => {
      const btn = el("btnMergePerson");
      const selP = getSelectedPerson();
      if (!selP || !state.overview) return;
      const targetName = window.prompt(
        `Merge "${selP.display_name}" into which existing child name?`
      );
      if (!targetName) return;
      const target = state.overview.people.find(
        (p) =>
          p.person_id !== selP.person_id &&
          p.display_name.toLowerCase() === targetName.trim().toLowerCase()
      );
      if (!target) {
        showToast(`No existing child named "${targetName.trim()}".`, "error");
        return;
      }
      runBusy(
        btn,
        async () => {
          const res = await api(`/api/people/${selP.person_id}`, {
            method: "PATCH",
            body: JSON.stringify({ merge_into_person_id: target.person_id }),
          });
          state.overview = res.overview;
          await selectPerson(target.person_id, "gallery");
          showToast(`Merged into ${target.display_name}!`);
        },
        { busyText: "Merging…", context: "Merge child" }
      );
    });

    el("btnMaterializeAll").addEventListener("click", () => {
      const btn = el("btnMaterializeAll");
      runBusy(
        btn,
        async () => {
          const res = await api("/api/materialize", { method: "POST", body: "{}" });
          showToast(
            `Synced ByChild/: ${res.links_created} new links, ${res.links_existing} existing, ${res.links_pruned} pruned.`
          );
        },
        { busyText: "Syncing ByChild/…", context: "Sync all folders" }
      );
    });

    el("btnSyncPersonFolder").addEventListener("click", () => {
      const btn = el("btnSyncPersonFolder");
      const selP = getSelectedPerson();
      if (!selP) return;
      runBusy(
        btn,
        async () => {
          const res = await api("/api/materialize", {
            method: "POST",
            body: JSON.stringify({ person: selP.display_name }),
          });
          showToast(
            `Synced ByChild/${selP.display_name}/: ${res.links_created} new, ${res.links_existing} existing.`
          );
        },
        { busyText: "Syncing…", context: "Sync folder" }
      );
    });

    document.querySelectorAll("[data-export]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const selP = getSelectedPerson();
        if (!selP) return;
        const fmt = btn.getAttribute("data-export") || "txt";
        const qs = new URLSearchParams({ person: selP.display_name, format: fmt });
        const fallback = `${selP.display_name.replace(/\s+/g, "_")}_urls.${fmt}`;
        runBusy(
          btn,
          async () => {
            const fname = await downloadApiFile(`/api/export-urls?${qs.toString()}`, fallback);
            showToast(`Downloaded ${fname}`);
          },
          { busyText: `Exporting .${fmt}…`, context: "Export URLs" }
        );
      });
    });

    el("btnDrawManualTag").addEventListener("click", () => {
      setManualDrawMode(!state.manualDrawMode);
    });
    bindManualBoxDrawing();

    el("btnClusterPrev").addEventListener("click", () => {
      if (state.modalNavContext) {
        runBusy(
          el("btnClusterPrev"),
          () => jumpToModalNavIndex(state.modalNavContext.index - 1),
          { context: "Previous photo" }
        );
      }
    });

    el("btnClusterNext").addEventListener("click", () => {
      if (state.modalNavContext) {
        runBusy(
          el("btnClusterNext"),
          () => jumpToModalNavIndex(state.modalNavContext.index + 1),
          { context: "Next photo" }
        );
      }
    });

    el("btnToggleSeenPhotos").addEventListener("click", () => {
      state.showSeenPhotos = !state.showSeenPhotos;
      rebuildCurrentDedupedNav();
    });

    el("btnResetSeenPhotos").addEventListener("click", () => {
      const n = state.seenPhotoIds.size;
      if (!n) return;
      if (!window.confirm(`Forget ${n} reviewed photo${n === 1 ? "" : "s"}? They will show up again in clusters.`)) {
        return;
      }
      state.seenPhotoIds = new Set();
      saveSeenPhotoIds();
      rebuildCurrentDedupedNav();
      if (state.mode === "childclusters") renderChildClusters();
      showToast("Reviewed-photo history cleared.");
    });

    el("btnClusterExcludeFace").addEventListener("click", () => {
      const ctx = state.modalNavContext;
      if (!ctx || ctx.type !== "cluster" || !ctx.faces.length) return;
      const curItem = ctx.faces[ctx.index];
      if (!curItem) return;
      runBusy(
        el("btnClusterExcludeFace"),
        async () => {
          const res = await api(`/api/clusters/${ctx.clusterId}/remove-face`, {
            method: "POST",
            body: JSON.stringify({ face_id: curItem.face_id }),
          });
          state.modalDirty = true;
          state.overview = res.overview;
          renderOverview();
          if (!res.cluster || !res.cluster.faces || !res.cluster.faces.length) {
            await closePhotoModal();
            showToast("Excluded photo — cluster is now empty.");
            return;
          }
          ctx.rawFaces = res.cluster.faces;
          ctx.photoCount = res.cluster.photo_count;
          const removedPhoto = curItem.photo_id;
          applyDedupedNav(ctx, null, removedPhoto, true);
          if (!ctx.faces.length) {
            await closePhotoModal();
            showToast("Excluded photo — no unseen photos left in this cluster.");
            return;
          }
          const stillThere = ctx.faces.findIndex((it) => Number(it.photo_id) === Number(removedPhoto));
          const nextIdx = stillThere >= 0 ? stillThere : Math.min(ctx.index, ctx.faces.length - 1);
          await jumpToModalNavIndex(nextIdx);
          showToast("Excluded photo from cluster — child will stay uncategorized when confirming.");
        },
        { busyText: "Excluding…", context: "Exclude photo from cluster" }
      );
    });

    el("modalClusterAssignForm").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const ctx = state.modalNavContext;
      if (!ctx || ctx.type !== "cluster") return;
      const inp = el("modalClusterNameInput");
      const submitBtn = ev.currentTarget.querySelector('button[type="submit"]');
      const name = inp.value.trim();
      if (!name) return;
      const cid = ctx.clusterId;
      runBusy(
        submitBtn,
        async () => {
          await assignCluster(cid, name, "classmate", true);
          state.modalDirty = false;
          await closePhotoModal();
        },
        {
          busyText: "Confirming…",
          alsoDisable: [inp, el("btnClusterExcludeFace")].filter(Boolean),
          context: "Confirm cluster",
        }
      );
    });

    el("btnCloseModal").addEventListener("click", () => {
      runBusy(el("btnCloseModal"), () => closePhotoModal(), { context: "Close modal" });
    });
    el("photoModal").addEventListener("click", (ev) => {
      if (ev.target === el("photoModal")) closePhotoModal();
    });

    window.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        if (!el("renamePersonModal").hidden) {
          closeRenameModal();
          return;
        }
        if (!el("reassignModal").hidden) {
          closeReassignModal();
          return;
        }
        if (!el("classesModal").hidden) {
          closeClassesModal();
          return;
        }
        if (!el("photoModal").hidden) {
          if (state.manualDraftBox) {
            state.manualDraftBox = null;
            syncManualDrawUi();
            renderModalFaceOverlays();
            return;
          }
          if (state.manualDrawMode) {
            setManualDrawMode(false);
            return;
          }
          closePhotoModal();
          return;
        }
      }
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (
        !el("photoModal").hidden &&
        el("reassignModal").hidden &&
        el("renamePersonModal").hidden &&
        (ev.key === "t" || ev.key === "T") &&
        !ev.metaKey &&
        !ev.ctrlKey &&
        !ev.altKey
      ) {
        ev.preventDefault();
        setManualDrawMode(!state.manualDrawMode);
        return;
      }
      if (!el("photoModal").hidden && state.modalNavContext) {
        const navCtx = state.modalNavContext;
        if (
          navCtx.type === "gallery" &&
          navCtx.ccKey &&
          el("reassignModal").hidden &&
          el("renamePersonModal").hidden &&
          !state.manualDrawMode &&
          !ev.metaKey &&
          !ev.ctrlKey &&
          !ev.altKey &&
          (ev.key === "b" || ev.key === "B" || ev.key === "n" || ev.key === "N")
        ) {
          ev.preventDefault();
          const act = ev.key.toLowerCase() === "b" ? "boost" : "penalize";
          const qBtn = el("modalBatchQuickRow").querySelector(`[data-ccact="${act}"]`);
          applyModalChildClusterAction(act, qBtn);
          return;
        }
        if (ev.key === "ArrowLeft") {
          ev.preventDefault();
          runBusy(
            el("btnClusterPrev"),
            () => jumpToModalNavIndex(state.modalNavContext.index - 1),
            { context: "Previous photo" }
          );
          return;
        }
        if (ev.key === "ArrowRight") {
          ev.preventDefault();
          runBusy(
            el("btnClusterNext"),
            () => jumpToModalNavIndex(state.modalNavContext.index + 1),
            { context: "Next photo" }
          );
          return;
        }
      }
      if (
        ev.key === "Enter" &&
        state.mode === "batch" &&
        el("photoModal").hidden &&
        el("reassignModal").hidden &&
        el("renamePersonModal").hidden &&
        el("classesModal").hidden &&
        state.batchCards.length > 0
      ) {
        ev.preventDefault();
        submitCurrentBatch();
      }
    });
  }

  async function init() {
    state.seenPhotoIds = loadSeenPhotoIds();
    bindEvents();
    connectEventStream();
    fetchQuietJson("/api/status").then((st) => {
      if (st) applyBgStatus(st);
    });
    try {
      state.overview = await api("/api/overview");
      const firstActive = (state.overview.people || []).find((p) => p.role !== "ignore");
      if (firstActive) {
        state.selectedPersonId = firstActive.person_id;
      } else {
        state.mode = "clusters";
      }
      renderOverview();
      syncModeTabs();
      if (state.mode === "batch" && state.selectedPersonId) {
        await loadBatchForSelected();
      } else {
        await loadClusters();
      }
    } catch (err) {
      reportError(err, "Startup");
    }
  }

  init();
})();
