// Header status: background-work chip, re-match pill, offline banner, and job-button disabling.
import { activeJobKinds } from "../api.js";
import { state, subscribe } from "../state.js";
import { el } from "../util/dom.js";
import { fmtSecs } from "../util/format.js";
import { showToast } from "./toasts.js";

const JOB_BUTTON_IDS = {
  recluster: ["btnRecluster"],
  materialize: ["btnMaterializeAll", "btnSyncPersonFolder"],
};

/** Buttons stay disabled while a job of their kind runs (bg_status + jobs started from this tab). */
export function syncJobButtons() {
  const kinds = activeJobKinds(state.bgStatus, state.pendingJobs);
  for (const [kind, ids] of Object.entries(JOB_BUTTON_IDS)) {
    const running = kinds.has(kind);
    for (const id of ids) {
      const btn = el(id);
      if (!btn || btn.dataset.busy === "1") continue;
      btn.disabled = running;
      btn.classList.toggle("is-loading", running);
      if (running) {
        btn.setAttribute("aria-busy", "true");
        btn.dataset.jobRunning = "1";
      } else {
        btn.removeAttribute("aria-busy");
        delete btn.dataset.jobRunning;
      }
    }
  }
}

function updateRematchPill() {
  const pill = el("rematchStatusPill");
  if (!pill) return;
  pill.hidden = !(state.rematchPending || state.rematchRunning);
  pill.textContent = state.rematchRunning ? "↻ syncing" : "↻ queued";
}

/** Chip label/state for a bg_status snapshot `sinceMs` old (pure). */
export function bgStatusView(st, sinceMs, offline) {
  if (offline) return { state: "error", label: "Backend offline", title: "Cannot reach the backend — it may be restarting." };
  if (!st) return { state: "idle", label: "Connecting…", title: "" };
  const lastTxt =
    st.last_rematch_ms != null
      ? `Last re-match took ${fmtSecs(st.last_rematch_ms)}s` +
        (st.last_rematch_at ? ` at ${new Date(st.last_rematch_at * 1000).toLocaleTimeString()}` : "")
      : "No re-match yet in this session";
  let stateName = "idle";
  let label = "✓ Up to date";
  if (st.sync_op) {
    stateName = "running";
    const jobs = Array.isArray(st.jobs) ? st.jobs : [];
    const longest = jobs.reduce((acc, j) => Math.max(acc, Number(j.running_ms || 0)), 0);
    label = jobs.length ? `${st.sync_op} ${fmtSecs(longest + sinceMs)}s` : st.sync_op;
  } else if (st.rematch === "running") {
    stateName = "running";
    const eta = st.last_rematch_ms ? ` / ~${fmtSecs(st.last_rematch_ms)}s` : "";
    label = `${st.warming ? "Warming up matcher" : "Re-matching"}… ${fmtSecs((st.rematch_running_ms || 0) + sinceMs)}s${eta}`;
  } else if (st.rematch === "queued") {
    stateName = "queued";
    const due = Math.max(0, (st.rematch_due_in_ms || 0) - sinceMs);
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
    label = `✓ Up to date · index save in ${fmtSecs(Math.max(0, (st.snapshot_due_in_ms || 0) - sinceMs))}s`;
  }
  return { state: stateName, label, title: `${lastTxt}.${st.last_error ? ` Last error: ${st.last_error}` : ""}` };
}

export function renderBgStatus() {
  const chip = el("bgStatusChip");
  const textEl = el("bgStatusText");
  if (!chip || !textEl) return;
  const view = bgStatusView(state.bgStatus, Date.now() - state.bgStatusAt, state.backendOffline);
  chip.dataset.state = view.state;
  textEl.textContent = view.label;
  if (view.title) chip.title = view.title;
}

function renderBackendBanner({ offline, message, reconnected }) {
  const banner = el("backendAlertBanner");
  const msgEl = el("backendAlertMsg");
  if (offline && msgEl && message) msgEl.textContent = `${message} — your current selections on screen are preserved.`;
  if (banner) banner.hidden = !offline;
  renderBgStatus();
  if (reconnected) showToast("✓ Backend reconnected! You can continue right where you left off.");
}

export function bindStatusChip() {
  subscribe("backend", renderBackendBanner);
  subscribe("sync", () => {
    updateRematchPill();
    syncJobButtons();
    renderBgStatus();
  });
}
