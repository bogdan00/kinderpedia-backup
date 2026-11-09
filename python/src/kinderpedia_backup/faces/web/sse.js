// Live server events (`/api/events`): bg_status, re-match lifecycle, overview pushes and background-job
// completion (with a reconciliation fallback for jobs that finished while the stream was down).
import { clearBackendDisconnected, fetchQuietJson, inFlightCount, jobStartsPending } from "./api.js";
import { refreshOpenPhotoModalQuietly } from "./components/lightbox.js";
import { renderBgStatus } from "./components/statusChip.js";
import { showToast } from "./components/toasts.js";
import { loadActiveView } from "./router.js";
import { emit, getSelectedPerson, setOverview, state } from "./state.js";
import { el } from "./util/dom.js";
import { plural } from "./util/format.js";
import { refreshBatchAfterRematch } from "./views/batch.js";
import { reloadChildClustersPreservingUi } from "./views/childClusters.js";
import { loadGallery } from "./views/gallery.js";

let bgStatusTicker = null;
let bgStatusPoller = null;

/** Completion toast for a finished job (pure). `wasMine`: started from this tab. */
export function jobCompletionMessage(job, wasMine) {
  const ok = job.ok !== undefined ? Boolean(job.ok) : job.state === "done";
  if (!ok) return { kind: "error", text: `${job.label || job.kind} failed: ${job.error || "unknown error"}` };
  const r = job.result || {};
  if (job.kind === "recluster") {
    const photos = r.overview ? Number(r.overview.photos_total || 0).toLocaleString() : "?";
    return { kind: "info", text: `Refreshed index: ${photos} photos, ${r.clusters_created ?? "?"} recurring clusters.` };
  }
  if (job.kind === "materialize") {
    const where = r.person ? `ByChild/${r.person}/` : "ByChild/";
    const ly = r.layout ? ` (${r.layout})` : "";
    return {
      kind: "info",
      text: `Synced ${where}${ly}: ${r.links_created ?? 0} new links, ${r.links_existing ?? 0} existing, ${r.links_pruned ?? 0} pruned.`,
    };
  }
  return wasMine ? { kind: "info", text: `${job.label || job.kind} finished.` } : null;
}

export async function handleJobComplete(job) {
  if (!job || !job.job_id || state.handledJobIds.has(job.job_id)) return;
  state.handledJobIds.add(job.job_id);
  const wasMine = state.pendingJobs.delete(job.job_id) || jobStartsPending();
  emit("sync");
  const msg = jobCompletionMessage(job, wasMine);
  const ok = !msg || msg.kind !== "error";
  if (ok && job.kind === "recluster") {
    setOverview((job.result || {}).overview);
    const userBusy = !el("photoModal").hidden || !el("reassignModal").hidden || (state.mode === "batch" && state.batchUserTouched);
    if (wasMine && !userBusy) await loadActiveView().catch(() => {});
  }
  if (msg) showToast(msg.text, msg.kind);
}

/** SSE fallback: a pending job that vanished from bg_status finished while we were not listening. */
async function reconcilePendingJobs(st) {
  if (!state.pendingJobs.size) return;
  const live = new Set(((st && st.jobs) || []).map((j) => j.job_id));
  for (const jobId of Array.from(state.pendingJobs.keys())) {
    if (live.has(jobId)) continue;
    const job = await fetchQuietJson(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (job && job.state && job.state !== "running") {
      await handleJobComplete(job);
    } else if (!job) {
      state.pendingJobs.delete(jobId); // server restarted: the job is gone
      emit("sync");
    }
  }
}

export function applyBgStatus(st) {
  if (!st || typeof st !== "object") return;
  state.bgStatus = st;
  state.bgStatusAt = Date.now();
  state.rematchRunning = st.rematch === "running";
  state.rematchPending = st.rematch !== "idle";
  emit("sync");
  reconcilePendingJobs(st);
  const ticking = Boolean(st.busy) || st.snapshot === "pending";
  if (ticking && !bgStatusTicker) {
    bgStatusTicker = setInterval(renderBgStatus, 500);
  } else if (!ticking && bgStatusTicker) {
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

function rematchToast(prevSel, prevQueue) {
  const nowSel = getSelectedPerson();
  if (!prevSel || !nowSel || prevQueue === null || nowSel.person_id !== prevSel.person_id) return;
  const delta = nowSel.review_ready_count - prevQueue;
  const left = `${nowSel.review_ready_count} left to review.`;
  if (delta > 0) {
    showToast(
      `↻ Re-match: ${plural(delta, "new possible match", "new possible matches")} for ${nowSel.display_name} (look-alikes of faces you just confirmed). ${left}`
    );
  } else if (delta < 0) {
    showToast(`↻ Re-match settled ${-delta} more for ${nowSel.display_name}. ${left}`);
  }
}

async function onRematchComplete(payload) {
  state.rematchRunning = false;
  state.rematchPending = Boolean(payload.pending);
  if (payload.overview) {
    const prevSel = getSelectedPerson();
    const prevQueue = prevSel ? prevSel.review_ready_count : null;
    setOverview(payload.overview);
    rematchToast(prevSel, prevQueue);
  }
  emit("sync");
  if (inFlightCount() > 0) return;
  const modalOpen = !el("photoModal").hidden;
  if (state.modalPhoto && modalOpen) await refreshOpenPhotoModalQuietly();
  if (!state.selectedPersonId) return;
  const calm = !modalOpen && el("reassignModal").hidden && inFlightCount() === 0;
  if (state.mode === "batch") {
    await refreshBatchAfterRematch();
  } else if (state.mode === "gallery" && calm && el("galleryStatusFilter").value !== "confirmed") {
    await loadGallery({ quiet: true }).catch(() => {});
  } else if (state.mode === "childclusters" && calm && el("renamePersonModal").hidden) {
    await reloadChildClustersPreservingUi({ quiet: true }).catch(() => {});
  }
}

const HANDLERS = {
  connected(data) {
    clearBackendDisconnected(true);
    state.rematchPending = Boolean(data.pending);
    if (data.status) applyBgStatus(data.status);
    emit("sync");
  },
  rematch_scheduled(data) {
    state.rematchPending = true;
    state.rematchRunning = Boolean(data.running);
    emit("sync");
  },
  rematch_status(data) {
    state.rematchRunning = Boolean(data.running);
    if (data.running) state.rematchPending = true;
    emit("sync");
  },
  bg_status(data) {
    applyBgStatus(data);
  },
  overview_updated(data) {
    setOverview(data.overview);
  },
  job_status(job) {
    if (job && job.state === "running" && state.bgStatus) {
      state.bgStatus.jobs = [...(state.bgStatus.jobs || []).filter((j) => j.job_id !== job.job_id), job];
    }
    emit("sync");
  },
  job_complete(job) {
    handleJobComplete(job);
  },
  data_changed(data) {
    if (data.source === "external") {
      showToast("↻ New scan data detected on disk — refreshing caches and re-matching in the background.");
    }
  },
  rematch_complete(data) {
    onRematchComplete(data);
  },
};

export function connectEventStream() {
  if (typeof window.EventSource !== "function") return;
  if (state.sseSource) {
    try {
      state.sseSource.close();
    } catch (_e) {}
  }
  const es = new EventSource("/api/events");
  state.sseSource = es;
  for (const [name, handler] of Object.entries(HANDLERS)) {
    es.addEventListener(name, (ev) => {
      let data;
      try {
        data = JSON.parse(ev.data || "{}");
      } catch (_e) {
        return;
      }
      handler(data || {});
    });
  }
}
