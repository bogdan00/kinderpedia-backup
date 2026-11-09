// Backend access: JSON fetch wrapper (CSRF header, timeout, offline/retryable policy), quiet GETs,
// file downloads and background-job starts. UI reacts through store topics ("backend", "sync").
import { emit, state } from "./state.js";
import { el } from "./util/dom.js";

const DEFAULT_TIMEOUT_MS = 45000;
const OFFLINE_MSG = "Cannot reach backend on 127.0.0.1:8765 (server may be restarting)";
const SLOW_MSG = "Request timed out — backend is still working in background";

let inFlightRequests = 0;
let jobStartsInFlight = 0;
let reconnectTimer = null;

export const inFlightCount = () => inFlightRequests;
export const jobStartsPending = () => jobStartsInFlight > 0;

function csrfToken() {
  const meta = document.querySelector('meta[name="kp-csrf"]');
  return meta ? meta.getAttribute("content") || "" : "";
}

function adjustInFlight(delta) {
  inFlightRequests = Math.max(0, inFlightRequests + delta);
  const bar = el("topProgressBar");
  if (bar) bar.hidden = inFlightRequests === 0;
}

/** `/api/x` + query params (empty / null values dropped). */
export function apiUrl(path, params = null) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== null && v !== undefined && v !== "") qs.set(k, String(v));
  }
  const q = qs.toString();
  return q ? `${path}?${q}` : path;
}

/** Request headers: mutating calls carry the CSRF token + JSON content type. */
export function requestHeaders(method, extra = {}, token = "") {
  const headers = { ...extra };
  if (method && method !== "GET") {
    headers["X-KP-Token"] = token;
    headers["Content-Type"] = "application/json";
  }
  return headers;
}

/**
 * How a response maps onto the UI: "offline" (backend gone / gateway errors / non-retryable 503),
 * "error" (an API error to show; 503 + retryable = DB briefly locked, not offline) or "ok".
 */
export function classifyResponse(status, data) {
  if (status === 502 || status === 504) return "offline";
  if (status === 503 && !(data && data.retryable)) return "offline";
  if (status < 200 || status >= 300) return "error";
  return "ok";
}

export function setBackendDisconnected(detailMsg) {
  state.backendOffline = true;
  emit("backend", { offline: true, message: detailMsg });
  if (reconnectTimer) return;
  reconnectTimer = setInterval(async () => {
    if (!state.backendOffline) return clearReconnectTimer();
    const ov = await fetchQuietJson("/api/overview");
    if (ov) {
      clearBackendDisconnected(true);
      emit("backend-overview", ov);
    }
  }, 3000);
}

function clearReconnectTimer() {
  if (reconnectTimer) clearInterval(reconnectTimer);
  reconnectTimer = null;
}

export function clearBackendDisconnected(announce = false) {
  const wasOffline = state.backendOffline;
  state.backendOffline = false;
  clearReconnectTimer();
  emit("backend", { offline: false, reconnected: wasOffline && announce });
}

function offline(msg) {
  setBackendDisconnected(msg);
  return new Error(msg);
}

export async function api(path, opts = {}) {
  const headers = requestHeaders(opts.method, opts.headers, csrfToken());
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), opts.timeoutMs || DEFAULT_TIMEOUT_MS);
  adjustInFlight(1);
  try {
    let resp;
    try {
      resp = await fetch(path, { ...opts, headers, signal: controller.signal });
    } catch (netErr) {
      if (netErr && netErr.name === "AbortError") {
        // A slow answer is not a dead backend: the status chip / SSE stream tell the real story.
        const slowErr = new Error(SLOW_MSG);
        slowErr.timedOut = true;
        throw slowErr;
      }
      throw offline(OFFLINE_MSG);
    }
    const data = await resp.json().catch(() => ({}));
    const kind = classifyResponse(resp.status, data);
    if (kind === "offline") throw offline(`Backend temporarily unavailable (HTTP ${resp.status})`);
    clearBackendDisconnected(true);
    if (kind === "error") {
      const apiErr = new Error(data.error || `Request failed (HTTP ${resp.status})`);
      apiErr.status = resp.status;
      apiErr.data = data;
      apiErr.retryable = Boolean(data && data.retryable);
      throw apiErr;
    }
    if (data && data.rematch_scheduled) {
      state.rematchPending = true;
      emit("sync");
    }
    return data;
  } finally {
    clearTimeout(timeoutId);
    adjustInFlight(-1);
  }
}

/** POST JSON helper. */
export function post(path, body = {}, method = "POST", opts = {}) {
  return api(path, { ...opts, method, body: JSON.stringify(body) });
}

/** GET that never throws or flags offline (background refreshes): JSON or null. */
export async function fetchQuietJson(path) {
  try {
    const resp = await fetch(path, { cache: "no-store" });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (_e) {
    return null;
  }
}

export async function downloadApiFile(path, fallbackFilename) {
  adjustInFlight(1);
  try {
    let resp;
    try {
      resp = await fetch(path, { cache: "no-store" });
    } catch (_netErr) {
      throw offline(OFFLINE_MSG);
    }
    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.error || `Export failed (HTTP ${resp.status})`);
    }
    clearBackendDisconnected(true);
    const match = (resp.headers.get("Content-Disposition") || "").match(/filename="?([^";]+)"?/i);
    const filename = (match && match[1]) || fallbackFilename;
    const objUrl = URL.createObjectURL(await resp.blob());
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

// ---------------------------------------------------------------- background jobs (202 + job id, done via SSE)
export function trackJob(job) {
  if (!job || !job.job_id || state.handledJobIds.has(job.job_id)) return;
  state.pendingJobs.set(job.job_id, job);
  emit("sync");
}

/**
 * Start a job; a 409 means the same kind is already running, which we then follow too.
 * Returns { job, started }: true = started, false = already running, null = finished before the 202 arrived.
 */
export async function startBackgroundJob(path, body) {
  jobStartsInFlight += 1;
  try {
    const job = await post(path, body || {});
    if (state.handledJobIds.has(job.job_id)) return { job, started: null };
    trackJob(job);
    return { job, started: true };
  } catch (err) {
    if (err && err.status === 409 && err.data && err.data.job) {
      trackJob(err.data.job);
      return { job: err.data.job, started: false };
    }
    throw err;
  } finally {
    jobStartsInFlight -= 1;
  }
}

/** Job kinds currently running (server bg_status + jobs this tab started). */
export function activeJobKinds(bgStatus, pendingJobs) {
  const kinds = new Set();
  for (const j of (bgStatus && bgStatus.jobs) || []) kinds.add(j.kind);
  for (const j of pendingJobs.values()) kinds.add(j.kind);
  return kinds;
}
