// Faces studio bootstrap: binds every module's listeners once, opens the SSE stream, loads the first view.
import { api, fetchQuietJson } from "./api.js";
import { bindLightbox } from "./components/lightbox.js";
import { bindLightboxNav, loadSeenPhotoIds } from "./components/lightboxNav.js";
import { bindManualBox } from "./components/manualBox.js";
import { bindReassignModal } from "./components/reassignModal.js";
import { bindRoster } from "./components/roster.js";
import { bindStatusChip } from "./components/statusChip.js";
import { bindToasts, reportError } from "./components/toasts.js";
import { bindKeyboard } from "./keyboard.js";
import { syncModeTabs } from "./router.js";
import { applyBgStatus, connectEventStream } from "./sse.js";
import { setOverview, state } from "./state.js";
import { bindBatch, loadBatchForSelected } from "./views/batch.js";
import { bindChildClusters } from "./views/childClusters.js";
import { bindClasses } from "./views/classes.js";
import { bindClusters, loadClusters } from "./views/clusters.js";
import { bindGallery } from "./views/gallery.js";
import { bindAlbumsEvents } from "./views/albums.js";
import { bindOverview } from "./views/overview.js";

for (const bind of [
  bindToasts,
  bindStatusChip,
  bindOverview,
  bindRoster,
  bindBatch,
  bindChildClusters,
  bindClusters,
  bindGallery,
  bindAlbumsEvents,
  bindClasses,
  bindReassignModal,
  bindLightbox,
  bindLightboxNav,
  bindManualBox,
  bindKeyboard,
]) {
  bind();
}

async function init() {
  state.seenPhotoIds = loadSeenPhotoIds();
  connectEventStream();
  fetchQuietJson("/api/status").then((st) => st && applyBgStatus(st));
  try {
    const overview = await api("/api/overview");
    const firstActive = (overview.people || []).find((p) => p.role !== "ignore");
    if (firstActive) state.selectedPersonId = firstActive.person_id;
    else state.mode = "clusters";
    setOverview(overview);
    syncModeTabs();
    if (state.mode === "batch" && state.selectedPersonId) await loadBatchForSelected();
    else await loadClusters();
  } catch (err) {
    reportError(err, "Startup");
  }
}

init();
