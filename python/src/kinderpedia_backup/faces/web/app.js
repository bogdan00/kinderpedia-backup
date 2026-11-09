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
import { handleUrlChange, loadActiveView, parseUrl, syncModeTabs, updateUrl } from "./router.js";
import { applyBgStatus, connectEventStream } from "./sse.js";
import { setOverview, state } from "./state.js";
import { bindBatch } from "./views/batch.js";
import { bindChildClusters } from "./views/childClusters.js";
import { bindClasses } from "./views/classes.js";
import { bindClusters } from "./views/clusters.js";
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

  window.addEventListener("popstate", () => handleUrlChange());
  window.addEventListener("hashchange", () => handleUrlChange());

  try {
    const overview = await api("/api/overview");
    const parsed = parseUrl();

    // 1. Resolve Person
    if (parsed.person) {
      const match = (overview.people || []).find((p) => p.person_id === parsed.person);
      if (match) state.selectedPersonId = match.person_id;
    }
    if (!state.selectedPersonId) {
      const firstActive = (overview.people || []).find((p) => p.role !== "ignore");
      if (firstActive) state.selectedPersonId = firstActive.person_id;
    }

    // 2. Resolve Mode
    if (parsed.mode) {
      state.mode = parsed.mode;
    } else if (!state.selectedPersonId) {
      state.mode = "clusters";
    } else {
      state.mode = "batch";
    }

    // 3. Album parameters
    if (parsed.child !== null && parsed.child !== undefined) state.albumFilterChild = parsed.child || "";
    if (parsed.year !== null && parsed.year !== undefined) state.albumFilterYear = parsed.year || "";
    if (parsed.album) state.selectedAlbumKey = parsed.album;

    setOverview(overview);
    syncModeTabs();

    await loadActiveView();

    // 4. Open photo in lightbox if requested in URL
    if (parsed.photo) {
      const { openPhotoModal } = await import("./components/lightbox.js");
      await openPhotoModal(parsed.photo, parsed.face);
    }

    updateUrl({ replace: true });
  } catch (err) {
    reportError(err, "Startup");
  }
}

init();
