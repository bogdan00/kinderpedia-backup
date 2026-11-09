// View switching & URL routing: selected child + active mode + open album + lightbox -> URL <-> UI sync.
import { emit, state } from "./state.js";
import { el } from "./util/dom.js";
import { loadAlbums, openAlbumByKey } from "./views/albums.js";
import { loadBatchForSelected } from "./views/batch.js";
import { loadChildClusters } from "./views/childClusters.js";
import { loadClusters } from "./views/clusters.js";
import { loadGallery } from "./views/gallery.js";

const VIEW_IDS = {
  batch: "viewBatch",
  childclusters: "viewChildClusters",
  clusters: "viewClusters",
  gallery: "viewGallery",
  albums: "viewAlbums",
};

let isNavigating = false;

export function syncModeTabs() {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.getAttribute("data-mode") === state.mode));
  for (const [mode, id] of Object.entries(VIEW_IDS)) el(id).hidden = state.mode !== mode;
}

export function parseUrl() {
  const result = {
    mode: null,
    person: null,
    album: null,
    child: null,
    year: null,
    filterPerson: null,
    photo: null,
    face: null,
  };

  // 1. Pathname fallback if direct hit on server (e.g. /albums or /batch)
  const pathname = window.location.pathname.replace(/^\/index\.html/, "").replace(/^\//, "");
  if (pathname && VIEW_IDS[pathname]) {
    result.mode = pathname;
  }

  // 2. Hash routing (e.g. #/albums?album=... or #/batch?person=1 or #albums)
  const hash = window.location.hash.replace(/^#\/?/, "");
  if (hash) {
    const [hashPath, hashQuery] = hash.split("?");
    const cleanPath = (hashPath || "").replace(/^\//, "").trim();
    if (cleanPath) {
      if (cleanPath.startsWith("albums/")) {
        result.mode = "albums";
        result.album = decodeURIComponent(cleanPath.slice("albums/".length));
      } else if (VIEW_IDS[cleanPath]) {
        result.mode = cleanPath;
      }
    }
    if (hashQuery) {
      const sp = new URLSearchParams(hashQuery);
      if (sp.has("mode") && VIEW_IDS[sp.get("mode")]) result.mode = sp.get("mode");
      if (sp.has("person")) result.person = Number(sp.get("person"));
      if (sp.has("album")) result.album = sp.get("album");
      if (sp.has("child")) result.child = sp.get("child");
      if (sp.has("year")) result.year = sp.get("year");
      if (sp.has("filterPerson")) {
        const val = sp.get("filterPerson");
        result.filterPerson = val === "unassigned" ? "unassigned" : (Number(val) || null);
      }
      if (sp.has("photo")) result.photo = Number(sp.get("photo"));
      if (sp.has("face")) result.face = Number(sp.get("face"));
    }
  }

  // 3. Search query fallback (e.g. ?mode=albums&album=...)
  if (window.location.search) {
    const sp = new URLSearchParams(window.location.search);
    if (!result.mode && sp.has("mode") && VIEW_IDS[sp.get("mode")]) result.mode = sp.get("mode");
    if (!result.person && sp.has("person")) result.person = Number(sp.get("person"));
    if (!result.album && sp.has("album")) result.album = sp.get("album");
    if (!result.child && sp.has("child")) result.child = sp.get("child");
    if (!result.year && sp.has("year")) result.year = sp.get("year");
    if (sp.has("filterPerson")) {
      const val = sp.get("filterPerson");
      result.filterPerson = val === "unassigned" ? "unassigned" : (Number(val) || null);
    }
    if (!result.photo && sp.has("photo")) result.photo = Number(sp.get("photo"));
    if (!result.face && sp.has("face")) result.face = Number(sp.get("face"));
  }

  return result;
}

export function buildUrl() {
  const mode = state.mode || "batch";
  const params = new URLSearchParams();

  if (mode === "albums") {
    if (state.selectedAlbumKey) params.set("album", state.selectedAlbumKey);
    if (state.albumFilterChild) params.set("child", state.albumFilterChild);
    if (state.albumFilterYear) params.set("year", state.albumFilterYear);
    if (state.albumPersonFilter !== null && state.albumPersonFilter !== undefined) {
      params.set("filterPerson", String(state.albumPersonFilter));
    }
  } else if (mode !== "clusters" && state.selectedPersonId) {
    params.set("person", String(state.selectedPersonId));
  }

  if (state.modalPhoto && state.modalPhoto.photo_id) {
    params.set("photo", String(state.modalPhoto.photo_id));
    if (state.modalHighlightFaceId) {
      params.set("face", String(state.modalHighlightFaceId));
    }
  }

  const qs = params.toString();
  return `#/${mode}${qs ? `?${qs}` : ""}`;
}

export function updateUrl({ replace = false } = {}) {
  if (isNavigating) return;
  const newHash = buildUrl();
  if (window.location.hash === newHash) return;

  if (replace) {
    history.replaceState(null, "", newHash);
  } else {
    history.pushState(null, "", newHash);
  }
}

export async function loadActiveView() {
  if (state.mode === "batch") await loadBatchForSelected();
  else if (state.mode === "childclusters") await loadChildClusters();
  else if (state.mode === "clusters") await loadClusters();
  else if (state.mode === "albums") {
    if (state.selectedAlbumKey) {
      await openAlbumByKey(state.selectedAlbumKey, state.albumPersonFilter !== null);
    } else {
      await loadAlbums();
    }
  } else await loadGallery();
}

export async function selectPerson(personId, mode = null) {
  state.selectedPersonId = personId;
  state.galleryOffset = 0;
  if (mode) state.mode = mode;
  else if (state.mode === "clusters") state.mode = "batch";
  emit("overview");
  syncModeTabs();
  updateUrl();
  if (state.mode !== "clusters") await loadActiveView();
}

export async function switchMode(newMode) {
  state.mode = newMode;
  syncModeTabs();
  updateUrl();
  await loadActiveView();
}

export async function handleUrlChange() {
  if (isNavigating) return;
  isNavigating = true;
  try {
    const parsed = parseUrl();
    let modeChanged = false;
    let personChanged = false;
    let albumChanged = false;

    if (parsed.mode && parsed.mode !== state.mode) {
      state.mode = parsed.mode;
      modeChanged = true;
    }
    if (parsed.person && parsed.person !== state.selectedPersonId) {
      state.selectedPersonId = parsed.person;
      personChanged = true;
    }
    if (parsed.child !== null) state.albumFilterChild = parsed.child || "";
    if (parsed.year !== null) state.albumFilterYear = parsed.year || "";
    let filterPersonChanged = false;
    if (parsed.filterPerson !== null) {
      if (state.albumPersonFilter !== parsed.filterPerson) {
        state.albumPersonFilter = parsed.filterPerson;
        filterPersonChanged = true;
      }
    }

    if (state.mode === "albums") {
      const targetAlbum = parsed.album || null;
      if (targetAlbum !== state.selectedAlbumKey) {
        state.selectedAlbumKey = targetAlbum;
        albumChanged = true;
      }
    }

    syncModeTabs();
    emit("overview");

    if (modeChanged || personChanged || albumChanged || filterPersonChanged) {
      await loadActiveView();
    }

    // Lightbox modal sync
    if (parsed.photo) {
      const curPid = state.modalPhoto ? state.modalPhoto.photo_id : null;
      if (curPid !== parsed.photo) {
        const { openPhotoModal } = await import("./components/lightbox.js");
        await openPhotoModal(parsed.photo, parsed.face, true);
      }
    } else if (state.modalPhoto) {
      const { closePhotoModal } = await import("./components/lightbox.js");
      await closePhotoModal();
    }
  } finally {
    isNavigating = false;
  }
}
