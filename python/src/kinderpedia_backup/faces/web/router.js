// View switching: selected child + active mode -> which view loads.
import { emit, state } from "./state.js";
import { el } from "./util/dom.js";
import { loadBatchForSelected } from "./views/batch.js";
import { loadChildClusters } from "./views/childClusters.js";
import { loadClusters } from "./views/clusters.js";
import { loadGallery } from "./views/gallery.js";

const VIEW_IDS = { batch: "viewBatch", childclusters: "viewChildClusters", clusters: "viewClusters", gallery: "viewGallery" };

export function syncModeTabs() {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.getAttribute("data-mode") === state.mode));
  for (const [mode, id] of Object.entries(VIEW_IDS)) el(id).hidden = state.mode !== mode;
}

export async function loadActiveView() {
  if (state.mode === "batch") await loadBatchForSelected();
  else if (state.mode === "childclusters") await loadChildClusters();
  else if (state.mode === "clusters") await loadClusters();
  else await loadGallery();
}

export async function selectPerson(personId, mode = null) {
  state.selectedPersonId = personId;
  state.galleryOffset = 0;
  if (mode) state.mode = mode;
  else if (state.mode === "clusters") state.mode = "batch";
  emit("overview");
  syncModeTabs();
  if (state.mode !== "clusters") await loadActiveView();
}

export async function switchMode(newMode) {
  state.mode = newMode;
  syncModeTabs();
  await loadActiveView();
}
