// Global keyboard shortcuts: Esc (close / cancel innermost), T (draw box), B/N (boost / penalize in
// child-cluster lightbox), ←/→ (lightbox navigation), Enter (submit batch).
import { closePhotoModal } from "./components/lightbox.js";
import { applyModalChildClusterAction, stepModal } from "./components/lightboxNav.js";
import { cancelManualDraft, setManualDrawMode } from "./components/manualBox.js";
import { closeReassignModal } from "./components/reassignModal.js";
import { closeRenameModal } from "./components/roster.js";
import { state } from "./state.js";
import { el } from "./util/dom.js";
import { submitCurrentBatch } from "./views/batch.js";
import { closeClassesModal } from "./views/classes.js";

const open = (id) => !el(id).hidden;

function onEscape() {
  if (open("renamePersonModal")) return closeRenameModal();
  if (open("reassignModal")) return closeReassignModal();
  if (open("classesModal")) return closeClassesModal();
  if (!open("photoModal")) return undefined;
  if (state.manualDraftBox) return cancelManualDraft();
  if (state.manualDrawMode) return setManualDrawMode(false);
  return closePhotoModal();
}

export function bindKeyboard() {
  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      onEscape();
      return;
    }
    const tag = (document.activeElement && document.activeElement.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    const plain = !ev.metaKey && !ev.ctrlKey && !ev.altKey;
    const lightboxOnly = open("photoModal") && !open("reassignModal") && !open("renamePersonModal");
    const key = ev.key.toLowerCase();
    if (lightboxOnly && plain && key === "t") {
      ev.preventDefault();
      setManualDrawMode(!state.manualDrawMode);
      return;
    }
    const navCtx = open("photoModal") ? state.modalNavContext : null;
    if (navCtx) {
      if (navCtx.type === "gallery" && navCtx.ccKey && lightboxOnly && !state.manualDrawMode && plain && (key === "b" || key === "n")) {
        ev.preventDefault();
        const act = key === "b" ? "boost" : "penalize";
        applyModalChildClusterAction(act, el("modalBatchQuickRow").querySelector(`[data-ccact="${act}"]`));
        return;
      }
      if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
        ev.preventDefault();
        stepModal(ev.key === "ArrowLeft" ? -1 : 1);
        return;
      }
    }
    const noDialogs = !open("photoModal") && !open("reassignModal") && !open("renamePersonModal") && !open("classesModal");
    if (ev.key === "Enter" && state.mode === "batch" && noDialogs && state.batchCards.length > 0) {
      ev.preventDefault();
      submitCurrentBatch();
    }
  });
}
