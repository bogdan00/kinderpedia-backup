// Toast notifications + error reporting.
import { el } from "../util/dom.js";

let toastTimer = null;

export function showToast(msg, kind = "info") {
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

export function reportError(err, context = "") {
  const raw = (err && err.message) || String(err || "Unknown error");
  showToast(`${context ? `${context}: ` : ""}${raw}`, "error");
}

export function bindToasts() {
  const node = el("toast");
  if (node) node.addEventListener("click", () => (node.hidden = true));
}
