// runBusy(): spinner / disabled state around an async UI action, errors reported as toasts.
import { reportError } from "../components/toasts.js";

export async function runBusy(controlEl, actionFn, opts = {}) {
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

  let origNodes = null;
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
      origNodes = Array.from(controlEl.childNodes);
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
      if (busyText !== null && origNodes !== null && !keepUpdatedLabel && controlEl.isConnected) {
        controlEl.replaceChildren(...origNodes);
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
