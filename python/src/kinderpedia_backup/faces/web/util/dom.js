// DOM helpers. All markup goes through the `html` tagged template: interpolated values are escaped unless
// they are themselves `html` results, so escaping is structural (no ad-hoc string concatenation of data).

export const el = (id) => document.getElementById(id);

export function escapeHtml(value) {
  return (value == null ? "" : String(value))
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Markup that is already safe (only ever produced by `html`). */
export class SafeHtml {
  constructor(markup) {
    this.markup = markup;
  }

  toString() {
    return this.markup;
  }
}

function renderValue(value) {
  if (value == null || value === false || value === true) return "";
  if (value instanceof SafeHtml) return value.markup;
  if (Array.isArray(value)) return value.map(renderValue).join("");
  return escapeHtml(value);
}

/**
 * html`<b title="${t}">${name}</b>` -> SafeHtml. Values: strings/numbers are escaped, SafeHtml is inserted
 * as-is, arrays are concatenated (each item by the same rules), null/undefined/booleans render nothing.
 */
export function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += renderValue(values[i]) + strings[i + 1];
  return new SafeHtml(out);
}

/** The only sanctioned innerHTML write. */
export function setHtml(node, safe) {
  if (!(safe instanceof SafeHtml)) throw new TypeError("setHtml() expects an html`...` value");
  node.innerHTML = safe.markup;
}

/** First element of an html`...` fragment. */
export function toElement(safe) {
  const tpl = document.createElement("template");
  setHtml(tpl, safe);
  return tpl.content.firstElementChild;
}

/**
 * One listener on `container` (element or id) for `event`, dispatched to the closest `selector` match
 * inside it: handler(ev, matchedEl). Survives any re-render of the container's children.
 */
export function delegate(container, event, selector, handler) {
  const root = typeof container === "string" ? el(container) : container;
  root.addEventListener(event, (ev) => {
    const target = ev.target instanceof Element ? ev.target.closest(selector) : null;
    if (target && root.contains(target)) handler(ev, target);
  });
}

/** Bind `handler` once to the element with `id`: handler(ev, element). */
export function on(id, event, handler) {
  const node = el(id);
  node.addEventListener(event, (ev) => handler(ev, node));
}

/** Controls inside `scope` other than `exclude` (to disable while an action runs). */
export function siblingControls(scope, exclude, selector = "input, select, button") {
  return scope ? Array.from(scope.querySelectorAll(selector)).filter((n) => n !== exclude) : [];
}
