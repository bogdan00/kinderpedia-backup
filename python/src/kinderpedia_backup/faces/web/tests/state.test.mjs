// Pure store reducers (node --test, no npm): default_decision -> card state, local decisions, labels
// persisted from the lightbox, rename / delete fan-out, submit lists, gallery clamp, strip window.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  applyLabelToCards,
  batchCounts,
  batchSubmitLists,
  cardFromServer,
  clampGalleryOffset,
  decideRange,
  dropPersonFromCards,
  emit,
  findPersonByName,
  getSelectedPerson,
  renamePersonInCards,
  resolvePersonName,
  select,
  state,
  stripWindow,
  subscribe,
  toggleAll,
  withDecision,
  wrapIndex,
} from "../state.js";

const card = (over = {}) => cardFromServer({ photo_id: 1, face_id: 11, similarity: 0.6, default_decision: "approved", ...over });

test("default_decision maps onto the initial card state", () => {
  for (const d of ["approved", "unsure", "disproved"]) {
    const c = cardFromServer({ photo_id: 1, face_id: 2, default_decision: d });
    assert.equal(c.decision, d);
    assert.deepEqual([c.targetName, c.targetPersonId, c.alreadySavedInModal], ["", null, false]);
  }
  assert.equal(cardFromServer({ photo_id: 1, face_id: 2 }).decision, "disproved"); // missing -> safe default
  assert.equal(cardFromServer({ photo_id: 1, face_id: 2, default_decision: "bogus" }).decision, "disproved");
});

test("withDecision clears reassignment target and the saved flag, without mutating", () => {
  const c = { ...card(), decision: "reassigned", targetPersonId: 7, targetName: "Maya", alreadySavedInModal: true };
  const next = withDecision(c, "unsure");
  assert.deepEqual([next.decision, next.targetPersonId, next.targetName, next.alreadySavedInModal], ["unsure", null, "", false]);
  assert.equal(c.decision, "reassigned");
});

test("decideRange covers both directions and reports changed indices", () => {
  const cards = [card(), card({ face_id: 12 }), card({ face_id: 13 }), card({ face_id: 14 })];
  const { cards: next, changed } = decideRange(cards, 3, 1, "disproved");
  assert.deepEqual(changed, [1, 2, 3]);
  assert.deepEqual(next.map((c) => c.decision), ["approved", "disproved", "disproved", "disproved"]);
  assert.equal(next[0], cards[0]); // untouched cards keep identity
});

test("toggleAll skips cards already saved from the lightbox", () => {
  const saved = { ...card({ face_id: 99 }), decision: "reassigned", targetPersonId: 5, alreadySavedInModal: true };
  const cards = [card(), card({ face_id: 12, default_decision: "disproved" }), saved];
  const off = toggleAll(cards, "approve");
  assert.deepEqual(off.map((c) => c.decision), ["disproved", "disproved", "reassigned"]);
  assert.deepEqual(toggleAll(off, "approve").map((c) => c.decision), ["approved", "approved", "reassigned"]);
  const unsure = toggleAll(cards, "unsure");
  assert.deepEqual(unsure.map((c) => c.decision), ["unsure", "unsure", "reassigned"]);
  assert.deepEqual(toggleAll(unsure, "unsure").map((c) => c.decision), ["approved", "approved", "reassigned"]);
  assert.equal(toggleAll([saved], "approve")[0], saved);
});

test("applyLabelToCards: lightbox tag / untag / unsure / reassign land on matching cards", () => {
  const cards = [card(), card({ photo_id: 2, face_id: 21 }), card({ photo_id: 3, face_id: null })];
  const sel = 1;
  const mine = applyLabelToCards(cards, { photoId: 1, faceId: 11, personId: sel, status: "confirmed" }, sel);
  assert.deepEqual(mine.changed, [0]);
  assert.deepEqual([mine.cards[0].decision, mine.cards[0].alreadySavedInModal], ["approved", true]);

  const other = applyLabelToCards(cards, { photoId: 2, faceId: 21, personId: 8, personName: "Maya", status: "confirmed" }, sel);
  assert.deepEqual([other.cards[1].decision, other.cards[1].targetPersonId, other.cards[1].targetName], ["reassigned", 8, "Maya"]);

  const untag = applyLabelToCards(cards, { photoId: 1, faceId: 11, personId: null, status: "rejected" }, sel);
  assert.equal(untag.cards[0].decision, "disproved");
  assert.equal(applyLabelToCards(cards, { photoId: 1, faceId: 11, status: "unsure" }, sel).cards[0].decision, "unsure");

  // face-less card matches by photo and adopts the new face id:
  const manual = applyLabelToCards(cards, { photoId: 3, faceId: 31, personId: sel, status: "confirmed" }, sel);
  assert.deepEqual([manual.changed, manual.cards[2].face_id], [[2], 31]);

  const none = applyLabelToCards(cards, { photoId: 9, faceId: 99, personId: sel, status: "confirmed" }, sel);
  assert.deepEqual(none.changed, []);
  assert.equal(none.cards[0], cards[0]);
});

test("delete / rename of a reassignment target", () => {
  const re = { ...card(), decision: "reassigned", targetPersonId: 4, targetName: "Leo" };
  const saved = { ...re, alreadySavedInModal: true };
  const [a, b] = dropPersonFromCards([re, saved], 4);
  assert.equal(a.decision, "unsure");
  assert.equal(b, saved);
  const byName = { ...card(), decision: "reassigned", targetPersonId: null, targetName: "leo" };
  const renamed = renamePersonInCards([re, byName, card()], 4, "Leo", "Theo");
  assert.deepEqual(renamed.slice(0, 2).map((c) => [c.targetPersonId, c.targetName]), [[4, "Theo"], [4, "Theo"]]);
  assert.equal(renamed[2].decision, "approved");
});

test("batch counts and submit lists", () => {
  const cards = [
    card(),
    withDecision(card({ face_id: 12 }), "disproved"),
    withDecision(card({ face_id: 13 }), "unsure"),
    { ...card({ face_id: 14 }), decision: "reassigned", targetPersonId: 7 },
    { ...card({ face_id: 15 }), decision: "reassigned", targetName: "Nina" },
    { ...card({ face_id: 16 }), decision: "reassigned" }, // no target -> hard negative
    { ...card({ face_id: 17 }), alreadySavedInModal: true },
  ];
  assert.deepEqual(batchCounts(cards), { approved: 2, disproved: 1, unsure: 1, reassigned: 3 });
  const lists = batchSubmitLists(cards, (_pid, name) => name.toUpperCase());
  assert.deepEqual(lists.approved, [{ photo_id: 1, face_id: 11 }]);
  assert.deepEqual(lists.unsure, [{ photo_id: 1, face_id: 13 }]);
  assert.deepEqual(lists.disproved.map((r) => r.face_id), [12, 16]);
  assert.deepEqual(lists.reassigned, [
    { photo_id: 1, face_id: 14, target_person_id: 7 },
    { photo_id: 1, face_id: 15, new_target_name: "NINA" },
  ]);
});

test("gallery offset clamp after removals", () => {
  // page 2 (offset 96) emptied, 96 left -> back to page 1
  assert.equal(clampGalleryOffset({ offset: 96, limit: 96, total: 96, itemsLen: 0 }), 0);
  // 3 pages, last emptied -> last non-empty page start
  assert.equal(clampGalleryOffset({ offset: 288, limit: 96, total: 200, itemsLen: 0 }), 192);
  assert.equal(clampGalleryOffset({ offset: 96, limit: 96, total: 97, itemsLen: 1 }), null); // has items
  assert.equal(clampGalleryOffset({ offset: 0, limit: 96, total: 5, itemsLen: 0 }), null); // first page
  assert.equal(clampGalleryOffset({ offset: 96, limit: 96, total: 0, itemsLen: 0 }), null); // nothing left at all
  assert.equal(clampGalleryOffset({ offset: 10, limit: 0, total: 3, itemsLen: 0 }), 2); // degenerate limit
});

test("lightbox strip windowing and wrap-around", () => {
  assert.deepEqual(stripWindow(0, 5, 30), { start: 0, end: 5 });
  assert.deepEqual(stripWindow(100, 500, 30), { start: 70, end: 131 });
  assert.deepEqual(stripWindow(499, 500, 30), { start: 469, end: 500 });
  assert.deepEqual(stripWindow(3, 0, 30), { start: 0, end: 0 });
  assert.deepEqual([wrapIndex(-1, 5), wrapIndex(5, 5), wrapIndex(7, 5), wrapIndex(3, 0)], [4, 0, 2, 0]);
});

test("selectors and pub/sub", () => {
  state.overview = { people: [{ person_id: 1, display_name: "Alex" }, { person_id: 2, display_name: "Sofia" }] };
  state.selectedPersonId = 2;
  assert.equal(getSelectedPerson().display_name, "Sofia");
  assert.equal(resolvePersonName(1, "x"), "Alex");
  assert.equal(resolvePersonName(9, "fallback"), "fallback");
  assert.equal(findPersonByName("  sofia ").person_id, 2);
  assert.equal(findPersonByName("Sofia", 2), null);
  assert.equal(select((s) => s.selectedPersonId), 2);
  const got = [];
  const off = subscribe("t", (p) => got.push(p));
  emit("t", 1);
  off();
  emit("t", 2);
  assert.deepEqual(got, [1]);
});
