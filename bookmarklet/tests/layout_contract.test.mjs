// Golden layout-contract conformance for the Bookmarklet (mirrors python/tests/test_layout_contract.py).
// Run from the repo root: `node --test bookmarklet/tests/`
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const kp = require(join(here, "..", "kinderpedia-backup.js"));
const contract = JSON.parse(
  readFileSync(join(here, "..", "..", "python", "tests", "fixtures", "layout_contract.json"), "utf8"),
);

const toMap = (obj) => new Map(Object.entries(obj).map(([k, v]) => [Number(k), v]));
const fromMap = (m) => Object.fromEntries([...m.entries()].map(([k, v]) => [String(k), v]));

test("sanitizeName", () => {
  for (const c of contract.sanitize_name) {
    assert.equal(kp.sanitizeName(c.input, c.max_len ?? 60), c.expected, JSON.stringify(c.input));
  }
});

test("parseDatePrefix", () => {
  for (const c of contract.parse_date_prefix) {
    const { prefix, timestamp } = kp.parseDatePrefix(c.input);
    assert.equal(prefix, c.prefix, JSON.stringify(c.input));
    let expected = c.timestamp;
    if (expected === "local") {
      const [y, mo, d, h, mi, s] = c.local;
      expected = new Date(y, mo - 1, d, h, mi, s).getTime() / 1000;
    }
    if (expected === null) {
      assert.equal(timestamp, null, JSON.stringify(c.input));
    } else {
      assert.ok(timestamp !== null && Math.abs(timestamp - expected) < 1e-3, `${JSON.stringify(c.input)}: ${timestamp} != ${expected}`);
    }
  }
});

test("normalizeMatchText & stemSimilarity", () => {
  for (const c of contract.normalize_match_text) {
    assert.equal(kp.normalizeMatchText(c.input), c.expected, JSON.stringify(c.input));
  }
  for (const c of contract.stem_similarity) {
    assert.equal(kp.stemSimilarity(c.video, c.album), c.expected, JSON.stringify(c));
  }
});

test("resolveChildFolderNames", () => {
  for (const c of contract.resolve_child_folder_names) {
    const contexts = c.contexts.map((x) => ({
      child_id: x.child_id,
      kindergarten_id: x.kindergarten_id,
      child_name: x.name,
      first_name: x.first_name,
      school_name: x.kindergarten_name,
    }));
    assert.deepEqual(Object.fromEntries(kp.resolveChildFolderNames(contexts)), c.expected);
  }
});

test("assignAlbumFolderNames", () => {
  for (const c of contract.assign_album_folder_names) {
    const got = kp.assignAlbumFolderNames(c.albums, toMap(c.existing));
    assert.deepEqual(fromMap(got), c.expected, c.name);
  }
});

test("planVideoTargets", () => {
  for (const c of contract.plan_video_targets) {
    const folders = kp.assignAlbumFolderNames(c.albums, toMap(c.existing_albums));
    const existingVideos = new Map(
      Object.entries(c.existing_videos).map(([k, [folderRel, fileName]]) => [Number(k), { folderRel, fileName }]),
    );
    const got = kp.planVideoTargets(c.videos, c.albums, folders, existingVideos);
    const asPairs = Object.fromEntries([...got.entries()].map(([k, t]) => [String(k), [t.folderRel, t.fileName]]));
    assert.deepEqual(asPairs, c.expected, c.name);
  }
});

test("albumFolderFromLeaf", () => {
  for (const c of contract.album_folder_from_leaf) {
    assert.equal(kp.albumFolderFromLeaf(c.input), c.expected);
  }
});

test("folder map parse/serialize round-trip rejects unsafe entries", () => {
  const parsed = kp.parseFolderMap({
    version: 1,
    albums: { 1: "2025/06/2025-06-12 Party", 2: "../escape", 3: "/abs/path", x: "2025/01/a" },
    videos: { 9: { folder: "2025/06/2025-06-12 Party", file: "Song.mp4" }, 10: { folder: "a/../b", file: "x.mp4" } },
  });
  assert.deepEqual([...parsed.albums.keys()], [1]);
  assert.deepEqual([...parsed.videos.keys()], [9]);
  const ser = kp.serializeFolderMap(parsed.albums, parsed.videos);
  assert.deepEqual(ser, {
    version: 1,
    albums: { 1: "2025/06/2025-06-12 Party" },
    videos: { 9: { folder: "2025/06/2025-06-12 Party", file: "Song.mp4" } },
  });
});

test("incremental listing merge keeps fresh copies first", () => {
  const merged = kp.mergeIncrementalListing(
    [{ id: 3, v: "new" }, { id: 1, v: "fresh" }],
    [{ id: 1, v: "old" }, { id: 2, v: "old" }],
  );
  assert.deepEqual(merged.map((x) => `${x.id}:${x.v}`), ["3:new", "1:fresh", "2:old"]);
  assert.equal(
    kp.albumListingSignature({ updated_at: "2025-01-01T00:00:00+0000", count_images: "4" }),
    kp.albumListingSignature({ updated_at: "2025-01-01T00:00:00+0000", count_images: 4 }),
  );
});
