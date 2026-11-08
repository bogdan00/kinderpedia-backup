# Local Facial Recognition, All-Classroom Index & Active-Learning Web UI — Design Doc

**Status:** Implemented (`python/src/kinderpedia_backup/faces/`)  
**Scope:** Offline SCRFD + ArcFace ONNX inference (`onnxruntime` or OpenCV `cv2.dnn`), per-year classroom clustering, `<20ms` non-blocking active-learning Web UI with Server-Sent Events (SSE), unified lightbox navigation, manual occluded-child bounding-box tagging, year-windowed sibling guard, hard-negative lookalike suppression, same-album outfit cohesion, social peer co-occurrence, flat hardlink materialization, and provenance-rich Kinderpedia URL export. *(Photos only in v1.)*

---

## 1. Core Workflow: "This IS `<Child>`" → Batch Approve / Disprove / Tag

Rather than reviewing thousands of group photos one at a time, `kinderpedia-backup faces ui` centers on an **active-learning batch confirmation loop** designed to categorize **every child in the classroom** with sub-`5ms` click latency and steadily decreasing false guesses:

1. **Seed Any Child in 1 Click:**
   - Family children auto-seed from `<Child>/avatar.png` (plus optional `References/<Person>/`).
   - Recurring classmates surface automatically in the **Unnamed Kids & Years** discovery grid (clustered per classroom & year, ranked by photo count, with outlier-photo exclusion before cluster confirmation), or by clicking/drawing a face bounding box inside any classroom photo.
   - Any person can be renamed in place, assigned a role (`family`, `classmate`, `educator`, `ignore`), or merged into another profile via `PATCH /api/people/<id>`.
2. **Batch Proposal ("Approve All Except 1–2 Mistakes"):**
   - The local Python backend serves a diversified batch of **24 proposed matches** (`max_per_album = 2`) for the selected child, pre-selected as **Approved `✓`** when confidence is strong.
   - Every card supports four quick decisions:
     - **`✓` Approved** (`confirmed`): adds the photo to the child's gallery and contributes pose/year-diverse positive anchors.
     - **`✗` Disproved** (`rejected`): records a **hard negative** for that child so lookalike classmates are suppressed across the archive.
     - **`?` Not Sure** (`unsure`): safely skips ambiguous/blurry crops from the review queue without teaching positive or negative anchors.
     - **`↪` Reassign ("Actually another child…")**: immediately re-tags the face box to another existing or new child (`replace_person_id`), records a **face-scoped hard negative** (`source = 'reassigned'`) so the displaced child can still match a *different* face in the same group photo, and refreshes `cover_face_id` if displaced.
3. **Unified Full-Photo Lightbox & Manual Occluded-Child Tagging (`+ Tag Child` / `T`):**
   - Opening any photo from **Batch Trainer**, **Unnamed Clusters**, or **Confirmed Gallery** reuses a single unified navigation context (`state.modalNavContext`) with `←` / `→` keyboard stepping, a scrollable thumbnail strip showing live `✓` / `✗` / `?` / `↪` badges, and inline quick-decision pills.
   - Any auto-detected face tag can be cleared via **`✗ Remove Tag`** (`POST /api/faces/untag`), leaving the face uncategorized with `suppress_auto = 1` so background sweeps never slap a runner-up guess back onto it.
   - When a child is turned away, wearing a costume hood, or missed by SCRFD, clicking **`+ Tag Child`** (or pressing `T`) lets the user drag a bounding box directly on the photo and pick a name chip. Manual boxes store a zero vector with `is_manual = 1, is_anchor = 0`, reusing all `/media/face/<id>.jpg` crop and `ByChild/` export logic while remaining 100% invisible to ArcFace biometric matching and clustering.

---

## 2. System Architecture & Real-Time Non-Blocking Pipeline

```mermaid
flowchart TD
    subgraph Archive["Backup Root"]
        IMG["Child/Albums/Year/Month/YYYY-MM-DD Title/001_id.jpg"]
        META["Metadata/Child/Albums/YYYY-MM-DD Title.json"]
        AV["Child/avatar.png + optional References/Person/*.jpg"]
    end

    subgraph Stage1["Stage 1: Batch Scan (kinderpedia-backup faces scan)"]
        ONNX["SCRFD-10GF + 5-Pt Umeyama + ArcFace R50 ONNX\n(onnxruntime or OpenCV cv2.dnn backend, Python 3.10-3.15+)"]
        THUMBS["Sharded Face Crop Cache (Metadata/face_thumbs/00..ff/*.jpg)"]
        CLUST["Per-(Classroom, Year) Cosine Graph Clustering"]
    end

    subgraph Store["Persistent Provenance & Categorization Index"]
        DB[("Metadata/faces_index.sqlite (WAL)\nphotos | faces (float16[512], suppress_auto, is_manual)\npeople | clusters | assignments | person_assignment_counts")]
        RAM["In-Memory Face Matrix Cache\nfloat32[N, 512] (is_manual = 0) + Album/Photo Lookup Maps"]
        JSON["Metadata/categorization_index.json\n(Throttled background portable provenance snapshot)"]
    end

    subgraph WebApp["Stage 2: Local Web App (kinderpedia-backup faces ui)"]
        API["ThreadingHTTPServer (127.0.0.1:8765)\n<5ms Mutation Endpoints + Lazy PIL Crop Server"]
        BG["Debounced Background Worker (1.6s)\nBatched BLAS SGEMM Matcher + Diff SQLite Upsert"]
        SSE["SSE Event Stream (GET /api/events)\nrematch_scheduled / rematch_complete / overview_updated"]
        UI["Interactive Browser Studio (Vanilla HTML/CSS/JS)\nBatch Trainer + Clusters + Gallery + Unified Lightbox + Draw Box (T)"]
    end

    subgraph Exports["Outputs & Sharing"]
        FLAT["ByChild/Person/YYYY-MM-DD Title_001_id.jpg\n(Flat zero-space hardlinks for Photos / Cloud Drive)"]
        URLS["Person_urls.txt / .csv / .json\n(Provenance-rich Kinderpedia CDN URL manifests)"]
    end

    IMG --> ONNX
    META --> DB
    AV --> ONNX
    ONNX --> DB
    ONNX --> THUMBS
    DB --> CLUST --> DB
    DB <--> API
    DB <--> RAM
    RAM --> BG --> DB
    BG --> JSON
    BG --> SSE --> UI
    THUMBS --> API
    API <-->|"<5ms JSON POST/PATCH/GET"| UI
    API --> FLAT
    API --> URLS
```

### `<20ms` Click Responsiveness & Background Sync Design

1. **Immediate HTTP Mutations (`~2ms`):**
   - `POST /api/faces/tag`, `POST /api/faces/untag`, `POST /api/faces/manual`, `POST /api/batch/submit`, `POST /api/clusters/<id>/assign`, and `PATCH /api/people/<id>` commit only the point SQLite change, read the 24-row trigger-maintained `person_assignment_counts` summary table, call `schedule_rematch()`, and return immediately.
2. **In-Memory Embedding Matrix & Batched BLAS `SGEMM`:**
   - `FacesIndexDB.load_all_faces_matrix()` caches the unpacked read-only `float32[N, 512]` auto-detected face matrix (`WHERE is_manual = 0`) plus pre-indexed `rows_by_album`, `fc_by_photo`, and `rows_by_photo` maps in RAM (`~0.3ms` warm vs `~870ms` SQLite BLOB unpack), reading live `suppress_auto` flags on each sweep.
   - `recompute_active_matching()` stacks positive and hard-negative anchor matrices across all active people into `<=512`-column single-pass BLAS `SGEMM` batches (`embs @ stacked_pos.T`) and writes only changed model rows via an in-memory diff against `assignments`.
3. **Debounced Worker + Server-Sent Events (`GET /api/events`):**
   - Rapid clicks within `1.6s` coalesce into a single background rematch sweep; portable `Metadata/categorization_index.json` writes are throttled separately (`20s` debounce + flush on shutdown/materialize).
   - When a background sweep completes, the server broadcasts `event: rematch_complete` over `GET /api/events` (`EventSource`), allowing `app.js` to refresh sidebar counters, open photo modals (preserving any in-progress input focus), and candidate queues without page reloads or cursor jumps.

---

## 3. Child-Matching Intelligence Layers

1. **Dual-Runtime SCRFD-10GF + ArcFace ResNet-50 ONNX (`Python 3.10–3.15+`):**
   - Runs `det_10g.onnx` + 5-point Umeyama similarity alignment + `w600k_r50.onnx` via `onnxruntime` (`CUDA` / `CoreML` / `CPU`) with automatic fallback to OpenCV's C++ SIMD engine (`cv2.dnn.readNetFromONNX` from `opencv-python-headless`).
   - Quality gate (`det_score >= 0.55` and `min(width, height) >= 28 px`) filters out classroom wall posters, cartoons, and dolls.
2. **Multi-Epoch Diverse Pose Anchors:**
   - Selects up to `max_anchors_per_year = 8` pose-diverse confirmed exemplars per `(person, year)` (`cos_sim < 0.88` diversity preference) rather than averaging toddler and preschool faces into a single blurry centroid.
3. **Year-Windowed Sibling Guard (`[Y-1, Y+1]`):**
   - Siblings at the same biological age (e.g., Older Sibling at age 2 in `2022` vs. Younger Sibling at age 2 in `2026`) look very similar to ArcFace. Scoring a photo from year $Y$ prioritizes anchors within $[Y-1, Y+1]$, applying a cross-era penalty (`0.06`) to distant-year anchors whenever nearby-year anchors exist.
4. **Hard-Negative Lookalike Suppression (Photo-Level & Face-Scoped):**
   - Disproved or removed faces (`status = 'rejected'`) form a per-person negative bank (the most recent `512`). Unreviewed faces strongly matching it (`neg >= 0.58` and closer to the negative than to the child's anchors) are penalized so a lookalike classmate only needs to be rejected once or twice.
   - **Soft negative tier:** faces with `neg >= 0.45` that are about as close to a removed/rejected face as to the child's own anchors (`neg >= pos - 0.02`) get a graded push-down of up to `-0.10`, so removing a tag also nudges the wider ring of similar-looking faces.
   - Reassigned or untagged boxes record `source = 'reassigned'`, locking out only `(face_id, person_id)` rather than `(photo_id, person_id)` so the displaced child can immediately match their actual face box elsewhere in the same group photo.
5. **Same-Album Outfit / Burst Cohesion & Album-Diversified Batches:**
   - Confirming even a single photo of child $P$ inside album $A$ scores all other faces in $A$ directly against that local outfit/lighting anchor (never diluted by global per-year anchor caps) and auto-absorbs clear burst matches (`source = 'album_cohesion'`) into $P$'s gallery without cluttering the manual review queue.
5b. **Temporal Cohesion (±3 days):**
   - Confirmed faces from the same few days (same haircut, outfit, season, lighting) act as extra local anchors that are never diluted by per-year anchor caps. A face's score is lifted to its best near-day match (discounted `0.02 + 0.02 × |Δdays|`) plus up to `+0.04` when that match is `>= 0.50` (raised from 0.40: at 0.40 same-day photos of *other* children were pulled into a child's queue after every batch, making the queue grow after submitting mostly rejections). Applied before hard negatives so rejections still win.
   - Holdout check on the real archive (15% of confirmed tags hidden, then re-matched): correct auto-tags 95.9% → 97.6%, wrong auto-tags 123 → 63.
6. **Social Peer Co-Occurrence & Small-Photo Cross-Circle Suppression:**
   - Learns a weighted social graph from confirmed appearances across intimate classroom photos (`2–8` faces). When classmate $Q$ (who regularly co-occurs with $R$) appears in a small group photo, $R$ receives up to `+0.030` peer boost while siblings/children from disjoint classroom circles receive up to `-0.045` suppression — tapering smoothly to `0` on larger events (`>= 9` faces) where classes and siblings mingle.
7. **Global Photo-Level Bipartite Exclusivity & `ignore` Role Isolation:**
   - Enforces at most 1 face box per active person and 1 active person per face box in every photo. Profiles with `role = 'ignore'` (e.g., `_Ignored_Background`) never pollute cluster name suggestions, peer graphs, or coverage percentages.
8. **Zero-Vector Manual Box Biometric Isolation (`is_manual = 1`):**
   - User-drawn boxes for occluded/costumed children store `is_manual = 1, suppress_auto = 1` with a zero `float16[512]` vector and `is_anchor = 0`. Both `load_all_faces_matrix()` (`WHERE is_manual = 0`) and `load_anchors_and_negatives()` (`LEFT JOIN faces ... AND is_manual = 0`) exclude them from biometric scoring while keeping their `(photo_id, person_id)` confirmation active for exports, thumbnails, and social peer co-occurrence.

8b. **Learned Label Vote (k-nearest labelled faces) — Trust & Veto:**
   - Every human-confirmed face of every profile (including `ignore` background profiles) is a voter. For each still-open face, the 15 most similar labelled faces vote (weight `(sim - 0.20)^2`).
   - **Veto:** if ≥ 80% of the vote (and a neighbour with `sim >= 0.45`) says *someone else*, or the closest labelled neighbour is a background face (`sim >= 0.55`), the face is removed from the other children's queues.
   - **Trust:** if ≥ 90% of the vote agrees with the matcher's `auto_high` pick (and the face is not closer to one of that child's rejected faces), the match is stored as `source = 'model_trusted'` and skips manual review (counted with album-cohesion autos). Candidates with ≥ 95% agreement are promoted.
   - Neighbour lists are cached across rematches and updated incrementally (new labels merged with one small SGEMM), so the warm rematch costs ~3 s; the cold build (~25 s) runs once in the background at `faces ui` start.
   - Holdout on the real archive (20% of labels hidden): wrong auto-tags on hidden confirmed faces 138 → 33, background faces auto-tagged to a child 276 → 111, review queue 40.4k → 27.5k while trusted matches rose 9.1k → 15.8k (≈ 0.3% wrong among trusted).
8c. **Weak-Match Gate & Same-Album Absence:**
   - Borderline matches (below `auto_high + 0.05`) for a child with ≥ 25 labelled faces are dropped when that child gets < 30% of the labelled-neighbour vote while another identity tops it, so faces of non-eligible or unenrolled kids stop piling into the "least bad" child's queue.
   - In albums where a child is already confirmed, borderline faces whose best similarity to her confirmed faces *from that same album* is < 0.40 are dropped (genuine same-day matches score ~0.7–0.9; measured: 7% of individually confirmed matches fall below 0.40 vs. ~85% of the leftover queue).
   - Holdout: review queue 20.3k → 16.8k for a loss of ~0.3% of correct matches.
9. **Hard Eligibility: Birthdays & Class Memberships:**
   - Each photo gets a *latest possible capture date* (`photo_date_ord`): the earlier of `album_date` / `image_dateadd`, falling back to the last day of the `<Year>/<Month>` folder. It is cached per face as `date_ords` in the RAM matrix.
   - `people.birth_date`: a person is never matched, suggested, batch-approved, cluster-confirmed or manually tagged on a photo dated before it. Setting a birthday immediately deletes that person's existing non-rejected tags on earlier photos.
   - `school_classes` (`archive_child` feed + optional `[start_date, end_date]`) and `class_members`: a person with ≥ 1 membership is only auto-matched, suggested and cluster-confirmed on photos inside one of their classes, so children of unrelated classrooms (e.g. siblings' feeds) are never auto-tagged together. People without memberships stay unrestricted. Explicit single manual tags remain allowed. `POST /api/classes/auto` seeds one class per feed × school year (Sep 1 – Aug 31).
   - Applied as a final `-1.0` mask on the score matrix just before bipartite exclusivity.
10. **Per-Child Visual Clusters & Soft Pressure:**
    - `get_person_face_clusters()` groups everything attributed to a child (confirmed + auto + candidate + pressured faces) per year with density-leader grouping (`threshold` 0.42 / 0.50 / 0.58) followed by centroid merging (`threshold + 0.12`); undersized groups fall into a per-year loose bucket (~0.2 s for an 11k-face child).
    - Actions: **Exclude** (face-scoped `rejected` / `reassigned` hard negatives), **Boost** (`person_face_pressure.weight = +1`: up to `+0.06` on faces with `sim ≥ 0.40` to the boosted set, without confirming), **Penalize** (`weight = -1`: up to `-0.12` on faces with `sim ≥ 0.42`, without rejecting), **Clear**.

---

## 4. Persistent Provenance & Categorization Schema (`Metadata/faces_index.sqlite`)

- **`photos`**: `photo_id`, `rel_path`, `archive_child`, `year`, `month`, `album_leaf`, `kinderpedia_album_id`, `album_title`, `album_date`, `classroom_groups`, `kinderpedia_image_id`, `seq_in_album`, `image_dateadd`, `best_live_url`, `download_url`, `fullsize_url`, `thumb_url`, `width`, `height`, `file_size`, `mtime_ns`, `detector_id`, `face_count`, `scanned_at`.
  - *Smart URL selection (`best_live_url`):* Compares local JPEG `(width, height)` against any `/<W>x<H>/` prefix in `fullsize_url` so pre-2024 albums where `download_url` returns `404` automatically export the working `fullsize_url`, while 2024+ albums export full-resolution `download_url`.
- **`faces`**: `face_id`, `photo_id`, `face_idx`, `det_score`, `bbox_top`, `bbox_right`, `bbox_bottom`, `bbox_left`, `embedding_f16` (`1,024 B` IEEE-754 `float16[512]`), `cluster_id`, `suppress_auto` (`0`/`1`), `is_manual` (`0`/`1`).
- **`people`**: `person_id`, `display_name` (`COLLATE NOCASE`), `role` (`family` | `classmate` | `educator` | `ignore`), `archive_hint`, `cover_face_id`, `birth_date` (`YYYY-MM-DD` or `''`), `created_at`.
- **`school_classes`**: `class_id`, `name` (unique, `COLLATE NOCASE`), `archive_child` (feed or `''` = any), `start_date`, `end_date`, `created_at`. **`class_members`**: `(class_id, person_id)` PK.
- **`person_face_pressure`**: `(person_id, face_id)` PK, `weight` (`+1` boost / `-1` penalize), `updated_at`.
- **`reference_seeds`**: `seed_id`, `person_id`, `source_key`, `anchor_year`, `embedding_f16`, `created_at`.
- **`clusters`**: `cluster_id`, `archive_child`, `year`, `person_id`, `medoid_face_id`, `sample_face_ids_json`, `face_count`, `photo_count`, `suggested_person_id`, `suggested_similarity`.
- **`assignments`**: `(photo_id, person_id)` PK, `face_id`, `status` (`confirmed` | `auto_high` | `candidate` | `unsure` | `rejected`), `source` (`avatar_seed` | `reference_file` | `cluster_confirm` | `batch_approve` | `manual_tag` | `model_match` | `album_cohesion` | `reassigned`), `similarity`, `is_anchor`, `updated_at`.
- **`person_assignment_counts`**: Trigger-maintained per-person summary table (`confirmed_count`, `auto_high_count`, `album_cohesion_count`, `candidate_count`, `unsure_count`, `rejected_count`, `categorized_face_count`) kept atomically in sync via `AFTER INSERT / UPDATE / DELETE` SQLite triggers on `assignments`.

---

## 5. CLI Subcommands

```bash
# 1. Incremental ONNX face scan + Metadata provenance/URL join + per-year clustering
kinderpedia-backup faces scan --input ~/KinderpediaBackup

# 2. Launch local HTTP backend + interactive active-learning Web UI on http://127.0.0.1:8765
kinderpedia-backup faces ui --input ~/KinderpediaBackup --port 8765

# 3. Materialize flat zero-space hardlinks: ByChild/<Person>/<YYYY-MM-DD Title>_001_<id>.jpg
kinderpedia-backup faces materialize --input ~/KinderpediaBackup [--person "Alex"] [--include-candidates]

# 4. Export provenance-rich Kinderpedia CDN URL listing for any single child or classmate
kinderpedia-backup faces export-urls --input ~/KinderpediaBackup --person "Leo" --format txt -o leo_urls.txt
```
