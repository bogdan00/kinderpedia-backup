"""Child-matching intelligence: per-year clustering, year-windowed sibling guard, hard negatives, social peer graph & bipartite assignment."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .index_db import FacesIndexDB
from .models import AssignmentSource, AssignmentStatus


def photo_intimacy_factor(face_count: int) -> float:
    """
    Scale social co-occurrence signals by how intimate the group photo is:
    - 1 face: 0.0 (solo portrait has no companions)
    - 2 faces: 1.00 (pair photo — strongest classroom peer signal)
    - 3..8 faces: linearly tapers from 0.86 down to 0.14
    - >= 9 faces: 0.0 (large playground / school-wide events where siblings & classes mingle)
    """
    if face_count <= 1 or face_count >= 9:
        return 0.0
    return max(0.0, min(1.0, (9.0 - float(face_count)) / 7.0))


def _build_social_context(
    people_info: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Learn the classroom social graph strictly from human decisions (`confirmed` & photo-level `rejected`):
    - `peer_bond[p, q]` in `[0, 1]`: 1-hop small-photo co-occurrence + 2-hop shared-classmate bridge.
    - `outsider_mat[p, q]` in `[0, 1]`: degree to which witness `q`'s presence in a small photo makes
      candidate `p` unlikely (e.g. `q` belongs to sibling `r`'s classroom circle while `p` belongs to
      a disjoint classroom cohort or separate social clique).
    """
    n_people = len(people_info)
    pid_to_idx: dict[int, int] = {}
    roles: list[str] = []
    arch_counts: list[Counter[str]] = []
    cohort_counts: list[Counter[tuple[str, int]]] = []

    for idx, p in enumerate(people_info):
        pid = int(p["person_id"])
        pid_to_idx[pid] = idx
        roles.append(str(p.get("role") or "classmate"))
        ac: Counter[str] = Counter()
        hint = str(p.get("archive_hint") or "").strip()
        if hint:
            ac[hint] += 3
        arch_counts.append(ac)
        cohort_counts.append(Counter())

    confirmed_count = np.zeros((n_people,), dtype=np.int32)
    confirmed_by_photo: dict[int, list[tuple[int | None, int]]] = defaultdict(list)
    rejected_photo_pids: dict[int, set[int]] = defaultdict(set)
    photo_fc_hint: dict[int, int] = {}

    for h in human_rows:
        per_id = int(h["person_id"])
        if per_id not in pid_to_idx:
            continue
        u = pid_to_idx[per_id]
        p_id = int(h["photo_id"])
        st = str(h["status"])
        fc = int(h.get("face_count") or 0)
        if fc > 0:
            photo_fc_hint[p_id] = fc

        if st == "confirmed":
            confirmed_count[u] += 1
            f_id: int | None = h["face_id"]
            confirmed_by_photo[p_id].append((f_id, u))
            arch = str(h.get("archive_child") or "").strip()
            yr = int(h.get("year") or 0)
            if arch:
                arch_counts[u][arch] += 1
                if yr > 0:
                    cohort_counts[u][(arch, yr)] += 1
        elif st == "rejected":
            src = str(h.get("source") or "")
            if src not in ("reassigned", "manual_tag"):
                rejected_photo_pids[p_id].add(u)

    co_mat = np.zeros((n_people, n_people), dtype=np.float32)
    neg_co_mat = np.zeros((n_people, n_people), dtype=np.float32)

    for p_id, entries in confirmed_by_photo.items():
        unique_u = sorted({u for _, u in entries})
        fc = max(len(unique_u), photo_fc_hint.get(p_id, len(unique_u)))
        if len(unique_u) >= 2:
            w_edge = 1.0 if fc <= 4 else (0.55 if fc <= 8 else 0.12)
            for i_pos in range(len(unique_u)):
                u = unique_u[i_pos]
                for j_pos in range(i_pos + 1, len(unique_u)):
                    v = unique_u[j_pos]
                    co_mat[u, v] += w_edge
                    co_mat[v, u] += w_edge

        if 2 <= fc <= 6 and p_id in rejected_photo_pids:
            for rej_u in rejected_photo_pids[p_id]:
                for q_u in unique_u:
                    if rej_u != q_u:
                        neg_co_mat[rej_u, q_u] += 1.0

    peer_bond = np.zeros((n_people, n_people), dtype=np.float32)
    if n_people >= 2:
        direct_aff = np.minimum(1.0, co_mat / 2.0)
        for u in range(n_people):
            for v in range(u + 1, n_people):
                two_hop = float(np.sum(np.minimum(co_mat[u], co_mat[v])))
                bond = min(1.0, float(direct_aff[u, v]) + 0.5 * min(1.0, two_hop / 2.0))
                peer_bond[u, v] = bond
                peer_bond[v, u] = bond

    def _dist_overlap(c1: Counter[Any], c2: Counter[Any]) -> float | None:
        t1 = sum(c1.values())
        t2 = sum(c2.values())
        if t1 <= 0 or t2 <= 0:
            return None
        keys = set(c1.keys()) & set(c2.keys())
        return sum(min(c1[k] / t1, c2[k] / t2) for k in keys)

    outsider_mat = np.zeros((n_people, n_people), dtype=np.float32)
    for p_idx in range(n_people):
        p_max_co = float(np.max(co_mat[p_idx])) if n_people > 1 else 0.0
        for q_idx in range(n_people):
            if p_idx == q_idx:
                continue
            # Never penalize two family siblings directly against each other in 2-kid sibling portraits;
            # third-party classmates (e.g. Leo/Nina) provide the classroom disambiguation.
            if roles[p_idx] == "family" and roles[q_idx] == "family":
                continue
            if float(peer_bond[p_idx, q_idx]) >= 0.25:
                continue

            # Witness `q_idx` must have an established peer tie to at least one OTHER person `r != p_idx, q_idx`:
            other_co = co_mat[q_idx].copy()
            other_co[p_idx] = 0.0
            other_co[q_idx] = 0.0
            best_other_val = float(np.max(other_co)) if n_people > 2 else 0.0
            if best_other_val < 0.8:
                continue
            circle_strength = min(1.0, best_other_val / 1.8)

            # Evidence that candidate `p_idx` belongs to a separate circle/classroom from witness `q_idx`:
            sep_a = min(1.0, p_max_co / 1.8) if p_max_co >= 1.0 else 0.0

            sep_b = 0.0
            arch_ov = _dist_overlap(arch_counts[p_idx], arch_counts[q_idx])
            coh_ov = _dist_overlap(cohort_counts[p_idx], cohort_counts[q_idx])
            if (arch_ov is not None and arch_ov < 0.20) or (coh_ov is not None and coh_ov < 0.20):
                sep_b = (
                    0.90
                    if int(confirmed_count[p_idx]) == 0
                    else min(1.0, 0.75 + 0.08 * float(confirmed_count[p_idx]))
                )

            sep_neg = min(1.0, 0.65 * float(neg_co_mat[p_idx, q_idx]))
            sep = max(sep_a, sep_b, sep_neg)
            if sep > 0.0:
                outsider_mat[p_idx, q_idx] = (
                    circle_strength * (1.0 - float(peer_bond[p_idx, q_idx])) * sep
                )

    return {
        "pid_to_idx": pid_to_idx,
        "peer_bond": peer_bond,
        "outsider_mat": outsider_mat,
        "confirmed_by_photo": confirmed_by_photo,
    }


def _chunked_neighbor_counts(
    embs: NDArray[np.float32],
    *,
    sim_thresh: float,
    chunk_size: int = 2048,
) -> NDArray[np.int32]:
    """Count neighbors with cosine similarity >= `sim_thresh` without O(N^2) peak RAM."""
    n = embs.shape[0]
    counts = np.zeros((n,), dtype=np.int32)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = embs[start:end] @ embs.T
        counts[start:end] = np.sum(sims >= sim_thresh, axis=1) - 1
    return np.maximum(counts, 0)


def _select_diverse_indices(
    embs: NDArray[np.float32],
    scores: NDArray[np.float32],
    *,
    max_items: int = 6,
    max_pair_sim: float = 0.88,
) -> list[int]:
    """Pick up to `max_items` high-clarity, pose-diverse exemplars from `embs`."""
    if embs.shape[0] == 0:
        return []
    order = np.argsort(-scores)
    picked: list[int] = [int(order[0])]
    for idx_raw in order[1:]:
        if len(picked) >= max_items:
            break
        idx = int(idx_raw)
        sims = embs[picked] @ embs[idx]
        if float(np.max(sims)) < max_pair_sim:
            picked.append(idx)
    # Fill up to max_items if strict diversity left fewer than available:
    if len(picked) < min(max_items, embs.shape[0]):
        picked_set = set(picked)
        for idx_raw in order:
            idx = int(idx_raw)
            if idx not in picked_set:
                picked.append(idx)
                picked_set.add(idx)
                if len(picked) >= max_items:
                    break
    return picked


def rebuild_yearly_clusters(
    db: FacesIndexDB,
    *,
    min_det_score: float = 0.63,
    min_face_px: int = 40,
    core_sim: float = 0.58,
    assign_sim: float = 0.54,
    merge_centroid_sim: float = 0.60,
    min_cluster_photos: int = 4,
) -> int:
    """
    Discover recurring children per `(archive_child, year)` bucket using core-medoid density
    clustering (avoiding single-linkage lookalike chaining) and match hints against enrolled seeds.
    """
    mat = db.load_all_faces_matrix()
    face_ids: NDArray[np.int64] = mat["face_ids"]
    photo_ids: NDArray[np.int64] = mat["photo_ids"]
    det_scores: NDArray[np.float32] = mat["det_scores"]
    min_dims: NDArray[np.int32] = mat["min_dims"]
    archive_children: list[str] = mat["archive_children"]
    years: list[str] = mat["years"]
    embs: NDArray[np.float32] = mat["embeddings"]
    cluster_date_ords: NDArray[np.int32] | None = mat.get("date_ords")

    if face_ids.shape[0] == 0:
        return db.save_clusters([])

    dim = int(embs.shape[1])
    elig_rules = db.load_eligibility_rules()
    seed_info = db.load_anchors_and_negatives(dim=dim)
    social_ctx = _build_social_context(seed_info["people"], seed_info["human"])
    soc_pid_to_idx: dict[int, int] = social_ctx["pid_to_idx"]
    soc_peer_bond: NDArray[np.float32] = social_ctx["peer_bond"]
    soc_outsider: NDArray[np.float32] = social_ctx["outsider_mat"]
    soc_confirmed_by_photo: dict[int, list[tuple[int | None, int]]] = social_ctx["confirmed_by_photo"]
    uniq_cluster_p, cluster_p_counts = np.unique(photo_ids, return_counts=True)
    fc_by_photo_cluster: dict[int, int] = dict(
        zip(uniq_cluster_p.tolist(), cluster_p_counts.tolist(), strict=True)
    )

    active_pids: set[int] = {int(p["person_id"]) for p in seed_info["people"]}
    person_centroids: list[tuple[int, NDArray[np.float32]]] = []
    vecs_by_pid: dict[int, list[NDArray[np.float32]]] = defaultdict(list)
    for s in seed_info["seeds"]:
        s_pid = int(s["person_id"])
        if s_pid in active_pids:
            vecs_by_pid[s_pid].append(s["emb"])
    for h in seed_info["human"]:
        h_pid = int(h["person_id"])
        if h["status"] == "confirmed" and h["emb"] is not None and h_pid in active_pids:
            vecs_by_pid[h_pid].append(h["emb"])
    for pid, vlist in vecs_by_pid.items():
        stacked = np.stack(vlist, axis=0)
        person_centroids.append((pid, stacked))

    # Map already-confirmed faces to their person_id so clusters predominantly confirmed for X link automatically:
    confirmed_face_to_pid: dict[int, int] = {
        int(h["face_id"]): int(h["person_id"])
        for h in seed_info["human"]
        if h["status"] == "confirmed" and h["face_id"] is not None
    }

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(face_ids.shape[0]):
        if float(det_scores[i]) >= min_det_score and int(min_dims[i]) >= min_face_px:
            buckets[(archive_children[i], years[i])].append(i)

    cluster_specs: list[dict[str, Any]] = []

    for (arch_child, yr), global_indices in sorted(buckets.items()):
        if len(global_indices) < min_cluster_photos:
            continue
        idx_arr = np.asarray(global_indices, dtype=np.int64)
        sub_embs = embs[idx_arr]
        sub_scores = det_scores[idx_arr]
        sub_photos = photo_ids[idx_arr]

        density = _chunked_neighbor_counts(sub_embs, sim_thresh=core_sim)
        # Prioritize dense, sharp faces as cluster medoids:
        priority = density.astype(np.float32) + 0.25 * sub_scores
        order = np.argsort(-priority)

        assigned = np.full((idx_arr.shape[0],), -1, dtype=np.int32)
        centroids: list[NDArray[np.float32]] = []
        medoid_local_idx: list[int] = []

        # Similarity columns of upcoming medoid candidates are computed in blocks (one SGEMM per
        # block instead of one matrix-vector product per candidate on 20k+ face buckets).
        elig_order = [int(i) for i in order if int(density[int(i)]) >= (min_cluster_photos - 1)]
        elig_pos = {j: p for p, j in enumerate(elig_order)}
        block_cols: dict[int, int] = {}
        block_sims: NDArray[np.float32] = np.zeros((idx_arr.shape[0], 0), dtype=np.float32)
        next_block = 0

        for loc_i in elig_order:
            if assigned[loc_i] != -1:
                continue
            if loc_i not in block_cols:
                upcoming = [j for j in elig_order[next_block:] if assigned[j] == -1][:128]
                next_block = elig_pos[upcoming[-1]] + 1
                block_cols = {j: c for c, j in enumerate(upcoming)}
                block_sims = sub_embs @ sub_embs[upcoming].T
            sims = block_sims[:, block_cols[loc_i]]
            core_members = np.where((assigned == -1) & (sims >= core_sim))[0]
            if core_members.size < (min_cluster_photos - 1):
                continue

            # Enforce 1 face per photo inside the core:
            seen_p: set[int] = set()
            dedup_members: list[int] = []
            for m_idx in core_members[np.argsort(-sims[core_members])]:
                p_id = int(sub_photos[int(m_idx)])
                if p_id not in seen_p:
                    seen_p.add(p_id)
                    dedup_members.append(int(m_idx))
            if len(dedup_members) < min_cluster_photos:
                continue

            c_vec = sub_embs[dedup_members].mean(axis=0)
            c_vec = c_vec / max(float(np.linalg.norm(c_vec)), 1e-8)

            # Check if this core merges into an existing centroid of the same kid in this year:
            merged_into = -1
            if centroids:
                c_mat = np.stack(centroids, axis=0)
                c_sims = c_mat @ c_vec
                best_c = int(np.argmax(c_sims))
                if float(c_sims[best_c]) >= merge_centroid_sim:
                    merged_into = best_c

            target_c = merged_into if merged_into >= 0 else len(centroids)
            for m_idx in dedup_members:
                assigned[m_idx] = target_c

            if merged_into < 0:
                centroids.append(c_vec.astype(np.float32))
                medoid_local_idx.append(loc_i)
            else:
                all_m = np.where(assigned == target_c)[0]
                upd = sub_embs[all_m].mean(axis=0)
                centroids[target_c] = (upd / max(float(np.linalg.norm(upd)), 1e-8)).astype(np.float32)

        if not centroids:
            continue

        # Second pass: assign remaining unassigned crisp faces in this year to closest cluster with margin:
        unassigned_loc = np.where(assigned == -1)[0]
        if unassigned_loc.size > 0:
            c_mat = np.stack(centroids, axis=0)
            u_sims = sub_embs[unassigned_loc] @ c_mat.T
            best_c_arr = np.argmax(u_sims, axis=1)
            for pos, loc_idx_raw in enumerate(unassigned_loc):
                loc_idx = int(loc_idx_raw)
                best_c = int(best_c_arr[pos])
                best_val = float(u_sims[pos, best_c])
                if best_val < assign_sim:
                    continue
                if c_mat.shape[0] >= 2:
                    row_copy = u_sims[pos].copy()
                    row_copy[best_c] = -1.0
                    second_val = float(np.max(row_copy))
                    if best_val - second_val < 0.04:
                        continue
                assigned[loc_idx] = best_c

        for c_idx, c_vec in enumerate(centroids):
            members_loc = np.where(assigned == c_idx)[0]
            if members_loc.size == 0:
                continue
            unique_photos = {int(sub_photos[int(m)]) for m in members_loc}
            if len(unique_photos) < min_cluster_photos:
                continue

            member_global = idx_arr[members_loc]
            member_fids = [int(face_ids[int(g)]) for g in member_global]

            # Recompute true medoid closest to final cluster centroid:
            c_aff = sub_embs[members_loc] @ c_vec
            medoid_pos = int(np.argmax(c_aff))
            medoid_fid = int(face_ids[int(member_global[medoid_pos])])

            diverse_rel = _select_diverse_indices(
                sub_embs[members_loc],
                sub_scores[members_loc],
                max_items=6,
                max_pair_sim=0.88,
            )
            sample_fids = [int(face_ids[int(member_global[r])]) for r in diverse_rel]
            if medoid_fid not in sample_fids:
                sample_fids = [medoid_fid, *sample_fids[:5]]

            # Check if majority of faces in this cluster are already human-confirmed for someone:
            pid_votes: dict[int, int] = defaultdict(int)
            for fid in member_fids:
                if fid in confirmed_face_to_pid:
                    pid_votes[confirmed_face_to_pid[fid]] += 1
            assigned_pid: int | None = None
            if pid_votes:
                top_pid, top_votes = max(pid_votes.items(), key=lambda kv: kv[1])
                if top_votes >= max(2, int(0.5 * len(member_fids))):
                    assigned_pid = top_pid

            sugg_pid: int | None = None
            sugg_sim: float | None = None
            if assigned_pid is None and person_centroids:
                best_p_sim = -1.0
                best_p_id: int | None = None
                c_ord = (
                    int(np.median(cluster_date_ords[member_global]))
                    if cluster_date_ords is not None
                    else 0
                )
                for pid, anchor_mat in person_centroids:
                    if pid in elig_rules and c_ord > 0:
                        b_ord, wins = elig_rules[pid]
                        if b_ord > 0 and c_ord < b_ord:
                            continue
                        if wins and not any(
                            (not w_f or w_f == arch_child) and w_s <= c_ord <= w_e
                            for w_f, w_s, w_e in wins
                        ):
                            continue
                    sim_val = float(np.max(anchor_mat @ c_vec))
                    if pid in soc_pid_to_idx and unique_photos:
                        p_soc = soc_pid_to_idx[pid]
                        soc_deltas: list[float] = []
                        for up_id in unique_photos:
                            intim = photo_intimacy_factor(fc_by_photo_cluster.get(up_id, 0))
                            if intim <= 0.0 or up_id not in soc_confirmed_by_photo:
                                continue
                            b_max = 0.0
                            o_max = 0.0
                            for _, q_soc in soc_confirmed_by_photo[up_id]:
                                if q_soc == p_soc:
                                    continue
                                b_max = max(b_max, float(soc_peer_bond[p_soc, q_soc]))
                                o_max = max(o_max, float(soc_outsider[p_soc, q_soc]))
                            soc_deltas.append(intim * (0.040 * b_max - 0.075 * o_max))
                        if soc_deltas:
                            sim_val += float(np.mean(soc_deltas))
                    if sim_val > best_p_sim:
                        best_p_sim = sim_val
                        best_p_id = pid
                if best_p_id is not None and best_p_sim >= 0.42:
                    sugg_pid = best_p_id
                    sugg_sim = best_p_sim

            cluster_specs.append(
                {
                    "archive_child": arch_child,
                    "year": yr,
                    "person_id": assigned_pid,
                    "medoid_face_id": medoid_fid,
                    "sample_face_ids": sample_fids,
                    "face_ids": member_fids,
                    "photo_count": len(unique_photos),
                    "suggested_person_id": sugg_pid,
                    "suggested_similarity": sugg_sim,
                }
            )

    return db.save_clusters(cluster_specs)


def _max_sim_to_others(embs: NDArray[np.float32], chunk_size: int = 2048) -> NDArray[np.float32]:
    """For each row, the highest cosine similarity to any OTHER row (chunked, -1 for a single row)."""
    n = int(embs.shape[0])
    out = np.full((n,), -1.0, dtype=np.float32)
    if n < 2:
        return out
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = embs[start:end] @ embs.T
        sims[np.arange(end - start), np.arange(start, end)] = -1.0
        out[start:end] = np.max(sims, axis=1)
    return out


def _robust_anchor_score(
    sims: NDArray[np.float32], top2_weight: float, min_anchors: int = 4
) -> NDArray[np.float32]:
    """
    Blend the best and second-best anchor similarity so a single atypical (or mistaken) anchor
    cannot pull in a whole ring of weak look-alikes on its own. Falls back to the max when a
    person has fewer than `min_anchors` anchors (e.g. only an avatar per age): there the second
    anchor is usually a different age, not a second opinion.
    """
    if top2_weight <= 0.0 or sims.shape[1] < max(2, min_anchors):
        return np.asarray(np.max(sims, axis=1), dtype=np.float32)
    top2 = -np.partition(-sims, 1, axis=1)[:, :2]
    best = np.max(top2, axis=1)
    second = np.min(top2, axis=1)
    return np.asarray((1.0 - top2_weight) * best + top2_weight * second, dtype=np.float32)


def _cached_label_neighbors(
    db: FacesIndexDB,
    embs: NDArray[np.float32],
    face_ids: NDArray[np.int64],
    open_rows: NDArray[np.int64],
    bank_rows: NDArray[np.int64],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """
    Top-`k` labelled neighbours (matrix row indices + cosine sims) for each of `open_rows`.
    Neighbour lists are cached on the DB object across rematches and updated incrementally:
    newly labelled faces are merged in with one small SGEMM, unlabelled ones are dropped, and only
    rows never seen before are computed from scratch. Labels change a handful at a time while the
    user clicks, so a warm rematch costs milliseconds instead of a full 50k x 30k neighbour search.
    """
    n_faces = int(face_ids.shape[0])
    cache = db._knn_neighbor_cache
    bank_set = set(bank_rows.tolist())
    if (
        cache is None
        or cache.get("n_faces") != n_faces
        or cache.get("k") != k
        or cache.get("face_ids") is not face_ids
    ):
        cache = {
            "n_faces": n_faces,
            "k": k,
            "face_ids": face_ids,
            "nn_rows": np.full((n_faces, k), -1, dtype=np.int64),
            "nn_sims": np.full((n_faces, k), -1.0, dtype=np.float32),
            "has": np.zeros((n_faces,), dtype=bool),
            "bank": set(),
        }
    nn_rows_c: NDArray[np.int64] = cache["nn_rows"]
    nn_sims_c: NDArray[np.float32] = cache["nn_sims"]
    has: NDArray[np.bool_] = cache["has"]
    old_bank: set[int] = cache["bank"]
    added = np.asarray(sorted(bank_set - old_bank), dtype=np.int64)
    removed = old_bank - bank_set

    def _full(rows_sub: NDArray[np.int64]) -> None:
        # Small query chunks keep the (chunk x bank) similarity block cache-friendly (~120 MB at a
        # 30k bank); partitioning `s_q` directly avoids a second negated copy of the block.
        bank_e = embs[bank_rows]
        n_bank = int(bank_rows.size)
        for c0 in range(0, int(rows_sub.size), 1024):
            q = rows_sub[c0 : c0 + 1024]
            s_q = embs[q] @ bank_e.T
            idx = np.argpartition(s_q, n_bank - k, axis=1)[:, n_bank - k :]
            nn_rows_c[q] = bank_rows[idx]
            nn_sims_c[q] = np.take_along_axis(s_q, idx, axis=1)
            has[q] = True

    # A bulk confirm adds thousands of voters at once; merging them is still far cheaper than a cold
    # rebuild as long as they are a minority of the bank.
    if added.size > max(4000, int(bank_rows.size) // 2) or len(removed) > 4000:
        has[:] = False
    else:
        if removed:
            rem_arr = np.fromiter(removed, dtype=np.int64, count=len(removed))
            cached_rows = np.where(has)[0]
            if cached_rows.size:
                hit = np.isin(nn_rows_c[cached_rows], rem_arr)
                if np.any(hit):
                    sub_r = nn_rows_c[cached_rows]
                    sub_s = nn_sims_c[cached_rows]
                    sub_r[hit] = -1
                    sub_s[hit] = -1.0
                    nn_rows_c[cached_rows] = sub_r
                    nn_sims_c[cached_rows] = sub_s
                    # Rows that lost too many neighbours get recomputed from scratch below:
                    lost = hit.sum(axis=1) > (k // 3)
                    has[cached_rows[lost]] = False
        if added.size:
            upd = open_rows[has[open_rows]]
            if upd.size:
                add_e = embs[added]
                step = max(256, min(16384, (1 << 25) // max(1, int(added.size) + k)))
                for c0 in range(0, int(upd.size), step):
                    q = upd[c0 : c0 + step]
                    s_new = embs[q] @ add_e.T
                    cand_r = np.concatenate([nn_rows_c[q], np.broadcast_to(added, s_new.shape)], axis=1)
                    cand_s = np.concatenate([nn_sims_c[q], s_new.astype(np.float32)], axis=1)
                    idx = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
                    nn_rows_c[q] = np.take_along_axis(cand_r, idx, axis=1)
                    nn_sims_c[q] = np.take_along_axis(cand_s, idx, axis=1)
            # Rows cached earlier but not open now are stale for the added faces:
            stale = np.where(has)[0]
            if stale.size:
                stale_mask = np.ones((n_faces,), dtype=bool)
                stale_mask[open_rows] = False
                has[stale[stale_mask[stale]]] = False
    missing = open_rows[~has[open_rows]]
    if missing.size:
        _full(missing)
    cache["bank"] = bank_set
    db._knn_neighbor_cache = cache
    return nn_rows_c[open_rows], nn_sims_c[open_rows]


def recompute_active_matching(
    db: FacesIndexDB,
    *,
    auto_high_threshold: float = 0.45,
    candidate_threshold: float = 0.33,
    max_anchors_per_year: int = 8,
    anchor_top2_weight: float = 0.35,
    distant_year_sibling_penalty: float = 0.075,
    mid_year_drift_penalty: float = 0.030,
    hard_neg_min_sim: float = 0.58,
    hard_neg_weight: float = 0.38,
    soft_neg_min_sim: float = 0.45,
    soft_neg_margin: float = 0.02,
    soft_neg_weight: float = 0.10,
    max_negatives_per_person: int = 512,
    social_peer_boost: float = 0.040,
    social_cross_circle_penalty: float = 0.075,
    album_cohesion_boost: float = 0.065,
    album_cohesion_absorb_sim: float = 0.55,
    pressure_boost: float = 0.06,
    pressure_penalty: float = 0.12,
    temporal_window_days: int = 3,
    temporal_min_sim: float = 0.50,
    temporal_boost: float = 0.04,
    temporal_day_decay: float = 0.02,
    knn_enabled: bool = True,
    knn_k: int = 15,
    knn_veto_frac: float = 0.80,
    knn_trust_frac: float = 0.90,
    knn_promote_frac: float = 0.95,
    knn_min_sim: float = 0.45,
    knn_bg_min_sim: float = 0.55,
    knn_gate_frac: float = 0.30,
    knn_gate_min_labels: int = 25,
    album_absent_sim: float = 0.40,
    album_absent_strong_margin: float = 0.05,
    auto_band_knn_upper: float = 0.60,
    auto_band_knn_min_sim: float = 0.52,
    knn_promote_min_score: float | None = None,
    album_cohesion_require_knn: bool = False,
    anchor_outlier_min_sim: float = 0.0,
    anchor_outlier_min_faces: int = 8,
    anchor_det_penalty: float = 0.0,
    knn_bank_cap_per_person: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """
    Re-score all archive faces against all active people in <0.25s using:
    1. Multi-year pose-diverse positive anchors (`avatar.png` + `References/` + `confirmed` faces)
    2. Year-Windowed Sibling Guard (`[Y-1, Y+1]` full weight; penalizes distant toddler anchors when nearby anchors exist)
    3. Hard-Negative Lookalike Suppression (penalizes faces strongly matching explicitly `rejected` faces)
    4. Same-Album Outfit/Burst Cohesion (direct local-album anchor matching + automatic burst absorption so you
       don't have to re-confirm the same child in the same outfit across one album)
    5. Social Peer Co-Occurrence & Small-Photo Cross-Circle Suppression (boosts known classroom peers and
       suppresses siblings/other-classroom kids in intimate 2-8 face photos while tapering to 0 on >=9 face events)
    6. Global Photo-Level Bipartite 1-to-1 Exclusivity (1 face box = at most 1 person, 1 person = at most 1 face box)

    Auto-acceptance policy (calibrated on hidden-label replays of the user's own decisions):
    - `auto_high_threshold <= score < auto_band_knn_upper` is auto-accepted only when the labelled
      neighbourhood agrees (`knn_trust_frac`) and its nearest labelled face is close enough to be
      the same child (`auto_band_knn_min_sim`; a closed-set vote alone cannot spot an un-enrolled
      look-alike); otherwise the face goes to the review queue.
    - `album_cohesion` absorbs same-album faces only at local similarity >= `album_cohesion_absorb_sim`.
    - kNN promotion of a candidate needs score >= `knn_promote_min_score`
      (default `auto_high_threshold - 0.05`).
    - Anchor scores blend the best and second-best anchor (`anchor_top2_weight`) so one atypical or
      mislabelled anchor cannot pull in its whole neighbourhood.
    The anchor outlier guard, det-score anchor discount and kNN voter cap are available but off: they
    did not improve hidden-label accuracy on a real archive.

    `diagnostics`, when given, is filled with per-section timings and the final score matrix
    (offline evaluation / calibration only; it does not change any result).
    """
    timings: dict[str, float] = {}
    t_mark = [time.perf_counter()]

    def _tick(section: str) -> None:
        now = time.perf_counter()
        timings[section] = timings.get(section, 0.0) + (now - t_mark[0])
        t_mark[0] = now

    mat = db.load_all_faces_matrix()
    face_ids: NDArray[np.int64] = mat["face_ids"]
    photo_ids: NDArray[np.int64] = mat["photo_ids"]
    year_ints: NDArray[np.int32] = mat["year_ints"]
    suppressed_auto: NDArray[np.bool_] = (
        mat["suppressed_auto"]
        if "suppressed_auto" in mat
        else np.zeros((int(face_ids.shape[0]),), dtype=bool)
    )
    album_keys: list[str] = mat.get("album_keys") or [""] * int(face_ids.shape[0])
    embs: NDArray[np.float32] = mat["embeddings"]

    n_faces = int(face_ids.shape[0])
    if n_faces == 0:
        return db.replace_model_assignments([], set())

    dim = int(embs.shape[1])
    _tick("load_matrix")
    state = db.load_anchors_and_negatives(dim=dim)
    _tick("load_state")
    active_people = [int(p["person_id"]) for p in state["people"]]
    if not active_people:
        return db.replace_model_assignments([], set())

    # 1. Build positive anchors (vec, anchor_year) + hard negatives + per-album local anchors:
    seeds_by_pid: dict[int, list[tuple[NDArray[np.float32], int]]] = defaultdict(list)
    for s in state["seeds"]:
        seeds_by_pid[int(s["person_id"])].append((s["emb"], int(s["year"])))

    confirmed_by_pid_year: dict[tuple[int, int], list[tuple[int, float, NDArray[np.float32]]]] = defaultdict(list)
    fallback_confirmed_by_pid_year: dict[tuple[int, int], list[tuple[int, float, NDArray[np.float32]]]] = defaultdict(list)
    neg_by_pid: dict[int, list[NDArray[np.float32]]] = defaultdict(list)
    album_pos_by_pid: dict[int, dict[str, list[NDArray[np.float32]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    album_neg_by_pid: dict[int, dict[str, list[NDArray[np.float32]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    touched_album_keys: set[str] = set()
    temporal_pos_by_pid: dict[int, list[tuple[int, NDArray[np.float32]]]] = defaultdict(list)
    locked_photo_people: set[tuple[int, int]] = set()
    locked_faces: set[int] = set()
    rejected_photo_people: set[tuple[int, int]] = set()
    rejected_face_people: set[tuple[int, int]] = set()

    for h in state["human"]:
        p_id = int(h["photo_id"])
        per_id = int(h["person_id"])
        f_id: int | None = h["face_id"]
        st: str = h["status"]
        src: str = str(h.get("source") or "")
        akey: str = str(h.get("album_key") or "")
        emb: NDArray[np.float32] | None = h["emb"]

        if st == "confirmed":
            locked_photo_people.add((p_id, per_id))
            if f_id is not None:
                locked_faces.add(f_id)
            if emb is not None and f_id is not None:
                det_sc = float(h["det_score"])
                if det_sc >= 0.56:
                    confirmed_by_pid_year[(per_id, int(h["year"]))].append((f_id, det_sc, emb))
                elif det_sc >= 0.38:
                    fallback_confirmed_by_pid_year[(per_id, int(h["year"]))].append(
                        (f_id, det_sc, emb)
                    )
                if akey and akey != "/" and det_sc >= 0.35:
                    album_pos_by_pid[per_id][akey].append(emb)
                if det_sc >= 0.35:
                    temporal_pos_by_pid[per_id].append((f_id, emb))
                    touched_album_keys.add(akey)
        elif st == "rejected":
            if f_id is not None and src in ("reassigned", "manual_tag"):
                # Reassigning a specific face box teaches a hard-negative on that box while still
                # allowing `per_id` to match a DIFFERENT face box in the same group photo!
                rejected_face_people.add((f_id, per_id))
            else:
                rejected_photo_people.add((p_id, per_id))
            if emb is not None:
                neg_by_pid[per_id].append(emb)
                if akey and akey != "/":
                    album_neg_by_pid[per_id][akey].append(emb)
                    touched_album_keys.add(akey)
        elif st == "unsure":
            # "I'm not sure": skip from `per_id`'s batch queue WITHOUT adding positive or negative anchors,
            # while still letting other children claim `f_id` if they match strongly.
            rejected_photo_people.add((p_id, per_id))

    chosen_anchor_face_ids: set[int] = set()
    anchors_by_pid: dict[int, tuple[NDArray[np.float32], NDArray[np.int32]]] = {}
    anchor_det_pen_by_pid: dict[int, NDArray[np.float32]] = {}

    for pid in active_people:
        vecs: list[NDArray[np.float32]] = []
        yrs: list[int] = []
        dets: list[float] = []
        for s_emb, s_yr in seeds_by_pid.get(pid, []):
            vecs.append(s_emb)
            yrs.append(s_yr)
            dets.append(1.0)  # curated reference photos

        # Select up to `max_anchors_per_year` pose-diverse confirmed faces per year:
        person_years = sorted({yr for (p, yr) in confirmed_by_pid_year if p == pid})
        source_map = confirmed_by_pid_year
        if not person_years and not vecs:
            person_years = sorted({yr for (p, yr) in fallback_confirmed_by_pid_year if p == pid})
            source_map = fallback_confirmed_by_pid_year

        for yr in person_years:
            items = source_map[(pid, yr)]
            sub_embs = np.stack([it[2] for it in items], axis=0)
            sub_sc = np.asarray([it[1] for it in items], dtype=np.float32)
            cand_rel = np.arange(len(items))
            if anchor_outlier_min_sim > 0.0 and len(items) >= anchor_outlier_min_faces:
                # Label-consistency guard: a face that looks like none of the person's other
                # confirmed faces that year is more likely a mislabel than a pose; never anchor on it.
                consistent = _max_sim_to_others(sub_embs) >= anchor_outlier_min_sim
                if np.any(consistent):
                    cand_rel = np.where(consistent)[0]
            keep_rel = _select_diverse_indices(
                sub_embs[cand_rel],
                sub_sc[cand_rel],
                max_items=max_anchors_per_year,
                max_pair_sim=0.86,
            )
            for rel_k in keep_rel:
                f_id, det_v, f_emb = items[int(cand_rel[rel_k])]
                chosen_anchor_face_ids.add(f_id)
                vecs.append(f_emb)
                yrs.append(yr)
                dets.append(det_v)

        if vecs:
            anchors_by_pid[pid] = (
                np.stack(vecs, axis=0).astype(np.float32),
                np.asarray(yrs, dtype=np.int32),
            )
            if anchor_det_penalty > 0.0:
                # Blurry / small anchors are noisier: discount their similarities slightly.
                det_arr = np.asarray(dets, dtype=np.float32)
                anchor_det_pen_by_pid[pid] = (
                    anchor_det_penalty * np.clip((0.80 - det_arr) / 0.25, 0.0, 1.0)
                ).astype(np.float32)

    enrolled_pids = [pid for pid in active_people if pid in anchors_by_pid]
    _tick("anchor_select")
    if not enrolled_pids:
        return db.replace_model_assignments([], chosen_anchor_face_ids)

    # Pre-group face row indices ONLY for albums that have human decisions:
    cached_rows_by_album: dict[str, NDArray[np.int64]] | None = mat.get("rows_by_album")
    rows_by_album: dict[str, NDArray[np.int64]] = {}
    if touched_album_keys:
        if cached_rows_by_album is not None:
            rows_by_album = cached_rows_by_album
        else:
            tmp_alb_rows: dict[str, list[int]] = defaultdict(list)
            for r_i, akey in enumerate(album_keys):
                if akey in touched_album_keys:
                    tmp_alb_rows[akey].append(r_i)
            for akey, r_list in tmp_alb_rows.items():
                rows_by_album[akey] = np.asarray(r_list, dtype=np.int64)

    cohesion_candidates: set[tuple[int, int]] = set()

    # 2. Vectorized scoring across all N_faces for each enrolled person (batched SGEMM):
    eff_scores_by_col = np.full((n_faces, len(enrolled_pids)), -1.0, dtype=np.float32)
    unique_photo_years = [int(y) for y in np.unique(year_ints) if int(y) > 0]
    unk_mask = year_ints <= 0
    has_unk = bool(np.any(unk_mask))
    year_masks = {py: (year_ints == py) for py in unique_photo_years}

    # Group positive anchor matrices into <=512-column SGEMM batches to maximize BLAS throughput
    # while bounding peak temporary buffer size (<250 MB on 120k faces):
    pos_batches: list[list[tuple[int, int, int, int]]] = []
    cur_batch: list[tuple[int, int, int, int]] = []
    cur_cols = 0
    for col_idx, pid in enumerate(enrolled_pids):
        k_cols = int(anchors_by_pid[pid][0].shape[0])
        if cur_batch and (cur_cols + k_cols) > 512:
            pos_batches.append(cur_batch)
            cur_batch = []
            cur_cols = 0
        cur_batch.append((col_idx, pid, cur_cols, cur_cols + k_cols))
        cur_cols += k_cols
    if cur_batch:
        pos_batches.append(cur_batch)

    for pbatch in pos_batches:
        stacked_pos = np.vstack([anchors_by_pid[pid][0] for _, pid, _, _ in pbatch]).astype(
            np.float32, copy=False
        )
        batch_raw_sims = embs @ stacked_pos.T

        for col_idx, pid, c_start, c_end in pbatch:
            _, a_years = anchors_by_pid[pid]
            raw_sims = batch_raw_sims[:, c_start:c_end]
            det_pen = anchor_det_pen_by_pid.get(pid)
            if det_pen is not None and np.any(det_pen > 0.0):
                raw_sims = raw_sims - det_pen[np.newaxis, :]

            # Apply Year-Windowed Sibling Guard when dated anchors exist:
            dated_mask = a_years > 0
            if np.any(dated_mask) and unique_photo_years:
                pos_scores = np.empty((n_faces,), dtype=np.float32)
                if has_unk:
                    pos_scores[unk_mask] = _robust_anchor_score(raw_sims[unk_mask], anchor_top2_weight)

                for py in unique_photo_years:
                    ymask = year_masks[py]
                    diffs = np.abs(a_years - py)
                    has_nearby = bool(np.any(dated_mask & (diffs <= 2)))
                    penalties = np.zeros((a_years.shape[0],), dtype=np.float32)
                    if has_nearby:
                        penalties[~dated_mask] = 0.015
                        penalties[dated_mask & (diffs == 2)] = mid_year_drift_penalty
                        penalties[dated_mask & (diffs >= 3)] = distant_year_sibling_penalty
                    adj = raw_sims[ymask] - penalties[np.newaxis, :]
                    pos_scores[ymask] = _robust_anchor_score(adj, anchor_top2_weight)
            else:
                pos_scores = _robust_anchor_score(raw_sims, anchor_top2_weight)

            eff_scores_by_col[:, col_idx] = pos_scores
    _tick("anchor_score")

    # 2b. Temporal Cohesion: confirmed faces from the same few days (same haircut, outfit, season,
    # lighting) act as extra local anchors that are never diluted by per-year anchor caps, with a
    # time-decayed bonus. Applied before hard negatives so rejections still win.
    day_nums: NDArray[np.int32] | None = mat.get("day_nums")
    if (
        temporal_window_days > 0
        and day_nums is not None
        and int(day_nums.shape[0]) == n_faces
        and temporal_pos_by_pid
    ):
        dated_rows = np.where(day_nums >= 0)[0]
        dated_days = day_nums[dated_rows]
        order_d = np.argsort(dated_days, kind="stable")
        sorted_rows = dated_rows[order_d]
        sorted_days = dated_days[order_d]
        for col_idx, pid in enumerate(enrolled_pids):
            t_items = temporal_pos_by_pid.get(pid)
            if not t_items:
                continue
            t_fids = np.asarray([fid for fid, _ in t_items], dtype=np.int64)
            t_pos = np.searchsorted(face_ids, t_fids)
            t_ok = (t_pos < n_faces) & (face_ids[np.minimum(t_pos, n_faces - 1)] == t_fids)
            by_day: dict[int, list[NDArray[np.float32]]] = defaultdict(list)
            for k, ok_k in enumerate(t_ok.tolist()):
                if not ok_k:
                    continue
                d_k = int(day_nums[int(t_pos[k])])
                if d_k >= 0:
                    by_day[d_k].append(t_items[k][1])
            if not by_day:
                continue
            best_t = np.full((n_faces,), -1.0, dtype=np.float32)
            for d_anchor, vecs in by_day.items():
                lo = int(np.searchsorted(sorted_days, d_anchor - temporal_window_days, side="left"))
                hi = int(np.searchsorted(sorted_days, d_anchor + temporal_window_days, side="right"))
                if hi <= lo:
                    continue
                w_rows = sorted_rows[lo:hi]
                a_mat = np.stack(vecs, axis=0).astype(np.float32, copy=False)
                s_loc = np.max(embs[w_rows] @ a_mat.T, axis=1)
                gap = np.abs(sorted_days[lo:hi] - d_anchor).astype(np.float32)
                # Discount local evidence slightly with distance in days (0 d: -0.02 ... 3 d: -0.08):
                s_eff = s_loc - 0.02 - temporal_day_decay * gap
                # Window rows are unique, so a gather/scatter max is exact (and much faster than ufunc.at):
                best_t[w_rows] = np.maximum(best_t[w_rows], s_eff.astype(np.float32))
            pos_col = eff_scores_by_col[:, col_idx]
            t_mask = best_t >= temporal_min_sim
            if np.any(t_mask):
                lifted = np.maximum(pos_col[t_mask], best_t[t_mask])
                bonus = temporal_boost * np.clip((best_t[t_mask] - temporal_min_sim) / 0.24, 0.0, 1.0)
                eff_scores_by_col[t_mask, col_idx] = (lifted + bonus).astype(np.float32)

    _tick("temporal")

    # 3. Hard-Negative Lookalike Suppression (batched SGEMM across all people with negatives):
    neg_mats_by_col: list[tuple[int, int, NDArray[np.float32]]] = []
    for col_idx, pid in enumerate(enrolled_pids):
        neg_list = neg_by_pid.get(pid)
        if neg_list:
            neg_mats_by_col.append(
                (col_idx, pid, np.stack(neg_list[-max_negatives_per_person:], axis=0).astype(np.float32))
            )

    if neg_mats_by_col:
        neg_batches: list[list[tuple[int, NDArray[np.float32], int, int]]] = []
        cur_nbatch: list[tuple[int, NDArray[np.float32], int, int]] = []
        cur_ncols = 0
        for col_idx, _pid, nmat in neg_mats_by_col:
            nk = int(nmat.shape[0])
            if cur_nbatch and (cur_ncols + nk) > 512:
                neg_batches.append(cur_nbatch)
                cur_nbatch = []
                cur_ncols = 0
            cur_nbatch.append((col_idx, nmat, cur_ncols, cur_ncols + nk))
            cur_ncols += nk
        if cur_nbatch:
            neg_batches.append(cur_nbatch)

        for nbatch in neg_batches:
            stacked_neg = np.vstack([nmat for _, nmat, _, _ in nbatch]).astype(
                np.float32, copy=False
            )
            batch_neg_sims = embs @ stacked_neg.T
            for col_idx, _, c_start, c_end in nbatch:
                neg_sims = np.max(batch_neg_sims[:, c_start:c_end], axis=1)
                pos_col = eff_scores_by_col[:, col_idx]
                suppress_mask = (neg_sims >= hard_neg_min_sim) & (neg_sims > (pos_col + 0.03))
                # Soft tier: faces about as close to a removed/rejected face as to the child's own
                # anchors get a graded push-down (up to `soft_neg_weight`) even below the hard cutoff.
                soft_mask = (
                    ~suppress_mask
                    & (neg_sims >= soft_neg_min_sim)
                    & (neg_sims >= (pos_col - soft_neg_margin))
                )
                if np.any(soft_mask):
                    eff_scores_by_col[soft_mask, col_idx] = pos_col[soft_mask] - soft_neg_weight * np.clip(
                        (neg_sims[soft_mask] - soft_neg_min_sim) / 0.25, 0.0, 1.0
                    )
                if np.any(suppress_mask):
                    eff_scores_by_col[suppress_mask, col_idx] = pos_col[
                        suppress_mask
                    ] - hard_neg_weight * (neg_sims[suppress_mask] - 0.48)

    album_absent_by_col: dict[int, list[NDArray[np.int64]]] = defaultdict(list)
    _tick("negatives")

    # 3b. Same-Album Local Outfit/Bunnysuit Cohesion & Local Negative Suppression:
    for col_idx, pid in enumerate(enrolled_pids):
        pid_alb_pos = album_pos_by_pid.get(pid)
        pid_alb_neg = album_neg_by_pid.get(pid)
        if not pid_alb_pos and not pid_alb_neg:
            continue
        pos_scores = eff_scores_by_col[:, col_idx].copy()
        relevant_akeys = set(pid_alb_pos.keys() if pid_alb_pos else ()) | set(
            pid_alb_neg.keys() if pid_alb_neg else ()
        )
        for akey in relevant_akeys:
            alb_rows = rows_by_album.get(akey)
            if alb_rows is None or alb_rows.size == 0:
                continue
            alb_embs = embs[alb_rows]
            loc_neg_sims: NDArray[np.float32] | None = None
            if pid_alb_neg and akey in pid_alb_neg:
                loc_nmat = np.stack(pid_alb_neg[akey], axis=0).astype(np.float32)
                loc_neg_sims = np.max(alb_embs @ loc_nmat.T, axis=1)
                loc_sup = (loc_neg_sims >= 0.48) & (loc_neg_sims > (pos_scores[alb_rows] - 0.02))
                if np.any(loc_sup):
                    sup_rows = alb_rows[loc_sup]
                    pos_scores[sup_rows] -= np.float32(0.50) * (loc_neg_sims[loc_sup] - 0.42)

            if pid_alb_pos and akey in pid_alb_pos:
                loc_pmat = np.stack(pid_alb_pos[akey], axis=0).astype(np.float32)
                loc_pos_sims = np.max(alb_embs @ loc_pmat.T, axis=1)
                if album_absent_sim > 0.0:
                    # Same album = same day / outfit / lighting: real matches score ~0.7-0.9 against the
                    # child's confirmed faces from this album. A face far below that is someone else.
                    absent_rows = alb_rows[loc_pos_sims < album_absent_sim]
                    if absent_rows.size:
                        album_absent_by_col[col_idx].append(absent_rows)
                not_loc_rej = (
                    loc_neg_sims < 0.45
                    if loc_neg_sims is not None
                    else np.ones((alb_rows.shape[0],), dtype=bool)
                )
                boost_mask = (loc_pos_sims >= 0.32) & not_loc_rej
                if np.any(boost_mask):
                    b_rows = alb_rows[boost_mask]
                    b_lpos = loc_pos_sims[boost_mask]
                    base_max = np.maximum(pos_scores[b_rows], b_lpos)
                    c_boost = np.clip(
                        album_cohesion_boost * ((b_lpos - 0.28) / 0.20),
                        0.020,
                        album_cohesion_boost,
                    )
                    pos_scores[b_rows] = (base_max + c_boost).astype(np.float32)

                absorb_mask = (loc_pos_sims >= album_cohesion_absorb_sim) & not_loc_rej
                if np.any(absorb_mask):
                    for r_abs in alb_rows[absorb_mask]:
                        cohesion_candidates.add((int(r_abs), col_idx))

        eff_scores_by_col[:, col_idx] = pos_scores

    _tick("album")

    # 4. Social Peer Co-Occurrence & Small-Photo Cross-Circle Suppression:
    social = _build_social_context(state["people"], state["human"])
    pid_to_soc_idx: dict[int, int] = social["pid_to_idx"]
    peer_bond: NDArray[np.float32] = social["peer_bond"]
    outsider_mat: NDArray[np.float32] = social["outsider_mat"]
    confirmed_by_photo: dict[int, list[tuple[int | None, int]]] = social["confirmed_by_photo"]

    if (social_peer_boost > 0.0 or social_cross_circle_penalty > 0.0) and (
        bool(np.any(peer_bond > 0.0)) or bool(np.any(outsider_mat > 0.0))
    ):
        cached_fc: dict[int, int] | None = mat.get("fc_by_photo")
        cached_rows_by_photo: dict[int, list[int]] | None = mat.get("rows_by_photo")
        if cached_fc is not None:
            fc_by_photo = cached_fc
        else:
            uniq_p, p_counts = np.unique(photo_ids, return_counts=True)
            fc_by_photo = dict(zip(uniq_p.tolist(), p_counts.tolist(), strict=True))

        vis_witness_by_row: dict[int, int] = {}
        n_enr = len(enrolled_pids)
        if n_enr == 1:
            vis_rows = np.where((eff_scores_by_col[:, 0] >= 0.50) & (~suppressed_auto))[0]
            for vr_raw in vis_rows:
                vis_witness_by_row[int(vr_raw)] = pid_to_soc_idx[enrolled_pids[0]]
        else:
            best_cols = np.argmax(eff_scores_by_col, axis=1)
            part = np.partition(eff_scores_by_col, -2, axis=1)
            top1 = part[:, -1]
            top2 = part[:, -2]
            vis_rows = np.where((top1 >= 0.50) & ((top1 - top2) >= 0.05) & (~suppressed_auto))[0]
            for vr_raw in vis_rows:
                r_int = int(vr_raw)
                vis_witness_by_row[r_int] = pid_to_soc_idx[enrolled_pids[int(best_cols[r_int])]]

        candidate_social_photos: set[int] = {
            p_id for p_id in confirmed_by_photo if 2 <= fc_by_photo.get(p_id, 0) <= 8
        }
        for r_int in vis_witness_by_row:
            p_id = int(photo_ids[r_int])
            if 2 <= fc_by_photo.get(p_id, 0) <= 8:
                candidate_social_photos.add(p_id)

        if candidate_social_photos:
            rows_by_social_photo: dict[int, list[int]] = {}
            if cached_rows_by_photo is not None:
                for p_id in candidate_social_photos:
                    pr_list = cached_rows_by_photo.get(p_id)
                    if pr_list:
                        rows_by_social_photo[p_id] = pr_list
            else:
                tmp_soc: dict[int, list[int]] = defaultdict(list)
                for r_idx in range(n_faces):
                    p_id = int(photo_ids[r_idx])
                    if p_id in candidate_social_photos:
                        tmp_soc[p_id].append(r_idx)
                rows_by_social_photo = tmp_soc

            # Gather (target row -> witnesses) once in Python, then score every (row, enrolled person)
            # pair with numpy: per pair, the strongest peer-bond witness and the strongest outsider
            # witness among the photo's other faces (a witness never vouches for itself).
            enrolled_soc = np.asarray([pid_to_soc_idx[pid] for pid in enrolled_pids], dtype=np.int64)
            t_rows: list[int] = []
            t_intim: list[float] = []
            t_wit: list[list[tuple[int, int, float]]] = []
            for p_id, p_rows in rows_by_social_photo.items():
                intimacy = photo_intimacy_factor(fc_by_photo[p_id])
                if intimacy <= 0.0:
                    continue
                witnesses: list[tuple[int, int, float]] = []
                conf_fids_used: set[int] = set()
                conf_soc_used: set[int] = set()
                for f_w, q_soc in confirmed_by_photo.get(p_id, []):
                    witnesses.append((-1 if f_w is None else int(f_w), q_soc, 1.0))
                    conf_soc_used.add(q_soc)
                    if f_w is not None:
                        conf_fids_used.add(f_w)
                for r_w in p_rows:
                    if r_w not in vis_witness_by_row:
                        continue
                    fid_w = int(face_ids[r_w])
                    q_soc = vis_witness_by_row[r_w]
                    if fid_w in conf_fids_used or q_soc in conf_soc_used:
                        continue
                    witnesses.append((fid_w, q_soc, 0.80))
                if not witnesses:
                    continue
                for r_t in p_rows:
                    if int(face_ids[r_t]) in locked_faces:
                        continue
                    t_rows.append(int(r_t))
                    t_intim.append(intimacy)
                    t_wit.append(witnesses)

            if t_rows:
                n_t = len(t_rows)
                m_w = max(len(w) for w in t_wit)
                w_fid = np.full((n_t, m_w), -2, dtype=np.int64)
                w_soc = np.zeros((n_t, m_w), dtype=np.int64)
                w_conf = np.zeros((n_t, m_w), dtype=np.float64)
                for i_t, wl in enumerate(t_wit):
                    for j_w, (f_w, q_soc, conf_q) in enumerate(wl):
                        w_fid[i_t, j_w] = f_w
                        w_soc[i_t, j_w] = q_soc
                        w_conf[i_t, j_w] = conf_q
                rows_arr = np.asarray(t_rows, dtype=np.int64)
                intim_arr = np.asarray(t_intim, dtype=np.float64)
                tgt_fid = face_ids[rows_arr]
                bond64 = peer_bond.astype(np.float64)[enrolled_soc]  # (cols, n_people)
                out64 = outsider_mat.astype(np.float64)[enrolled_soc]
                for c0 in range(0, n_t, 4096):
                    sl = slice(c0, c0 + 4096)
                    soc_c = w_soc[sl]
                    usable = (w_fid[sl] != tgt_fid[sl, np.newaxis])[:, np.newaxis, :] & (
                        enrolled_soc[np.newaxis, :, np.newaxis] != soc_c[:, np.newaxis, :]
                    )
                    conf_c = w_conf[sl][:, np.newaxis, :]
                    b_sig = np.where(usable, conf_c * bond64[:, soc_c].transpose(1, 0, 2), 0.0).max(axis=2)
                    o_sig = np.where(usable, conf_c * out64[:, soc_c].transpose(1, 0, 2), 0.0).max(axis=2)
                    b_sig = np.maximum(b_sig, 0.0)
                    o_sig = np.maximum(o_sig, 0.0)
                    base = eff_scores_by_col[rows_arr[sl]].astype(np.float64)
                    scale_b = (social_peer_boost * intim_arr[sl])[:, np.newaxis]
                    scale_o = (social_cross_circle_penalty * intim_arr[sl])[:, np.newaxis]
                    boost = np.where(base >= 0.27, scale_b * b_sig, 0.0)
                    eff_scores_by_col[rows_arr[sl]] = ((base + boost) - scale_o * o_sig).astype(np.float32)

    _tick("social")

    # 4b. Soft per-child pressure from the Child Clusters view (boost / penalize look-alikes):
    pressure_by_pid: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for pr in state.get("pressure", []):
        bucket = pressure_by_pid[int(pr["person_id"])]
        (bucket[0] if float(pr["weight"]) > 0 else bucket[1]).append(int(pr["face_id"]))
    if pressure_by_pid:
        for col_idx, pid in enumerate(enrolled_pids):
            if pid not in pressure_by_pid:
                continue
            boost_fids, pen_fids = pressure_by_pid[pid]
            for fids_list, is_boost in ((boost_fids, True), (pen_fids, False)):
                if not fids_list:
                    continue
                q = np.asarray(sorted(set(fids_list)), dtype=np.int64)
                pos = np.searchsorted(face_ids, q)
                ok = (pos < n_faces) & (face_ids[np.minimum(pos, n_faces - 1)] == q)
                rows_sel = pos[ok]
                if rows_sel.size == 0:
                    continue
                # Later stages only ever lower scores, so pressure can only matter for faces that can
                # still reach the candidate threshold: skip the rest of the archive.
                col_now = eff_scores_by_col[:, col_idx]
                reach = candidate_threshold - (pressure_boost if is_boost else 0.0) - 1e-3
                tgt_rows = np.where(col_now >= np.float32(reach))[0]
                if tgt_rows.size == 0:
                    continue
                tgt_e = embs[tgt_rows]
                best = np.full((int(tgt_rows.size),), -1.0, dtype=np.float32)
                for c0 in range(0, int(rows_sel.size), 256):
                    chunk = embs[rows_sel[c0 : c0 + 256]]
                    np.maximum(best, np.max(tgt_e @ chunk.T, axis=1), out=best)
                if is_boost:
                    w = np.clip((best - 0.40) / 0.30, 0.0, 1.0) * pressure_boost
                else:
                    w = -np.clip((best - 0.42) / 0.30, 0.0, 1.0) * pressure_penalty
                eff_scores_by_col[tgt_rows, col_idx] += w.astype(np.float32)

    _tick("pressure")

    # 4c. Hard eligibility: birthdays (never before birth) and class memberships (only own classes):
    rules = db.load_eligibility_rules()
    if rules:
        date_ords: NDArray[np.int32] | None = mat.get("date_ords")
        arch_list: list[str] = mat.get("archive_children") or [""] * n_faces
        if date_ords is not None and int(date_ords.shape[0]) == n_faces:
            feed_names, feed_codes = np.unique(np.asarray(arch_list, dtype=object), return_inverse=True)
            feed_index = {str(nm): i for i, nm in enumerate(feed_names.tolist())}
            for col_idx, pid in enumerate(enrolled_pids):
                if pid not in rules:
                    continue
                birth_ord, windows = rules[pid]
                allowed = np.ones((n_faces,), dtype=bool)
                if birth_ord > 0:
                    allowed &= date_ords >= birth_ord
                if windows:
                    in_cls = np.zeros((n_faces,), dtype=bool)
                    for w_feed, w_start, w_end in windows:
                        w_mask = (date_ords >= w_start) & (date_ords <= w_end)
                        if w_feed:
                            code = feed_index.get(w_feed)
                            if code is None:
                                continue
                            w_mask &= feed_codes == code
                        in_cls |= w_mask
                    allowed &= in_cls
                eff_scores_by_col[~allowed, col_idx] = -1.0

    # 4e. Same-album absence: in albums where a child is already confirmed, borderline matches that look
    # nothing like her confirmed faces from that very album are dropped from her queue.
    if album_absent_by_col:
        weak_cap = np.float32(candidate_threshold - 0.01)
        for col_idx, row_lists in album_absent_by_col.items():
            a_rows = np.unique(np.concatenate(row_lists))
            col_vals = eff_scores_by_col[a_rows, col_idx]
            weak = col_vals < np.float32(auto_high_threshold + album_absent_strong_margin)
            if np.any(weak):
                eff_scores_by_col[a_rows[weak], col_idx] = np.minimum(col_vals[weak], weak_cap)

    _tick("eligibility")

    # 4d. Learned Label Vote (k-nearest labelled faces): every human-confirmed face of every person
    # (including `ignore` background profiles) votes on the identity of each still-open face.
    # - Veto: when the labelled neighbourhood clearly says "someone else" (or "background"), the face
    #   is removed from the other children's queues.
    # - Trust: when it clearly agrees with the matcher, the auto match skips manual review.
    knn_top = np.full((n_faces,), -1, dtype=np.int64)
    knn_frac = np.zeros((n_faces,), dtype=np.float32)
    knn_sim = np.full((n_faces,), -1.0, dtype=np.float32)
    if knn_enabled:
        pid_by_row = np.full((n_faces,), -1, dtype=np.int64)
        for h in state["human"]:
            if h["status"] == "confirmed" and h["emb"] is not None and h["face_id"] is not None:
                b_pos = int(np.searchsorted(face_ids, int(h["face_id"])))
                if b_pos < n_faces and int(face_ids[b_pos]) == int(h["face_id"]):
                    pid_by_row[b_pos] = int(h["person_id"])
        labelled_row = pid_by_row >= 0
        if knn_bank_cap_per_person > 0:
            # Thin out very large label sets (one child can hold a quarter of all labels) with a
            # face-id hash so the bank stays stable while labels are added one at a time.
            lab_counts = Counter(int(x) for x in pid_by_row[pid_by_row >= 0].tolist())
            keep_frac = np.zeros((max(lab_counts, default=0) + 2,), dtype=np.float64)
            for pid_c, cnt in lab_counts.items():
                keep_frac[pid_c] = min(1.0, knn_bank_cap_per_person / float(cnt))
            hashed = ((face_ids.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)).astype(np.float64)
            thin = (pid_by_row >= 0) & (hashed >= keep_frac[np.maximum(pid_by_row, 0)] * float(1 << 32))
            pid_by_row = np.where(thin, -1, pid_by_row)  # thinned faces stay locked, they just don't vote
        bank_rows = np.where(pid_by_row >= 0)[0]
        open_mask = np.max(eff_scores_by_col, axis=1) >= candidate_threshold
        open_mask &= ~labelled_row  # labelled faces are locked / are the voters themselves
        if locked_faces:
            open_mask &= ~np.isin(face_ids, np.fromiter(locked_faces, dtype=np.int64, count=len(locked_faces)))
        open_rows = np.where(open_mask & ~suppressed_auto)[0]
        if bank_rows.size >= 2 and open_rows.size:
            nn_rows_all, nn_sim_all = _cached_label_neighbors(
                db, embs, face_ids, open_rows, bank_rows, min(knn_k, int(bank_rows.size) - 1)
            )
            cls_ids = sorted({int(x) for x in pid_by_row[bank_rows].tolist()})
            cls_index_arr = np.full((max(cls_ids) + 2,), -1, dtype=np.int64)
            for i_c, pid_c in enumerate(cls_ids):
                cls_index_arr[pid_c] = i_c
            n_cls = len(cls_ids)
            cls_pid_arr = np.asarray(cls_ids, dtype=np.int64)
            ignore_pids = set(state.get("ignore_person_ids", []))
            cls_is_bg = np.asarray([pid_c in ignore_pids for pid_c in cls_ids], dtype=bool)
            col_of_pid = {pid_c: ci for ci, pid_c in enumerate(enrolled_pids)}
            cls_to_col = np.asarray([col_of_pid.get(pid_c, -1) for pid_c in cls_ids], dtype=np.int64)
            n_cols = len(enrolled_pids)
            veto_cap = np.float32(candidate_threshold - 0.01)
            k_eff = int(nn_rows_all.shape[1])
            bank_counts = Counter(int(x) for x in pid_by_row[bank_rows].tolist())
            gate_pairs = [
                (ci, int(cls_index_arr[pid_c]))
                for ci, pid_c in enumerate(enrolled_pids)
                if bank_counts.get(pid_c, 0) >= knn_gate_min_labels
            ]
            gate_cols = np.asarray([g[0] for g in gate_pairs], dtype=np.int64)
            gate_cls = np.asarray([g[1] for g in gate_pairs], dtype=np.int64)
            for c0 in range(0, int(open_rows.size), 8192):
                q_rows = open_rows[c0 : c0 + 8192]
                nn_r = nn_rows_all[c0 : c0 + 8192]
                nn_sim = nn_sim_all[c0 : c0 + 8192]
                valid_nn = nn_r >= 0
                nn_pid = np.where(valid_nn, pid_by_row[np.maximum(nn_r, 0)], -1)
                valid_nn &= nn_pid >= 0
                nn_cls = np.where(valid_nn, cls_index_arr[np.maximum(nn_pid, 0)], 0)
                w_nn = np.where(valid_nn, np.clip(nn_sim - 0.20, 0.0, None) ** 2, 0.0)
                m_q = int(q_rows.shape[0])
                flat = np.repeat(np.arange(m_q), k_eff) * n_cls + nn_cls.ravel()
                cls_w = np.bincount(flat, weights=w_nn.ravel(), minlength=m_q * n_cls).reshape(m_q, n_cls)
                cls_s_flat = np.full((m_q * n_cls,), -1.0, dtype=np.float32)
                np.maximum.at(cls_s_flat, flat, np.where(valid_nn, nn_sim, -1.0).ravel().astype(np.float32))
                cls_s = cls_s_flat.reshape(m_q, n_cls)
                top_c = np.argmax(cls_w, axis=1)
                ar = np.arange(m_q)
                frac_q = (cls_w[ar, top_c] / (cls_w.sum(axis=1) + 1e-6)).astype(np.float32)
                sim_q = cls_s[ar, top_c]
                knn_top[q_rows] = cls_pid_arr[top_c]
                knn_frac[q_rows] = frac_q
                knn_sim[q_rows] = sim_q

                # Identity veto: the labelled neighbourhood clearly says "someone else".
                veto = (frac_q >= knn_veto_frac) & (sim_q >= knn_min_sim)
                keep_col = cls_to_col[top_c]
                # Background veto: closest labelled neighbour is an `ignore` (background) face.
                if np.any(cls_is_bg):
                    bg_s = np.max(np.where(cls_is_bg[np.newaxis, :], cls_s, -1.0), axis=1)
                    kid_s = np.max(np.where(cls_is_bg[np.newaxis, :], -1.0, cls_s), axis=1)
                    bg_veto = (bg_s >= knn_bg_min_sim) & (bg_s >= kid_s + 0.02)
                    keep_col = np.where(bg_veto & ~veto, -1, keep_col)
                    veto |= bg_veto
                if np.any(veto):
                    v_rows = q_rows[veto]
                    sub = eff_scores_by_col[v_rows]
                    capped = np.minimum(sub, veto_cap)
                    keep_mask = np.arange(n_cols)[np.newaxis, :] == keep_col[veto][:, np.newaxis]
                    eff_scores_by_col[v_rows] = np.where(keep_mask, sub, capped)

                # Weak-match gate: a borderline match (below `auto_high + 0.05`) for a well-labelled child
                # is dropped when that child gets < `knn_gate_frac` of the labelled-neighbour vote while
                # the vote clearly points at someone else. This stops faces of other (or not-yet-eligible)
                # kids from piling into one child's queue just because she is the "least bad" option.
                if gate_cols.size:
                    other_top = sim_q >= knn_min_sim
                    frac_cols = (cls_w[:, gate_cls] / (cls_w.sum(axis=1, keepdims=True) + 1e-6)).astype(np.float32)
                    sub_g = eff_scores_by_col[np.ix_(q_rows, gate_cols)]
                    top_is_col = top_c[:, np.newaxis] == gate_cls[np.newaxis, :]
                    gate_hit = (
                        other_top[:, np.newaxis]
                        & ~top_is_col
                        & (frac_cols < knn_gate_frac)
                        & (sub_g >= candidate_threshold)
                        & (sub_g < np.float32(auto_high_threshold + 0.05))
                    )
                    if np.any(gate_hit):
                        eff_scores_by_col[np.ix_(q_rows, gate_cols)] = np.where(gate_hit, veto_cap, sub_g)

    _tick("knn")
    neg_mat_cache: dict[int, NDArray[np.float32] | None] = {}

    def _knn_trusts(r_idx: int, pid: int, promote: bool) -> bool:
        if int(knn_top[r_idx]) != pid:
            return False
        need = knn_promote_frac if promote else knn_trust_frac
        f_v = float(knn_frac[r_idx])
        s_v = float(knn_sim[r_idx])
        if f_v < need or s_v < (knn_min_sim + (0.05 if promote else 0.0)):
            return False
        if pid not in neg_mat_cache:
            n_list = neg_by_pid.get(pid)
            neg_mat_cache[pid] = (
                np.stack(n_list[-max_negatives_per_person:], axis=0).astype(np.float32)
                if n_list
                else None
            )
        n_mat = neg_mat_cache[pid]
        return not (n_mat is not None and float(np.max(n_mat @ embs[r_idx])) >= s_v)

    # 5. Global Photo-Level Bipartite 1-to-1 Exclusivity:
    # Find all (face_row, person_col) pairs >= candidate_threshold and group by photo_id.
    cand_rows, cand_cols = np.where(eff_scores_by_col >= candidate_threshold)
    if cand_rows.size == 0:
        return db.replace_model_assignments([], chosen_anchor_face_ids)

    by_photo: dict[int, list[tuple[float, int, int, int, int]]] = defaultdict(list)
    for r_idx_raw, c_idx_raw in zip(cand_rows, cand_cols, strict=True):
        r_idx = int(r_idx_raw)
        c_idx = int(c_idx_raw)
        fid = int(face_ids[r_idx])
        if fid in locked_faces or bool(suppressed_auto[r_idx]):
            continue
        p_id = int(photo_ids[r_idx])
        pid = enrolled_pids[c_idx]
        if (
            (p_id, pid) in locked_photo_people
            or (p_id, pid) in rejected_photo_people
            or (fid, pid) in rejected_face_people
        ):
            continue
        score = float(eff_scores_by_col[r_idx, c_idx])
        by_photo[p_id].append((score, fid, pid, r_idx, c_idx))

    n_enr_cols = len(enrolled_pids)
    promote_min = auto_high_threshold - 0.05 if knn_promote_min_score is None else float(knn_promote_min_score)
    auto_assignments: list[tuple[int, int, int, AssignmentStatus, float, AssignmentSource]] = []
    for p_id, candidates in by_photo.items():
        if len(candidates) > 1:
            candidates.sort(key=lambda x: x[0], reverse=True)
        used_fids: set[int] = set()
        used_pids: set[int] = set()
        for score, fid, pid, r_idx, c_idx in candidates:
            if fid in used_fids or pid in used_pids:
                continue
            used_fids.add(fid)
            used_pids.add(pid)
            status: AssignmentStatus = "auto_high" if score >= auto_high_threshold else "candidate"
            src_kind: AssignmentSource = "model_match"
            if status == "auto_high" and (r_idx, c_idx) in cohesion_candidates:
                if n_enr_cols == 1:
                    src_kind = "album_cohesion"
                else:
                    row_vals = eff_scores_by_col[r_idx]
                    runner_up = float(
                        np.max(np.concatenate((row_vals[:c_idx], row_vals[c_idx + 1 :])))
                    )
                    if (score - runner_up) >= 0.06:
                        src_kind = "album_cohesion"
            if knn_enabled:
                trusts = status == "auto_high" and _knn_trusts(r_idx, pid, False)
                band_trusts = trusts and (
                    score >= auto_band_knn_upper or float(knn_sim[r_idx]) >= auto_band_knn_min_sim
                )
                if status == "auto_high" and not band_trusts and (
                    score < auto_band_knn_upper
                    or (src_kind == "album_cohesion" and album_cohesion_require_knn)
                ):
                    # Calibration band: human decisions at these scores are wrong ~1 in 5 times,
                    # so only auto-accept when the labelled neighbourhood agrees; else ask.
                    status = "candidate"
                    src_kind = "model_match"
                elif trusts and src_kind != "album_cohesion":
                    src_kind = "model_trusted"
                elif (
                    status == "candidate"
                    and score >= promote_min
                    and float(knn_sim[r_idx]) >= auto_band_knn_min_sim
                    and _knn_trusts(r_idx, pid, True)
                ):
                    status = "auto_high"
                    src_kind = "model_trusted"
            auto_assignments.append((p_id, pid, fid, status, score, src_kind))
    _tick("assign")

    result = db.replace_model_assignments(auto_assignments, chosen_anchor_face_ids)
    _tick("write")
    if diagnostics is not None:
        diagnostics.update(
            timings=timings,
            face_ids=face_ids,
            enrolled_pids=list(enrolled_pids),
            eff_scores=eff_scores_by_col,
            knn_top=knn_top,
            knn_frac=knn_frac,
            knn_sim=knn_sim,
            assignments=auto_assignments,
        )
    return result
