"""Hidden-label replay of the matcher on a COPY of a faces index (calibration / regression tool).

Usage::

    python -m kinderpedia_backup.faces.evaluate --db <faces_index.sqlite copy> \\
        [--grid "min_sim=0.25,0.30;min_faces=4,8;det_penalty=0,0.05"] [--seeds 1,2] \\
        [--folds 5] [--only-folds 0,1] [--json out.json] [--markdown out.md]

Safety
    The tool never writes the given database. It first copies it (SQLite online-backup API, source
    opened read-only) into a private temp directory and works only on copies. It also refuses a path
    named `faces_index.sqlite` inside a `Metadata` directory (the live archive layout) unless
    `--i-know-this-is-a-copy` is given.

Protocol (pre-declared, deterministic for a given `--seeds` / `--folds`)
    1. Decision units are FACES carrying human decisions: confirmed tags (batch approve, manual tag,
       cluster confirm), rejections (batch reject, untag / reassign negatives) and "unsure" marks.
       Truth of a face = its confirmed person (if any); its rejected persons are known negatives.
    2. Every face gets a stratum `(kind, person, year)`, kind = `cluster` (confirmed via
       cluster_confirm) or `batch` (everything else), person = confirmed person or first rejected one.
       Within a stratum faces are shuffled with `numpy.random.default_rng(seed)` and dealt round-robin
       from a random start into `--folds` folds, so every fold hides a matching share of batch
       decisions AND of the bulk cluster confirms of every person-year.
    3. Per fold: all human decisions on the fold's faces are deleted from a fresh copy (plus their
       cluster pressure / keep marks; `suppress_auto` is cleared), the rest stays as labels, the matcher
       re-scores the archive, and ONLY the hidden faces are scored.
    4. Metrics per (grid cell, fold): auto-accept precision = correct / (correct + wrong) over hidden
       faces auto-accepted (`auto_high`) for a person whose truth is known (wrong = another person's
       face, a background (`ignore`) face, or a hidden rejection); recall = hidden faces of enrolled
       people auto-accepted for their true person / all such faces. Also reported: per score band,
       per channel (`model_match` / `album_cohesion` / `model_trusted`), background faces
       auto-tagged, hidden rejections re-auto-accepted, labels flagged by the label-consistency guard,
       and the human reject rate per score band of every hidden (face, person) pair (threshold check).
    5. Aggregation: mean over folds (and seeds) +- standard error (std / sqrt(n replicates)); deltas
       vs the guard-off baseline are paired per replicate.

Selection rule (pre-declared)
    Baseline = guard off (`label_outlier_min_sim = -1`: nothing can be flagged), det penalty 0.
    A grid cell is NON-LOSING when its mean paired precision delta vs the baseline is >= -SE of that
    delta (a difference within the fold-to-fold standard error is a tie, not a loss). Among
    non-losing cells the one with the highest mean recall wins; cells whose paired recall deficit
    to it is within its SE tie, and ties go to the SIMPLER cell: lower det_penalty, then lower
    min_faces, then lower min_sim.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .index_db import FacesIndexDB
from .matcher import recompute_active_matching

#: Grid axis name -> `recompute_active_matching` keyword.
GRID_PARAMS: dict[str, str] = {
    "min_sim": "label_outlier_min_sim",
    "min_faces": "label_outlier_min_faces",
    "det_penalty": "anchor_det_penalty",
}
BASELINE_NAME = "guard-off"
BASELINE_KW: dict[str, Any] = {"label_outlier_min_sim": -1.0, "anchor_det_penalty": 0.0}
SCORE_BANDS: tuple[tuple[float, float], ...] = ((0.38, 0.45), (0.45, 0.50), (0.50, 0.60), (0.60, 9.0))
THRESHOLD_BANDS: tuple[tuple[float, float], ...] = (
    (0.33, 0.40),
    (0.40, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 9.0),
)
CHANNELS: tuple[str, ...] = ("model_match", "album_cohesion", "model_trusted")


def _matcher_default(name: str) -> Any:
    return inspect.signature(recompute_active_matching).parameters[name].default


def is_live_archive_path(path: Path) -> bool:
    """True for `.../Metadata/faces_index.sqlite` (the live archive layout)."""
    resolved = path.expanduser().resolve()
    return resolved.name == "faces_index.sqlite" and resolved.parent.name == "Metadata"


def copy_database(src: Path, dst: Path) -> None:
    """Consistent copy via the SQLite online-backup API (source opened read-only)."""
    src_conn = sqlite3.connect(f"{src.expanduser().resolve().as_uri()}?mode=ro", uri=True)
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def parse_grid(spec: str) -> list[dict[str, Any]]:
    """`"min_sim=0.25,0.3;min_faces=4,8"` -> cartesian product of matcher keyword dicts (sorted)."""
    axes: list[tuple[str, list[float]]] = []
    for part in [p.strip() for p in spec.split(";") if p.strip()]:
        name, _, vals = part.partition("=")
        name = name.strip()
        if name not in GRID_PARAMS:
            raise ValueError(f"Unknown grid axis {name!r} (known: {', '.join(GRID_PARAMS)}).")
        values = [float(v) for v in vals.split(",") if v.strip()]
        if not values:
            raise ValueError(f"Grid axis {name!r} has no values.")
        axes.append((name, sorted(set(values))))
    cells: list[dict[str, Any]] = [{}]
    for name, values in axes:
        kw = GRID_PARAMS[name]
        cells = [
            {**c, kw: (int(v) if kw == "label_outlier_min_faces" else float(v))} for c in cells for v in values
        ]
    return cells


def cell_params(kw: dict[str, Any]) -> dict[str, Any]:
    """Full guard parameters of a cell (matcher defaults filled in)."""
    return {k: kw.get(k, _matcher_default(k)) for k in GRID_PARAMS.values()}


def cell_label(kw: dict[str, Any]) -> str:
    p = cell_params(kw)
    return (
        f"sim={p['label_outlier_min_sim']:.2f} faces={p['label_outlier_min_faces']} "
        f"det={p['anchor_det_penalty']:.2f}"
    )


def simplicity_key(kw: dict[str, Any]) -> tuple[float, int, float]:
    p = cell_params(kw)
    return (float(p["anchor_det_penalty"]), int(p["label_outlier_min_faces"]), float(p["label_outlier_min_sim"]))


@dataclass
class HiddenFace:
    face_id: int
    truth: int | None  # confirmed person
    negatives: set[int]
    kind: str  # "cluster" | "batch"
    truth_is_background: bool


@dataclass
class Decisions:
    faces: dict[int, HiddenFace]
    strata: dict[tuple[str, int, int], list[int]]
    background_pids: set[int]
    active_pids: set[int]


def load_decisions(db_path: Path) -> Decisions:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        roles = {int(r[0]): str(r[1]) for r in conn.execute("SELECT person_id, role FROM people")}
        rows = conn.execute(
            """
            SELECT a.face_id, a.person_id, a.status, a.source, p.year
            FROM assignments a
            JOIN faces f ON f.face_id = a.face_id AND f.is_manual = 0
            JOIN photos p ON p.photo_id = a.photo_id
            WHERE a.status IN ('confirmed', 'rejected', 'unsure')
            ORDER BY a.face_id, a.person_id
            """
        ).fetchall()
    finally:
        conn.close()
    by_face: dict[int, list[tuple[int, str, str, int]]] = defaultdict(list)
    for fid, pid, st, src, yr in rows:
        y = str(yr)
        by_face[int(fid)].append((int(pid), str(st), str(src), int(y) if y.isdigit() and len(y) == 4 else 0))
    background = {pid for pid, role in roles.items() if role == "ignore"}
    faces: dict[int, HiddenFace] = {}
    strata: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for fid, decs in by_face.items():
        conf = [d for d in decs if d[1] == "confirmed"]
        negs = {d[0] for d in decs if d[1] == "rejected"}
        truth = conf[0][0] if conf else None
        kind = "cluster" if conf and conf[0][2] == "cluster_confirm" else "batch"
        primary = truth if truth is not None else (min(negs) if negs else decs[0][0])
        faces[fid] = HiddenFace(fid, truth, negs, kind, truth in background)
        strata[(kind, primary, decs[0][3])].append(fid)
    return Decisions(faces, dict(strata), background, {p for p, r in roles.items() if r != "ignore"})


def assign_folds(dec: Decisions, n_folds: int, seed: int) -> dict[int, int]:
    """Deterministic stratified fold of every decision face."""
    rng = np.random.default_rng(seed)
    fold_of: dict[int, int] = {}
    for key in sorted(dec.strata):
        fids = np.asarray(sorted(dec.strata[key]), dtype=np.int64)
        perm = rng.permutation(fids.size)
        start = int(rng.integers(0, n_folds))
        for i, j in enumerate(perm.tolist()):
            fold_of[int(fids[j])] = (start + i) % n_folds
    return fold_of


def hide_faces(db_path: Path, face_ids: Sequence[int]) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute("CREATE TEMP TABLE hidden (face_id INTEGER PRIMARY KEY)")
            conn.executemany("INSERT INTO hidden (face_id) VALUES (?)", [(int(f),) for f in face_ids])
            conn.execute(
                "DELETE FROM assignments WHERE status IN ('confirmed', 'rejected', 'unsure') "
                "AND face_id IN (SELECT face_id FROM hidden)"
            )
            conn.execute("DELETE FROM person_face_pressure WHERE face_id IN (SELECT face_id FROM hidden)")
            conn.execute("DELETE FROM label_keeps WHERE face_id IN (SELECT face_id FROM hidden)")
            conn.execute(
                "UPDATE faces SET suppress_auto = 0 WHERE is_manual = 0 AND face_id IN (SELECT face_id FROM hidden)"
            )
    finally:
        conn.close()


@dataclass
class FoldResult:
    correct: int = 0
    wrong: int = 0
    unknown: int = 0
    correct_truth: int = 0
    recall_den: int = 0
    background_hidden: int = 0
    background_auto: int = 0
    rejections_hidden: int = 0
    rejections_reaccepted: int = 0
    flagged_labels: int = 0
    band: dict[str, list[int]] = field(default_factory=dict)  # band -> [correct, wrong]
    channel: dict[str, list[int]] = field(default_factory=dict)
    kind: dict[str, list[int]] = field(default_factory=dict)  # kind -> [correct, wrong, recall_num, recall_den]
    pairs: dict[str, list[int]] = field(default_factory=dict)  # "kind|band|knn" -> [approved, rejected]
    seconds: float = 0.0

    @property
    def precision(self) -> float:
        n = self.correct + self.wrong
        return self.correct / n if n else math.nan

    @property
    def recall(self) -> float:
        return self.correct_truth / self.recall_den if self.recall_den else math.nan


def _band_name(lo: float, hi: float) -> str:
    return f">={lo:.2f}" if hi > 2.0 else f"{lo:.2f}-{hi:.2f}"


def _band_of(score: float, bands: tuple[tuple[float, float], ...]) -> str | None:
    for lo, hi in bands:
        if lo <= score < hi:
            return _band_name(lo, hi)
    return None


def score_fold(diag: dict[str, Any], hidden: list[HiddenFace], dec: Decisions) -> FoldResult:
    res = FoldResult(flagged_labels=len(diag["flagged_labels"]))
    face_ids: np.ndarray = diag["face_ids"]
    col = {p: i for i, p in enumerate(diag["enrolled_pids"])}
    eff: np.ndarray = diag["eff_scores"]
    knn_top: np.ndarray = diag["knn_top"]
    knn_frac: np.ndarray = diag["knn_frac"]
    knn_sim: np.ndarray = diag["knn_sim"]
    trust_frac = float(_matcher_default("knn_trust_frac"))
    knn_min = float(_matcher_default("knn_min_sim"))
    auto: dict[int, tuple[int, float, str]] = {
        int(a[2]): (int(a[1]), float(a[4]), str(a[5])) for a in diag["assignments"] if a[3] == "auto_high"
    }
    for hf in hidden:
        got = auto.get(hf.face_id)
        kind_c = res.kind.setdefault(hf.kind, [0, 0, 0, 0])
        if hf.truth is not None and not hf.truth_is_background and hf.truth in dec.active_pids:
            res.recall_den += 1
            kind_c[3] += 1
            if got is not None and got[0] == hf.truth:
                res.correct_truth += 1
                kind_c[2] += 1
        if hf.truth_is_background:
            res.background_hidden += 1
            if got is not None:
                res.background_auto += 1
        res.rejections_hidden += len(hf.negatives)
        if got is not None:
            a_pid, a_score, a_src = got
            if a_pid in hf.negatives:
                res.rejections_reaccepted += 1
            ok: bool | None
            if hf.truth is not None:
                ok = a_pid == hf.truth
            elif a_pid in hf.negatives:
                ok = False
            else:
                ok = None
            if ok is None:
                res.unknown += 1
            else:
                idx = 0 if ok else 1
                if ok:
                    res.correct += 1
                else:
                    res.wrong += 1
                kind_c[idx] += 1
                band = _band_of(a_score, SCORE_BANDS) or "<0.38"
                res.band.setdefault(band, [0, 0])[idx] += 1
                res.channel.setdefault(a_src, [0, 0])[idx] += 1
        # Threshold check: human approve / reject per score band of every hidden (face, person) pair.
        r = int(np.searchsorted(face_ids, hf.face_id))
        if r >= face_ids.shape[0] or int(face_ids[r]) != hf.face_id:
            continue
        labelled = [(p, 0) for p in hf.negatives]
        if hf.truth is not None and not hf.truth_is_background:
            labelled.append((hf.truth, 1))
        for pid, lab in labelled:
            if pid not in col:
                continue
            b = _band_of(float(eff[r, col[pid]]), THRESHOLD_BANDS)
            if b is None:
                continue
            agree = int(knn_top[r]) == pid and float(knn_frac[r]) >= trust_frac and float(knn_sim[r]) >= knn_min
            key = f"{hf.kind}|{b}|{'knn' if agree else 'noknn'}"
            res.pairs.setdefault(key, [0, 0])[0 if lab else 1] += 1
    return res


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))


def run_replay(
    db: Path,
    *,
    cells: list[dict[str, Any]],
    seeds: Sequence[int],
    n_folds: int,
    only_folds: Sequence[int] | None,
    workdir: Path,
    log: Any = None,
) -> dict[str, Any]:
    """Run the protocol; returns `{"cells": [...], "replicates": [...], ...}` (see module doc)."""

    def _log(msg: str) -> None:
        if log is not None:
            print(msg, file=log, flush=True)

    base = workdir / "base.sqlite"
    copy_database(db, base)
    FacesIndexDB(base).close()  # bring the private copy to the current schema
    dec = load_decisions(base)
    all_cells = [dict(BASELINE_KW), *cells]
    folds = list(only_folds) if only_folds is not None else list(range(n_folds))
    per_cell: list[list[FoldResult]] = [[] for _ in all_cells]
    replicates: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for seed in seeds:
        fold_of = assign_folds(dec, n_folds, seed)
        for fold in folds:
            hidden_ids = sorted(f for f, k in fold_of.items() if k == fold)
            hidden = [dec.faces[f] for f in hidden_ids]
            fold_db = workdir / f"fold_s{seed}_f{fold}.sqlite"
            copy_database(base, fold_db)
            hide_faces(fold_db, hidden_ids)
            n_kind: dict[str, int] = defaultdict(int)
            for hf in hidden:
                n_kind[hf.kind] += 1
            replicates.append({"seed": seed, "fold": fold, "hidden_faces": len(hidden), **dict(n_kind)})
            _log(f"[evaluate] seed {seed} fold {fold}: {len(hidden)} hidden faces {dict(n_kind)}")
            with FacesIndexDB(fold_db) as fdb:
                for ci, kw in enumerate(all_cells):
                    diag: dict[str, Any] = {}
                    t0 = time.perf_counter()
                    recompute_active_matching(fdb, diagnostics=diag, **kw)
                    res = score_fold(diag, hidden, dec)
                    res.seconds = time.perf_counter() - t0
                    per_cell[ci].append(res)
                    _log(
                        f"[evaluate]   {BASELINE_NAME if ci == 0 else cell_label(kw):34s} "
                        f"P={res.precision:.4f} R={res.recall:.4f} flagged={res.flagged_labels} "
                        f"({res.seconds:.1f}s)"
                    )
            for suffix in ("", "-wal", "-shm"):
                Path(f"{fold_db}{suffix}").unlink(missing_ok=True)
    elapsed = time.perf_counter() - t_start
    return summarize(all_cells, per_cell, replicates, dec, elapsed)


def _pooled(results: list[FoldResult], attr: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in results:
        for k, v in getattr(r, attr).items():
            acc = out.setdefault(k, [0] * len(v))
            for i, x in enumerate(v):
                acc[i] += x
    return out


def summarize(
    all_cells: list[dict[str, Any]],
    per_cell: list[list[FoldResult]],
    replicates: list[dict[str, Any]],
    dec: Decisions,
    elapsed: float,
) -> dict[str, Any]:
    base = per_cell[0]
    rows: list[dict[str, Any]] = []
    for ci, (kw, results) in enumerate(zip(all_cells, per_cell, strict=True)):
        p_mean, p_se = _mean_se([r.precision for r in results])
        r_mean, r_se = _mean_se([r.recall for r in results])
        dp_mean, dp_se = _mean_se([a.precision - b.precision for a, b in zip(results, base, strict=True)])
        dr_mean, dr_se = _mean_se([a.recall - b.recall for a, b in zip(results, base, strict=True)])
        rows.append(
            {
                "name": BASELINE_NAME if ci == 0 else cell_label(kw),
                "is_baseline": ci == 0,
                "params": cell_params(kw),
                "precision": p_mean,
                "precision_se": p_se,
                "recall": r_mean,
                "recall_se": r_se,
                "d_precision": dp_mean,
                "d_precision_se": dp_se,
                "d_recall": dr_mean,
                "d_recall_se": dr_se,
                "correct": sum(r.correct for r in results),
                "wrong": sum(r.wrong for r in results),
                "unknown": sum(r.unknown for r in results),
                "recall_den": sum(r.recall_den for r in results),
                "background_hidden": sum(r.background_hidden for r in results),
                "background_auto": sum(r.background_auto for r in results),
                "rejections_hidden": sum(r.rejections_hidden for r in results),
                "rejections_reaccepted": sum(r.rejections_reaccepted for r in results),
                "flagged_labels_mean": float(np.mean([r.flagged_labels for r in results])),
                "bands": _pooled(results, "band"),
                "channels": _pooled(results, "channel"),
                "kinds": _pooled(results, "kind"),
                "pairs": _pooled(results, "pairs"),
                "seconds_mean": float(np.mean([r.seconds for r in results])),
                "per_replicate": [{"precision": r.precision, "recall": r.recall} for r in results],
            }
        )
    chosen = select_cell(rows, per_cell)
    for i, row in enumerate(rows):
        row["chosen"] = i == chosen
    return {
        "protocol": "stratified k-fold hidden-face replay (see `python -m kinderpedia_backup.faces.evaluate --help`)",
        "replicates": replicates,
        "decision_faces": len(dec.faces),
        "cells": rows,
        "chosen_index": chosen,
        "elapsed_seconds": round(elapsed, 1),
    }


def select_cell(rows: list[dict[str, Any]], per_cell: list[list[FoldResult]]) -> int | None:
    """Pre-declared selection rule (module doc); returns the row index or None if no cell qualifies."""
    for row in rows[1:]:
        se = row["d_precision_se"] if not math.isnan(row["d_precision_se"]) else 0.0
        row["non_losing"] = (not math.isnan(row["d_precision"])) and row["d_precision"] >= -se
    rows[0]["non_losing"] = True
    cand = [i for i in range(1, len(rows)) if rows[i]["non_losing"]]
    if not cand:
        return None
    best = max(cand, key=lambda i: rows[i]["recall"])
    tied: list[int] = []
    for i in cand:
        d_mean, d_se = _mean_se(
            [b.recall - a.recall for a, b in zip(per_cell[i], per_cell[best], strict=True)]
        )
        if i == best or d_mean <= d_se:
            tied.append(i)
    return min(tied, key=lambda i: simplicity_key(rows[i]["params"]))


def _fmt(mean: float, se: float, pct: bool = True) -> str:
    if math.isnan(mean):
        return "n/a"
    k = 100.0 if pct else 1.0
    return f"{k * mean:.2f} ± {k * se:.2f}"


def _ratio(c: int, w: int) -> str:
    n = c + w
    return f"{100.0 * c / n:.1f}% (n={n})" if n else "- (n=0)"


def render_markdown(summary: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = summary["cells"]
    reps = summary["replicates"]
    out: list[str] = []
    seeds = sorted({r["seed"] for r in reps})
    out.append(
        f"Replicates: {len(reps)} (seeds {', '.join(map(str, seeds))}; folds "
        f"{', '.join(str(r['fold']) for r in reps if r['seed'] == seeds[0])}); "
        f"hidden faces per replicate ≈ {int(np.mean([r['hidden_faces'] for r in reps]))} of "
        f"{summary['decision_faces']} decision faces; runtime {summary['elapsed_seconds']} s."
    )
    out.append("")
    out.append(
        "| cell | precision % | recall % | Δprecision pp (paired) | Δrecall pp (paired) | auto ok / wrong | "
        "bg auto-tagged | hidden rejections re-accepted | flagged labels | verdict |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        verdict = "baseline" if row["is_baseline"] else ("non-losing" if row.get("non_losing") else "LOSS")
        if row.get("chosen"):
            verdict = "**chosen**"
        out.append(
            f"| {row['name']} | {_fmt(row['precision'], row['precision_se'])} | "
            f"{_fmt(row['recall'], row['recall_se'])} | "
            f"{'-' if row['is_baseline'] else _fmt(row['d_precision'], row['d_precision_se'])} | "
            f"{'-' if row['is_baseline'] else _fmt(row['d_recall'], row['d_recall_se'])} | "
            f"{row['correct']} / {row['wrong']} | {row['background_auto']} / {row['background_hidden']} | "
            f"{row['rejections_reaccepted']} / {row['rejections_hidden']} | {row['flagged_labels_mean']:.1f} | "
            f"{verdict} |"
        )
    show = [rows[0]] + [r for r in rows[1:] if r.get("chosen")]
    out.append("")
    out.append("Auto-accept precision per score band and channel (pooled over replicates):")
    out.append("")
    band_names = [_band_name(lo, hi) for lo, hi in SCORE_BANDS]
    out.append("| cell | " + " | ".join(["<0.38", *band_names, *CHANNELS, "batch kind", "cluster kind"]) + " |")
    out.append("|---|" + "---|" * (len(band_names) + len(CHANNELS) + 3))
    for row in show:
        cells = [_ratio(*row["bands"].get(b, [0, 0])) for b in ["<0.38", *band_names]]
        cells += [_ratio(*row["channels"].get(c, [0, 0])) for c in CHANNELS]
        for kind in ("batch", "cluster"):
            k = row["kinds"].get(kind, [0, 0, 0, 0])
            cells.append(f"P {_ratio(k[0], k[1])}, R {100.0 * k[2] / k[3]:.1f}%" if k[3] else _ratio(k[0], k[1]))
        out.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    out.append("")
    out.append(
        "Threshold check — human decisions on hidden (face, person) pairs per model score band "
        "(reject rate; `kNN` = labelled neighbourhood agrees):"
    )
    for row in show:
        out.append("")
        out.append(f"{row['name']}:")
        out.append("")
        out.append("| score band | batch: approved/rejected (reject %) | batch kNN-agree | batch no-kNN | all kinds (reject %) |")
        out.append("|---|---|---|---|---|")
        for lo, hi in THRESHOLD_BANDS:
            b = _band_name(lo, hi)

            def _agg(
                kinds: tuple[str, ...], knns: tuple[str, ...], b: str = b, pairs: dict[str, list[int]] = row["pairs"]
            ) -> tuple[int, int]:
                a = j = 0
                for kd in kinds:
                    for kn in knns:
                        v = pairs.get(f"{kd}|{b}|{kn}", [0, 0])
                        a += v[0]
                        j += v[1]
                return a, j

            def _cell(t: tuple[int, int]) -> str:
                a, j = t
                return f"{a}/{j} ({100.0 * j / (a + j):.1f}%)" if a + j else "0/0"

            out.append(
                f"| {b} | {_cell(_agg(('batch',), ('knn', 'noknn')))} | {_cell(_agg(('batch',), ('knn',)))} | "
                f"{_cell(_agg(('batch',), ('noknn',)))} | {_cell(_agg(('batch', 'cluster'), ('knn', 'noknn')))} |"
            )
    return "\n".join(out) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kinderpedia_backup.faces.evaluate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", type=Path, required=True, help="faces_index.sqlite to replay (copied first).")
    parser.add_argument(
        "--grid",
        default="",
        help='Guard grid, e.g. "min_sim=0.25,0.30;min_faces=4,8;det_penalty=0,0.05". '
        "Default: only the current matcher policy (plus the guard-off baseline, always run).",
    )
    parser.add_argument("--seeds", default="1", help="Comma-separated fold seeds (default: 1).")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (shorthand for --seeds N).")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds k (default: 5).")
    parser.add_argument("--only-folds", default="", help="Run only these fold indices (e.g. 0,1).")
    parser.add_argument("--json", type=Path, default=None, help="Also write the full summary as JSON.")
    parser.add_argument("--markdown", type=Path, default=None, help="Write the markdown report here (default: stdout).")
    parser.add_argument("--workdir", type=Path, default=None, help="Parent directory for temp copies.")
    parser.add_argument(
        "--i-know-this-is-a-copy",
        action="store_true",
        help="Allow a path named faces_index.sqlite inside a Metadata/ directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db: Path = args.db
    if not db.is_file():
        print(f"error: {db} does not exist", file=sys.stderr)
        return 2
    if is_live_archive_path(db) and not args.i_know_this_is_a_copy:
        print(
            "error: refusing to replay a live archive index (Metadata/faces_index.sqlite). Make a copy "
            "(sqlite3 <db> \".backup copy.sqlite\") or pass --i-know-this-is-a-copy.",
            file=sys.stderr,
        )
        return 2
    if args.folds < 2:
        print("error: --folds must be >= 2", file=sys.stderr)
        return 2
    seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",") if s.strip()]
    only = [int(s) for s in args.only_folds.split(",") if s.strip()] or None
    if only is not None and any(not 0 <= f < args.folds for f in only):
        print("error: --only-folds out of range", file=sys.stderr)
        return 2
    cells = parse_grid(args.grid) if args.grid.strip() else [{}]
    with tempfile.TemporaryDirectory(prefix="faces-eval-", dir=args.workdir) as tmp:
        summary = run_replay(
            db, cells=cells, seeds=seeds, n_folds=args.folds, only_folds=only, workdir=Path(tmp), log=sys.stderr
        )
    md = render_markdown(summary)
    if args.markdown is not None:
        args.markdown.write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
