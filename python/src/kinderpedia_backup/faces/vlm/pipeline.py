"""High-level album clothing continuity and gender consistency pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..index_db import FacesIndexDB
from .base import ClothingSignature, VlmProvider

logger = logging.getLogger("kinderpedia.faces.vlm.pipeline")


def analyze_album_continuity(
    db: FacesIndexDB,
    vlm: VlmProvider,
    archive_child: str,
    album_leaf: str,
    backup_dir: Path,
    *,
    target_person_id: int | None = None,
) -> dict[str, Any]:
    """Execute clothing continuity & gender verification across an album.

    1. Inspects confirmed child faces in the album to build/cache appearance signatures.
    2. Verifies candidate and auto_high faces against the appearance signature.
    3. Prunes gender-conflicting labels and upgrades outfit-matched candidates.
    """
    if not vlm.is_available():
        return {
            "ok": False,
            "error": "VLM provider is not available.",
            "album_key": f"{archive_child}/{album_leaf}",
        }

    detail = db.get_album_detail(album_key=f"{archive_child}/{album_leaf}")
    photos = detail.get("photos", [])
    if not photos:
        return {
            "ok": True,
            "album_key": f"{archive_child}/{album_leaf}",
            "verified_count": 0,
            "gender_rejected_count": 0,
            "message": "Album has no photos.",
        }

    # Map photo_id -> rel_path
    photo_rel_paths = {int(p["photo_id"]): str(p["rel_path"]) for p in photos}
    people_map = {int(p["person_id"]): p for p in db.list_people()}

    # Group faces by person and status
    confirmed_by_person: dict[int, list[dict[str, Any]]] = {}
    candidates_by_person: dict[int, list[dict[str, Any]]] = {}

    for p in photos:
        ph_id = int(p["photo_id"])
        for f in p.get("faces", []):
            pid = f.get("person_id")
            st = f.get("status")
            if pid is None:
                continue
            item = {
                "photo_id": ph_id,
                "face_id": int(f["face_id"]),
                "rel_path": photo_rel_paths.get(ph_id, ""),
                "box": [
                    int(f.get("bbox_left", 0)),
                    int(f.get("bbox_top", 0)),
                    int(f.get("bbox_right", 0)),
                    int(f.get("bbox_bottom", 0)),
                ],
                "status": st,
            }
            if st == "confirmed":
                confirmed_by_person.setdefault(pid, []).append(item)
            elif st in ("candidate", "auto_high"):
                candidates_by_person.setdefault(pid, []).append(item)

    # Determine which people to analyze
    target_pids = [target_person_id] if target_person_id is not None else list(confirmed_by_person.keys())
    # Exclude ignore profiles
    target_pids = [pid for pid in target_pids if people_map.get(pid, {}).get("role") != "ignore"]

    verified_count = 0
    gender_rejected_count = 0
    signatures_recorded = 0
    results_detail: list[dict[str, Any]] = []

    for pid in target_pids:
        p_info = people_map.get(pid)
        if not p_info:
            continue
        pname = str(p_info["display_name"])
        target_gender = str(p_info.get("gender") or "unknown")

        # 1. Obtain clothing signature (cached or fresh)
        existing_apps = [a for a in detail.get("appearances", []) if a["person_id"] == pid]
        sig: ClothingSignature | None = None

        if existing_apps:
            app = existing_apps[0]
            sig = ClothingSignature(
                clothing_summary=app["clothing_summary"],
                top_color=app.get("top_color", ""),
                bottom_color=app.get("bottom_color", ""),
                distinctive_features=app.get("distinctive_features", ""),
                apparent_gender=app.get("apparent_gender", "unknown"),
            )
        elif pid in confirmed_by_person:
            # Pick best sample confirmed face
            sample = confirmed_by_person[pid][0]
            img_path = backup_dir / sample["rel_path"]
            if img_path.is_file():
                sig = vlm.extract_clothing(
                    image_path=img_path,
                    face_box=sample["box"],
                    person_name=pname,
                    target_gender=target_gender,
                )
                if sig:
                    db.record_album_appearance(
                        archive_child=archive_child,
                        album_leaf=album_leaf,
                        person_id=pid,
                        sample_photo_id=sample["photo_id"],
                        sample_face_id=sample["face_id"],
                        clothing_summary=sig.clothing_summary,
                        top_color=sig.top_color,
                        bottom_color=sig.bottom_color,
                        distinctive_features=sig.distinctive_features,
                        apparent_gender=sig.apparent_gender,
                        verified_count=1,
                    )
                    signatures_recorded += 1

        if not sig:
            continue

        # 2. Check candidate faces for this person
        person_cands = candidates_by_person.get(pid, [])
        if not person_cands:
            continue

        cand_payload = [
            {
                "path": str(backup_dir / c["rel_path"]),
                "box": c["box"],
                "face_id": c["face_id"],
                "photo_id": c["photo_id"],
            }
            for c in person_cands
            if (backup_dir / c["rel_path"]).is_file()
        ]

        verifications = vlm.batch_verify_album_candidates(
            candidates=cand_payload,
            target_person_name=pname,
            target_gender=target_gender,
            signature=sig,
        )

        for v in verifications:
            fid = v.get("face_id")
            if not fid:
                continue
            is_match = bool(v.get("clothing_match", False))
            has_gender_conflict = bool(v.get("gender_conflict", False))
            conf = float(v.get("confidence", 0.0))
            notes = str(v.get("notes", ""))

            # Find matching candidate photo_id
            c_entry = next((c for c in cand_payload if c["face_id"] == fid), None)
            target_photo_id: int | None = int(c_entry["photo_id"]) if c_entry is not None else None

            res_item = {
                "face_id": fid,
                "photo_id": target_photo_id,
                "person_id": pid,
                "person_name": pname,
                "clothing_match": is_match,
                "gender_conflict": has_gender_conflict,
                "confidence": conf,
                "notes": notes,
            }
            results_detail.append(res_item)

            if has_gender_conflict and target_photo_id is not None:
                # Reject/unlink candidate label
                db.tag_single_face(
                    photo_id=target_photo_id,
                    face_id=fid,
                    person_id=pid,
                    status="rejected",
                    source="reassigned",
                )
                gender_rejected_count += 1
            elif is_match and conf >= 0.75 and target_photo_id is not None:
                # Upgrade to confirmed / auto_high with album cohesion source
                db.tag_single_face(
                    photo_id=target_photo_id,
                    face_id=fid,
                    person_id=pid,
                    status="confirmed",
                    source="album_cohesion",
                )
                verified_count += 1

    return {
        "ok": True,
        "album_key": f"{archive_child}/{album_leaf}",
        "signatures_recorded": signatures_recorded,
        "verified_count": verified_count,
        "gender_rejected_count": gender_rejected_count,
        "results": results_detail,
    }
