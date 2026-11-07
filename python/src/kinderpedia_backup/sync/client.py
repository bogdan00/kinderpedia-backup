"""Kinderpedia cloud API & Vimeo download client."""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from ..console import ffmpeg_install_hint, find_ffmpeg, log
from ..models import ChildContext
from ..security import default_state_dir, redact_secrets, replace_atomic, scrub_metadata, write_json_atomic
from .rate_limit import AdaptiveRateController

USER_GATEWAY_BASE = "https://usergateway-services.kinderpedia.co"
WEB_API_BASE = "https://apiv20.kinderpedia.co"
VIMEO_API_BASE = "https://api.vimeo.com"

# Public client identifier strings sent by Kinderpedia's web & mobile apps.
# Overridable via environment variables if Kinderpedia bumps client versions.
DEFAULT_MOBILE_CLIENT_ID = "M01bxi3l4em|v3.2"
DEFAULT_WEB_API_KEY = "Web01Pari3l4em|v1.01"


class TwoFactorRequiredError(RuntimeError):
    """Raised when Kinderpedia requires a 2FA confirmation code during login."""


class KinderpediaClient:
    """Authenticated HTTP client for Kinderpedia parent accounts and Vimeo embeds."""

    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        persist_session: bool = True,
        pool_size: int = 32,
        initial_interval: float = 3.15,
        ffmpeg_bin: str | None = None,
    ) -> None:
        self.state_dir = (state_dir or default_state_dir()).resolve()
        self.persist_session = persist_session
        self.session_file = self.state_dir / "session.json"
        self.ffmpeg_bin = ffmpeg_bin if ffmpeg_bin is not None else find_ffmpeg()
        self._warned_missing_ffmpeg = False

        self.web_api_key = os.environ.get("KINDERPEDIA_WEB_API_KEY", DEFAULT_WEB_API_KEY)
        self.mobile_client_id = os.environ.get("KINDERPEDIA_MOBILE_CLIENT_ID", DEFAULT_MOBILE_CLIENT_ID)

        self.http = requests.Session()
        adapter = HTTPAdapter(pool_connections=24, pool_maxsize=pool_size, max_retries=0)
        self.http.mount("https://", adapter)
        self.http.mount("http://", adapter)

        self.token: str = ""
        self.refresh_token: str = ""
        self.api_key: str = ""
        self.monolith_session: str = ""
        self.user_id: int | None = None
        self.vimeo_tokens: list[str] = []
        self.device_uuid: str = str(uuid.uuid4())

        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self.url_cache: dict[str, Path] = {}
        self.rate_ctrl = AdaptiveRateController(
            self.state_dir / "adaptive_rate.json" if persist_session else None,
            initial_interval=initial_interval,
        )

    def load_session(self, legacy_session_file: Path | None = None) -> bool:
        """Load persisted session from private state_dir (or migrate from legacy path)."""
        candidates = [self.session_file]
        if legacy_session_file is not None and legacy_session_file != self.session_file:
            candidates.append(legacy_session_file)

        for cand in candidates:
            if not cand.is_file():
                continue
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                tok = data.get("token", "")
                if not tok:
                    continue
                self.token = tok
                self.refresh_token = data.get("refresh_token", "")
                self.api_key = data.get("api_key", "")
                self.monolith_session = data.get("monolith_session", "")
                self.user_id = data.get("user_id")
                self.vimeo_tokens = data.get("vimeo_tokens", [])
                self.device_uuid = data.get("device_uuid") or self.device_uuid
                if cand != self.session_file and self.persist_session:
                    self.save_session()
                    log(
                        f"[Security] Imported session into private state directory {self.session_file}. "
                        f"Consider deleting legacy {cand} if your backup directory is cloud-synced."
                    )
                return True
            except (OSError, ValueError) as exc:
                log(f"[WARN] Skipping unreadable session file {cand}: {redact_secrets(str(exc))}")
        return False

    def save_session(self) -> None:
        if not self.persist_session:
            return
        data = {
            "token": self.token,
            "refresh_token": self.refresh_token,
            "api_key": self.api_key,
            "monolith_session": self.monolith_session,
            "user_id": self.user_id,
            "vimeo_tokens": self.vimeo_tokens,
            "device_uuid": self.device_uuid,
            "saved_at": datetime.now().isoformat(),
        }
        write_json_atomic(self.session_file, data, private=True)

    def _http_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        form_body: dict[str, str] | None = None,
        retry_auth: bool = True,
        retries: int = 10,
    ) -> dict[str, Any] | list[Any]:
        hdrs = {
            "User-Agent": "kinderpedia-api",
            "Accept": "application/json, application/ld+json",
        }
        if headers:
            hdrs.update(headers)

        safe_url = redact_secrets(url)
        is_web_api = "apiv20.kinderpedia.co" in url
        last_err: Exception | None = None

        for attempt in range(retries):
            if is_web_api:
                self.rate_ctrl.acquire()
            used_tok = self.token
            if "Authorization" in hdrs and hdrs["Authorization"].startswith("Bearer ") and retry_auth:
                hdrs["Authorization"] = f"Bearer {used_tok}"
            try:
                resp = self.http.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=hdrs,
                    json=json_body,
                    data=form_body,
                    timeout=(15, 45),
                )
                if (
                    resp.status_code == 401
                    and retry_auth
                    and self.refresh_token
                    and self.refresh_jwt(failed_token=used_tok)
                ):
                    hdrs["Authorization"] = f"Bearer {self.token}"
                    return self._http_json(
                        method,
                        url,
                        params=params,
                        headers=hdrs,
                        json_body=json_body,
                        form_body=form_body,
                        retry_auth=False,
                        retries=retries,
                    )
                if resp.status_code == 429 and attempt + 1 < retries:
                    retry_after = resp.headers.get("Retry-After", "")
                    if is_web_api:
                        pause_s = self.rate_ctrl.on_rate_limit(attempt, retry_after)
                        snap = self.rate_ctrl.snapshot()
                        log(
                            f"  [AdaptiveRate] HTTP 429 on apiv20; exp-backoff {pause_s:.1f}s "
                            f"(new pace={snap.api_interval_s:.2f}s/req, floor={snap.soft_floor_s:.2f}s, "
                            f"attempt {attempt + 1}/{retries})"
                        )
                    else:
                        pause_s = min(120.0, 10.0 * (2**attempt)) + random.uniform(0.5, 3.0)
                        log(
                            f"  [RateLimit] HTTP 429 on {safe_url}; sleeping {pause_s:.1f}s "
                            f"(attempt {attempt + 1}/{retries})"
                        )
                        time.sleep(pause_s)
                    last_err = RuntimeError(f"HTTP 429 on {safe_url}")
                    continue
                if resp.status_code in (500, 502, 503, 504) and attempt + 1 < retries:
                    pause_s = min(90.0, 2.5 * (2**attempt)) + random.uniform(0.5, 2.5)
                    log(f"  [Transient] HTTP {resp.status_code} on {safe_url}; retrying in {pause_s:.1f}s...")
                    time.sleep(pause_s)
                    last_err = RuntimeError(f"HTTP {resp.status_code} on {safe_url}")
                    continue
                if resp.status_code >= 400:
                    snippet = redact_secrets(resp.text[:350])
                    raise RuntimeError(f"HTTP {resp.status_code} on {safe_url}: {snippet}")
                if is_web_api:
                    self.rate_ctrl.on_success()
                raw = resp.text
                return resp.json() if raw.strip() else {}
            except requests.RequestException as exc:
                sanitized_exc = RuntimeError(redact_secrets(str(exc)))
                last_err = sanitized_exc
                if attempt + 1 < retries:
                    pause_s = min(90.0, 2.5 * (2**attempt)) + random.uniform(0.5, 2.5)
                    time.sleep(pause_s)
                    continue
                raise sanitized_exc from None
        raise RuntimeError(f"Request failed for {safe_url}: {redact_secrets(str(last_err))}")

    def login(self, email: str, password: str, confirmation_code: str | None = None) -> None:
        payload: dict[str, str] = {
            "email": email,
            "password": password,
            "uuid": self.device_uuid,
        }
        if confirmation_code:
            payload["confirmationCode"] = confirmation_code

        headers = {
            "user-type": "parent",
            "App-Version": "5.4.1",
            "Platform": "Android-release",
            "Accept": "application/ld+json",
        }
        try:
            resp = self._http_json(
                "POST",
                f"{USER_GATEWAY_BASE}/api/login",
                headers=headers,
                json_body=payload,
                retry_auth=False,
            )
        except RuntimeError as err:
            msg_low = str(err).lower()
            if not confirmation_code and any(k in msg_low for k in ("confirmationcode", "2fa", "two_factor", "otp")):
                raise TwoFactorRequiredError("Account requires a 2FA confirmation code.") from err
            raise

        if not isinstance(resp, dict):
            raise RuntimeError("Unexpected non-JSON login response from Kinderpedia.")

        token = resp.get("token") or resp.get("jwToken")
        if not token:
            keys_low = " ".join(str(k).lower() for k in resp) + " " + str(resp.get("message", "")).lower()
            if not confirmation_code and any(k in keys_low for k in ("code", "2fa", "confirm", "otp", "twofactor")):
                raise TwoFactorRequiredError("Account requires a 2FA confirmation code.")
            raise RuntimeError(f"Login failed (no token returned): {redact_secrets(str(resp.get('message') or list(resp.keys())))}")

        self.token = str(token)
        self.refresh_token = str(
            resp.get("refresh_token")
            or resp.get("refreshToken")
            or resp.get("refreshJwToken")
            or ""
        )
        self.api_key = str(resp.get("apiKey") or self.api_key)
        self.monolith_session = str(resp.get("monolithSession") or self.monolith_session)
        self.save_session()
        log(f"Authenticated successfully as {email}")

    def refresh_jwt(self, failed_token: str | None = None) -> bool:
        with self._lock:
            if failed_token and self.token and self.token != failed_token:
                return True
            if not self.refresh_token:
                return False
            try:
                resp = self._http_json(
                    "GET",
                    f"{USER_GATEWAY_BASE}/api/token/refresh",
                    params={"refreshToken": self.refresh_token},
                    headers={"user-type": "parent", "Accept": "application/ld+json"},
                    retry_auth=False,
                )
                new_tok = resp.get("token") or resp.get("jwToken") if isinstance(resp, dict) else None
                if isinstance(resp, dict) and new_tok:
                    self.token = str(new_tok)
                    new_ref = (
                        resp.get("refresh_token")
                        or resp.get("refreshToken")
                        or resp.get("refreshJwToken")
                    )
                    if new_ref:
                        self.refresh_token = str(new_ref)
                    self.save_session()
                    log("Refreshed Kinderpedia session token.")
                    return True
            except Exception as exc:
                log(f"Session token refresh failed: {redact_secrets(str(exc))}")
            return False

    def get_me_and_children(self) -> tuple[dict[str, Any], list[ChildContext]]:
        me = self._http_json(
            "GET",
            f"{USER_GATEWAY_BASE}/api/users/me",
            headers={
                "Authorization": f"Bearer {self.token}",
                "user-type": "parent",
                "Accept": "application/ld+json",
            },
        )
        if not isinstance(me, dict):
            raise RuntimeError("Unexpected /api/users/me response type.")

        user_obj = me.get("user") or {}
        if isinstance(user_obj, dict):
            if user_obj.get("id"):
                self.user_id = int(user_obj["id"])
            if user_obj.get("apiKey"):
                self.api_key = str(user_obj["apiKey"])

        app_keys = me.get("appKeys") or me.get("app_keys") or {}
        if isinstance(app_keys, dict):
            vjwt = app_keys.get("vimeoJwt") or app_keys.get("vimeo_jwt") or {}
            if isinstance(vjwt, dict):
                tokens: list[str] = []
                if isinstance(vjwt.get("school_account"), str) and vjwt["school_account"]:
                    tokens.append(vjwt["school_account"])
                for key in ("app_ids", "appIds", "rotating_accounts", "rotatingAccounts"):
                    items = vjwt.get(key) or []
                    if isinstance(items, list):
                        for it in items:
                            if isinstance(it, dict) and it.get("token"):
                                tokens.append(str(it["token"]))
                            elif isinstance(it, str) and it:
                                tokens.append(it)
                if tokens:
                    self.vimeo_tokens = list(dict.fromkeys(tokens))
        self.save_session()

        children_contexts: list[ChildContext] = []
        seen_pairs: set[tuple[int, int]] = set()

        try:
            fam = self._http_json(
                "GET",
                f"{USER_GATEWAY_BASE}/api/families/my-members",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "user-type": "parent",
                    "Accept": "application/ld+json",
                },
            )
            self._extract_children_from_gateway(fam, me, children_contexts, seen_pairs)
        except Exception as exc:
            log(f"Note: families/my-members lookup returned: {redact_secrets(str(exc))}")

        if not children_contexts and self.user_id and self.api_key:
            try:
                sess = self._http_json(
                    "POST",
                    f"{WEB_API_BASE}/api/check-session",
                    form_body={
                        "client_id": self.mobile_client_id,
                        "user_id": str(self.user_id),
                        "api_key": self.api_key,
                    },
                )
                self._extract_children_from_monolith(sess, children_contexts, seen_pairs)
            except Exception as exc:
                log(f"Note: monolith check-session lookup returned: {redact_secrets(str(exc))}")

        child_ids = me.get("childrenIds") or []
        kg_ids = me.get("kindergartenIds") or []
        if child_ids and kg_ids and not children_contexts:
            for kg_id in kg_ids:
                for cid in child_ids:
                    pair = (int(cid), int(kg_id))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        children_contexts.append(
                            ChildContext(
                                child_id=int(cid),
                                kindergarten_id=int(kg_id),
                                name=f"child_{cid}",
                            )
                        )

        return me, children_contexts

    def _extract_children_from_monolith(
        self,
        sess: dict[str, Any] | list[Any],
        out: list[ChildContext],
        seen: set[tuple[int, int]],
    ) -> None:
        if not isinstance(sess, dict):
            return
        candidates: list[dict[str, Any]] = [sess]
        for key in ("response", "result", "data"):
            sub = sess.get(key)
            if isinstance(sub, dict):
                candidates.append(sub)
        for obj in candidates:
            raw_user = obj.get("user")
            user: dict[str, Any] = raw_user if isinstance(raw_user, dict) else obj
            children = user.get("children") or obj.get("children") or []
            if not isinstance(children, list):
                continue
            for ch in children:
                if not isinstance(ch, dict):
                    continue
                cid_raw = ch.get("id") or ch.get("user_id") or ch.get("child_id")
                if not cid_raw:
                    continue
                cid = int(cid_raw)
                given = str(ch.get("first_name") or ch.get("firstName") or "").strip()
                fname = f"{given} {ch.get('last_name') or ch.get('lastName') or ''}".strip()
                if not fname:
                    fname = str(ch.get("name") or f"child_{cid}")

                kgs: list[tuple[dict[str, Any], Any]] = []
                if isinstance(ch.get("kindergarten"), dict):
                    kgs.append((ch["kindergarten"], ch.get("group") or ch.get("main_group") or {}))
                if isinstance(ch.get("kindergartens"), list):
                    for k in ch["kindergartens"]:
                        if isinstance(k, dict):
                            kgs.append((k, k.get("group") or {}))
                if ch.get("kindergarten_id"):
                    kgs.append(({"id": ch["kindergarten_id"], "name": ch.get("kindergarten_name", "")}, {}))
                if isinstance(ch.get("past_kindergartens"), list):
                    for pk in ch["past_kindergartens"]:
                        if isinstance(pk, dict):
                            kgs.append((pk, pk.get("group") or {}))

                for kg_obj, grp_obj in kgs:
                    kid_raw = kg_obj.get("id") or kg_obj.get("kindergarten_id")
                    if not kid_raw:
                        continue
                    kid = int(kid_raw)
                    if (cid, kid) not in seen:
                        seen.add((cid, kid))
                        grp_name = str(grp_obj.get("name", "")) if isinstance(grp_obj, dict) else str(grp_obj or "")
                        out.append(
                            ChildContext(
                                child_id=cid,
                                kindergarten_id=kid,
                                name=fname,
                                first_name=given,
                                group_name=grp_name,
                                kindergarten_name=str(kg_obj.get("name") or ""),
                                avatar_url=str(ch.get("photo") or ""),
                            )
                        )

    def _extract_children_from_gateway(
        self,
        fam: dict[str, Any] | list[Any],
        me: dict[str, Any],
        out: list[ChildContext],
        seen: set[tuple[int, int]],
    ) -> None:
        members: list[Any] = []
        if isinstance(fam, dict):
            raw_members = fam.get("hydra:member") or fam.get("members") or fam.get("children") or []
            if isinstance(raw_members, dict):
                members = list(raw_members.values())
            elif isinstance(raw_members, list):
                members = raw_members
        elif isinstance(fam, list):
            members = fam

        default_kgs = me.get("kindergartenIds") or []
        allowed_child_ids = {int(x) for x in (me.get("childrenIds") or []) if str(x).isdigit()}

        for m in members:
            if not isinstance(m, dict):
                continue
            raw_prof = m.get("profile")
            prof: dict[str, Any] = raw_prof if isinstance(raw_prof, dict) else m
            role_or_type = str(
                m.get("entityType")
                or prof.get("type")
                or m.get("type")
                or m.get("userType")
                or ""
            ).lower()
            cid_raw = prof.get("id") or m.get("entityId") or m.get("userId") or m.get("id")
            if not cid_raw:
                continue
            cid = int(cid_raw)
            if cid in allowed_child_ids or role_or_type in ("child", "student", "copil", "elev"):
                given = str(prof.get("firstName") or prof.get("first_name") or "").strip()
                fname = f"{given} {prof.get('lastName') or prof.get('last_name') or ''}".strip()
                raw_mg = prof.get("mainGroup")
                mg: dict[str, Any] = raw_mg if isinstance(raw_mg, dict) else {}
                raw_kg = mg.get("kindergarten")
                kg_obj: dict[str, Any] = raw_kg if isinstance(raw_kg, dict) else {}
                kg_name = str(kg_obj.get("name") or "")
                grp_name = str(mg.get("name") or "")
                avatar_url = str(prof.get("photo") or "")

                kids_kgs: list[int] = []
                if mg.get("kindergartenId"):
                    kids_kgs.append(int(mg["kindergartenId"]))
                if kg_obj.get("id"):
                    kids_kgs.append(int(kg_obj["id"]))
                if prof.get("lastKindergarten"):
                    kids_kgs.append(int(prof["lastKindergarten"]))
                for k in default_kgs:
                    kids_kgs.append(int(k))

                for kid in dict.fromkeys(kids_kgs):
                    if (cid, kid) not in seen:
                        seen.add((cid, kid))
                        out.append(
                            ChildContext(
                                child_id=cid,
                                kindergarten_id=kid,
                                name=fname or f"child_{cid}",
                                first_name=given,
                                group_name=grp_name,
                                kindergarten_name=kg_name,
                                avatar_url=avatar_url,
                            )
                        )

    def web_api_headers(self, kindergarten_id: int, child_id: int) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Requested-With": "XMLHttpRequest",
            "x-api-key": self.web_api_key,
            "X-KINDERGARTEN-ID": str(kindergarten_id),
            "X-CHILD-ID": str(child_id),
            "User-Agent": "kinderpedia-api",
        }

    @staticmethod
    def _unwrap_web_api(data: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
        if isinstance(data, dict):
            if "response" in data and data["response"] is not None:
                return data["response"]  # type: ignore[no-any-return]
            if "result" in data and data["result"] is not None:
                return data["result"]  # type: ignore[no-any-return]
        return data

    def list_albums(
        self,
        kindergarten_id: int,
        child_id: int,
        *,
        partial_file: Path | None = None,
    ) -> list[dict[str, Any]]:
        albums: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        page = 1

        if partial_file is not None and partial_file.exists():
            with contextlib.suppress(OSError, ValueError):
                pdata = json.loads(partial_file.read_text(encoding="utf-8"))
                if isinstance(pdata, dict) and isinstance(pdata.get("items"), list) and pdata["items"]:
                    for item in pdata["items"]:
                        if isinstance(item, dict) and "id" in item:
                            aid_init = int(item["id"])
                            if aid_init not in seen_ids:
                                seen_ids.add(aid_init)
                                albums.append(item)
                    if pdata.get("complete") is True:
                        return albums
                    page = max(1, int(pdata.get("next_page") or 1))
                    log(f"  [Resume] Resuming album cataloging from page {page} ({len(albums)} cached)...")

        while True:
            raw = self._http_json(
                "GET",
                f"{WEB_API_BASE}/web-api/data/gallery",
                params={
                    "mode": "albums",
                    "page": str(page),
                    "items_per_page": "30",
                    "order_by": "updated_at",
                    "order_type": "DESC",
                },
                headers=self.web_api_headers(kindergarten_id, child_id),
            )
            payload = self._unwrap_web_api(raw)
            page_items: list[Any] = []
            if isinstance(payload, dict):
                page_items = payload.get("albums") or payload.get("items") or []
            elif isinstance(payload, list):
                page_items = payload

            if not page_items:
                break

            new_count = 0
            for a in page_items:
                if not isinstance(a, dict) or "id" not in a:
                    continue
                aid = int(a["id"])
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    albums.append(a)
                    new_count += 1
            log(f"  [Albums] page {page}: +{new_count} albums (total so far: {len(albums)})")

            if partial_file is not None:
                total_photos_est = sum(int(x.get("count_images") or 0) for x in albums)
                write_json_atomic(
                    partial_file,
                    {
                        "complete": False,
                        "next_page": page + 1,
                        "total_albums": len(albums),
                        "total_photos_est": total_photos_est,
                        "updated_at": datetime.now().isoformat(),
                        "items": scrub_metadata(albums),
                    },
                )

            if new_count == 0 or len(page_items) < 30:
                break
            page += 1

        if partial_file is not None:
            total_photos_est = sum(int(x.get("count_images") or 0) for x in albums)
            write_json_atomic(
                partial_file,
                {
                    "complete": True,
                    "next_page": page,
                    "total_albums": len(albums),
                    "total_photos_est": total_photos_est,
                    "updated_at": datetime.now().isoformat(),
                    "items": scrub_metadata(albums),
                },
            )
        return albums

    def get_album_images(
        self, kindergarten_id: int, child_id: int, album_id: int, expected_count: int = 0
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        hdrs = self.web_api_headers(kindergarten_id, child_id)
        album_meta: dict[str, Any] = {}
        images: list[dict[str, Any]] = []

        try:
            raw_imgs = self._http_json(
                "GET",
                f"{WEB_API_BASE}/web-api/data/images-plugin/{album_id}",
                params={"entity_type": "media"},
                headers=hdrs,
            )
            unwrapped = self._unwrap_web_api(raw_imgs)
            cand = unwrapped.get("images", []) if isinstance(unwrapped, dict) else []
            if isinstance(cand, list):
                images = [img for img in cand if isinstance(img, dict)]
        except Exception as exc:
            log(f"  [Warn] images-plugin/{album_id}: {redact_secrets(str(exc))}")

        if not images and expected_count > 0:
            try:
                raw_det = self._http_json(
                    "GET",
                    f"{WEB_API_BASE}/web-api/data/gallery/{album_id}",
                    headers=hdrs,
                )
                det = self._unwrap_web_api(raw_det)
                if isinstance(det, dict):
                    if isinstance(det.get("album"), dict):
                        album_meta = det["album"]
                    if isinstance(det.get("images"), list) and len(det["images"]) > len(images):
                        images = [img for img in det["images"] if isinstance(img, dict)]
            except Exception as exc:
                log(f"  [Warn] gallery/{album_id} details: {redact_secrets(str(exc))}")

        return album_meta, images

    def list_videos(
        self,
        kindergarten_id: int,
        child_id: int,
        *,
        partial_file: Path | None = None,
    ) -> list[dict[str, Any]]:
        videos: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        page = 1

        if partial_file is not None and partial_file.exists():
            with contextlib.suppress(OSError, ValueError):
                pdata = json.loads(partial_file.read_text(encoding="utf-8"))
                if isinstance(pdata, dict) and isinstance(pdata.get("items"), list) and pdata["items"]:
                    for item in pdata["items"]:
                        if isinstance(item, dict) and "id" in item:
                            vid_init = int(item["id"])
                            if vid_init not in seen_ids:
                                seen_ids.add(vid_init)
                                videos.append(item)
                    if pdata.get("complete") is True:
                        return videos
                    page = max(1, int(pdata.get("next_page") or 1))
                    log(f"  [Resume] Resuming video cataloging from page {page} ({len(videos)} cached)...")

        while True:
            raw = self._http_json(
                "GET",
                f"{WEB_API_BASE}/web-api/data/gallery",
                params={
                    "mode": "videos",
                    "page": str(page),
                    "items_per_page": "30",
                    "order_by": "dateadd",
                    "order_type": "DESC",
                },
                headers=self.web_api_headers(kindergarten_id, child_id),
            )
            payload = self._unwrap_web_api(raw)
            page_items: list[Any] = []
            if isinstance(payload, dict):
                page_items = payload.get("videos") or payload.get("albums") or payload.get("items") or []
            elif isinstance(payload, list):
                page_items = payload

            if not page_items:
                break

            new_count = 0
            for v in page_items:
                if not isinstance(v, dict) or "id" not in v:
                    continue
                vid = int(v["id"])
                if vid not in seen_ids:
                    seen_ids.add(vid)
                    videos.append(v)
                    new_count += 1
            log(f"  [Videos] page {page}: +{new_count} videos (total so far: {len(videos)})")

            if partial_file is not None:
                write_json_atomic(
                    partial_file,
                    {
                        "complete": False,
                        "next_page": page + 1,
                        "total_videos": len(videos),
                        "updated_at": datetime.now().isoformat(),
                        "items": scrub_metadata(videos),
                    },
                )

            if new_count == 0 or len(page_items) < 30:
                break
            page += 1

        if partial_file is not None:
            write_json_atomic(
                partial_file,
                {
                    "complete": True,
                    "next_page": page,
                    "total_videos": len(videos),
                    "updated_at": datetime.now().isoformat(),
                    "items": scrub_metadata(videos),
                },
            )
        return videos

    def get_video_details(self, kindergarten_id: int, child_id: int, video_id: int) -> dict[str, Any]:
        raw = self._http_json(
            "GET",
            f"{WEB_API_BASE}/web-api/data/gallery/{video_id}",
            params={"mode": "video"},
            headers=self.web_api_headers(kindergarten_id, child_id),
        )
        payload = self._unwrap_web_api(raw)
        if isinstance(payload, dict):
            if isinstance(payload.get("video"), dict):
                return payload["video"]  # type: ignore[no-any-return]
            return payload
        return {}

    def list_documents(
        self,
        kindergarten_id: int,
        child_id: int,
        *,
        partial_file: Path | None = None,
    ) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        completed_entities: set[str] = set()

        if partial_file is not None and partial_file.exists():
            with contextlib.suppress(OSError, ValueError):
                pdata = json.loads(partial_file.read_text(encoding="utf-8"))
                if isinstance(pdata, dict) and isinstance(pdata.get("items"), list):
                    for item in pdata["items"]:
                        if isinstance(item, dict):
                            did_init = str(item.get("id") or item.get("file_id") or item.get("url") or "")
                            if did_init and did_init not in seen_ids:
                                seen_ids.add(did_init)
                                docs.append(item)
                    if pdata.get("complete") is True:
                        return docs
                    if isinstance(pdata.get("completed_entities"), list):
                        completed_entities = {str(e) for e in pdata["completed_entities"]}

        top_entities: list[tuple[str, int]] = [
            ("documents", 0),
            ("child-portfolio-folders", child_id),
            ("children_public", child_id),
        ]

        for root_etype, root_eid in top_entities:
            if root_etype in completed_entities:
                continue
            visited_folders: set[tuple[str, int]] = set()
            queue: list[tuple[str, int]] = [(root_etype, root_eid)]

            while queue:
                entity_type, entity_id = queue.pop(0)
                if (entity_type, entity_id) in visited_folders:
                    continue
                visited_folders.add((entity_type, entity_id))

                page = 1
                while True:
                    try:
                        raw = self._http_json(
                            "GET",
                            f"{WEB_API_BASE}/web-api/data/documents",
                            params={
                                "entity_type": entity_type,
                                "entity_id": str(entity_id),
                                "page": str(page),
                                "items_per_page": "30",
                                "order_by": "created_date",
                                "order_type": "desc",
                            },
                            headers=self.web_api_headers(kindergarten_id, child_id),
                        )
                    except Exception:
                        break
                    payload = self._unwrap_web_api(raw)
                    if not isinstance(payload, dict):
                        break

                    subfolders = payload.get("folders") or []
                    if isinstance(subfolders, list):
                        for fl in subfolders:
                            if isinstance(fl, dict) and fl.get("id") is not None:
                                sub_etype = "documents" if entity_type == "documents" else entity_type
                                queue.append((sub_etype, int(fl["id"])))

                    items = payload.get("documents") or payload.get("files") or payload.get("items") or []
                    if not isinstance(items, list) or not items:
                        break

                    new_count = 0
                    for d in items:
                        if not isinstance(d, dict):
                            continue
                        if (d.get("type") == "folder" or d.get("is_folder")) and not d.get("url"):
                            if d.get("id") is not None:
                                queue.append((entity_type, int(d["id"])))
                            continue
                        did = str(d.get("id") or d.get("file_id") or d.get("url") or "")
                        if did and did not in seen_ids:
                            seen_ids.add(did)
                            docs.append(d)
                            new_count += 1
                    if new_count == 0 or len(items) < 30:
                        break
                    page += 1

            completed_entities.add(root_etype)
            if partial_file is not None:
                write_json_atomic(
                    partial_file,
                    {
                        "complete": False,
                        "completed_entities": sorted(completed_entities),
                        "total_docs": len(docs),
                        "updated_at": datetime.now().isoformat(),
                        "items": scrub_metadata(docs),
                    },
                )

        if partial_file is not None:
            write_json_atomic(
                partial_file,
                {
                    "complete": True,
                    "completed_entities": sorted(completed_entities),
                    "total_docs": len(docs),
                    "updated_at": datetime.now().isoformat(),
                    "items": scrub_metadata(docs),
                },
            )
        return docs

    def register_cached_url(self, url: str, path: Path) -> None:
        if not url:
            return
        with self._cache_lock:
            if url not in self.url_cache:
                self.url_cache[url] = path

    def download_binary(
        self,
        url: str,
        dest: Path,
        extra_headers: dict[str, str] | None = None,
        retries: int = 6,
    ) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_lock:
            cached = self.url_cache.get(url)
        if cached and cached != dest and cached.exists() and cached.stat().st_size > 0:
            shutil.copy2(cached, dest)
            return dest.stat().st_size

        safe_url = redact_secrets(url)
        tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.{threading.get_ident()}.part")
        hdrs = {"User-Agent": "Mozilla/5.0 (Linux; Android 17) KinderpediaBackup/1.0"}
        if extra_headers:
            hdrs.update(extra_headers)
        for attempt in range(retries):
            try:
                with self.http.get(url, headers=hdrs, stream=True, timeout=(15, 60)) as resp:
                    if resp.status_code in (401, 403) and "Authorization" not in hdrs and "kinderpedia.co" in url:
                        hdrs["Authorization"] = f"Bearer {self.token}"
                        continue
                    if resp.status_code in (404, 410):
                        raise FileNotFoundError(f"HTTP {resp.status_code} for {safe_url}")
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                        pause_s = min(90.0, 2.0 * (2**attempt)) + random.uniform(0.5, 2.5)
                        time.sleep(pause_s)
                        continue
                    if resp.status_code >= 400:
                        raise RuntimeError(f"HTTP {resp.status_code} downloading {safe_url}")
                    with open(tmp, "wb") as out:
                        for chunk in resp.iter_content(chunk_size=256 * 1024):
                            if chunk:
                                out.write(chunk)
                sz = tmp.stat().st_size
                if sz == 0:
                    raise RuntimeError(f"Downloaded 0 bytes from {safe_url}")
                replace_atomic(tmp, dest)
                self.register_cached_url(url, dest)
                return sz
            except FileNotFoundError:
                with contextlib.suppress(OSError):
                    if tmp.exists():
                        tmp.unlink()
                raise
            except Exception as exc:
                with contextlib.suppress(OSError):
                    if tmp.exists():
                        tmp.unlink()
                if attempt + 1 < retries:
                    pause_s = min(90.0, 2.0 * (2**attempt)) + random.uniform(0.5, 2.5)
                    time.sleep(pause_s)
                    continue
                raise RuntimeError(redact_secrets(str(exc))) from None
        return 0

    def download_with_fallbacks(self, urls: list[str], dest: Path) -> int:
        last_err: Exception | None = None
        for u in urls:
            with self._cache_lock:
                cached = self.url_cache.get(u)
            if cached and cached != dest and cached.exists() and cached.stat().st_size > 0:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached, dest)
                return dest.stat().st_size
        for u in urls:
            try:
                sz = self.download_binary(u, dest)
                for alias in urls:
                    self.register_cached_url(alias, dest)
                return sz
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(f"All candidate URLs failed for {dest.name}: {redact_secrets(str(last_err))}")

    def download_vimeo_video(self, video_url: str, dest: Path, retries: int = 5) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_lock:
            cached = self.url_cache.get(video_url)
        if cached and cached != dest and cached.exists() and cached.stat().st_size > 0:
            shutil.copy2(cached, dest)
            return dest.stat().st_size

        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                return self._download_vimeo_once(video_url, dest)
            except FileNotFoundError:
                raise
            except Exception as exc:
                last_err = exc
                if attempt + 1 < retries:
                    pause_s = min(90.0, 3.0 * (2**attempt)) + random.uniform(1.0, 3.5)
                    log(f"    [VimeoRetry] attempt {attempt + 1}/{retries} in {pause_s:.1f}s ({redact_secrets(str(exc))})")
                    time.sleep(pause_s)
                    continue
                raise
        raise RuntimeError(f"Vimeo download failed for {redact_secrets(video_url)}: {redact_secrets(str(last_err))}")

    def _download_vimeo_once(self, video_url: str, dest: Path) -> int:
        m_id = re.search(r"vimeo\.com/(?:video/)?(\d+)", video_url)
        vimeo_id = m_id.group(1) if m_id else None

        if vimeo_id and self.vimeo_tokens:
            for vtok in self.vimeo_tokens:
                with contextlib.suppress(Exception):
                    vmeta = self._http_json(
                        "GET",
                        f"{VIMEO_API_BASE}/videos/{vimeo_id}",
                        headers={
                            "Authorization": f"Bearer {vtok}",
                            "Accept": "application/vnd.vimeo.*+json;version=3.4",
                        },
                        retry_auth=False,
                        retries=1,
                    )
                    if isinstance(vmeta, dict) and isinstance(vmeta.get("download"), list) and vmeta["download"]:
                        dl_list = sorted(
                            [d for d in vmeta["download"] if isinstance(d, dict) and d.get("link")],
                            key=lambda d: (int(d.get("height") or 0), int(d.get("size") or 0)),
                            reverse=True,
                        )
                        if dl_list:
                            sz = self.download_binary(str(dl_list[0]["link"]), dest)
                            self.register_cached_url(video_url, dest)
                            return sz

        player_url = video_url
        if vimeo_id and "player.vimeo.com" not in player_url:
            m_hash = re.search(r"vimeo\.com/(?:video/)?\d+/([0-9a-zA-Z]+)", video_url)
            q_hash = urllib.parse.parse_qs(urllib.parse.urlparse(video_url).query).get("h", [None])[0]
            uhash = m_hash.group(1) if m_hash else q_hash
            player_url = f"https://player.vimeo.com/video/{vimeo_id}" + (f"?h={uhash}" if uhash else "")

        r_html = self.http.get(
            player_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Referer": "https://app.kinderpedia.co/",
            },
            timeout=(15, 45),
        )
        if r_html.status_code >= 400:
            raise RuntimeError(f"HTTP {r_html.status_code} fetching Vimeo embed {redact_secrets(player_url)}")
        html = r_html.text

        m_cfg = re.search(r"window\.playerConfig\s*=\s*(\{.*?\});?\s*</script>", html, re.S)
        if not m_cfg:
            raise RuntimeError(f"Could not locate window.playerConfig in {redact_secrets(player_url)}")

        cfg = json.loads(m_cfg.group(1))
        files = (cfg.get("request") or {}).get("files") or {}

        prog = files.get("progressive") or []
        if isinstance(prog, list) and prog:
            best_prog = sorted(
                [p for p in prog if isinstance(p, dict) and p.get("url")],
                key=lambda p: (int(p.get("height") or 0), int(p.get("width") or 0)),
                reverse=True,
            )
            if best_prog:
                sz = self.download_binary(str(best_prog[0]["url"]), dest, {"Referer": "https://app.kinderpedia.co/"})
                self.register_cached_url(video_url, dest)
                return sz

        hls = files.get("hls") or {}
        cdns = hls.get("cdns") or {}
        hls_master_url = ""
        default_cdn = hls.get("default_cdn")
        if default_cdn and isinstance(cdns.get(default_cdn), dict):
            hls_master_url = str(cdns[default_cdn].get("avc_url") or cdns[default_cdn].get("url") or "")
        if not hls_master_url:
            for cdn_obj in cdns.values():
                if isinstance(cdn_obj, dict) and (cdn_obj.get("avc_url") or cdn_obj.get("url")):
                    hls_master_url = str(cdn_obj.get("avc_url") or cdn_obj.get("url"))
                    break

        if not hls_master_url:
            raise RuntimeError(f"No progressive or HLS URL found for Vimeo video {redact_secrets(video_url)}")

        if not self.ffmpeg_bin:
            with self._lock:
                if not self._warned_missing_ffmpeg:
                    self._warned_missing_ffmpeg = True
                    log(
                        f"  [WARN] 'ffmpeg' is required to remux HLS Vimeo streams into MP4 files.\n"
                        f"         Install it with:  {ffmpeg_install_hint()}\n"
                        f"         Then re-run this command to resume and download the remaining videos."
                    )
            raise FileNotFoundError("ffmpeg binary not installed")

        master_m3u8 = self._fetch_text(hls_master_url)
        best_variant_url = self._pick_best_hls_variant(hls_master_url, master_m3u8)
        tmp = dest.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part.mp4")
        if tmp.exists():
            tmp.unlink()

        if best_variant_url and "#EXT-X-MEDIA:TYPE=AUDIO" not in master_m3u8:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-headers",
                "Referer: https://app.kinderpedia.co/\r\n",
                "-i",
                best_variant_url,
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(tmp),
            ]
        else:
            best_prog_id = self._pick_best_hls_program_index(master_m3u8)
            v_map = f"0:p:{best_prog_id}:v:0" if best_prog_id is not None else "0:v:0"
            a_map = f"0:p:{best_prog_id}:a:0?" if best_prog_id is not None else "0:a:0?"
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-headers",
                "Referer: https://app.kinderpedia.co/\r\n",
                "-i",
                hls_master_url,
                "-map",
                v_map,
                "-map",
                a_map,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(tmp),
            ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            with contextlib.suppress(OSError):
                if tmp.exists():
                    tmp.unlink()
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {redact_secrets(proc.stderr[:350])}")
        sz = tmp.stat().st_size
        replace_atomic(tmp, dest)
        self.register_cached_url(video_url, dest)
        return sz

    def _fetch_text(self, url: str) -> str:
        r = self.http.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://app.kinderpedia.co/",
            },
            timeout=(15, 30),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} fetching playlist {redact_secrets(url)}")
        return r.text

    @staticmethod
    def _pick_best_hls_variant(master_url: str, m3u8: str) -> str:
        best_bw = -1
        best_uri = ""
        lines = [ln.strip() for ln in m3u8.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if ln.startswith("#EXT-X-STREAM-INF:") and i + 1 < len(lines):
                m_bw = re.search(r"BANDWIDTH=(\d+)", ln)
                bw = int(m_bw.group(1)) if m_bw else 0
                uri = lines[i + 1]
                if not uri.startswith("#") and bw > best_bw:
                    best_bw = bw
                    best_uri = urllib.parse.urljoin(master_url, uri)
        return best_uri or master_url

    @staticmethod
    def _pick_best_hls_program_index(m3u8: str) -> int | None:
        best_bw = -1
        best_idx = None
        prog_idx = 0
        for ln in m3u8.splitlines():
            ln = ln.strip()
            if ln.startswith("#EXT-X-STREAM-INF:"):
                m_res = re.search(r"RESOLUTION=(\d+)x(\d+)", ln)
                m_bw = re.search(r"BANDWIDTH=(\d+)", ln)
                pixels = (int(m_res.group(1)) * int(m_res.group(2))) if m_res else 0
                bw = int(m_bw.group(1)) if m_bw else 0
                score = pixels * 10_000_000 + bw
                if score > best_bw:
                    best_bw = score
                    best_idx = prog_idx
                prog_idx += 1
        return best_idx
