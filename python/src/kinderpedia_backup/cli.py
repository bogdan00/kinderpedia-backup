"""Command-line interface for Kinderpedia Cloud Media Backup & Face Organization."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .console import (
    LiveProgress,
    ffmpeg_install_hint,
    find_ffmpeg,
    keep_system_awake,
    log,
    set_active_progress,
)
from .faces import (
    MissingFacesBackendError,
    discover_backup_photos,
    discover_child_avatars,
    discover_reference_photos,
    load_default_recognizer,
)
from .models import ChildBackupStats, ChildContext
from .security import (
    default_state_dir,
    ensure_within_directory,
    load_dotenv,
    redact_secrets,
    write_json_atomic,
)
from .sync import (
    AuthExpiredError,
    KinderpediaClient,
    StatusTracker,
    TwoFactorRequiredError,
    backup_child_context,
    cleanup_stale_parts,
    resolve_child_folder_names,
)


def _add_sync_arguments(parser: argparse.ArgumentParser) -> None:
    parser.epilog = (
        "Exit codes: 0 = backup complete; 1 = partial (some photos/videos/documents/album manifests "
        "failed -- re-run to retry) or nothing to back up; 2 = usage/configuration error; "
        "3 = aborted because the Kinderpedia session expired and could not be refreshed; "
        "130 = interrupted."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Target backup directory (default: ~/kinderpedia_bk or $KINDERPEDIA_OUTPUT)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Parent account email (or set KINDERPEDIA_EMAIL in .env; prompts interactively if omitted)",
    )
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read parent account password from stdin (e.g. `op read ... | kinderpedia-backup sync --password-stdin`)",
    )
    parser.add_argument(
        "--code",
        default=None,
        help="Optional 2FA confirmation code (prompts automatically if required)",
    )
    parser.add_argument(
        "--child",
        action="append",
        default=[],
        help="Only back up child(ren) whose name matches this substring (repeatable)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent photo download threads (default: 6)",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=2,
        help="Concurrent video download jobs (default: 2)",
    )
    parser.add_argument(
        "--api-interval",
        type=float,
        default=3.15,
        help="Initial apiv20 pacing interval in seconds before adaptive tuning (default: 3.15)",
    )
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help=(
            "Force a full refresh: re-list every album/video/document and refetch every album's photo "
            "manifest. A plain `sync` is already incremental (it fetches listing pages until it reaches "
            "a page with no new/changed albums and only refetches albums whose (updated_at, "
            "count_images) changed)."
        ),
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Dry run: list children and media counts without downloading files",
    )
    parser.add_argument("--skip-photos", action="store_true", help="Skip photo albums")
    parser.add_argument("--skip-videos", action="store_true", help="Skip videos")
    parser.add_argument("--skip-documents", action="store_true", help="Skip documents & portfolio files")
    parser.add_argument(
        "--no-save-session",
        action="store_true",
        help="Do not persist auth tokens on disk (for shared/ephemeral machines)",
    )
    parser.add_argument(
        "--logout",
        action="store_true",
        help="Remove saved session tokens from disk and exit",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable interactive terminal progress bar")
    parser.add_argument("--no-keep-awake", action="store_true", help="Do not inhibit OS idle sleep during backup")


def _add_faces_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        nargs="?",
        choices=("scan", "ui", "materialize", "export-urls"),
        default="scan",
        help=(
            "Action to run: `scan` (incremental ONNX face scan + per-year clustering), "
            "`ui` (launch local active-learning Web UI on 127.0.0.1), "
            "`materialize` (sync flat ByChild/<Person>/ hardlinks), or "
            "`export-urls` (export Kinderpedia CDN URLs for a single child). Default: scan."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        "--backup-dir",
        dest="input",
        default=None,
        help="Backup directory containing downloaded albums (default: ~/kinderpedia_bk or $KINDERPEDIA_OUTPUT)",
    )
    parser.add_argument(
        "-r",
        "--refs",
        default=None,
        help="Optional directory of manual reference portraits <refs>/<Person>/*.jpg (default: <input>/References)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output folder for `materialize` (default: <input>/ByChild) or output file path for `export-urls`",
    )
    parser.add_argument(
        "--person",
        default=None,
        help="Filter `materialize` or target `export-urls` for a single named child/classmate",
    )
    parser.add_argument(
        "--format",
        dest="export_format",
        choices=("txt", "csv", "json"),
        default="txt",
        help="Output format for `faces export-urls` (default: txt — one URL per line)",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Also include unreviewed inclusive candidate matches (`0.33 <= sim < 0.45`) in materialize / export-urls",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="High-confidence cosine similarity threshold in (0.0, 1.0] (default: 0.45)",
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.33,
        help="Inclusive candidate similarity threshold for active-learning review (default: 0.33)",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        choices=(640, 1024),
        default=640,
        help="SCRFD detector input resolution (default: 640; use 1024 for ultra-wide outdoor group shots)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent photo inference threads during `faces scan` (default: up to 6 CPU cores)",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How matched photos are materialized into ByChild/<Person>/ (default: hardlink — zero extra disk space)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Local ONNX execution backend: auto, onnxruntime, or opencv (default: auto)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for `faces ui` (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local HTTP port for `faces ui` (default: 8765)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open a browser tab when launching `faces ui`",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable interactive terminal progress bar during `faces scan`",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect discovered album photos, avatars, and reference portraits without running ML inference",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="`faces scan`: re-detect every photo (labels are preserved by re-linking faces to their old ids)",
    )
    prune_group = parser.add_mutually_exclusive_group()
    prune_group.add_argument(
        "--prune-missing",
        dest="prune_missing",
        action="store_const",
        const=True,
        default=None,
        help="`faces scan`: always delete index rows of photos no longer on disk (default: auto-guarded)",
    )
    prune_group.add_argument(
        "--keep-missing",
        dest="prune_missing",
        action="store_const",
        const=False,
        help="`faces scan`: never delete index rows of photos missing on disk",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=0.0,
        help="`faces scan`: seconds to wait for another running scan to finish (default: 0 = refuse)",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinderpedia-backup",
        description=(
            "Kinderpedia Cloud Media Toolkit: download full-resolution photo albums, Vimeo videos, "
            "and documents (`sync`), or organize classroom albums per child using local facial recognition (`faces`)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="{sync,faces}")

    sync_parser = subparsers.add_parser(
        "sync",
        help="Download and incrementally sync photo albums, videos, and documents from Kinderpedia",
        description="Resumable, adaptive Kinderpedia cloud media downloader.",
    )
    _add_sync_arguments(sync_parser)

    faces_parser = subparsers.add_parser(
        "faces",
        help="Scan backed-up albums locally and organize photos per child via facial recognition",
        description=(
            "Local-only facial recognition & photo organizer. Works on backups created by either "
            "the Browser Bookmarklet or `kinderpedia-backup sync`."
        ),
    )
    _add_faces_arguments(faces_parser)
    return parser


def run_sync(args: argparse.Namespace) -> int:
    if args.workers < 1 or args.video_workers < 1:
        log("ERROR: --workers and --video-workers must be >= 1.")
        return 2

    dotenv, perm_warnings = load_dotenv(Path.cwd() / ".env")
    for w in perm_warnings:
        log(f"[Security Warning] {w}")

    raw_out = args.output or os.environ.get("KINDERPEDIA_OUTPUT") or dotenv.get("KINDERPEDIA_OUTPUT")
    backup_dir = Path(raw_out).expanduser().resolve() if raw_out else (Path.home() / "kinderpedia_bk").resolve()
    state_dir = default_state_dir()

    if args.logout:
        removed_any = False
        for sp in (state_dir / "session.json", backup_dir / ".session.json"):
            if sp.exists():
                sp.unlink()
                log(f"Removed saved session: {sp}")
                removed_any = True
        if not removed_any:
            log(f"No saved session found in {state_dir}")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=True)
    dotenv_bk, bk_warnings = load_dotenv(Path.cwd() / ".env", backup_dir / ".env")
    for w in bk_warnings:
        if w not in perm_warnings:
            log(f"[Security Warning] {w}")
    dotenv = dotenv_bk

    cleaned = cleanup_stale_parts(backup_dir)
    if cleaned:
        log(f"[Resume] Cleaned up {cleaned} incomplete .part temporary file(s) from previous run.")

    email = (args.email or os.environ.get("KINDERPEDIA_EMAIL") or dotenv.get("KINDERPEDIA_EMAIL") or "").strip()
    password: str | None = None
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = os.environ.get("KINDERPEDIA_PASSWORD") or dotenv.get("KINDERPEDIA_PASSWORD")
    code = args.code or os.environ.get("KINDERPEDIA_2FA_CODE") or dotenv.get("KINDERPEDIA_2FA_CODE")

    ffmpeg_bin = find_ffmpeg()
    if not args.skip_videos and not ffmpeg_bin:
        log(
            f"[Note] 'ffmpeg' was not found on PATH. Photos & documents will download normally, "
            f"but HLS Vimeo videos require ffmpeg ({ffmpeg_install_hint()})."
        )

    client = KinderpediaClient(
        state_dir=state_dir,
        persist_session=not args.no_save_session,
        pool_size=max(24, args.workers + args.video_workers + 8),
        initial_interval=args.api_interval,
        ffmpeg_bin=ffmpeg_bin,
    )

    logged_in = False
    children: list[ChildContext] = []
    interactive_tty = sys.stdin.isatty() and not args.password_stdin

    def _login(email_in: str, password_in: str | None, code_in: str | None) -> int:
        """Authenticate (prompting on a TTY when needed). Returns 0 on success or an exit code."""
        nonlocal email
        email = email_in
        pw = password_in
        if not email:
            if interactive_tty:
                email = input("Kinderpedia parent email: ").strip()
            if not email:
                log(
                    "ERROR: No saved session and no email provided. "
                    "Pass --email, set KINDERPEDIA_EMAIL in .env, or run interactively."
                )
                return 2
        if not pw:
            if interactive_tty:
                pw = getpass.getpass(f"Kinderpedia password for {email}: ")
            if not pw:
                log(
                    "ERROR: No active session and no password provided. "
                    "Prompt interactively, pipe via --password-stdin, or set KINDERPEDIA_PASSWORD in .env (chmod 600)."
                )
                return 2
        try:
            client.login(email=email, password=pw, confirmation_code=code_in)
        except TwoFactorRequiredError:
            if not interactive_tty:
                log("ERROR: Account requires a 2FA code; pass --code or KINDERPEDIA_2FA_CODE.")
                return 2
            code_prompt = input("Enter Kinderpedia 2FA confirmation code: ").strip()
            try:
                client.login(email=email, password=pw, confirmation_code=code_prompt)
            except Exception as exc:
                log(f"Authentication failed: {redact_secrets(str(exc))}")
                return 1
        except Exception as exc:
            log(f"Authentication failed: {redact_secrets(str(exc))}")
            return 1
        finally:
            pw = None
        return 0

    if not args.password_stdin and client.load_session(legacy_session_file=backup_dir / ".session.json"):
        try:
            _, children = client.get_me_and_children()
            logged_in = True
            log(f"Reusing valid saved session ({client.session_file})")
        except Exception:
            log("Saved session expired; attempting token refresh...")
            if client.refresh_jwt():
                _, children = client.get_me_and_children()
                logged_in = True

    if not logged_in:
        rc_login = _login(email, password, code)
        password = None
        if rc_login != 0:
            return rc_login
        _, children = client.get_me_and_children()

    if not children:
        log("WARNING: No children were found on this Kinderpedia parent account.")
        return 1

    if args.child:
        needles = [n.lower() for n in args.child]
        children = [c for c in children if any(n in c.name.lower() for n in needles)]
        if not children:
            log(f"ERROR: No child matched filter {args.child}.")
            return 1

    folder_names = resolve_child_folder_names(children)
    log(f"Discovered {len(children)} child/kindergarten context(s) -> saving media to {backup_dir}:")
    for c in children:
        target_folder = folder_names[(c.child_id, c.kindergarten_id)]
        log(
            f"  - {c.name} -> folder '{target_folder}' | child_id={c.child_id} "
            f"| kindergarten_id={c.kindergarten_id} | group={c.group_name} | school={c.kindergarten_name}"
        )

    progress = LiveProgress(enabled=not args.no_progress)
    set_active_progress(progress)
    status_path = (state_dir / "backup_status.json") if not args.no_save_session else None
    tracker = StatusTracker(
        status_path,
        client.rate_ctrl,
        args.workers,
        args.video_workers,
        live_progress=progress,
    )

    all_stats: list[ChildBackupStats] = []
    auth_abort: str | None = None
    reauth_used = False
    try:
        with keep_system_awake(enabled=not args.no_keep_awake and not args.discover_only):
            pending = list(children)
            while pending:
                ctx = pending[0]
                try:
                    st = backup_child_context(
                        client,
                        ctx,
                        backup_dir=backup_dir,
                        tracker=tracker,
                        child_folder_name=folder_names[(ctx.child_id, ctx.kindergarten_id)],
                        workers=args.workers,
                        video_workers=args.video_workers,
                        discover_only=args.discover_only,
                        refresh_index=args.refresh_index,
                        skip_photos=args.skip_photos,
                        skip_videos=args.skip_videos,
                        skip_documents=args.skip_documents,
                    )
                except AuthExpiredError as exc:
                    # Circuit breaker: the JWT expired mid-run and refreshing it failed. Rather than
                    # failing every remaining album one by one, re-authenticate once on a TTY or abort.
                    log(f"\n[Auth] {redact_secrets(str(exc))}")
                    if interactive_tty and not reauth_used:
                        reauth_used = True
                        progress.close()
                        log("[Auth] Please sign in again to continue (already-downloaded media is kept).")
                        if _login(email, None, None) == 0:
                            continue  # retry the same child; cached metadata makes this cheap
                    auth_abort = (
                        "Kinderpedia session expired and could not be refreshed. Already-downloaded media "
                        "is kept; sign in again (or remove the saved session with --logout) and re-run "
                        "the same command to resume."
                    )
                    break
                all_stats.append(st)
                pending.pop(0)
    except KeyboardInterrupt:
        progress.close()
        set_active_progress(None)
        log("\nBackup interrupted (Ctrl+C). Re-run the same command anytime to resume cleanly.")
        return 130
    finally:
        progress.close()
        set_active_progress(None)

    total_failures = sum(st.total_failures for st in all_stats)
    if auth_abort is not None:
        status = "FAILED"
    elif total_failures > 0:
        status = "PARTIAL"
    else:
        status = "COMPLETE"

    tracker.update(phase="complete" if status == "COMPLETE" else status.lower(), child="", force=True)
    if not args.discover_only:
        manifest_path = ensure_within_directory(backup_dir, backup_dir / "Metadata" / "backup_summary.json")
        write_json_atomic(
            manifest_path,
            {
                "completed_at": datetime.now().isoformat(),
                "status": status.lower(),
                "total_failures": total_failures,
                "aborted_reason": auth_abort,
                "adaptive_rate": client.rate_ctrl.snapshot().to_dict(),
                "children": [s.to_public_dict() for s in all_stats],
            },
        )

    if status == "COMPLETE":
        log("\n=== BACKUP COMPLETE ===")
    elif status == "PARTIAL":
        log(f"\n=== BACKUP PARTIAL: {total_failures} item(s) failed -- re-run the same command to retry ===")
    else:
        log(f"\n=== BACKUP FAILED: {auth_abort} ===")
    for st in all_stats:
        log(
            f"  {st.child_name} (KG {st.kindergarten_id}): "
            f"Albums={st.albums_count} (new photos={st.photos_downloaded}, "
            f"existing={st.photos_skipped_existing}, failed={st.photos_failed}, "
            f"album manifests failed={st.albums_failed}) | "
            f"Videos={st.videos_count} (new={st.videos_downloaded}, "
            f"existing={st.videos_skipped_existing}, failed={st.videos_failed}) | "
            f"Docs={st.documents_downloaded} new / {st.documents_skipped_existing} existing"
            f" / {st.documents_failed} failed"
        )
    if auth_abort is not None:
        return 3
    return 1 if total_failures > 0 else 0


def run_faces(args: argparse.Namespace) -> int:
    dotenv, _ = load_dotenv(Path.cwd() / ".env")
    raw_in = args.input or os.environ.get("KINDERPEDIA_OUTPUT") or dotenv.get("KINDERPEDIA_OUTPUT")
    backup_dir = Path(raw_in).expanduser().resolve() if raw_in else (Path.home() / "kinderpedia_bk").resolve()
    refs_dir = Path(args.refs).expanduser().resolve() if args.refs else (backup_dir / "References").resolve()

    if not backup_dir.exists():
        log(f"ERROR: Backup directory does not exist: {backup_dir}")
        return 2

    action = str(getattr(args, "action", "scan") or "scan")

    if args.dry_run:
        out_dir = Path(args.output).expanduser().resolve() if args.output else (backup_dir / "ByChild").resolve()
        photos = discover_backup_photos(backup_dir, exclude_dir=out_dir)
        avatars = discover_child_avatars(backup_dir)
        refs = discover_reference_photos(refs_dir)
        log(f"[Faces] Backup directory: {backup_dir} ({len(photos):,} album photo(s) discovered)")
        log(f"[Faces] Auto-seed avatars: {len(avatars)} ({', '.join(avatars) or 'none'})")
        log(f"[Faces] Reference directory: {refs_dir} ({len(refs)} manual profile(s))")
        for person, ref_files in refs.items():
            log(f"  - {person}: {len(ref_files)} reference portrait(s)")
        return 0

    db_path = ensure_within_directory(backup_dir, backup_dir / "Metadata" / "faces_index.sqlite")

    try:
        from .faces import FacesIndexDB, run_archive_face_scan, serve_faces_ui
    except MissingFacesBackendError as exc:
        log(f"[Faces] {exc}")
        return 2

    if action == "ui":
        out_dir = Path(args.output).expanduser().resolve() if args.output else (backup_dir / "ByChild").resolve()
        try:
            serve_faces_ui(
                backup_dir=backup_dir,
                output_dir=out_dir,
                host=args.host,
                port=args.port,
                open_browser=not args.no_open,
                auto_high_threshold=args.threshold,
                candidate_threshold=args.candidate_threshold,
                link_mode=args.mode,
            )
        except FileNotFoundError as exc:
            log(f"ERROR: {exc}")
            return 2
        return 0

    if action == "materialize":
        if not db_path.is_file():
            log(f"ERROR: No face index at {db_path}. Run `kinderpedia-backup faces scan` first.")
            return 2
        out_dir = Path(args.output).expanduser().resolve() if args.output else (backup_dir / "ByChild").resolve()
        with FacesIndexDB(db_path) as db:
            stats = db.materialize_flat_folders(
                backup_dir=backup_dir,
                output_dir=out_dir,
                link_mode=args.mode,
                person_filter=args.person,
                include_candidates=args.include_candidates,
                prune=True,
            )
        log(
            f"[Faces] Materialized {stats.reference_profiles} person folder(s) in {out_dir}: "
            f"photos={stats.photos_matched:,}, created={stats.links_created:,}, "
            f"existing={stats.links_skipped_existing:,}, pruned={stats.links_pruned:,}"
        )
        return 0

    if action == "export-urls":
        if not args.person:
            log("ERROR: `faces export-urls` requires `--person <Child Name>`.")
            return 2
        if not db_path.is_file():
            log(f"ERROR: No face index at {db_path}. Run `kinderpedia-backup faces scan` first.")
            return 2
        with FacesIndexDB(db_path) as db:
            try:
                disp_name, content, count = db.format_person_urls(
                    args.person,
                    fmt=args.export_format,
                    include_candidates=args.include_candidates,
                )
            except KeyError as exc:
                log(f"ERROR: {exc}")
                return 2
        if args.output:
            out_file = Path(args.output).expanduser().resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(content, encoding="utf-8")
            log(f"[Faces] Exported {count:,} URL(s) for {disp_name} -> {out_file}")
        else:
            sys.stdout.write(content)
        return 0

    # Default action == "scan":
    out_dir = Path(args.output).expanduser().resolve() if args.output else (backup_dir / "ByChild").resolve()
    try:
        recognizer = load_default_recognizer(args.backend, det_size=args.det_size)
    except MissingFacesBackendError as exc:
        log(f"[Faces] {exc}")
        return 2

    from .faces.db.scan_lock import ScanLockBusyError

    progress = LiveProgress(enabled=not args.no_progress)
    set_active_progress(progress)
    try:
        with keep_system_awake(enabled=True):
            summary = run_archive_face_scan(
                backup_dir=backup_dir,
                recognizer=recognizer,
                refs_dir=refs_dir,
                output_dir=out_dir,
                workers=args.workers,
                auto_high_threshold=args.threshold,
                candidate_threshold=args.candidate_threshold,
                progress=progress,
                force=bool(getattr(args, "force", False)),
                prune_missing=getattr(args, "prune_missing", None),
                lock_timeout_s=float(getattr(args, "lock_timeout", 0.0) or 0.0),
            )
    except ScanLockBusyError as exc:
        log(f"[Faces] {exc}")
        return 3
    except KeyboardInterrupt:
        progress.close()
        set_active_progress(None)
        log("\n[Faces] Scan interrupted (Ctrl+C). Progress is saved in SQLite — re-run anytime to resume.")
        return 130
    finally:
        progress.close()
        set_active_progress(None)

    log(
        f"[Faces] Scan complete: total={summary.photos_total:,} (new={summary.photos_scanned:,}, "
        f"cached={summary.photos_cached:,}), faces={summary.faces_total:,}, "
        f"clusters={summary.clusters_created:,}, seeded_people={summary.people_seeded}, "
        f"auto_matches={summary.auto_high_matches:,}, review_candidates={summary.candidate_matches:,}"
    )
    if summary.photos_verified or summary.photos_rekeyed or summary.photos_removed or summary.faces_relinked:
        log(
            f"[Faces] Reconcile: verified-unchanged={summary.photos_verified:,}, "
            f"re-keyed={summary.photos_rekeyed:,}, removed={summary.photos_removed:,}, "
            f"faces re-linked={summary.faces_relinked:,}, labels without face box={summary.labels_orphaned:,}"
        )
    log("[Faces] Next step: run `kinderpedia-backup faces ui` to batch-confirm children & classmates!")
    return 0


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)

    # Default to `sync` when invoked with no subcommand or directly with sync flags
    # (e.g. `kinderpedia-backup` or `kinderpedia-backup --discover-only`).
    if not effective_argv or (
        effective_argv[0] not in ("sync", "faces", "-h", "--help", "--version")
        and effective_argv[0].startswith("-")
    ):
        effective_argv = ["sync", *effective_argv]

    parser = build_arg_parser()
    args = parser.parse_args(effective_argv)

    if args.command == "faces":
        return run_faces(args)
    return run_sync(args)


def cli() -> None:
    sys.exit(main())
