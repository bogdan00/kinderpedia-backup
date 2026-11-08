"""Cloud backup & synchronization engine for Kinderpedia."""

from __future__ import annotations

from .client import (
    WEB_API_BASE,
    AuthExpiredError,
    IncompleteDownloadError,
    KinderpediaClient,
    TwoFactorRequiredError,
    album_listing_signature,
    hls_playlist_duration,
)
from .engine import (
    StatusTracker,
    backup_child_context,
    candidate_image_urls,
    cleanup_stale_parts,
    merge_incremental_listing,
    set_file_mtime,
)
from .folder_map import (
    FOLDER_MAP_FILE,
    FolderMap,
    album_folder_from_leaf,
    load_folder_map,
    parse_folder_map,
    rebuild_folder_map,
    save_folder_map,
)
from .naming import (
    assign_album_folder_names,
    parse_date_prefix,
    plan_video_targets,
    resolve_child_folder_names,
    sanitize_extension,
    sanitize_name,
    split_year_month,
    stem_similarity,
)
from .rate_limit import AdaptiveRateController

__all__ = [
    "FOLDER_MAP_FILE",
    "WEB_API_BASE",
    "AdaptiveRateController",
    "AuthExpiredError",
    "FolderMap",
    "IncompleteDownloadError",
    "KinderpediaClient",
    "StatusTracker",
    "TwoFactorRequiredError",
    "album_folder_from_leaf",
    "album_listing_signature",
    "assign_album_folder_names",
    "backup_child_context",
    "candidate_image_urls",
    "cleanup_stale_parts",
    "hls_playlist_duration",
    "load_folder_map",
    "merge_incremental_listing",
    "parse_date_prefix",
    "parse_folder_map",
    "plan_video_targets",
    "rebuild_folder_map",
    "resolve_child_folder_names",
    "sanitize_extension",
    "sanitize_name",
    "save_folder_map",
    "set_file_mtime",
    "split_year_month",
    "stem_similarity",
]
