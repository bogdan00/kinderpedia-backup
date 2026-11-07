"""Kinderpedia Cloud Media Backup & Face Organization package."""

from __future__ import annotations

__version__ = "1.0.0"

from .models import ChildBackupStats, ChildContext, RateSnapshot
from .sync import (
    AdaptiveRateController,
    KinderpediaClient,
    StatusTracker,
    TwoFactorRequiredError,
    backup_child_context,
)

__all__ = [
    "AdaptiveRateController",
    "ChildBackupStats",
    "ChildContext",
    "KinderpediaClient",
    "RateSnapshot",
    "StatusTracker",
    "TwoFactorRequiredError",
    "__version__",
    "backup_child_context",
]
